"""diagnose_why.py -- WHY does frozen ESM-2 lose to the conventional baseline?

A negative result with a mechanism is a finding; one without is a shrug. The
headline so far (median Spearman over 21 groove-held-out folds):

    A  supervised NN, BLOSUM/one-hot, no foundation model   0.277
    B  ESM-2 150M peptide + pseudo-seq, ridge               0.060
    D  peptide-only control, no allele information at all   0.074

This file asks what the frozen embedding actually lost, with five experiments:

  P0  HARNESS CHECK   re-implement arm B's ridge here and check it lands on
                      0.060. Everything below shares this code path, so if P0
                      misses, nothing below is worth reading.
  P1  PROBE TASKS     can a linear head read trivially sequence-determined
                      facts (hydropathy, mass, residue identity at P2 / P9)
                      out of the peptide embedding at all?
  P2  FACTORIAL       peptide encoding x allele encoding on the REAL task,
                      same head throughout, so the 0.277 -> 0.060 gap can be
                      attributed to one side or the other.
  P3  ALLELE GEOMETRY how much do the 75 alleles separate, and does embedding
                      distance track groove identity?
  P4  POOLING         mean over the 9 tokens vs the lossless per-residue
                      concatenation of the same forward pass.

ONE head everywhere: standardise, then ridge, alpha chosen from the same grid
as arm B by MSE on a PEPTIDE-GROUPED inner split of the training rows only.
Reusing arm B's head is the point -- the only thing that varies between cells
is the representation.

Run:    python diagnose_why.py            # everything (~20 min on an M5)
        python diagnose_why.py p1 p3       # named parts only
Writes: results_H_why.csv            every measurement, one row each
        predictions_H_why.parquet    per-row predictions for the P2/P4 cells
        diagnose_why_probe.png  _factorial.png  _alleles.png  _pooling.png

Nothing runs at import time. Reads the frozen modules, writes nothing else.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cho_factor, cho_solve

import data
import embed
import metrics
import splits
import supertypes

# arm_A is import-safe (its work is behind __main__) and holds the BLOSUM62
# matrix and pseudo-sequence map the conventional baseline actually used. Taking
# them from there rather than retyping guarantees the comparison is against the
# same encoding, not a lookalike.
import arm_A_supervised_nn as armA

AA = armA.AA
AA_IDX = armA.AA_IDX
B62 = armA.B62                       # (20, 20), already /5 as arm A scales it
EYE = np.eye(20)

ALPHAS = (1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0)   # arm B's grid
INNER_FRAC, INNER_SEED = 0.2, 0                             # arm B's inner split

RESULTS = []                 # every measured number ends up here
_T0 = time.time()


def log(*a):
    print(f"[{time.time() - _T0:7.1f}s]", *a, flush=True)


def rec(part, experiment, encoding, metric, value, n=np.nan, note=""):
    RESULTS.append(dict(part=part, experiment=experiment, encoding=encoding,
                        metric=metric, value=value, n=n, note=note))


# ===========================================================================
# ridge: one head, used by every experiment
# ===========================================================================

def _solve(Xf, Yf, alphas):
    """Standardise + ridge for a list of alphas. Returns (predict_fns, mus).

    Y may be (n,) or (n, k) so the same solver does regression and the
    least-squares linear classifier used by the probe tasks.
    """
    # float64 statistics even when Xf is stored float32: every chunk below then
    # upcasts on the subtraction, so the Gram is accumulated in full precision
    # while the feature matrix itself stays half the size in memory.
    mu, sd = Xf.mean(0, dtype=np.float64), Xf.std(0, dtype=np.float64)
    sd[sd < 1e-8] = 1.0
    xc = np.zeros_like(mu)      # standardising by the fit mean already centres
    Y = Yf if Yf.ndim == 2 else Yf[:, None]
    yc = Y.mean(0)
    # Accumulate the Gram in row chunks. Materialising the whole standardised
    # copy costs ~1.2 GB at d=5760 and this machine is deep in swap.
    d = Xf.shape[1]
    G = np.zeros((d, d))
    b = np.zeros((d, Y.shape[1]))
    for i in range(0, len(Xf), 4096):
        C = (Xf[i:i + 4096] - mu) / sd
        G += C.T @ C
        b += C.T @ (Y[i:i + 4096] - yc)
    out = {}
    for a in alphas:
        c = cho_factor(G + a * np.eye(d), lower=True, check_finite=False)
        out[a] = cho_solve(c, b, check_finite=False)
    return out, (mu, sd, xc, yc)


def _apply(X, coef, p):
    mu, sd, xc, yc = p
    return ((X - mu) / sd - xc) @ coef + yc


def ridge_fit_predict(Xtr, ytr, Xte, groups=None, alphas=ALPHAS):
    """Arm B's head. alpha by MSE on a peptide-grouped inner split of TRAIN.

    groups: integer peptide id per training row. Rows sharing a peptide never
    straddle the inner boundary, or alpha is picked against a leaked
    validation set. groups=None falls back to a plain row split (used only
    where the rows ARE the peptides, i.e. the probe tasks).
    """
    n = len(Xtr)
    g = np.arange(n) if groups is None else np.asarray(groups)
    ug = np.unique(g)
    rng = np.random.default_rng(INNER_SEED)
    val_g = set(rng.permutation(len(ug))[: max(1, int(round(len(ug) * INNER_FRAC)))].tolist())
    gi = np.searchsorted(ug, g)
    va = np.fromiter((x in val_g for x in gi), dtype=bool, count=n)
    tr = ~va

    coefs, p = _solve(Xtr[tr], ytr[tr], alphas)
    best, best_mse = alphas[0], np.inf
    for a in alphas:
        e = _apply(Xtr[va], coefs[a], p) - (ytr[va] if ytr.ndim == 2 else ytr[va][:, None])
        m = float(np.mean(e ** 2))
        if m < best_mse:
            best, best_mse = a, m

    coefs, p = _solve(Xtr, ytr, [best])
    pred = _apply(Xte, coefs[best], p)
    return (pred.ravel() if ytr.ndim == 1 else pred), best


def _check_solver():
    """Our ridge must equal sklearn's, or every number below is our bug."""
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(0)
    X, y = rng.standard_normal((400, 30)), rng.standard_normal(400)
    mu, sd = X.mean(0), X.std(0)
    ref = Ridge(alpha=10.0).fit((X - mu) / sd, y).predict((X[:50] - mu) / sd)
    coefs, p = _solve(X, y, [10.0])
    ours = _apply(X[:50], coefs[10.0], p).ravel()
    err = float(np.abs(ours - ref).max())
    log(f"P0 solver vs sklearn Ridge: max |diff| = {err:.3e}")
    assert err < 1e-8, "ridge solver disagrees with sklearn"
    rec("P0", "solver self-check", "-", "max_abs_diff_vs_sklearn_ridge", err)


# ===========================================================================
# encodings
# ===========================================================================

_C: dict = {}


def _tok_embed_150M():
    """Per-residue ESM-2 150M token embeddings for every distinct 9-mer.

    (5633, 9, 640). embed.py only ever stored the MEAN over these 9 positions,
    which is the thing P4 is testing, so the tokens are recomputed here. The
    mean of what we compute is checked against emb_peptides.npy to prove it is
    the same forward pass and not a second, differently-configured model.
    """
    if "tok150" in _C:
        return _C["tok150"]
    import torch
    from transformers import AutoModel, AutoTokenizer
    peps = sorted(data_df().Pep.unique())
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(embed.MODEL)
    mdl = AutoModel.from_pretrained(embed.MODEL).eval().to(dev)
    out, t0 = [], time.time()
    with torch.no_grad():
        for i in range(0, len(peps), 256):
            enc = tok(peps[i:i + 256], return_tensors="pt", padding=True).to(dev)
            h = mdl(**enc).last_hidden_state          # [BOS, 9 residues, EOS]
            out.append(h[:, 1:10, :].float().cpu().numpy())
    T = np.concatenate(out).astype(np.float64)
    log(f"P4 token embeddings {T.shape} in {time.time() - t0:.1f}s on {dev}")

    ref, ridx = embed.load("peptides")
    order = np.array([ridx[p] for p in peps])
    err = float(np.abs(T.mean(1) - ref[order]).max())
    log(f"P4 mean(tokens) vs stored emb_peptides.npy: max |diff| = {err:.3e}")
    # Recorded under its own part label: several parts build these tokens, and
    # the results file merges by part, so sharing "P4" here would make any
    # other part's merge delete P4's rows.
    rec("TOK", "token embeddings reproduce the stored mean-pool", "esm150",
        "max_abs_diff", err, note="same forward pass, confirmed")
    assert err < 1e-4, "recomputed tokens do not reproduce the stored mean-pool"
    _C["tok150"] = (T, {p: i for i, p in enumerate(peps)})
    return _C["tok150"]


def data_df():
    if "df" not in _C:
        _C["df"] = data.load()
    return _C["df"]


def _seq_block(seqs, tables):
    """(len(seqs), L*20*len(tables)) encoding of equal-length sequences."""
    idx = np.array([[AA_IDX[c] for c in s] for s in seqs])
    return np.concatenate([t[idx].reshape(len(seqs), -1) for t in tables], axis=1)


def peptide_table(kind):
    """kind -> (matrix, {peptide: row}). One row per DISTINCT 9-mer."""
    if kind in _C:
        return _C[kind]
    peps = sorted(data_df().Pep.unique())
    index = {p: i for i, p in enumerate(peps)}
    if kind == "esm150_mean":
        M, i2 = embed.load("peptides")
        M = M[[i2[p] for p in peps]].astype(np.float64)
    elif kind == "esm650_mean":
        M = np.load("emb650_peptides.npy")
        i2 = json.load(open("emb650_peptides_index.json"))
        M = M[[i2[p] for p in peps]].astype(np.float64)
    elif kind == "esm150_concat":
        # kept float32 -- which is the model's own output precision, so this
        # is lossless -- because 5,760 columns x 26k rows in float64 is the
        # one allocation big enough to matter on this machine.
        T, i2 = _tok_embed_150M()
        M = T[[i2[p] for p in peps]].reshape(len(peps), -1).astype(np.float32)
    elif kind == "onehot":
        M = _seq_block(peps, [EYE])
    elif kind == "blosum":
        M = _seq_block(peps, [B62])
    else:
        raise KeyError(kind)
    _C[kind] = (M, index)
    return _C[kind]


def allele_table(kind):
    """kind -> (matrix, {allele: row}). One row per allele."""
    key = "A/" + kind
    if key in _C:
        return _C[key]
    pm = armA.pseudo_map()
    names = sorted(data_df().HLA.unique())
    index = {a: i for i, a in enumerate(names)}
    if kind == "esm150_pseudo":
        M, i2 = embed.load("pseudo")
        M = M[[i2[a] for a in names]].astype(np.float64)
    elif kind == "esm650_pseudo":
        M = np.load("emb650_pseudo.npy")
        i2 = json.load(open("emb650_pseudo_index.json"))
        M = M[[i2[a] for a in names]].astype(np.float64)
    elif kind == "esm150_chain":
        M, i2 = embed.load("alleles")
        M = M[[i2[a] for a in names]].astype(np.float64)
    elif kind == "onehot_pseudo":
        M = _seq_block([pm[a] for a in names], [EYE])
    elif kind == "blosum_pseudo":
        M = _seq_block([pm[a] for a in names], [B62])
    elif kind == "identity":
        M = np.eye(len(names))           # allele ID only: cannot transfer
    elif kind == "none":
        M = np.zeros((len(names), 0))
    else:
        raise KeyError(kind)
    _C[key] = (M, index)
    return _C[key]


def featurize(sub, pep_kind, hla_kind):
    P, pi = peptide_table(pep_kind)
    blocks = [P[[pi[p] for p in sub.Pep]]]
    if hla_kind != "none":
        A, ai = allele_table(hla_kind)
        blocks.append(A[[ai[h] for h in sub.HLA]])
    return np.hstack(blocks)


def _rank_reduce(M, tol=1e-9):
    """Exact lossless reduction of a table with few distinct rows.

    75 alleles span at most a 75-dimensional subspace however wide the
    encoding is, so projecting onto that subspace changes no inner product and
    no ridge solution, but turns a 21,760-column allele block into 75 columns.
    Returns (reduced, max reconstruction error) so losslessness is measured,
    not asserted.
    """
    mu = M.mean(0)
    U, s, Vt = np.linalg.svd(M - mu, full_matrices=False)
    k = int((s > tol * max(1.0, s[0])).sum())
    R = (U[:, :k] * s[:k])
    err = float(np.abs(R @ Vt[:k] + mu - M).max())
    return R, err


# ===========================================================================
# the real task: 21 groove-held-out folds, one row per fold
# ===========================================================================

def _pep_groups(sub):
    return pd.factorize(sub.Pep.to_numpy())[0]


def run_cells(cells, tag, save_predictions=False):
    """Each cell is (label, pep_kind, hla_kind). Returns {label: per-fold Series}.

    Protocol identical to run_experiment.run_arm: folds from
    splits.choose_held_out, metrics.spearman_per_allele per held-out allele,
    censored='tied', then the MEDIAN over the alleles of the fold. Ridge is
    deterministic so one fit per fold replaces run_arm's 5 seeds; the arms it
    is compared against are reproduced in P0 under exactly this loop.
    """
    df = data_df()
    folds = splits.choose_held_out(df)
    out, preds = {}, []
    for label, pk, hk in cells:
        t0, per_fold = time.time(), {}
        for fname, alleles in folds:
            tr_idx, te_idx = splits.split_by_allele(df, alleles)
            tr, te = df.loc[tr_idx], df.loc[te_idx]
            Xtr, Xte = featurize(tr, pk, hk), featurize(te, pk, hk)
            if Xtr.shape[1] > 8000:          # allele blocks are low-rank; use it
                R, err = _rank_reduce(np.vstack([Xtr, Xte]))
                assert err < 1e-6, f"rank reduction lost information ({err:.2e})"
                Xtr, Xte = R[:len(Xtr)], R[len(Xtr):]
            pred, alpha = ridge_fit_predict(Xtr, tr.y.to_numpy(), Xte,
                                            groups=_pep_groups(tr))
            rho = metrics.spearman_per_allele(te, pred, censored="tied")
            per_fold[fname] = np.nan if not len(rho) else float(np.median(rho))
            if save_predictions:
                preds.append(pd.DataFrame({
                    "cell": label, "fold_name": fname,
                    "row_id": te.index.to_numpy(np.int64),
                    "HLA": te.HLA.values, "Pep": te.Pep.values,
                    "y_true": te.y.to_numpy(), "y_pred": pred,
                    "alpha": alpha}))
        s = pd.Series(per_fold)
        out[label] = s
        rec(tag, label, f"pep={pk} hla={hk}", "median_spearman_over_folds",
            float(s.median()), n=int(s.notna().sum()),
            note=f"IQR [{s.quantile(.25):.3f}, {s.quantile(.75):.3f}] "
                 f"worst {s.min():.3f}  {time.time() - t0:.0f}s")
        log(f"{tag} {label:<34s} median rho {s.median():+.4f}  "
            f"IQR [{s.quantile(.25):+.3f},{s.quantile(.75):+.3f}]  "
            f"worst {s.min():+.3f}  ({time.time() - t0:.0f}s)")
    if save_predictions and preds:
        pd.concat(preds, ignore_index=True).to_parquet("predictions_H_why.parquet")
    return out


# ===========================================================================
# P0  harness check
# ===========================================================================

def p0():
    log("=" * 72)
    log("P0  HARNESS CHECK -- does this file's ridge reproduce arm B and arm D?")
    log("=" * 72)
    _check_solver()
    got = run_cells([("arm B replica (ESM pep + ESM pseudo)", "esm150_mean", "esm150_pseudo"),
                     ("arm D replica (ESM pep only)", "esm150_mean", "none")], "P0")
    for label, published in [("arm B replica (ESM pep + ESM pseudo)", 0.060),
                             ("arm D replica (ESM pep only)", 0.074)]:
        m = float(got[label].median())
        log(f"P0 {label:<38s} here {m:+.4f}   published {published:+.3f}")
        rec("P0", label, "-", "published_for_comparison", published)
    return got


# ===========================================================================
# P1  probe tasks -- is the information in there at all?
# ===========================================================================

KD = dict(zip("ARNDCQEGHILKMFPSTWYV",
              [1.8, -4.5, -3.5, -3.5, 2.5, -3.5, -3.5, -0.4, -3.2, 4.5,
               3.8, -3.9, 1.9, 2.8, -1.6, -0.8, -0.7, -0.9, -1.3, 4.2]))
# average residue masses in Da (amino acid minus water)
MASS = dict(zip("GASPVTCLINDQKEMHFRYW",
                [57.0519, 71.0788, 87.0782, 97.1167, 99.1326, 101.1051, 103.1388,
                 113.1594, 113.1594, 114.1038, 115.0886, 128.1307, 128.1741,
                 129.1155, 131.1926, 137.1411, 147.1766, 156.1875, 163.1760,
                 186.2132]))

PROBE_ENCODINGS = ["onehot", "blosum", "esm150_mean", "esm650_mean", "esm150_concat"]


def p1():
    log("=" * 72)
    log("P1  PROBE TASKS -- can a linear head read sequence facts off the embedding?")
    log("=" * 72)
    peps = sorted(data_df().Pep.unique())
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(peps))
    cut = int(len(peps) * 0.8)
    tr, te = perm[:cut], perm[cut:]
    log(f"P1 {len(peps)} distinct 9-mers, {len(tr)} train / {len(te)} test, split by peptide")

    hyd = np.array([np.mean([KD[c] for c in p]) for p in peps])
    mw = np.array([sum(MASS[c] for c in p) for p in peps])
    pos = {k: np.array([AA_IDX[p[k - 1]] for p in peps]) for k in (2, 9)}

    reg = {"hydropathy (mean Kyte-Doolittle)": hyd, "molecular weight (Da)": mw}
    cls = {f"residue identity at P{k}": pos[k] for k in (2, 9)}
    for k in (2, 9):
        c = np.bincount(pos[k][te], minlength=20)
        base = c.max() / c.sum()
        rec("P1", f"residue identity at P{k}", "majority class", "accuracy", float(base),
            n=len(te), note="chance level for this probe")
        log(f"P1 chance (majority class) for P{k} identity: {base:.3f}")

    table = {}
    for enc in PROBE_ENCODINGS:
        X, _ = peptide_table(enc)
        row = {}
        for name, y in reg.items():
            pred, a = ridge_fit_predict(X[tr], y[tr], X[te])
            r2 = 1 - np.sum((pred - y[te]) ** 2) / np.sum((y[te] - y[te].mean()) ** 2)
            rho = stats.spearmanr(pred, y[te]).statistic
            row[name] = r2
            rec("P1", name, enc, "r2", float(r2), n=len(te), note=f"spearman {rho:.3f} alpha {a:g}")
        for name, y in cls.items():
            Y = np.eye(20)[y]
            pred, a = ridge_fit_predict(X[tr], Y[tr], X[te])
            acc = float((pred.argmax(1) == y[te]).mean())
            row[name] = acc
            rec("P1", name, enc, "accuracy", acc, n=len(te), note=f"alpha {a:g}")
        table[enc] = row
        log(f"P1 {enc:<16s} " + "  ".join(f"{k.split('(')[0].strip()[:14]}={v:.3f}"
                                          for k, v in row.items()))
    _fig_probe(table, cls, pos, te)
    return table


# ===========================================================================
# P2  factorial -- which SIDE of the pair does the embedding lose?
# ===========================================================================

def p2():
    log("=" * 72)
    log("P2  FACTORIAL -- peptide encoding x allele encoding, one head")
    log("=" * 72)
    peps = ["esm150_mean", "blosum", "onehot"]
    hlas = ["esm150_pseudo", "blosum_pseudo", "onehot_pseudo", "identity", "none"]
    cells = [(f"{p} | {h}", p, h) for p in peps for h in hlas]
    cells += [("esm650_mean | esm650_pseudo", "esm650_mean", "esm650_pseudo")]
    got = run_cells(cells, "P2", save_predictions=True)
    _fig_factorial(got, peps, hlas)
    return got


# ===========================================================================
# P3  allele geometry
# ===========================================================================

def p3():
    log("=" * 72)
    log("P3  ALLELE GEOMETRY -- how far apart are the 75 grooves?")
    log("=" * 72)
    names = sorted(data_df().HLA.unique())
    pm = armA.pseudo_map()
    iden = np.array([[sum(x == y for x, y in zip(pm[a], pm[b])) / 34 for b in names]
                     for a in names])
    iu = np.triu_indices(len(names), 1)

    stats_rows = {}
    for kind in ["esm150_chain", "esm150_pseudo", "esm650_pseudo",
                 "blosum_pseudo", "onehot_pseudo", "identity"]:
        M, _ = allele_table(kind)
        Z = M / np.linalg.norm(M, axis=1, keepdims=True)
        cos = Z @ Z.T
        mc = float(cos[iu].mean())
        # Participation ratio of the centred covariance spectrum: how many
        # directions the 75 alleles actually occupy (max 74). Taken from the
        # singular values of the centred table, which is exact at any width.
        s = np.linalg.svd(M - M.mean(0), compute_uv=False)
        ev = s ** 2
        eff = float(ev.sum() ** 2 / (ev ** 2).sum()) if ev.sum() > 0 else np.nan
        # does embedding distance track groove identity?
        mantel = float(stats.spearmanr(1 - cos[iu], 1 - iden[iu]).statistic)
        stats_rows[kind] = dict(mean_cos=mc, eff_dim=eff, mantel=mantel)
        for k, v in stats_rows[kind].items():
            rec("P3", "allele separation", kind, k, v, n=len(names))
        log(f"P3 {kind:<16s} mean pairwise cos {mc:+.4f}   effective dims {eff:5.1f}"
            f"   rho(embedding dist, groove dist) {mantel:+.3f}")

    # Linear probe: can the 34 polymorphic contact residues be read back out of
    # the allele embedding? Leave-one-allele-out, so it measures transfer to an
    # unseen allele, which is what the groove folds demand.
    log("P3 leave-one-allele-out probe: recover the 34 contact residues")
    Y = np.stack([[AA_IDX[c] for c in pm[a]] for a in names])
    maj = float(np.mean([np.bincount(Y[:, j]).max() / len(names) for j in range(34)]))
    rec("P3", "recover 34 contact residues (LOAO)", "majority class", "accuracy", maj,
        n=34 * len(names), note="chance level")
    log(f"P3   chance (per-position majority residue): {maj:.3f}")
    for kind in ["esm150_pseudo", "esm650_pseudo", "esm150_chain"]:
        M, _ = allele_table(kind)
        hit = []
        for i in range(len(names)):
            m = np.ones(len(names), bool)
            m[i] = False
            Yo = np.concatenate([np.eye(20)[Y[m, j]] for j in range(34)], axis=1)
            pred, _ = ridge_fit_predict(M[m], Yo, M[~m])
            p = pred.reshape(34, 20)
            hit.append((p.argmax(1) == Y[i]).mean())
        acc = float(np.mean(hit))
        rec("P3", "recover 34 contact residues (LOAO)", kind, "accuracy", acc,
            n=34 * len(names))
        log(f"P3   {kind:<16s} accuracy {acc:.3f}")
    _fig_alleles(stats_rows, iden, iu, names)
    return stats_rows


# ===========================================================================
# P4  pooling -- is mean over the 9 tokens the thing that breaks it?
# ===========================================================================

def p4():
    log("=" * 72)
    log("P4  POOLING -- mean over 9 tokens vs lossless per-residue concatenation")
    log("=" * 72)
    _tok_embed_150M()
    cells = [
        ("mean-pool | no allele", "esm150_mean", "none"),
        ("per-residue concat | no allele", "esm150_concat", "none"),
        ("mean-pool | ESM pseudo", "esm150_mean", "esm150_pseudo"),
        ("per-residue concat | ESM pseudo", "esm150_concat", "esm150_pseudo"),
        ("per-residue concat | one-hot pseudo", "esm150_concat", "onehot_pseudo"),
    ]
    got = run_cells(cells, "P4")
    _fig_pooling(got)
    return got


# ===========================================================================
# P5  is the ridge head even CAPABLE of using an allele encoding?
# ===========================================================================
#
# [peptide | allele] concatenated into an ADDITIVE model has no peptide-allele
# interaction term. Every row of a given allele carries the same allele block,
# so that block contributes a per-allele CONSTANT at test time -- and a
# per-allele Spearman is invariant to a per-allele constant. If that is right,
# the allele column of the P2 factorial is not measuring groove transfer at
# all, and no encoding can win it. P5a measures the invariance; P5b removes
# the limitation with an explicit bilinear interaction block and re-asks the
# question with a head that can actually answer it.

K_INTER = 32          # PCA components per side before the outer product


def _pca_fit(X, k):
    """Top-k PCA basis, fitted on TRAIN rows only. Via the d x d covariance
    eigendecomposition, which is the same subspace as the SVD and far cheaper
    here (n = 26k rows, d <= 680 columns)."""
    mu = X.mean(0)
    G = np.zeros((X.shape[1],) * 2)
    for i in range(0, len(X), 4096):           # chunked: X can be 26k x 5760
        C = X[i:i + 4096] - mu
        G += C.T @ C
    if G.shape[0] > 2000:                      # only the top k are needed
        from scipy.sparse.linalg import eigsh
        V = eigsh(G, k=k, which="LA")[1][:, ::-1]
    else:
        V = np.linalg.eigh(G)[1][:, ::-1][:, :k]
    return lambda A: (A - mu) @ V


def p5():
    log("=" * 72)
    log("P5  ADDITIVITY -- can a concatenated ridge express allele specificity?")
    log("=" * 72)
    df = data_df()
    folds = splits.choose_held_out(df)

    # ---- P5a: swap the test-time allele block for a constant -------------
    diffs = []
    for pk, hk in [("onehot", "onehot_pseudo"), ("esm150_mean", "esm150_pseudo")]:
        worst = 0.0
        for fname, alleles in folds:
            tr_idx, te_idx = splits.split_by_allele(df, alleles)
            tr, te = df.loc[tr_idx], df.loc[te_idx]
            Xtr, Xte = featurize(tr, pk, hk), featurize(te, pk, hk)
            npep = peptide_table(pk)[0].shape[1]
            Xflat = Xte.copy()
            Xflat[:, npep:] = Xtr[:, npep:].mean(0)       # one fixed pseudo allele
            p1_, a = ridge_fit_predict(Xtr, tr.y.to_numpy(), Xte, groups=_pep_groups(tr))
            p2_, _ = ridge_fit_predict(Xtr, tr.y.to_numpy(), Xflat, groups=_pep_groups(tr))
            r1 = metrics.spearman_per_allele(te, p1_, censored="tied")
            r2 = metrics.spearman_per_allele(te, p2_, censored="tied")
            if len(r1):
                worst = max(worst, float(np.abs(r1.values - r2.values).max()))
        diffs.append((f"{pk} | {hk}", worst))
        rec("P5", "per-allele rho with the test allele block blanked", f"{pk} | {hk}",
            "max_abs_change_in_per_allele_rho", worst, n=len(folds),
            note="0 means the allele block cannot affect within-allele ranking")
        log(f"P5a {pk} | {hk}: blanking the test allele block changes per-allele "
            f"rho by at most {worst:.2e}")

    # ---- P5b: same encodings, but with a bilinear interaction block ------
    log(f"P5b adding a {K_INTER}x{K_INTER} peptide-allele interaction block")
    out = {}
    for pk, hk in [("onehot", "onehot_pseudo"), ("onehot", "esm150_pseudo"),
                   ("onehot", "identity"), ("esm150_mean", "esm150_pseudo"),
                   ("esm150_mean", "onehot_pseudo"), ("esm150_mean", "identity")]:
        t0, per_fold = time.time(), {}
        for fname, alleles in folds:
            tr_idx, te_idx = splits.split_by_allele(df, alleles)
            tr, te = df.loc[tr_idx], df.loc[te_idx]
            P, pi = peptide_table(pk)
            A, ai = allele_table(hk)
            ptr, pte = P[[pi[p] for p in tr.Pep]], P[[pi[p] for p in te.Pep]]
            atr, ate = A[[ai[h] for h in tr.HLA]], A[[ai[h] for h in te.HLA]]
            fp, fa = _pca_fit(ptr, K_INTER), _pca_fit(atr, K_INTER)      # TRAIN only
            def build(p, a):
                u, v = fp(p), fa(a)
                return np.hstack([p, a, (u[:, :, None] * v[:, None, :]).reshape(len(p), -1)])
            pred, _ = ridge_fit_predict(build(ptr, atr), tr.y.to_numpy(),
                                        build(pte, ate), groups=_pep_groups(tr))
            rho = metrics.spearman_per_allele(te, pred, censored="tied")
            per_fold[fname] = np.nan if not len(rho) else float(np.median(rho))
        s = pd.Series(per_fold)
        label = f"{pk} | {hk} + interaction"
        out[label] = s
        rec("P5", label, f"pep={pk} hla={hk} bilinear k={K_INTER}",
            "median_spearman_over_folds", float(s.median()), n=int(s.notna().sum()),
            note=f"IQR [{s.quantile(.25):.3f}, {s.quantile(.75):.3f}] "
                 f"worst {s.min():.3f}  {time.time() - t0:.0f}s")
        log(f"P5b {label:<42s} median rho {s.median():+.4f}  "
            f"IQR [{s.quantile(.25):+.3f},{s.quantile(.75):+.3f}]  ({time.time() - t0:.0f}s)")
    _fig_interaction(out)
    return out


# ===========================================================================
# P7  the best shot: everything the diagnosis says frozen ESM-2 needs
# ===========================================================================
#
# P4 says the peptide side needs per-residue tokens, not a mean. P5 says the
# head needs a peptide-allele interaction. Neither is a change to the
# foundation model -- both are changes to how its output is consumed. This is
# the strongest frozen-ESM-2 configuration this diagnosis can justify, and it
# is the number that decides whether the negative result is about the model or
# about the wrapper around it.
#
# The design matrix here is assembled in float32 (ESM-2's own output
# precision, so the embedding blocks lose nothing; only the interaction
# products are rounded, at 1e-7 relative). The Gram matrix is still
# accumulated in float64. This is a memory concession: at 7,424 columns the
# float64 matrix does not fit alongside everything else on this machine.

def p7():
    log("=" * 72)
    log("P7  BEST SHOT -- per-residue tokens + peptide-allele interaction")
    log("=" * 72)
    df = data_df()
    folds = splits.choose_held_out(df)
    out = {}
    for label, pk, hk in [
        ("ESM concat | ESM pseudo + interaction", "esm150_concat", "esm150_pseudo"),
        ("one-hot | one-hot pseudo + interaction (conventional)", "onehot", "onehot_pseudo"),
    ]:
        t0, per_fold = time.time(), {}
        for fname, alleles in folds:
            tr_idx, te_idx = splits.split_by_allele(df, alleles)
            tr, te = df.loc[tr_idx], df.loc[te_idx]
            P, pi = peptide_table(pk)
            A, ai = allele_table(hk)
            ptr, pte = P[[pi[p] for p in tr.Pep]], P[[pi[p] for p in te.Pep]]
            atr, ate = A[[ai[h] for h in tr.HLA]], A[[ai[h] for h in te.HLA]]
            fp, fa = _pca_fit(ptr, K_INTER), _pca_fit(atr, K_INTER)
            def build(p, a):
                u, v = fp(p), fa(a)
                inter = (u[:, :, None] * v[:, None, :]).reshape(len(p), -1)
                return np.hstack([p.astype(np.float32), a.astype(np.float32),
                                  inter.astype(np.float32)])
            pred, _ = ridge_fit_predict(build(ptr, atr), tr.y.to_numpy(),
                                        build(pte, ate), groups=_pep_groups(tr))
            rho = metrics.spearman_per_allele(te, pred, censored="tied")
            per_fold[fname] = np.nan if not len(rho) else float(np.median(rho))
        s = pd.Series(per_fold)
        out[label] = s
        rec("P7", label, f"pep={pk} hla={hk} bilinear k={K_INTER}",
            "median_spearman_over_folds", float(s.median()), n=int(s.notna().sum()),
            note=f"IQR [{s.quantile(.25):.3f}, {s.quantile(.75):.3f}] "
                 f"worst {s.min():.3f}  {time.time() - t0:.0f}s")
        log(f"P7 {label:<54s} median rho {s.median():+.4f}  "
            f"IQR [{s.quantile(.25):+.3f},{s.quantile(.75):+.3f}]  ({time.time() - t0:.0f}s)")
    return out


# ===========================================================================
# P6  are the factorial differences bigger than fold-to-fold noise?
# ===========================================================================
#
# Re-scored from predictions_H_why.parquet, so no model is refitted and the
# paired structure is exact: the same 21 folds, the same held-out alleles,
# only the representation differs. Wilcoxon signed-rank on the 21 paired fold
# medians. n=21 is small, so the p-value is a sanity check on the sign, not a
# headline.

CONTRASTS = [
    ("PEPTIDE side, allele held fixed at one-hot pseudo",
     "onehot | onehot_pseudo", "esm150_mean | onehot_pseudo"),
    ("PEPTIDE side, allele held fixed at ESM pseudo",
     "onehot | esm150_pseudo", "esm150_mean | esm150_pseudo"),
    ("PEPTIDE side, BLOSUM vs ESM, allele one-hot pseudo",
     "blosum | onehot_pseudo", "esm150_mean | onehot_pseudo"),
    ("ALLELE side, ESM pseudo vs allele identity (one-hot peptide)",
     "onehot | esm150_pseudo", "onehot | identity"),
    ("ALLELE side, one-hot pseudo vs allele identity (one-hot peptide)",
     "onehot | onehot_pseudo", "onehot | identity"),
    ("ALLELE side, ESM pseudo vs NO allele block (ESM peptide)",
     "esm150_mean | esm150_pseudo", "esm150_mean | none"),
    ("MODEL SIZE, ESM-2 650M vs 150M (both sides)",
     "esm650_mean | esm650_pseudo", "esm150_mean | esm150_pseudo"),
]


def p6():
    log("=" * 72)
    log("P6  PAIRED TESTS over the 21 folds, re-scored from stored predictions")
    log("=" * 72)
    df = data_df()
    pr = pd.read_parquet("predictions_H_why.parquet")
    per_fold = {}
    for cell, sub in pr.groupby("cell", sort=False):
        d = {}
        for fname, s in sub.groupby("fold_name", sort=False):
            te = df.loc[s.row_id.to_numpy()]
            rho = metrics.spearman_per_allele(te, s.y_pred.to_numpy(), censored="tied")
            d[fname] = np.nan if not len(rho) else float(np.median(rho))
        per_fold[cell] = pd.Series(d)
    for label, a, b in CONTRASTS:
        if a not in per_fold or b not in per_fold:
            log(f"P6 SKIP {label}: missing a cell")
            continue
        x, y = per_fold[a].align(per_fold[b], join="inner")
        m = x.notna() & y.notna()
        d = (x[m] - y[m]).to_numpy()
        p = float(stats.wilcoxon(d).pvalue)
        rec("P6", label, f"{a}  minus  {b}", "median_paired_difference",
            float(np.median(d)), n=int(m.sum()),
            note=f"wins {int((d > 0).sum())}/{int(m.sum())} folds, wilcoxon p={p:.4g}")
        log(f"P6 {label}\n      {a}  minus  {b}"
            f"\n      median diff {np.median(d):+.4f}   wins {int((d > 0).sum())}/{int(m.sum())}"
            f" folds   wilcoxon p = {p:.4g}")
    return per_fold


# ===========================================================================
# figures
# ===========================================================================

BG, FG, MUTED = "#0d0d0f", "#f2f2f2", "#8a8a93"
GOOD, BAD, WARM = "#4ad3a0", "#6f7380", "#e8b04b"


def _axes(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(MUTED)
    ax.tick_params(colors=FG, labelsize=11)
    return ax


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _fig_probe(table, cls, pos, te):
    plt = _plt()
    tasks = list(next(iter(table.values())).keys())
    encs = list(table)
    fig, axes = plt.subplots(1, len(tasks), figsize=(19, 6.5), facecolor=BG)
    for ax, t in zip(axes, tasks):
        _axes(ax)
        v = [table[e][t] for e in encs]
        cols = [GOOD if e.startswith("esm") else BAD for e in encs]
        ax.barh(range(len(encs)), v, color=cols, height=.62)
        ax.set_yticks(range(len(encs)))
        ax.set_yticklabels(encs, color=FG)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        for i, x in enumerate(v):
            ax.text(min(x, 1.0) + .02, i, f"{x:.3f}", color=FG, va="center", fontsize=11)
        if t in cls:
            k = int(t[-1])
            c = np.bincount(pos[k][te], minlength=20)
            ax.axvline(c.max() / c.sum(), color=WARM, lw=1.4, ls="--")
        ax.set_title(t, color=FG, fontsize=13, pad=12)
        ax.set_xlabel("R2" if t not in cls else "accuracy", color=MUTED)
    fig.suptitle("Can a linear head read a sequence fact off the peptide representation?",
                 color=FG, fontsize=17, y=.99)
    fig.text(.5, .02, "1,127 held-out 9-mers. dashed line = majority-class chance. "
                      "green = ESM-2, grey = plain encoding.", color=MUTED, ha="center", fontsize=11)
    fig.tight_layout(rect=[0, .05, 1, .95])
    fig.savefig("diagnose_why_probe.png", dpi=150, facecolor=BG)
    log("wrote diagnose_why_probe.png")


def _fig_factorial(got, peps, hlas):
    plt = _plt()
    M = np.array([[got.get(f"{p} | {h}", pd.Series([np.nan])).median() for h in hlas]
                  for p in peps])
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG)
    _axes(ax)
    im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0, vmax=max(0.3, np.nanmax(M)))
    ax.set_xticks(range(len(hlas)))
    ax.set_xticklabels([h.replace("_", "\n") for h in hlas], color=FG)
    ax.set_yticks(range(len(peps)))
    ax.set_yticklabels(peps, color=FG)
    for i in range(len(peps)):
        for j in range(len(hlas)):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                    color="#000000" if M[i, j] > .18 else FG, fontsize=14, fontweight="bold")
    ax.set_xlabel("allele encoding", color=MUTED, fontsize=12, labelpad=10)
    ax.set_ylabel("peptide encoding", color=MUTED, fontsize=12)
    ax.set_title("Median Spearman over 21 groove-held-out folds, one ridge head throughout",
                 color=FG, fontsize=15, pad=14)
    cb = fig.colorbar(im)
    cb.ax.tick_params(colors=FG)
    cb.outline.set_edgecolor(MUTED)
    fig.tight_layout()
    fig.savefig("diagnose_why_factorial.png", dpi=150, facecolor=BG)
    log("wrote diagnose_why_factorial.png")


def _fig_alleles(rows, iden, iu, names):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), facecolor=BG)
    ax = _axes(axes[0])
    ks = list(rows)
    ax.barh(range(len(ks)), [rows[k]["mean_cos"] for k in ks], color=GOOD, height=.6)
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels(ks, color=FG)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    for i, k in enumerate(ks):
        ax.text(rows[k]["mean_cos"] + .015, i, f"{rows[k]['mean_cos']:.3f}", color=FG,
                va="center", fontsize=12)
    ax.set_title("mean pairwise cosine between the 75 alleles", color=FG, fontsize=13, pad=10)
    ax.set_xlabel("1.0 = every allele looks identical", color=MUTED)

    ax = _axes(axes[1])
    M, _ = allele_table("esm150_pseudo")
    Z = M / np.linalg.norm(M, axis=1, keepdims=True)
    ax.scatter(1 - iden[iu], 1 - (Z @ Z.T)[iu], s=7, color=GOOD, alpha=.35, linewidths=0)
    r = stats.spearmanr(1 - (Z @ Z.T)[iu], 1 - iden[iu]).statistic
    ax.set_xlabel("groove distance  (1 - identity over the 34 contact residues)", color=MUTED)
    ax.set_ylabel("ESM-2 pseudo-seq cosine distance", color=MUTED)
    ax.set_title(f"does the embedding preserve groove similarity?   rho = {r:+.3f}",
                 color=FG, fontsize=13, pad=10)
    fig.tight_layout()
    fig.savefig("diagnose_why_alleles.png", dpi=150, facecolor=BG)
    log("wrote diagnose_why_alleles.png")


def _fig_interaction(got):
    plt = _plt()
    ks = list(got)
    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor=BG)
    _axes(ax)
    med = [got[k].median() for k in ks]
    ax.barh(range(len(ks)), med, height=.6,
            color=[WARM if "identity" in k else (GOOD if "esm150_pseudo" in k else BAD)
                   for k in ks])
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([k.replace(" + interaction", "") for k in ks], color=FG)
    ax.invert_yaxis()
    for i, m in enumerate(med):
        ax.text(m + .004, i, f"{m:.3f}", color=FG, va="center", fontsize=12)
    ax.set_xlabel("median Spearman over 21 groove-held-out folds", color=MUTED)
    ax.set_title("With a peptide-allele interaction term: does any allele encoding "
                 "beat 'identity'?", color=FG, fontsize=14, pad=14)
    fig.text(.5, .02, "orange = allele identity one-hot, which carries NO information "
                      "about an unseen groove and is therefore the floor",
             color=MUTED, ha="center", fontsize=11)
    fig.tight_layout(rect=[0, .05, 1, 1])
    fig.savefig("diagnose_why_interaction.png", dpi=150, facecolor=BG)
    log("wrote diagnose_why_interaction.png")


def _fig_pooling(got):
    plt = _plt()
    ks = list(got)
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG)
    _axes(ax)
    med = [got[k].median() for k in ks]
    q1 = [got[k].quantile(.25) for k in ks]
    q3 = [got[k].quantile(.75) for k in ks]
    ax.barh(range(len(ks)), med, color=[GOOD if "concat" in k else BAD for k in ks], height=.6)
    ax.errorbar(med, range(len(ks)),
                xerr=[np.array(med) - np.array(q1), np.array(q3) - np.array(med)],
                fmt="none", ecolor=FG, capsize=5, lw=1.4)
    ax.axvline(0.277, color=WARM, lw=1.6, ls="--")
    # inside the axes: at y below the top the label collided with the suptitle
    ax.text(0.273, len(ks) - 0.6, "arm A, no foundation model", color=WARM,
            fontsize=11, ha="right", va="bottom")
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels(ks, color=FG)
    ax.invert_yaxis()
    for i, m in enumerate(med):
        ax.text(m + .006, i, f"{m:.3f}", color=FG, va="center", fontsize=12)
    ax.set_xlabel("median Spearman over 21 groove-held-out folds (whisker = IQR)", color=MUTED)
    ax.set_title("Is mean-pooling the 9 tokens what loses the information?",
                 color=FG, fontsize=15, pad=14)
    fig.tight_layout()
    fig.savefig("diagnose_why_pooling.png", dpi=150, facecolor=BG)
    log("wrote diagnose_why_pooling.png")


# ===========================================================================

PARTS = {"p0": p0, "p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5, "p6": p6,
         "p7": p7}


def main():
    want = [a for a in sys.argv[1:] if a in PARTS] or list(PARTS)
    for k in want:
        PARTS[k]()
    out = pd.DataFrame(RESULTS)
    # Parts can be run separately, so merge rather than clobber: rows from the
    # parts just run replace their previous versions, everything else survives.
    import os
    if os.path.exists("results_H_why.csv"):
        old = pd.read_csv("results_H_why.csv")
        out = pd.concat([old[~old.part.isin(out.part.unique())], out], ignore_index=True)
    out.to_csv("results_H_why.csv", index=False)
    log(f"wrote results_H_why.csv  ({len(out)} measurements)")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
