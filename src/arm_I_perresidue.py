"""Arm I: ESM-2 WITHOUT mean pooling.

HYPOTHESIS. Peptides here are 9-mers and the signal concentrates in anchor
positions (classically P2 and the C-terminus, P9). The conventional net
(BLOSUM/one-hot, arm A) preserves residue identity AT EACH POSITION. Every
frozen ESM-2 arm so far MEAN-POOLS the 9 token embeddings, which averages
positional identity away. If pooling is the cause of the gap, un-pooling should
recover most of it.

So: re-embed the 5,633 peptides with facebook/esm2_t30_150M_UR50D and keep all
9 per-residue token embeddings instead of averaging. 9 x 640 = 5760 dims,
concatenated in positional order. For a fixed-length 9-mer that is positionally
lossless -- a strict superset of the information in the mean, which is just the
average of the 9 blocks.

Two feature sets, mirroring the existing arms:
  I_perres_D   peptide only                                   (5760)
  I_perres_B   peptide per-residue | allele pseudo-seq (mean) (5760 + 640)

The allele side stays the cached mean-pooled pseudo-sequence embedding, exactly
as arm B uses it. This arm changes ONE thing (peptide pooling) so the delta is
attributable.

Head: StandardScaler + Ridge, alpha swept on a PEPTIDE-GROUPED inner split of
the TRAIN fold only -- 5760 dims on ~24k rows needs real regularisation, and
the test fold must never see it. Same inner-split machinery and alpha grid as
arm B, widened upwards because the dimensionality is 4.5x higher.

Usage:
    python arm_I_perresidue.py --embed     # build the per-residue cache
    python arm_I_perresidue.py             # both feature sets, tied
    python arm_I_perresidue.py --drop      # robustness re-score
    python arm_I_perresidue.py --report

Nothing runs at import time.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import embed as embed_mod
import metrics
import run_experiment as R

PEP_LEN = 9
EMB_PATH = "arm_I_perres_emb.npy"
IDX_PATH = "arm_I_perres_index.json"

ARM_D = "I_perres_D_peptide_only"
ARM_B = "I_perres_B_pseudo"
# Matched controls: the SAME head and folds on the MEAN-POOLED peptide, so the
# un-pooling delta is attributable to pooling and nothing else. (The project's
# existing arms D and B use an MLP head, so they are not a matched comparison.)
ARM_D0 = "I_pooled_D_peptide_only"
ARM_B0 = "I_pooled_B_pseudo"

# Arm B's grid, extended upward: 5760 features on ~24k rows sits much further
# into the over-parameterised regime, so the useful alpha is larger.
ALPHAS = (10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0, 10_000_000.0)
INNER_FRAC = 0.2
INNER_SEED = 0


# ---------------------------------------------------------------------------
# the embedding: all 9 token vectors, no pooling
# ---------------------------------------------------------------------------

def build_embeddings(batch=256):
    """Encode every distinct 9-mer, keep the 9 residue token vectors.

    ESM-2 tokenises as [BOS] r1..r9 [EOS], so the residues are positions 1..9 of
    the last hidden state. Every peptide is exactly 9 long, so there is no
    padding and no mask to get wrong -- asserted below rather than assumed.
    """
    import torch

    import data

    df = data.load()
    peps = sorted(df.Pep.unique())
    assert all(len(p) == PEP_LEN for p in peps), "not every peptide is a 9-mer"

    tok, model = embed_mod.load_model()
    dev = embed_mod.device()
    print(f"device {dev}  model {embed_mod.MODEL}  {len(peps)} peptides", flush=True)

    out, t0 = [], time.time()
    with torch.no_grad():
        for i in range(0, len(peps), batch):
            chunk = peps[i: i + batch]
            enc = tok(chunk, return_tensors="pt", padding=True).to(dev)
            # fixed length => every row has the same token count
            assert int(enc["attention_mask"].sum(1).min()) == PEP_LEN + 2
            h = model(**enc).last_hidden_state[:, 1: PEP_LEN + 1, :]   # (n, 9, 640)
            out.append(h.reshape(len(chunk), -1).float().cpu().numpy())
            if i % (batch * 5) == 0:
                print(f"  {i + len(chunk)}/{len(peps)}  {time.time() - t0:.1f}s", flush=True)

    mat = np.vstack(out)
    dt = time.time() - t0
    print(f"per-residue: {mat.shape} in {dt:.1f}s ({len(peps) / dt:.0f} pep/s)", flush=True)

    # Sanity: the mean over the 9 position blocks must reproduce the cached
    # mean-pooled embedding. If it does not, these are not the same vectors and
    # no comparison against the pooled arms would be meaningful.
    P, pi = embed_mod.load("peptides")
    recon = mat.reshape(len(peps), PEP_LEN, -1).mean(1)
    ref = P[[pi[p] for p in peps]]
    err = float(np.abs(recon - ref).max())
    print(f"max |mean(per-residue) - cached mean-pooled| = {err:.2e}", flush=True)
    assert err < 1e-3, "per-residue tokens do not average to the cached mean pooling"

    np.save(EMB_PATH, mat.astype(np.float32))
    json.dump({p: i for i, p in enumerate(peps)}, open(IDX_PATH, "w"))
    print(f"wrote {EMB_PATH} {mat.shape} and {IDX_PATH}", flush=True)
    return mat


_TABLES = None


def _tables():
    global _TABLES
    if _TABLES is None:
        if not os.path.exists(EMB_PATH):
            raise FileNotFoundError(
                f"{EMB_PATH} missing -- run `python arm_I_perresidue.py --embed` first")
        Pr = np.load(EMB_PATH)
        pri = json.load(open(IDX_PATH))
        S, si = embed_mod.load("pseudo")
        _TABLES = (Pr, pri, S, si)
    return _TABLES


def featurize_D(df_subset):
    """9 x 640 per-residue peptide embedding, positional order -> (n, 5760)."""
    Pr, pri, _, _ = _tables()
    p = df_subset.Pep.map(pri).to_numpy()
    if pd.isna(p).any():
        raise KeyError("peptide missing from the per-residue index")
    return Pr[p.astype(int)].astype(np.float64)


def featurize_D0(df_subset):
    """Control: mean-pooled peptide only -> (n, 640)."""
    _, _, _, _ = _tables()
    P, pi = embed_mod.load("peptides")
    p = df_subset.Pep.map(pi).to_numpy()
    if pd.isna(p).any():
        raise KeyError("peptide missing from the mean-pooled index")
    return P[p.astype(int)].astype(np.float64)


def featurize_B0(df_subset):
    """Control: mean-pooled peptide | mean-pooled pseudo -> (n, 1280)."""
    P, pi = embed_mod.load("peptides")
    _, _, S, si = _tables()
    p = df_subset.Pep.map(pi).to_numpy()
    s = df_subset.HLA.map(si).to_numpy()
    if pd.isna(p).any() or pd.isna(s).any():
        raise KeyError("peptide or allele missing from an embedding index")
    return np.hstack([P[p.astype(int)], S[s.astype(int)]]).astype(np.float64)


def featurize_B(df_subset):
    """[per-residue peptide (5760) | mean-pooled pseudo-sequence (640)]."""
    Pr, pri, S, si = _tables()
    p = df_subset.Pep.map(pri).to_numpy()
    s = df_subset.HLA.map(si).to_numpy()
    if pd.isna(p).any() or pd.isna(s).any():
        raise KeyError("peptide or allele missing from an embedding index")
    return np.hstack([Pr[p.astype(int)], S[s.astype(int)]]).astype(np.float64)


# ---------------------------------------------------------------------------
# head: scaler + ridge, alpha on a peptide-grouped inner split of TRAIN
# ---------------------------------------------------------------------------

def _peptide_groups(X, n_pep_cols):
    """Peptide identity recovered from the feature matrix (arm B's trick).

    The peptide block is bit-identical for two rows sharing a peptide, so one
    fixed random projection of it gives each distinct peptide its own float.
    """
    r = np.random.default_rng(12345).standard_normal(n_pep_cols)
    return np.unique(X[:, :n_pep_cols] @ r, return_inverse=True)[1]


def _inner_split(X, n_pep_cols, frac=INNER_FRAC, seed=INNER_SEED):
    g = _peptide_groups(X, n_pep_cols)
    n_g = g.max() + 1
    rng = np.random.default_rng(seed)
    val_g = set(rng.permutation(n_g)[: max(1, int(round(n_g * frac)))].tolist())
    val = np.fromiter((x in val_g for x in g), dtype=bool, count=len(g))
    return ~val, val


class _Scaler:
    def fit(self, X):
        self.mu = X.mean(0)
        self.sd = X.std(0)
        self.sd[self.sd < 1e-8] = 1.0
        return self

    def __call__(self, X):
        return (X - self.mu) / self.sd


def _ridge_path(X, y, alphas):
    """Closed-form ridge for a whole alpha grid from ONE Gram matrix.

    Identical model to sklearn's Ridge with fit_intercept=True (centre X and y,
    penalise the weights only, intercept recovered from the means) -- checked
    against sklearn in _check_ridge below. The point is cost: these feature
    matrices are 5,760 to 8,120 columns and n > d, so forming X'X once (one
    BLAS call) and Cholesky-solving it per alpha is several times cheaper than
    refitting from scratch for each alpha.

    Returns {alpha: (w, b)}.
    """
    from scipy.linalg import cho_factor, cho_solve
    mx, my = X.mean(0), float(y.mean())
    Xc = X - mx
    G = Xc.T @ Xc
    rhs = Xc.T @ (y - my)
    d = G.shape[0]
    out = {}
    for a in alphas:
        G[np.diag_indices(d)] += a
        w = cho_solve(cho_factor(G, lower=True, check_finite=False), rhs,
                      check_finite=False)
        G[np.diag_indices(d)] -= a
        out[a] = (w, my - float(mx @ w))
    return out


def _check_ridge(n=400, d=60, alpha=10.0, tol=1e-6):
    """_ridge_path must agree with sklearn.linear_model.Ridge. Run by --selftest."""
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, d))
    y = X @ rng.standard_normal(d) + 0.3 * rng.standard_normal(n) + 2.0
    w, b = _ridge_path(X, y, [alpha])[alpha]
    r = Ridge(alpha=alpha).fit(X, y)
    e = max(float(np.abs(w - r.coef_).max()), abs(b - float(r.intercept_)))
    print(f"_ridge_path vs sklearn.Ridge: max abs diff {e:.2e}")
    assert e < tol, "closed-form ridge disagrees with sklearn"
    return e


class RidgeHead:
    """StandardScaler + Ridge. Deterministic: every seed gives the same fit, so
    run_experiment blanks the calibration columns rather than reporting a 0 that
    would read as catastrophic miscalibration. Same convention as arm B's ridge.
    """

    chosen_alpha: list = []

    def __init__(self, seed=0, n_pep_cols=PEP_LEN * 640, alphas=ALPHAS):
        self.seed, self.n_pep_cols, self.alphas = seed, n_pep_cols, alphas

    def fit(self, X, y):
        tr, va = _inner_split(X, self.n_pep_cols)
        sc = _Scaler().fit(X[tr])
        Xi, Xv = sc(X[tr]), sc(X[va])
        path = _ridge_path(Xi, y[tr], self.alphas)
        best = min(self.alphas,
                   key=lambda a: float(np.mean(
                       (Xv @ path[a][0] + path[a][1] - y[va]) ** 2)))
        self.alpha_ = best
        RidgeHead.chosen_alpha.append(best)
        self.sc = _Scaler().fit(X)
        self.w, self.b = _ridge_path(self.sc(X), y, [best])[best]
        return self

    def predict(self, X):
        return self.sc(X) @ self.w + self.b


def _runner(arm, featurize, n_pep_cols, seeds, censored, head="ridge"):
    RidgeHead.chosen_alpha = []
    make = (make_mlp(n_pep_cols) if head == "mlp"
            else (lambda s: RidgeHead(seed=s, n_pep_cols=n_pep_cols)))
    t0 = time.time()
    res = R.run_arm(arm, featurize, make, seeds=seeds, censored=censored)
    if head == "ridge":
        print(f"[{arm}] alphas chosen: "
              f"{pd.Series(RidgeHead.chosen_alpha).value_counts().to_dict()}")
    print(f"[{arm}] wall clock {time.time() - t0:.1f}s\n", flush=True)
    return res


# ---------------------------------------------------------------------------
# second head: arm A's tuned MLP, so the un-pooling delta can also be read
# against the project's actual conventional headline instead of only against a
# ridge-matched control. Identical hyperparameters to arm A (A.BEST), identical
# allele-grouped early stopping; only the features differ.
# ---------------------------------------------------------------------------

def make_mlp(n_pep_cols):
    import arm_A_supervised_nn as A

    def make(seed):
        return A.TorchMLP(seed=seed, n_pep_cols=n_pep_cols, **A.BEST)
    return make


ARM_B_MLP = "I_perres_B_pseudo_mlp"
ARM_B0_MLP = "I_pooled_B_pseudo_mlp"

# key -> (arm name, featurize, width of the leading PEPTIDE block, head)
SPEC = {
    "D":      (ARM_D,      featurize_D,  PEP_LEN * 640, "ridge"),
    "B":      (ARM_B,      featurize_B,  PEP_LEN * 640, "ridge"),
    "D0":     (ARM_D0,     featurize_D0, 640,           "ridge"),
    "B0":     (ARM_B0,     featurize_B0, 640,           "ridge"),
    "Bmlp":   (ARM_B_MLP,  featurize_B,  PEP_LEN * 640, "mlp"),
    "B0mlp":  (ARM_B0_MLP, featurize_B0, 640,           "mlp"),
}


def run(seeds=1, censored="tied", which="both"):
    keys = (["D", "B"] if which == "both"
            else list(SPEC) if which == "all" else [which])
    out = {}
    for k in keys:
        arm, f, npc, head = SPEC[k]
        out[arm] = _runner(arm, f, npc, seeds, censored, head=head)
    return out


def rescore(arm, censored="drop"):
    """Re-score stored predictions under a different censoring policy.

    Censoring is an EVALUATION choice -- metrics.py applies it, the model never
    sees it, y is identical either way -- so this re-scores rather than refits.
    """
    import pyarrow.parquet as pq
    df = R.load_df()
    p = pq.read_table(f"predictions_{arm}.parquet").to_pandas()
    rows = []
    for (fold, seed), g in p.groupby(["fold_name", "seed"], sort=False):
        test = df.loc[g.row_id.values]
        rho = metrics.spearman_per_allele(test, g.y_pred.values, censored=censored)
        top = metrics.top10_precision_per_allele(test, g.y_pred.values, censored=censored)
        rows.append({"arm": arm, "fold_name": fold, "seed": int(seed),
                     "censored_policy": censored,
                     "spearman": np.nanmedian(rho) if len(rho) else np.nan,
                     "top10_precision": np.nanmedian(top) if len(top) else np.nan,
                     "n_alleles_scored": len(rho)})
    out = pd.DataFrame(rows)
    path = f"results_{arm}__{censored}.csv"
    out.to_csv(path, index=False)
    print(f"[{arm}] censored={censored}: per-fold median Spearman "
          f"{out.groupby('fold_name').spearman.mean().median():+.4f}  -> {path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true", help="build the per-residue cache")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--which", default="both", choices=("both", "all", *SPEC))
    ap.add_argument("--drop", action="store_true", help="re-score stored preds, censored='drop'")
    a = ap.parse_args()

    if a.selftest:
        _check_ridge()
        return
    if a.embed:
        build_embeddings()
        return
    if a.drop:
        arms = [SPEC[k][0] for k in (list(SPEC) if a.which in ("both", "all") else [a.which])]
        for arm in arms:
            rescore(arm, "drop")
        return
    run(seeds=a.seeds, which=a.which)


if __name__ == "__main__":
    main()
