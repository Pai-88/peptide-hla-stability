"""Arm L (ProtT5): the same evaluation, a DIFFERENT model family.

WHY THIS ARM EXISTS. Every protein-language-model arm in this repo so far is
ESM-2 -- arms B, C, D, E, G, I, J, K and the fine-tuning arms F/FG all call the
same encoder family, varying only scale (150M -> 650M), pooling, head and
whether the weights are frozen. So the project's conclusion so far is strictly
"a conventional supervised net beats ESM-2 on this task", and the obvious
objection is that ESM-2 is simply the wrong encoder. This arm answers that
objection by swapping the encoder family and changing nothing else.

ProtT5 (Rostlab/prot_t5_xl_half_uniref50-enc) is a 1.2B-parameter T5 ENCODER
trained on UniRef50 with a BART-like MLM objective -- a different architecture
(T5 relative-position encoder, d_model 1024), a different pre-training corpus
slice and a different objective from ESM-2's. It is ungated, encoder-only, and
the model card's documented recipe is followed exactly:
  * residues separated by single spaces ("S L Y N T V A T L"),
  * the rare/ambiguous residues U, Z, O and B rewritten to X,
  * add_special_tokens=True, which appends </s> and adds NO BOS, so for a
    fixed-length 9-mer the residues are token positions 0..8. (ESM-2 prepends
    BOS, so arm I slices 1..9; getting this off by one would silently embed the
    EOS token as a residue, so it is asserted, not assumed.)
The half-precision checkpoint is loaded and cast to float32: the card warns
half precision is unusable on CPU, and these sequences are 9 and 34 tokens long
so fp32 costs seconds, not minutes.

WHAT IS HELD FIXED. Everything that is not the encoder:
  * the 21 leave-one-groove-cluster-out folds (splits.choose_held_out),
  * censored='tied',
  * metrics.py, per held-out allele then median,
  * the ridge head is arm I's RidgeHead, imported, not reimplemented,
  * the MLP head is arm A's TorchMLP with arm A's tuned BEST hyperparameters,
    imported, not reimplemented.
Only featurize() differs. That is the whole point: any delta is attributable to
the encoder.

FOUR ARMS, mirroring the ESM-2 ladder so each new number has a counterpart:
  L_prott5_pooled_B_ridge   mean-pooled peptide (1024) | pooled pseudo (1024)
  L_prott5_pooled_B_mlp     same features, arm A's tuned MLP head
  L_prott5_perres_B_ridge   per-residue peptide (9x1024) | pooled pseudo (1024)
  L_prott5_perres_B_mlp     same features, arm A's tuned MLP head
Un-pooling was worth more than fine-tuning for ESM-2 (0.195 vs 0.170), so a
pooled-only comparison would understate ProtT5 and would not be a fair test.

The ridge head is DETERMINISTIC -- every seed gives a bit-identical fit, so it
is run with one seed and the "5-seed ensemble" of it is that same fit. The MLP
is stochastic (init, dropout, batch order) and is run with 5 seeds, scored as
the 5-seed ensemble mean exactly like arms A, B, G and I.

Usage:
    python arm_L_prott5.py --embed            # build the caches (resumable)
    python arm_L_prott5.py --which pooled_B_ridge --seeds 1
    python arm_L_prott5.py --all              # every arm, skipping finished ones
    python arm_L_prott5.py --report

Nothing runs at import time.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd

import metrics
import run_experiment as R

MODEL = "Rostlab/prot_t5_xl_half_uniref50-enc"
DIM = 1024
PEP_LEN = 9
PSEUDO_LEN = 34

POOLED_PEP = "emb_prott5_peptides.npy"
POOLED_PEP_IDX = "emb_prott5_peptides_index.json"
PERRES_PEP = "emb_prott5_perres.npy"
PERRES_PEP_IDX = "emb_prott5_perres_index.json"
PSEUDO = "emb_prott5_pseudo.npy"
PSEUDO_IDX = "emb_prott5_pseudo_index.json"


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------

def _prep(seqs):
    """The model card's preprocessing, verbatim: U/Z/O/B -> X, spaces between
    every residue. Returns (prepared, n_rewritten) so the rewrite is reported
    rather than silent -- on this dataset it should be zero."""
    n = sum(len(re.findall(r"[UZOB]", s)) for s in seqs)
    return [" ".join(list(re.sub(r"[UZOB]", "X", s))) for s in seqs], n


def load_model(dtype="float32"):
    import torch
    from transformers import T5EncoderModel, T5Tokenizer
    tok = T5Tokenizer.from_pretrained(MODEL, do_lower_case=False, legacy=False)
    model = T5EncoderModel.from_pretrained(
        MODEL, dtype=getattr(torch, dtype)).eval()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    return tok, model.to(dev), dev


def _encode(seqs, length, tok, model, dev, batch, log=""):
    """(n, length, DIM) real-residue token states. Every sequence must be
    exactly `length` long, so there is no padding and no mask to get wrong."""
    import torch

    assert all(len(s) == length for s in seqs), f"not every sequence is {length} long"
    prepared, n_rw = _prep(seqs)
    if n_rw:
        print(f"  NOTE: rewrote {n_rw} U/Z/O/B residues to X", flush=True)
    out, t0 = [], time.time()
    with torch.no_grad():
        for i in range(0, len(prepared), batch):
            chunk = prepared[i: i + batch]
            enc = tok(chunk, add_special_tokens=True, padding="longest",
                      return_tensors="pt").to(dev)
            # T5 appends </s> and adds NO BOS, so a length-L sequence is
            # exactly L+1 tokens and the residues are positions 0..L-1.
            assert int(enc["attention_mask"].sum(1).min()) == length + 1, \
                "unexpected token count: residue slice would be off by one"
            h = model(input_ids=enc["input_ids"],
                      attention_mask=enc["attention_mask"]).last_hidden_state
            out.append(h[:, :length, :].float().cpu().numpy())
            if log and i % (batch * 4) == 0:
                print(f"  {log} {i + len(chunk)}/{len(prepared)}  "
                      f"{time.time() - t0:.1f}s", flush=True)
    return np.concatenate(out, axis=0), time.time() - t0


def build_embeddings(batch=128, force=False):
    """Cache all three encodings. Resumable: already-written files are kept."""
    import data

    have = all(os.path.exists(p) for p in
               (POOLED_PEP, POOLED_PEP_IDX, PERRES_PEP, PERRES_PEP_IDX,
                PSEUDO, PSEUDO_IDX))
    if have and not force:
        print("ProtT5 caches already present, nothing to do (use --force to rebuild)")
        return

    df = data.load()
    peps = sorted(df.Pep.unique())
    alle = json.load(open("alleles.json"))
    pnames = sorted(n for n in alle if alle[n]["pseudo"])

    tok, model, dev = load_model()
    print(f"device {dev}  model {MODEL}  "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params", flush=True)

    # --- peptides: keep all 9 token vectors, derive the mean from them -------
    tokens, dt = _encode(peps, PEP_LEN, tok, model, dev, batch, log="pep")
    print(f"peptides: {len(peps)} in {dt:.1f}s ({len(peps) / dt:.0f}/s)", flush=True)
    assert tokens.shape == (len(peps), PEP_LEN, DIM), tokens.shape
    assert np.isfinite(tokens).all(), "non-finite values in the peptide encoding"
    perres = tokens.reshape(len(peps), PEP_LEN * DIM)
    pooled = tokens.mean(axis=1)
    # The pooled matrix is the mean of the per-residue blocks BY CONSTRUCTION,
    # so the two caches cannot drift apart. Asserted anyway: a silent mismatch
    # would make the pooled/un-pooled delta uninterpretable.
    recon = perres.reshape(len(peps), PEP_LEN, DIM).mean(1)
    err = float(np.abs(recon - pooled).max())
    print(f"max |mean(per-residue) - pooled| = {err:.2e}", flush=True)
    assert err < 1e-5

    np.save(PERRES_PEP, perres.astype(np.float32))
    json.dump({p: i for i, p in enumerate(peps)}, open(PERRES_PEP_IDX, "w"))
    np.save(POOLED_PEP, pooled.astype(np.float32))
    json.dump({p: i for i, p in enumerate(peps)}, open(POOLED_PEP_IDX, "w"))
    print(f"wrote {PERRES_PEP} {perres.shape} and {POOLED_PEP} {pooled.shape}", flush=True)

    # --- the 34-residue allele pseudo-sequences ------------------------------
    tokens, dt = _encode([alle[n]["pseudo"] for n in pnames], PSEUDO_LEN,
                         tok, model, dev, batch=64)
    ps = tokens.mean(axis=1)
    assert np.isfinite(ps).all(), "non-finite values in the pseudo-sequence encoding"
    np.save(PSEUDO, ps.astype(np.float32))
    json.dump({n: i for i, n in enumerate(pnames)}, open(PSEUDO_IDX, "w"))
    print(f"pseudo: {len(pnames)} in {dt:.1f}s -> {PSEUDO} {ps.shape}", flush=True)

    # A sanity floor: the 75 allele vectors must not collapse together, or the
    # "allele shown" arms are secretly peptide-only arms.
    #
    # Exactly ONE collision is expected and correct. B*14:01(C67S) and
    # B*14:02(C67S) differ only outside the 34 groove-contact positions, so
    # their pseudo-sequences are character-identical and every encoder must map
    # them to the same vector. The cached ESM-2 pseudo matrix has the same single
    # duplicate pair, so this is a property of the data, not of ProtT5. Anything
    # beyond one pair would mean the encoder is collapsing distinct grooves.
    d = np.linalg.norm(ps[:, None, :] - ps[None, :, :], axis=-1)
    off = d[~np.eye(len(ps), dtype=bool)]
    n_dup = int(((d < 1e-6).sum() - len(ps)) // 2)
    seq_dup = len(pnames) - len({alle[n]["pseudo"] for n in pnames})
    print(f"pseudo pairwise L2: min {off.min():.3f} median {np.median(off):.3f} "
          f"max {off.max():.3f}  identical pairs {n_dup} "
          f"(duplicate pseudo-sequences in alleles.json: {seq_dup})", flush=True)
    assert n_dup == seq_dup, (
        f"{n_dup} identical embedding pairs but only {seq_dup} identical "
        f"pseudo-sequences: the encoder is collapsing distinct grooves")


# ---------------------------------------------------------------------------
# featurizers -- arm I's conventions: peptide block FIRST, allele block second
# ---------------------------------------------------------------------------

_T = None


def _tables():
    global _T
    if _T is None:
        missing = [p for p in (POOLED_PEP, PERRES_PEP, PSEUDO) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"{missing} missing -- run `python arm_L_prott5.py --embed` first")
        _T = (np.load(POOLED_PEP), json.load(open(POOLED_PEP_IDX)),
              np.load(PERRES_PEP), json.load(open(PERRES_PEP_IDX)),
              np.load(PSEUDO), json.load(open(PSEUDO_IDX)))
    return _T


def _rows(df_subset, mat, idx, col):
    k = df_subset[col].map(idx).to_numpy()
    if pd.isna(k).any():
        raise KeyError(f"{col} value missing from the ProtT5 index")
    return mat[k.astype(int)]


def featurize_pooled_B(df_subset):
    """[mean-pooled peptide (1024) | mean-pooled pseudo (1024)] -> (n, 2048)."""
    P, pi, _, _, S, si = _tables()
    return np.hstack([_rows(df_subset, P, pi, "Pep"),
                      _rows(df_subset, S, si, "HLA")]).astype(np.float64)


def featurize_perres_B(df_subset):
    """[per-residue peptide (9x1024) | mean-pooled pseudo (1024)] -> (n, 10240)."""
    _, _, Pr, pri, S, si = _tables()
    return np.hstack([_rows(df_subset, Pr, pri, "Pep"),
                      _rows(df_subset, S, si, "HLA")]).astype(np.float64)


# ---------------------------------------------------------------------------
# heads: imported from the arms that defined them, never reimplemented
# ---------------------------------------------------------------------------

def make_ridge(n_pep_cols):
    import arm_I_perresidue as I
    return lambda s: I.RidgeHead(seed=s, n_pep_cols=n_pep_cols)


def make_mlp(n_pep_cols):
    import arm_A_supervised_nn as A
    return lambda s: A.TorchMLP(seed=s, n_pep_cols=n_pep_cols, **A.BEST)


# key -> (arm name, featurize, peptide-block width, head, default seeds)
SPEC = {
    "pooled_B_ridge": ("L_prott5_pooled_B_ridge", featurize_pooled_B, DIM, "ridge", 1),
    "pooled_B_mlp":   ("L_prott5_pooled_B_mlp",   featurize_pooled_B, DIM, "mlp",   5),
    "perres_B_ridge": ("L_prott5_perres_B_ridge", featurize_perres_B, PEP_LEN * DIM, "ridge", 1),
    "perres_B_mlp":   ("L_prott5_perres_B_mlp",   featurize_perres_B, PEP_LEN * DIM, "mlp",   5),
}
ORDER = ["pooled_B_ridge", "pooled_B_mlp", "perres_B_ridge", "perres_B_mlp"]


def is_done(arm, seeds):
    """True only if a COMPLETE, readable result pair is already on disk.

    Overnight and unattended, so a half-written arm from a killed process must
    not be mistaken for a finished one: we require 21 folds x `seeds` rows all
    with status 'ok' AND a parquet that actually opens.
    """
    rc, pq = f"results_{arm}.csv", f"predictions_{arm}.parquet"
    if not (os.path.exists(rc) and os.path.exists(pq)):
        return False
    try:
        res = pd.read_csv(rc)
        ok = res[res.status == "ok"]
        if len(ok) < 21 * seeds or ok.fold_name.nunique() < 21:
            return False
        pd.read_parquet(pq, columns=["row_id"])
        return True
    except Exception:
        return False


def run_one(key, seeds=None, censored="tied", skip_done=True):
    arm, f, npc, head, dflt = SPEC[key]
    seeds = dflt if seeds is None else seeds
    if skip_done and is_done(arm, seeds):
        print(f"[{arm}] already complete on disk, skipping", flush=True)
        return None
    make = make_mlp(npc) if head == "mlp" else make_ridge(npc)
    if head == "ridge":
        import arm_I_perresidue as I
        I.RidgeHead.chosen_alpha = []
    t0 = time.time()
    res = R.run_arm(arm, f, make, seeds=seeds, censored=censored)
    if head == "ridge":
        import arm_I_perresidue as I
        print(f"[{arm}] alphas chosen: "
              f"{pd.Series(I.RidgeHead.chosen_alpha).value_counts().to_dict()}", flush=True)
    print(f"[{arm}] wall clock {time.time() - t0:.1f}s\n", flush=True)
    return res


def run_all(seeds=None, censored="tied", keys=None, skip_done=True):
    out = {}
    for k in (keys or ORDER):
        try:
            out[k] = run_one(k, seeds=seeds, censored=censored, skip_done=skip_done)
        except Exception as e:                      # one arm must not kill the night
            import traceback
            print(f"[{SPEC[k][0]}] ARM FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            out[k] = None
    return out


# ---------------------------------------------------------------------------
# reporting -- the SAME estimator the project's headline table uses
# ---------------------------------------------------------------------------

def headline(arm, censored="tied"):
    """Median over folds of the per-fold median per-allele Spearman, computed on
    the across-seed MEAN prediction. Identical estimator to pitch_figure.py, so
    the new rows are directly comparable to the existing table. Returns
    (median, n_folds, n_seeds) or None if the arm has no predictions."""
    import arm_B_esm_pseudo as B
    if not os.path.exists(f"predictions_{arm}.parquet"):
        return None
    d = B.rescore_ensemble(arm, censored)
    d = d[d.n_alleles_scored > 0]
    if not len(d):
        return None
    return (float(np.median(d.spearman.to_numpy(dtype=float))),
            int(len(d)), int(d.n_seeds.max()))


def report(censored="tied"):
    rows = []
    for k in ORDER:
        arm = SPEC[k][0]
        h = headline(arm, censored)
        if h is None:
            print(f"{arm:30s} (no predictions on disk)")
            continue
        med, nf, ns = h
        rows.append({"arm": arm, "spearman": med, "folds": nf, "seeds": ns})
        print(f"{arm:30s} {med:+.4f}  folds {nf}  seeds {ns}")
    if rows:
        out = pd.DataFrame(rows)
        out.to_csv("results_L_prott5_summary.csv", index=False)
        print("\n-> results_L_prott5_summary.csv")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--force", action="store_true", help="rebuild caches")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--which", choices=list(SPEC))
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--censored", default="tied", choices=("tied", "drop"))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--redo", action="store_true", help="ignore finished arms on disk")
    a = ap.parse_args()

    if a.embed:
        build_embeddings(force=a.force)
    elif a.report:
        report(a.censored)
    elif a.which:
        run_one(a.which, seeds=a.seeds, censored=a.censored, skip_done=not a.redo)
    elif a.all:
        run_all(seeds=a.seeds, censored=a.censored, skip_done=not a.redo)
        report(a.censored)
    else:
        ap.error("pick one of --embed / --which / --all / --report")


if __name__ == "__main__":
    main()
