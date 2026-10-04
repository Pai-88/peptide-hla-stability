"""Arm L (SaProt): the negative control for the family sweep, run on purpose.

READ THIS BEFORE QUOTING ANY NUMBER THIS FILE PRODUCES.

SaProt is a STRUCTURE-AWARE protein language model. Its vocabulary is 20 amino
acids x 20 Foldseek 3Di structure tokens (plus "#", meaning "structure unknown
at this position"), so a residue is encoded as a PAIR, e.g. "Md", "Ev", "Qp".
Its power comes from the structure half of that pair.

WE CANNOT SUPPLY THE STRUCTURE HALF. Foldseek 3Di tokens are read off a 3D
structure, and we have no structure for these 28,166 peptide-HLA pairs. Nor is
one cheaply obtainable: a free 9-mer peptide has no independent fold -- its
conformation is INDUCED by the HLA groove it sits in -- so the only meaningful
structure would be of the complex, i.e. 28,166 runs of a structure predictor,
which the brief already ruled out on cost. No structure tokens are faked here;
every position is given the model's own documented "#" unknown-structure token.

AND ITS AUTHORS SAY THIS MODE DOES NOT WORK FROZEN. The SaProt model card
states that amino-acid-sequence-only mode works but must be FINE-TUNED, and
that frozen embeddings work only for structure-aware tokens, not for AA-only
sequences. This project's pipeline is frozen embeddings plus a head -- exactly
the mode the authors disown.

So why run it at all? Because "the model card says it would not work" is an
assertion, and a measurement is better than an assertion. This arm exists ONLY
to turn that assertion into a number, and the number means:

    "SaProt, deprived of the structure channel that is its entire point,
     scores X" -- NOT "SaProt scores X".

It is NOT evidence about SaProt's ability on this task, and it must never be
put in the headline ladder next to ESM-2, ProtT5 and ESM-C, which were all
given everything they need. Two pooled arms only; spending an hour on the
un-pooled variants of a deliberately crippled model would be waste.

Usage:
    python arm_L_saprot.py --embed
    python arm_L_saprot.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import run_experiment as R

MODEL = "westlake-repl/SaProt_650M_AF2"
DIM = 1280
PEP_LEN = 9
PSEUDO_LEN = 34

# The model's own token for "structure at this position is unknown". Using it is
# the documented AA-only mode, not an invented structure.
UNK_STRUCT = "#"

POOLED = "emb_saprot_peptides.npy"
POOLED_IDX = "emb_saprot_peptides_index.json"
PSEUDO = "emb_saprot_pseudo.npy"
PSEUDO_IDX = "emb_saprot_pseudo_index.json"


def sa_encode(seq):
    """'SLYNT' -> 'S#L#Y#N#T#': every residue paired with 'structure unknown'."""
    return "".join(c + UNK_STRUCT for c in seq)


def load_model(dtype="float32"):
    import torch
    from transformers import EsmForMaskedLM, EsmTokenizer
    tok = EsmTokenizer.from_pretrained(MODEL)
    m = EsmForMaskedLM.from_pretrained(MODEL, dtype=getattr(torch, dtype)).eval()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    return tok, m.to(dev), dev


def _encode(seqs, length, tok, model, dev, batch, log=""):
    """Mean-pooled hidden states over the `length` real residue-pair tokens."""
    import torch

    assert all(len(s) == length for s in seqs)
    sa = [sa_encode(s) for s in seqs]
    out, t0 = [], time.time()
    with torch.no_grad():
        for i in range(0, len(sa), batch):
            chunk = sa[i: i + batch]
            enc = tok(chunk, add_special_tokens=True, padding="longest",
                      return_tensors="pt").to(dev)
            # ESM alphabet: <cls> t1..tL <eos>, one token per RESIDUE PAIR.
            assert int(enc["attention_mask"].sum(1).min()) == length + 2, \
                "SaProt did not give one token per residue pair"
            h = model(**enc, output_hidden_states=True).hidden_states[-1]
            out.append(h[:, 1: length + 1, :].float().mean(1).cpu().numpy())
            if log and i % (batch * 8) == 0:
                print(f"  {log} {i + len(chunk)}/{len(sa)}  {time.time() - t0:.1f}s",
                      flush=True)
    return np.concatenate(out, axis=0), time.time() - t0


def build_embeddings(batch=64, force=False):
    import data

    if all(os.path.exists(p) for p in (POOLED, POOLED_IDX, PSEUDO, PSEUDO_IDX)) \
            and not force:
        print("SaProt caches already present (use --force to rebuild)")
        return

    df = data.load()
    peps = sorted(df.Pep.unique())
    alle = json.load(open("alleles.json"))
    pnames = sorted(n for n in alle if alle[n]["pseudo"])

    tok, model, dev = load_model()
    print(f"device {dev}  model {MODEL}  "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params", flush=True)
    probe = tok(sa_encode("ACDEFGHIKLMNPQRSTVWY"), add_special_tokens=False)["input_ids"]
    assert len(probe) == 20, f"expected 20 residue-pair tokens, got {len(probe)}"
    assert tok.unk_token_id not in probe, "a residue-pair token hit <unk>"
    print(f"alphabet check OK: 'A#' -> id {tok(sa_encode('A'), add_special_tokens=False)['input_ids']}",
          flush=True)

    P, dt = _encode(peps, PEP_LEN, tok, model, dev, batch, log="pep")
    print(f"peptides: {len(peps)} in {dt:.1f}s", flush=True)
    assert np.isfinite(P).all() and P.shape == (len(peps), DIM), P.shape
    np.save(POOLED, P.astype(np.float32))
    json.dump({p: i for i, p in enumerate(peps)}, open(POOLED_IDX, "w"))

    S, dt = _encode([alle[n]["pseudo"] for n in pnames], PSEUDO_LEN,
                    tok, model, dev, batch=32)
    assert np.isfinite(S).all()
    np.save(PSEUDO, S.astype(np.float32))
    json.dump({n: i for i, n in enumerate(pnames)}, open(PSEUDO_IDX, "w"))
    print(f"pseudo: {len(pnames)} in {dt:.1f}s -> {PSEUDO} {S.shape}", flush=True)

    d = np.linalg.norm(S[:, None, :] - S[None, :, :], axis=-1)
    n_dup = int(((d < 1e-6).sum() - len(S)) // 2)
    seq_dup = len(pnames) - len({alle[n]["pseudo"] for n in pnames})
    print(f"pseudo identical pairs {n_dup} (expected {seq_dup})", flush=True)
    assert n_dup == seq_dup, "encoder is collapsing distinct grooves"


_T = None


def _tables():
    global _T
    if _T is None:
        missing = [p for p in (POOLED, PSEUDO) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"{missing} missing -- run --embed first")
        _T = (np.load(POOLED), json.load(open(POOLED_IDX)),
              np.load(PSEUDO), json.load(open(PSEUDO_IDX)))
    return _T


def featurize_pooled_B(df_subset):
    P, pi, S, si = _tables()
    p = df_subset.Pep.map(pi).to_numpy()
    s = df_subset.HLA.map(si).to_numpy()
    if pd.isna(p).any() or pd.isna(s).any():
        raise KeyError("peptide or allele missing from the SaProt index")
    return np.hstack([P[p.astype(int)], S[s.astype(int)]]).astype(np.float64)


SPEC = {
    "pooled_B_ridge": ("L_saprot_nostruct_pooled_B_ridge", DIM, "ridge", 1),
    "pooled_B_mlp":   ("L_saprot_nostruct_pooled_B_mlp",   DIM, "mlp",   5),
}
ORDER = ["pooled_B_ridge", "pooled_B_mlp"]


def is_done(arm, seeds):
    rc, pq = f"results_{arm}.csv", f"predictions_{arm}.parquet"
    if not (os.path.exists(rc) and os.path.exists(pq)):
        return False
    try:
        ok = pd.read_csv(rc).query("status == 'ok'")
        if len(ok) < 21 * seeds or ok.fold_name.nunique() < 21:
            return False
        pd.read_parquet(pq, columns=["row_id"])
        return True
    except Exception:
        return False


def run_one(key, seeds=None, censored="tied", skip_done=True):
    import arm_A_supervised_nn as A
    import arm_I_perresidue as I
    arm, npc, head, dflt = SPEC[key]
    seeds = dflt if seeds is None else seeds
    if skip_done and is_done(arm, seeds):
        print(f"[{arm}] already complete, skipping", flush=True)
        return None
    make = ((lambda s: A.TorchMLP(seed=s, n_pep_cols=npc, **A.BEST)) if head == "mlp"
            else (lambda s: I.RidgeHead(seed=s, n_pep_cols=npc)))
    t0 = time.time()
    res = R.run_arm(arm, featurize_pooled_B, make, seeds=seeds, censored=censored)
    print(f"[{arm}] wall clock {time.time() - t0:.1f}s\n", flush=True)
    return res


def headline(arm, censored="tied"):
    import arm_B_esm_pseudo as B
    if not os.path.exists(f"predictions_{arm}.parquet"):
        return None
    d = B.rescore_ensemble(arm, censored)
    d = d[d.n_alleles_scored > 0]
    if not len(d):
        return None
    return (float(np.median(d.spearman.to_numpy(dtype=float))),
            int(len(d)), int(d.n_seeds.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.embed:
        build_embeddings(force=a.force)
    elif a.report:
        for k in ORDER:
            print(SPEC[k][0], headline(SPEC[k][0]))
    elif a.all:
        for k in ORDER:
            try:
                run_one(k)
            except Exception as e:
                import traceback
                print(f"[{SPEC[k][0]}] ARM FAILED: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
        for k in ORDER:
            print(SPEC[k][0], headline(SPEC[k][0]))
    else:
        ap.error("pick --embed / --all / --report")


if __name__ == "__main__":
    main()
