"""Arm L (ESM-C): the third model family, on the same evaluation.

WHY. Arm L/ProtT5 already shows the result is not an artefact of ESM-2's
architecture. ESM-C closes the remaining gap in the family sweep: it is
EvolutionaryScale's SUCCESSOR to ESM-2 -- same masked-language-model objective
on protein sequence, but a redesigned, much stronger encoder (SwiGLU MLP,
rotary position embeddings, pre-norm, no learned position table) trained on a
larger and better-curated corpus. ESM-C 300M is reported to match or beat
ESM-2 650M, and ESM-C 600M to approach ESM-2 3B. So if the ESM-2 numbers in
this repo are "the encoder was simply too weak", ESM-C is the cheapest decisive
test of that: a strictly better sequence encoder, same family lineage, same
everything else.

LICENCE / ACCESS STATUS (checked 2026-10-03, matters because the brief flagged
ESM-C as possibly gated): the weights are NOT gated. The old
`EvolutionaryScale/esmc-*-2024-12` repos now redirect to `biohub/esmc-*`, and
the HF API reports `gated: false` for both those and the transformers-native
`biohub/ESMC-300M` / `biohub/ESMC-600M` repos. No licence click, no token, no
account was needed, and none was given.

The `biohub/ESMC-*` configs carry an `auto_map` that would route
`AutoModel.from_pretrained(..., trust_remote_code=True)` into a shim that calls
EvolutionaryScale's separate `esm` package. We do NOT use it: transformers has
shipped a native ESM-C port since 5.x (`transformers.EsmcModel`), so the class
is named explicitly and no remote code is downloaded or executed.

WHAT IS HELD FIXED. Identical to arm L/ProtT5 and to the ESM-2 arms: the 21
groove-cluster folds, censored='tied', metrics.py, arm I's RidgeHead and arm A's
tuned TorchMLP, both imported rather than reimplemented. Only the encoder moves.

TOKEN LAYOUT. ESM-C, like ESM-2 and unlike ProtT5, prepends <cls> and appends
<eos>, so a 9-mer is 11 tokens and the residues are positions 1..9. Asserted,
not assumed -- off by one here would silently embed <cls> as a residue.

Usage:
    python arm_L_esmc.py --model 600m --embed
    python arm_L_esmc.py --model 600m --all
    python arm_L_esmc.py --model 600m --report

Nothing runs at import time.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import run_experiment as R

PEP_LEN = 9
PSEUDO_LEN = 34

# tag -> (hf repo, hidden size)
MODELS = {
    "600m": ("biohub/ESMC-600M", 1152),
    "300m": ("biohub/ESMC-300M", 960),
}


def paths(tag):
    p = f"emb_esmc{tag}"
    return {"pooled": f"{p}_peptides.npy", "pooled_idx": f"{p}_peptides_index.json",
            "perres": f"{p}_perres.npy", "perres_idx": f"{p}_perres_index.json",
            "pseudo": f"{p}_pseudo.npy", "pseudo_idx": f"{p}_pseudo_index.json"}


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------

def load_model(tag, dtype="float32"):
    """Native transformers ESM-C. No trust_remote_code: the class is explicit."""
    import torch
    from transformers import AutoTokenizer, EsmcModel
    repo, dim = MODELS[tag]
    tok = AutoTokenizer.from_pretrained(repo)
    model = EsmcModel.from_pretrained(repo, dtype=getattr(torch, dtype)).eval()
    assert model.config.hidden_size == dim, \
        f"expected hidden {dim}, checkpoint says {model.config.hidden_size}"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    return tok, model.to(dev), dev, dim


def _encode(seqs, length, tok, model, dev, dim, batch, log=""):
    """(n, length, dim) real-residue token states, <cls>/<eos> excluded."""
    import torch

    assert all(len(s) == length for s in seqs), f"not every sequence is {length} long"
    out, t0 = [], time.time()
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            chunk = seqs[i: i + batch]
            enc = tok(chunk, add_special_tokens=True, padding="longest",
                      return_tensors="pt").to(dev)
            # <cls> r1..rL <eos>  ->  L + 2 tokens, residues at 1..L
            assert int(enc["attention_mask"].sum(1).min()) == length + 2, \
                "unexpected token count: residue slice would be off by one"
            h = model(**enc).last_hidden_state
            out.append(h[:, 1: length + 1, :].float().cpu().numpy())
            if log and i % (batch * 8) == 0:
                print(f"  {log} {i + len(chunk)}/{len(seqs)}  "
                      f"{time.time() - t0:.1f}s", flush=True)
    m = np.concatenate(out, axis=0)
    assert m.shape == (len(seqs), length, dim), m.shape
    return m, time.time() - t0


def build_embeddings(tag, batch=128, force=False):
    import data

    P = paths(tag)
    if all(os.path.exists(v) for v in P.values()) and not force:
        print(f"ESM-C {tag} caches already present (use --force to rebuild)")
        return

    df = data.load()
    peps = sorted(df.Pep.unique())
    alle = json.load(open("alleles.json"))
    pnames = sorted(n for n in alle if alle[n]["pseudo"])

    tok, model, dev, dim = load_model(tag)
    print(f"device {dev}  model {MODELS[tag][0]}  dim {dim}  "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params", flush=True)
    # The vocabulary must cover the 20 standard residues as single tokens, or
    # the per-position slice would not line up with the peptide's residues.
    probe = tok("ACDEFGHIKLMNPQRSTVWY", add_special_tokens=False)["input_ids"]
    assert len(probe) == 20, f"alphabet is not one token per residue: {len(probe)}"
    assert tok.unk_token_id not in probe, "a standard residue tokenised as <unk>"

    tokens, dt = _encode(peps, PEP_LEN, tok, model, dev, dim, batch, log="pep")
    print(f"peptides: {len(peps)} in {dt:.1f}s ({len(peps) / dt:.0f}/s)", flush=True)
    assert np.isfinite(tokens).all(), "non-finite values in the peptide encoding"
    perres = tokens.reshape(len(peps), PEP_LEN * dim)
    pooled = tokens.mean(axis=1)
    err = float(np.abs(perres.reshape(len(peps), PEP_LEN, dim).mean(1) - pooled).max())
    print(f"max |mean(per-residue) - pooled| = {err:.2e}", flush=True)
    assert err < 1e-5

    np.save(P["perres"], perres.astype(np.float32))
    json.dump({p: i for i, p in enumerate(peps)}, open(P["perres_idx"], "w"))
    np.save(P["pooled"], pooled.astype(np.float32))
    json.dump({p: i for i, p in enumerate(peps)}, open(P["pooled_idx"], "w"))
    print(f"wrote {P['perres']} {perres.shape} and {P['pooled']} {pooled.shape}",
          flush=True)

    tokens, dt = _encode([alle[n]["pseudo"] for n in pnames], PSEUDO_LEN,
                         tok, model, dev, dim, batch=64)
    ps = tokens.mean(axis=1)
    assert np.isfinite(ps).all(), "non-finite values in the pseudo-sequence encoding"
    np.save(P["pseudo"], ps.astype(np.float32))
    json.dump({n: i for i, n in enumerate(pnames)}, open(P["pseudo_idx"], "w"))
    print(f"pseudo: {len(pnames)} in {dt:.1f}s -> {P['pseudo']} {ps.shape}", flush=True)

    # Exactly one collision is expected: B*14:01(C67S) and B*14:02(C67S) have
    # character-identical 34-mer pseudo-sequences, so every encoder must map
    # them to the same point. More than that would mean collapsed grooves.
    d = np.linalg.norm(ps[:, None, :] - ps[None, :, :], axis=-1)
    off = d[~np.eye(len(ps), dtype=bool)]
    n_dup = int(((d < 1e-6).sum() - len(ps)) // 2)
    seq_dup = len(pnames) - len({alle[n]["pseudo"] for n in pnames})
    print(f"pseudo pairwise L2: min {off.min():.3f} median {np.median(off):.3f} "
          f"max {off.max():.3f}  identical pairs {n_dup} (expected {seq_dup})", flush=True)
    assert n_dup == seq_dup, "encoder is collapsing distinct grooves"


# ---------------------------------------------------------------------------
# featurizers -- peptide block FIRST, allele block second (arm I's convention)
# ---------------------------------------------------------------------------

_T: dict = {}


def _tables(tag):
    if tag not in _T:
        P = paths(tag)
        missing = [v for k, v in P.items() if not os.path.exists(v)]
        if missing:
            raise FileNotFoundError(
                f"{missing} missing -- run `python arm_L_esmc.py --model {tag} --embed`")
        _T[tag] = (np.load(P["pooled"]), json.load(open(P["pooled_idx"])),
                   np.load(P["perres"]), json.load(open(P["perres_idx"])),
                   np.load(P["pseudo"]), json.load(open(P["pseudo_idx"])))
    return _T[tag]


def _rows(df_subset, mat, idx, col):
    k = df_subset[col].map(idx).to_numpy()
    if pd.isna(k).any():
        raise KeyError(f"{col} value missing from the ESM-C index")
    return mat[k.astype(int)]


def make_featurizers(tag):
    def pooled_B(df_subset):
        P, pi, _, _, S, si = _tables(tag)
        return np.hstack([_rows(df_subset, P, pi, "Pep"),
                          _rows(df_subset, S, si, "HLA")]).astype(np.float64)

    def perres_B(df_subset):
        _, _, Pr, pri, S, si = _tables(tag)
        return np.hstack([_rows(df_subset, Pr, pri, "Pep"),
                          _rows(df_subset, S, si, "HLA")]).astype(np.float64)

    return pooled_B, perres_B


# ---------------------------------------------------------------------------
# heads, imported from the arms that defined them
# ---------------------------------------------------------------------------

def make_ridge(n_pep_cols):
    import arm_I_perresidue as I
    return lambda s: I.RidgeHead(seed=s, n_pep_cols=n_pep_cols)


def make_mlp(n_pep_cols):
    import arm_A_supervised_nn as A
    return lambda s: A.TorchMLP(seed=s, n_pep_cols=n_pep_cols, **A.BEST)


def spec(tag):
    dim = MODELS[tag][1]
    pooled_B, perres_B = make_featurizers(tag)
    return {
        "pooled_B_ridge": (f"L_esmc{tag}_pooled_B_ridge", pooled_B, dim, "ridge", 1),
        "pooled_B_mlp":   (f"L_esmc{tag}_pooled_B_mlp",   pooled_B, dim, "mlp",   5),
        "perres_B_ridge": (f"L_esmc{tag}_perres_B_ridge", perres_B, PEP_LEN * dim, "ridge", 1),
        "perres_B_mlp":   (f"L_esmc{tag}_perres_B_mlp",   perres_B, PEP_LEN * dim, "mlp",   5),
    }


ORDER = ["pooled_B_ridge", "pooled_B_mlp", "perres_B_ridge", "perres_B_mlp"]


def is_done(arm, seeds):
    """Complete AND readable, so a killed run is never mistaken for a finished one."""
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


def run_one(tag, key, seeds=None, censored="tied", skip_done=True):
    arm, f, npc, head, dflt = spec(tag)[key]
    seeds = dflt if seeds is None else seeds
    if skip_done and is_done(arm, seeds):
        print(f"[{arm}] already complete on disk, skipping", flush=True)
        return None
    import arm_I_perresidue as I
    make = make_mlp(npc) if head == "mlp" else make_ridge(npc)
    if head == "ridge":
        I.RidgeHead.chosen_alpha = []
    t0 = time.time()
    res = R.run_arm(arm, f, make, seeds=seeds, censored=censored)
    if head == "ridge":
        print(f"[{arm}] alphas chosen: "
              f"{pd.Series(I.RidgeHead.chosen_alpha).value_counts().to_dict()}", flush=True)
    print(f"[{arm}] wall clock {time.time() - t0:.1f}s\n", flush=True)
    return res


def run_all(tag, seeds=None, censored="tied", keys=None, skip_done=True):
    out = {}
    for k in (keys or ORDER):
        try:
            out[k] = run_one(tag, k, seeds=seeds, censored=censored, skip_done=skip_done)
        except Exception as e:
            import traceback
            print(f"[{spec(tag)[k][0]}] ARM FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            out[k] = None
    return out


def headline(arm, censored="tied"):
    """Same estimator as pitch_figure.py: median over folds of the per-fold
    median per-allele Spearman, on the across-seed mean prediction."""
    import arm_B_esm_pseudo as B
    if not os.path.exists(f"predictions_{arm}.parquet"):
        return None
    d = B.rescore_ensemble(arm, censored)
    d = d[d.n_alleles_scored > 0]
    if not len(d):
        return None
    return (float(np.median(d.spearman.to_numpy(dtype=float))),
            int(len(d)), int(d.n_seeds.max()))


def report(tag, censored="tied"):
    rows = []
    for k in ORDER:
        arm = spec(tag)[k][0]
        h = headline(arm, censored)
        if h is None:
            print(f"{arm:32s} (no predictions on disk)")
            continue
        med, nf, ns = h
        rows.append({"arm": arm, "spearman": med, "folds": nf, "seeds": ns})
        print(f"{arm:32s} {med:+.4f}  folds {nf}  seeds {ns}")
    if rows:
        pd.DataFrame(rows).to_csv(f"results_L_esmc{tag}_summary.csv", index=False)
        print(f"\n-> results_L_esmc{tag}_summary.csv")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="600m", choices=list(MODELS))
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--which", choices=ORDER)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--censored", default="tied", choices=("tied", "drop"))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()

    if a.embed:
        build_embeddings(a.model, force=a.force)
    elif a.report:
        report(a.model, a.censored)
    elif a.which:
        run_one(a.model, a.which, seeds=a.seeds, censored=a.censored, skip_done=not a.redo)
    elif a.all:
        run_all(a.model, seeds=a.seeds, censored=a.censored, skip_done=not a.redo)
        report(a.model, a.censored)
    else:
        ap.error("pick one of --embed / --which / --all / --report")


if __name__ == "__main__":
    main()
