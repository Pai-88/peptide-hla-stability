"""Single-mutant design scan: which substitution most stabilises this peptide?

METHOD, and why it is enumeration rather than optimisation.

A 9-mer has 9 positions x 19 alternative residues = 171 single mutants. That is small
enough to score exhaustively. Sequence space for a free 9-mer is 20^9, about 5e11; pointing
a search algorithm at that against a forward model with ~0.28 Spearman on unseen grooves
finds the model's blind spots, not good peptides. Staying inside single mutants of a real,
measured peptide keeps every candidate near the data the model was fitted on. That
restriction IS the safety mechanism.

The forward model is the conventional supervised net (arm A), not a foundation model,
because it is the one that works: 0.277 vs 0.096 for frozen ESM-2 on groove-held-out folds.

The scan REFUSES when the allele is out of distribution. A ranking we cannot stand behind
is worse than no ranking.
"""

import json
import os

import numpy as np
import pandas as pd

import arm_A_supervised_nn as armA
import data

AA = "ACDEFGHIKLMNPQRSTVWY"
CACHE = "design_ensemble.npz"
N_SEEDS = 5


def _ood(allele):
    """('in'|'borderline'|'out', headline). Degrades to 'unknown' if ood.py is mid-edit."""
    try:
        import ood
        v = ood.verdict(allele)
        return v[0], v[1]
    except Exception as e:
        return "unknown", f"out-of-distribution check unavailable ({type(e).__name__})"


def _label(peptide):
    try:
        import sources
        return sources.label(peptide, default="")
    except Exception:
        return ""


class Ensemble:
    """N seeds trained on the whole dataset. Spread across seeds is the uncertainty."""

    def __init__(self, models, featurize):
        self.models, self.featurize = models, featurize

    def predict(self, df):
        X = self.featurize(df)
        P = np.stack([m.predict(X) for m in self.models])
        return P.mean(0), P.std(0)


def train(n_seeds=N_SEEDS, verbose=True):
    df = data.load()
    f = armA.featurizer(armA.ENCODING)
    X, y = f(df), df.y.to_numpy()
    models = []
    for s in range(n_seeds):
        m = armA.make_model(s)
        m.fit(X, y)
        models.append(m)
        if verbose:
            print(f"  seed {s} trained on {len(df):,} rows")
    return Ensemble(models, f)


def mutants(peptide):
    """All 171 single-point mutants, in (position, residue) order."""
    out = []
    for i, wt in enumerate(peptide):
        for aa in AA:
            if aa != wt:
                out.append((i + 1, wt, aa, peptide[:i] + aa + peptide[i + 1:]))
    return out


def scan(ens, peptide, allele, top=10):
    """Rank every single mutant. Returns (verdict, headline, DataFrame or None)."""
    peptide = peptide.strip().upper()
    if len(peptide) != 9 or any(c not in AA for c in peptide):
        raise ValueError("peptide must be 9 standard amino acids")

    v, headline = _ood(allele)
    if v == "out":
        return v, headline, None

    muts = mutants(peptide)
    df = pd.DataFrame({
        "HLA": allele,
        "Pep": [peptide] + [m[3] for m in muts],
        "pos": [0] + [m[0] for m in muts],
        "wt": [""] + [m[1] for m in muts],
        "mut": [""] + [m[2] for m in muts],
    })
    mean, sd = ens.predict(df)
    df["pred_thalf_h"] = 10 ** mean
    df["sd_log10"] = sd

    wt_row = df.iloc[0]
    df = df.iloc[1:].copy()
    df["fold_change"] = df.pred_thalf_h / wt_row.pred_thalf_h
    df["substitution"] = df.wt + df.pos.astype(str) + df.mut
    df = df.sort_values("pred_thalf_h", ascending=False)

    df.attrs["wt_pred"] = float(wt_row.pred_thalf_h)
    df.attrs["wt_sd"] = float(wt_row.sd_log10)
    return v, headline, df.head(top)[
        ["substitution", "Pep", "pred_thalf_h", "fold_change", "sd_log10"]]


if __name__ == "__main__":
    import sys
    pep = sys.argv[1] if len(sys.argv) > 1 else "VTTEVAFGL"
    hla = sys.argv[2] if len(sys.argv) > 2 else "HLA-A*02:01"

    print(f"training {N_SEEDS}-seed ensemble on all measured data")
    ens = train()

    src = _label(pep)
    print(f"\n{pep}  on  {hla}" + (f"\n{src}" if src else ""))

    v, headline, top = scan(ens, pep, hla)
    print(f"\nconfidence: {v.upper()}  |  {headline}")
    if top is None:
        print("\nREFUSED. This groove is unlike anything with measured stability.")
        print("A ranking here would be a guess, so the tool does not produce one.")
        raise SystemExit(0)

    _wt = float(top.pred_thalf_h.iloc[0]) / float(top.fold_change.iloc[0])
    print(f"\nwild type {pep} predicted half-life: {_wt:.2f} h")
    print(top.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print("\nsd_log10 is the spread across seeds. Wide spread means do not trust the row.")
