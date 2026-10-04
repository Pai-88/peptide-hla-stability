"""Arm J: does ESM-2 add ANYTHING on top of conventional features?

The sharper, more defensible question. "Does ESM-2 beat BLOSUM" confounds the
representation with the head and the tuning budget. "Does ESM-2 add information
the conventional encoding does not already carry" does not: same head, same
folds, same inner-split protocol, one thing changed.

Four feature sets, all through an IDENTICAL StandardScaler + Ridge head:

  J_conv        conventional only   BLOSUM62 + one-hot, peptide and 34-mer
                                    pseudo-sequence, arm A's `both` encoding   1720
  J_esm         ESM-2 only          mean-pooled peptide | mean-pooled pseudo
                                    (arm B's features)                         1280
  J_hybrid      conventional | ESM-2 mean-pooled                               3000
  J_hybrid_pr   conventional | ESM-2 PER-RESIDUE peptide (arm I) | ESM pseudo  8120

Reading:
  hybrid  > conv   -> ESM-2 carries complementary information. A real finding.
  hybrid == conv   -> ESM-2 adds nothing on top. Also real, and stronger.

J_conv is deliberately NOT arm A. Arm A is a tuned MLP; putting the conventional
features through the SAME ridge head as everything else is what makes the delta
attributable to the features. Arm A's number stays the project's headline for
the conventional approach and is reported alongside, not replaced.

Caveat stated up front: one ridge alpha is shared across blocks of very
different width and scale, which can only penalise the hybrid relative to the
single-block arms. A hybrid that fails to beat conv under this head is therefore
weak evidence of "adds nothing"; a hybrid that beats it is strong evidence of
"adds something".

Usage:
    python arm_J_hybrid.py                 # all four, censored='tied'
    python arm_J_hybrid.py --which hybrid
    python arm_J_hybrid.py --drop          # robustness re-score
    python arm_J_hybrid.py --paired        # paired per-fold test vs J_conv

Nothing runs at import time.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy import stats

import arm_A_supervised_nn as A
import arm_B_esm_pseudo as B
import arm_I_perresidue as I
import metrics
import run_experiment as R

CONV_MODE = "both"                 # arm A's chosen encoding
N_CONV = 9 * 40 + 34 * 40          # 1720
N_CONV_PEP = 9 * 40                # 360, the peptide block (first)

# arm -> (description, width of the leading PEPTIDE block). Every featurizer
# puts a peptide-only block first, so the peptide-grouped inner split is
# recovered the same way in all four arms.
ARMS = {
    "J_conv":      ("conventional only (BLOSUM+one-hot)",            N_CONV_PEP),
    "J_esm":       ("ESM-2 only (mean-pooled pep | pseudo)",         640),
    "J_hybrid":    ("conventional | ESM-2 mean-pooled",              N_CONV_PEP),
    "J_hybrid_pr": ("conventional | ESM-2 per-residue | ESM pseudo", N_CONV_PEP),
}

ALPHAS = (1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0)


# ---------------------------------------------------------------------------
# features: every one starts with the conventional peptide block, so the same
# column count identifies the peptide for the inner split in every arm.
# ---------------------------------------------------------------------------

_CONV = None


def _conv():
    global _CONV
    if _CONV is None:
        _CONV = A.featurizer(CONV_MODE)
    return _CONV


def f_conv(d):
    X = _conv()(d)
    assert X.shape[1] == N_CONV, X.shape
    return X


def f_esm(d):
    """Arm B's features exactly: [mean-pooled peptide (640) | pseudo (640)].
    Reproduced here under this arm's head so the comparison is head-controlled."""
    return B.featurize(d)


def f_hybrid(d):
    return np.hstack([f_conv(d), B.featurize(d)])


def f_hybrid_pr(d):
    return np.hstack([f_conv(d), I.featurize_B(d)])


FEATS = {"J_conv": f_conv, "J_esm": f_esm,
         "J_hybrid": f_hybrid, "J_hybrid_pr": f_hybrid_pr}


# ---------------------------------------------------------------------------
# the one head, shared by all four arms
# ---------------------------------------------------------------------------

class RidgeHead(I.RidgeHead):
    def __init__(self, seed=0, n_pep_cols=N_CONV_PEP, alphas=ALPHAS):
        super().__init__(seed=seed, n_pep_cols=n_pep_cols, alphas=alphas)


def run(which="all", seeds=1, censored="tied"):
    names = list(ARMS) if which == "all" else [which]
    out = {}
    for arm in names:
        RidgeHead.chosen_alpha = []
        I.RidgeHead.chosen_alpha = []
        t0 = time.time()
        npc = ARMS[arm][1]
        out[arm] = R.run_arm(arm, FEATS[arm], lambda s, n=npc: RidgeHead(seed=s, n_pep_cols=n),
                             seeds=seeds, censored=censored)
        print(f"[{arm}] alphas: "
              f"{pd.Series(I.RidgeHead.chosen_alpha).value_counts().to_dict()}")
        print(f"[{arm}] {ARMS[arm][0]}  wall clock {time.time() - t0:.1f}s\n", flush=True)
    return out


def rescore(arm, censored="drop"):
    return I.rescore(arm, censored)


# ---------------------------------------------------------------------------
# the paired test -- the actual deliverable of this arm
# ---------------------------------------------------------------------------

def _perfold(arm, censored="tied"):
    """fold -> median-over-alleles Spearman. Ridge is deterministic, so the
    5-seed ensemble mean prediction is identical to the single fit and the two
    estimator conventions coincide; one seed is run and that is stated."""
    if censored == "tied":
        r = pd.read_csv(f"results_{arm}.csv")
        r = r[r.status == "ok"]
    else:
        r = pd.read_csv(f"results_{arm}__{censored}.csv")
    return r.groupby("fold_name").spearman.mean()


def paired(arms=("J_esm", "J_hybrid", "J_hybrid_pr"), ref="J_conv", censored="tied"):
    base = _perfold(ref, censored)
    rows = []
    for arm in arms:
        a = _perfold(arm, censored)
        common = base.index.intersection(a.index)
        d = (a[common] - base[common]).to_numpy()
        w = stats.wilcoxon(d) if len(d) > 5 else None
        t = stats.ttest_rel(a[common], base[common])
        rows.append({
            "arm": arm, "ref": ref, "censored": censored, "n_folds": len(common),
            "median_rho": float(np.median(a[common])),
            "ref_median_rho": float(np.median(base[common])),
            "median_delta": float(np.median(d)),
            "mean_delta": float(d.mean()),
            "folds_improved": int((d > 0).sum()),
            "wilcoxon_p": float(w.pvalue) if w is not None else np.nan,
            "paired_t_p": float(t.pvalue),
        })
    out = pd.DataFrame(rows)
    path = f"results_J_paired__{censored}.csv"
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    print(f"-> {path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--drop", action="store_true")
    ap.add_argument("--paired", action="store_true")
    a = ap.parse_args()
    if a.paired:
        paired(censored="tied")
        try:
            paired(censored="drop")
        except FileNotFoundError:
            print("(no drop re-score yet)")
        return
    if a.drop:
        for arm in (list(ARMS) if a.which == "all" else [a.which]):
            rescore(arm, "drop")
        return
    run(which=a.which, seeds=a.seeds)


if __name__ == "__main__":
    main()
