"""Supertype-merge ROBUSTNESS CHECK. ADDITIONAL analysis, never a replacement.

WHY THIS FILE EXISTS
--------------------
The headline uses 21 folds = groove clusters from 34-residue pseudo-sequence
identity at 0.80 (`supertypes.clusters(0.80)`). Two sets of those clusters share
a single P2 anchor preference. Radin Moradi, the team's biochemist, was asked
directly whether they are single supertypes split too finely by the 0.80 cut and
confirmed these are single supertypes for both sets. The merge below follows his
supertype assignment; the P2 preferences in our own measured data (verified, see
MERGED_CHECK.md section 1) corroborate it but are not the basis for it.

  B44-like, all preferring E/Q at P2:
      {B*40:01, B*40:02, B*41:01, B*45:01} , {B*18:01} , {B*44:05}
  B07-like, all preferring P at P2:
      {B*07:02, B*42:01, B*42:02, B*81:01} , {B*54:01, B*55:01, B*56:01} , {B*51:01}

If each set is really ONE supertype then the 21-fold design leaks: hold out
B*44:05 while B*40:01 stays in training and a groove that takes the same key is
still visible to the model. Merging makes the test HARDER and more honest.
21 folds -> 17 folds.

WHAT IT DOES NOT DO
-------------------
`supertypes.py` and `splits.py` are NOT modified. The merge is defined here, in
`merged_folds()`, on top of the unmodified `splits.groove_folds()` output.
Fine-tuning is NOT re-run. Only four arms are re-run, through the SAME
`run_experiment.run_arm` code path, the same featurizers and the same model
constructors as the main results.

ESTIMATOR (copied from PITCH.md line 4 / RESULTS.md section 2, unchanged)
-------------------------------------------------------------------------
  seed-ensemble MEAN prediction across seeds, censored='tied',
  per-fold = median per-allele Spearman over that fold's held-out alleles
             (metrics.MIN_N = 20 test rows per allele),
  headline = median over folds.
Implemented by `_estimator()` below, which calls `run_experiment._score`, which
calls `metrics.spearman_per_allele`. Identical to `run_experiment.ensemble_metrics`
except that nothing is written to `ensemble_*.csv`.

USAGE
-----
  python merged_check.py folds                 # print the merged fold table
  python merged_check.py worker <key> <outdir> # run one arm (meant to be run in
                                               # a scratch cwd; see run_all.sh)
  python merged_check.py combine <dir> [...]   # collect workers -> results_merged.csv,
                                               # predictions_merged.parquet, and the
                                               # comparison table on stdout

WRITES exactly: results_merged.csv, predictions_merged.parquet (plus whatever a
worker writes inside its own scratch directory). Nothing else in the project is
touched.
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

import data
import run_experiment as R
import splits
import supertypes

# ---------------------------------------------------------------------------
# 1. the merge
# ---------------------------------------------------------------------------

# The two groups of 0.80-clusters to collapse, written as frozensets of allele
# names so the merge cannot silently match the wrong cluster if the clustering
# ever changes. Each entry must appear EXACTLY as a cluster in
# supertypes.clusters(0.80), or merged_folds() raises.
MERGES = {
    "B07-like": [
        frozenset({"HLA-B*07:02", "HLA-B*42:01", "HLA-B*42:02", "HLA-B*81:01"}),
        frozenset({"HLA-B*54:01", "HLA-B*55:01", "HLA-B*56:01"}),
        frozenset({"HLA-B*51:01"}),
    ],
    "B44-like": [
        frozenset({"HLA-B*40:01", "HLA-B*40:02", "HLA-B*41:01", "HLA-B*45:01"}),
        frozenset({"HLA-B*18:01"}),
        frozenset({"HLA-B*44:05"}),
    ],
}


def merged_folds(df=None, cut=splits.CUT, min_test_rows=100):
    """The 21 groove-cluster folds with the six listed clusters collapsed into two.

    Returns [(fold_name, [allele, ...]), ...] in the same shape
    `splits.choose_held_out` returns, biggest fold first, so it can be handed
    straight to `run_experiment.run_arm(folds=...)`.
    """
    df = data.load() if df is None else df

    # verify the six clusters exist exactly as stated, at the unmodified cut
    live = {frozenset(v) for v in supertypes.clusters(cut).values()}
    for group, parts in MERGES.items():
        for p in parts:
            if p not in live:
                raise ValueError(
                    f"{group}: {sorted(p)} is not a cluster of "
                    f"supertypes.clusters({cut}); the merge is not defined")

    base = splits.groove_folds(df, cut, min_test_rows)
    to_merge = {a for parts in MERGES.values() for p in parts for a in p}

    out = []
    for name, members in base:
        if not (set(members) & to_merge):
            out.append((name, members))

    for group, parts in sorted(MERGES.items()):
        members = sorted({a for p in parts for a in p})
        n = int(df[df.HLA.isin(members)].shape[0])
        if n < min_test_rows:                               # pragma: no cover
            raise ValueError(f"{group}: only {n} test rows")
        out.append((f"MERGED {group}", members))

    out.sort(key=lambda t: -int(df[df.HLA.isin(t[1])].shape[0]))
    return out


# ---------------------------------------------------------------------------
# 2. the arms. Same featurizers, same model constructors, same code path.
# ---------------------------------------------------------------------------

def _arm_A():
    import arm_A_supervised_nn as A
    return A.ARM, A.featurizer(A.ENCODING), A.make_model, 5


def _arm_B_mlp():
    import arm_B_esm_pseudo as B
    return B.MLP_ARM, B.featurize, B.MLPHead, 5


def _arm_D():
    import arm_D_peptide_only as D
    return D.ARM, D.featurize, D.make_model, 5


def _arm_E_null():
    import arm_E_allele_mean as E
    return E.TIEBREAK, E.allele_code, E.make_tiebreak, 5


def _arm_I_mlp():
    """The BEST frozen ESM-2 arm in the full ladder: same peptide | pseudo-seq
    encoding, un-pooled (per-residue), through arm A's own tuned MLP."""
    import arm_I_perresidue as I
    name, featurize, npc, head = I.SPEC["Bmlp"]
    assert head == "mlp"
    return name, featurize, I.make_mlp(npc), 5


ARMS = {
    "A": ("A_supervised_nn", _arm_A),
    "B": ("B_esm_pseudo_mlp", _arm_B_mlp),
    "D": ("D_peptide_only", _arm_D),
    "E": ("E_allele_mean__tiebreak", _arm_E_null),
    "I": ("I_perres_B_pseudo_mlp", _arm_I_mlp),
}

LABEL = {
    "A_supervised_nn": "arm A, conventional supervised net (BLOSUM62 + one-hot, no FM)",
    "I_perres_B_pseudo_mlp": "BEST frozen ESM-2 150M, peptide | pseudo-seq un-pooled, tuned MLP",
    "B_esm_pseudo_mlp": "frozen ESM-2 150M, peptide | pseudo-seq pooled, MLP head",
    "D_peptide_only": "CONTROL, ESM-2 150M peptide only (allele never shown)",
    "E_allele_mean__tiebreak": "NULL, allele mean + tie-break",
}


def worker(key):
    """Run one arm over the merged folds. Writes results_/predictions_ for
    `merged__<arm>` INTO THE CURRENT DIRECTORY, which is expected to be a scratch
    directory, not the project."""
    arm, maker = ARMS[key]
    name, featurize, make_model, seeds = maker()
    df = R.load_df()
    folds = merged_folds(df)
    assert len(folds) == 17, f"expected 17 merged folds, got {len(folds)}"
    R.run_arm(f"merged__{name}", featurize, make_model, seeds=seeds,
              censored="tied", folds=folds, df=df, verbose=True)


# ---------------------------------------------------------------------------
# 3. the estimator. Identical to run_experiment.ensemble_metrics, minus its
#    side-effect of writing ensemble_<name>.csv.
# ---------------------------------------------------------------------------

def _estimator(pred, df, censored="tied"):
    """(per-fold Series of median per-allele Spearman, headline median over folds).

    pred must have columns fold_name, row_id, y_pred, seed.
    """
    g = pred.groupby(["fold_name", "row_id"], sort=False)
    agg = g.y_pred.agg(["mean", "count"]).reset_index()
    agg.columns = ["fold_name", "row_id", "pred_mean", "n_seeds"]
    rows = {}
    for fold_name, sub in agg.groupby("fold_name", sort=False):
        test_df = df.loc[sub.row_id.to_numpy()]
        sc, _rho, _top = R._score(test_df, sub.pred_mean.to_numpy(dtype=np.float64),
                                  censored, None)
        rows[fold_name] = sc["spearman"]
    s = pd.Series(rows, dtype=float).sort_index()
    return s, float(np.nanmedian(s.to_numpy()))


def _orig(name, df, censored="tied"):
    """The ORIGINAL 21-fold number for the same arm, recomputed from its stored
    parquet with the same `_estimator`. Never copied from a table."""
    return _estimator(pd.read_parquet(f"predictions_{name}.parquet"), df, censored)


# ---------------------------------------------------------------------------
# 4. combine + report
# ---------------------------------------------------------------------------

def combine(dirs):
    df = data.load()
    res_all, pred_all = [], []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "results_merged__*.csv"))):
            arm = os.path.basename(p)[len("results_merged__"):-len(".csv")]
            q = os.path.join(d, f"predictions_merged__{arm}.parquet")
            if not os.path.exists(q):
                print(f"  !! no predictions for {arm}, skipping", file=sys.stderr)
                continue
            r = pd.read_csv(p)
            r["arm"] = arm
            res_all.append(r)
            pq = pd.read_parquet(q)
            pq.insert(0, "arm", arm)
            pred_all.append(pq)
    if not res_all:
        raise SystemExit("no worker output found")

    res = pd.concat(res_all, ignore_index=True)
    pred = pd.concat(pred_all, ignore_index=True)
    res.to_csv("results_merged.csv", index=False)
    pred.to_parquet("predictions_merged.parquet", index=False)
    print(f"wrote results_merged.csv ({len(res)} rows) and "
          f"predictions_merged.parquet ({len(pred)} rows)\n")

    folds17 = merged_folds(df)
    folds21 = splits.choose_held_out(df)
    print(f"merged folds: {len(folds17)}   original folds: {len(folds21)}\n")

    table = []
    per_fold = {}
    for arm in pred.arm.unique():
        m_s, m_med = _estimator(pred[pred.arm == arm], df)
        o_s, o_med = _orig(arm, df)
        per_fold[arm] = (m_s, o_s)
        table.append({
            "arm": arm, "label": LABEL.get(arm, ""),
            "merged_17": round(m_med, 4), "original_21": round(o_med, 4),
            "delta": round(m_med - o_med, 4),
            "n_folds_merged": int(m_s.notna().sum()),
            "n_folds_orig": int(o_s.notna().sum()),
        })
    t = pd.DataFrame(table)
    order = ["A_supervised_nn", "I_perres_B_pseudo_mlp", "B_esm_pseudo_mlp",
             "D_peptide_only", "E_allele_mean__tiebreak"]
    t["_o"] = t.arm.map({a: i for i, a in enumerate(order)}).fillna(99)
    t = t.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    print(t.to_string(index=False))

    print("\nTHE TWO MERGED FOLDS, individually:")
    comp = {"MERGED B07-like": ["B*07:02 +3", "B*54:01 +2", "B*51:01"],
            "MERGED B44-like": ["B*40:01 +3", "B*18:01", "B*44:05"]}
    for arm in order:
        if arm not in per_fold:
            continue
        m_s, o_s = per_fold[arm]
        print(f"\n  {arm}")
        for f, parts in comp.items():
            got = m_s.get(f, np.nan)
            old = {p: o_s.get(p, np.nan) for p in parts}
            oldtxt = ", ".join(f"{k} {v:+.4f}" for k, v in old.items())
            print(f"    {f:<18} merged {got:+.4f}   <- was [{oldtxt}]")

    print("\nUNCHANGED FOLDS (15): merged vs original, same fold, retrained "
          "because the training side grew smaller")
    for arm in order:
        if arm not in per_fold:
            continue
        m_s, o_s = per_fold[arm]
        shared = sorted(set(m_s.index) & set(o_s.index))
        d = pd.Series({f: m_s[f] - o_s[f] for f in shared}).dropna()
        print(f"  {arm:<26} n={len(d)}  median delta {d.median():+.4f}  "
              f"min {d.min():+.4f}  max {d.max():+.4f}")

    print("\nPAIRED OVER THE 17 MERGED FOLDS (Wilcoxon signed rank, two-sided), "
          "and the same pair over the 21 original folds for reference")
    pairs = [("A_supervised_nn", "I_perres_B_pseudo_mlp", "conventional net - BEST frozen ESM-2"),
             ("A_supervised_nn", "B_esm_pseudo_mlp", "conventional net - pooled ESM-2 MLP"),
             ("A_supervised_nn", "D_peptide_only", "conventional net - peptide-only control"),
             ("A_supervised_nn", "E_allele_mean__tiebreak", "conventional net - null"),
             ("I_perres_B_pseudo_mlp", "D_peptide_only", "BEST frozen ESM-2 - peptide-only control"),
             ("B_esm_pseudo_mlp", "D_peptide_only", "pooled ESM-2 MLP - peptide-only control"),
             ("B_esm_pseudo_mlp", "E_allele_mean__tiebreak", "pooled ESM-2 MLP - null")]
    print(f"  {'comparison':<42} {'folds':>6} {'medianD':>9} {'won':>7} {'p':>10}   "
          f"| {'21-fold medianD':>15} {'won':>7} {'p':>10}")
    for a, b, lab in pairs:
        if a not in per_fold or b not in per_fold:
            continue
        line = f"  {lab:<42}"
        for which, n_tot in ((0, 17), (1, 21)):
            x = per_fold[a][which].dropna()
            y = per_fold[b][which].dropna()
            k = sorted(set(x.index) & set(y.index))
            d = (x[k] - y[k]).to_numpy()
            p = _wilcoxon(d)
            won = int((d > 0).sum())
            if which == 0:
                line += f" {len(k):>6} {np.median(d):>+9.4f} {won:>4}/{len(k):<2} {p:>10.2e}   |"
            else:
                line += f" {np.median(d):>+15.4f} {won:>4}/{len(k):<2} {p:>10.2e}"
        print(line)
    return t


def _wilcoxon(d):
    from scipy import stats
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    if not len(d) or np.allclose(d, 0):                     # pragma: no cover
        return float("nan")
    return float(stats.wilcoxon(d).pvalue)


def print_folds():
    df = data.load()
    f17 = merged_folds(df)
    f21 = splits.choose_held_out(df)
    print(f"original {len(f21)} folds -> merged {len(f17)} folds\n")
    print(f"{'rows':>6}  {'n':>2}  fold")
    for n, m in f17:
        print(f"{int(df[df.HLA.isin(m)].shape[0]):>6}  {len(m):>2}  {n:<18} "
              f"{', '.join(x.replace('HLA-', '') for x in m)}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "folds"
    if cmd == "folds":
        print_folds()
    elif cmd == "worker":
        worker(sys.argv[2])
    elif cmd == "combine":
        combine(sys.argv[2:] or ["."])
    else:                                                   # pragma: no cover
        raise SystemExit(__doc__)
