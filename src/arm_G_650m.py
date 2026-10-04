"""ARM G -- does a 4x bigger protein foundation model change the conclusion?

Arms B and D used ESM-2 150M (esm2_t30_150M_UR50D, 640 dims). This arm re-runs
the SAME code against ESM-2 650M (esm2_t33_650M_UR50D, 1280 dims). Nothing else
moves: same folds, same splits, same inner-split protocol, same heads, same
hyperparameters, same metrics, same censoring policy, same seeds.

HOW "nothing else moves" IS ENFORCED
  This file defines no featurizer and no model. It imports arm_B_esm_pseudo and
  arm_D_peptide_only and calls THEIR featurize / RidgeHead / MLPHead /
  make_model objects. The single intervention is the documented shim:

      embed.load = lambda n: (np.load(f"emb650_{n}.npy"),
                              json.load(open(f"emb650_{n}_index.json")))

  applied before either module's lazy embedding cache is filled. So if arm B's
  head is wrong, arm G is wrong in exactly the same way, which is the point.

THE THREE SUB-ARMS
  G_650m_B_ridge          [peptide 1280 | pseudo-seq 1280] -> ridge   (vs B 0.060)
  G_650m_B_mlp            same 2560 features -> 2560-512-128-1 MLP    (vs B 0.096)
  G_650m_D_peptide_only   peptide 1280 only, no allele at all         (vs D 0.074)

  emb650_joint.npy does not exist (the joint encoding needs a GPU that is
  payment-gated), so there is no 650M equivalent of arm C and none is faked.

THE TEST THAT MATTERS
  Not "did 650M beat 150M". The test is whether 650M-with-the-allele beats
  650M-without-the-allele -- its OWN control, at its OWN scale. An arm that
  cannot beat a model forbidden from seeing which HLA it is predicting for has
  not learned peptide-HLA specificity, however the headline Spearman moves.

HEADLINE CONVENTION
  Median over all (fold, seed) rows with status == 'ok' of the per-allele median
  Spearman, censored='tied'. Verified in main() to reproduce the published
  150M numbers (A 0.277, B 0.060 / 0.096, C 0.071, D 0.074, E 0.005) from the
  existing results_*.csv before any 650M number is reported.

Run:    python arm_G_650m.py                 # all three sub-arms, then report
        python arm_G_650m.py --folds 2       # smoke, first 2 folds
        python arm_G_650m.py --report        # re-report from existing files
Writes: results_G_650m.csv              headline comparison, one row per arm
        results_G_650m_perfold.csv      paired 650M-vs-150M per fold
        results_G_650m_<sub>.csv        raw run_arm output, per sub-arm
        predictions_G_650m_<sub>.parquet
        ensemble_G_650m_<sub>.csv       written by run_experiment.ensemble_metrics
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

import embed

# ---------------------------------------------------------------------------
# THE SHIM. Must land before arm_B._tables() or arm_D._emb() caches anything.
# Neither module touches embed.load at import time, but the caches are cleared
# below as well so this file is safe to import after them in any order.
# ---------------------------------------------------------------------------

_EMB150_LOAD = embed.load


def load650(name):
    return (np.load(f"emb650_{name}.npy"),
            json.load(open(f"emb650_{name}_index.json")))


embed.load = load650

import arm_B_esm_pseudo as B          # noqa: E402  (must follow the shim)
import arm_D_peptide_only as D        # noqa: E402
import metrics                        # noqa: E402
import run_experiment as R            # noqa: E402
import splits                         # noqa: E402

B._TABLES = None
D._EMB = None

RIDGE_ARM = "G_650m_B_ridge"
MLP_ARM = "G_650m_B_mlp"
PEPONLY_ARM = "G_650m_D_peptide_only"

# (650M arm, the 150M arm it is the direct equivalent of, what it is)
PAIRS = [
    (RIDGE_ARM,   "B_esm_pseudo",     "peptide + pseudo-seq, ridge"),
    (MLP_ARM,     "B_esm_pseudo_mlp", "peptide + pseudo-seq, MLP"),
    (PEPONLY_ARM, "D_peptide_only",   "peptide only (control, no allele)"),
]

# RESULTS.md section 2 is the authority. It mandates ONE estimator and says the
# arms originally quoted three different ones. The per-(fold, seed) median this
# file used in its first revision is one of the RETIRED conventions; the numbers
# in the original task brief (A 0.277, B 0.060/0.096, D 0.074) are that retired
# estimator. Do not reinstate them. Published section-2 values, ensemble:
LOOKUP = "lookup_table"

PUBLISHED = {
    "A_supervised_nn":  (0.307, "supervised NN, no FM            (row 1)"),
    LOOKUP:             (0.130, "peptide-mean lookup  CONTROL    (row 2)"),
    "B_esm_pseudo_mlp": (0.106, "ESM 150M pep|pseudo, MLP        (row 3)"),
    RIDGE_ARM:          (0.090, "ESM 650M pep|pseudo, ridge      (row 4)"),
    "C_esm_joint":      (0.077, "ESM 150M joint encoding         (row 5)"),
    "D_peptide_only":   (0.064, "ESM 150M peptide-only CONTROL   (row 6)"),
    "B_esm_pseudo":     (0.061, "ESM 150M pep|pseudo, ridge      (row 7)"),
}


# ---------------------------------------------------------------------------
# THE estimator, as RESULTS.md section 2 defines it:
#   5-seed ensemble MEAN prediction, censored='tied', per-fold = median
#   per-allele Spearman over that fold's held-out alleles, headline = median
#   over the 21 folds.
# B.rescore_ensemble is that computation and writes nothing.
# ---------------------------------------------------------------------------

_ENS = {}


def lookup_table_folds(df=None):
    """Ladder row 2: predict the mean training y of that exact peptide.

    No model, no embedding, never the allele. Deterministic, so its 5-seed
    ensemble IS the single fit. Recomputed here because the repo stores no
    predictions_*.parquet for it; verify() checks it reproduces 0.130.
    """
    df = R.load_df() if df is None else df
    out = {}
    for fold, alle in splits.choose_held_out(df):
        tr, te = splits.split_by_allele(df, alle)
        trd, ted = df.loc[tr], df.loc[te]
        pm = ted.Pep.map(trd.groupby("Pep").y.mean()).fillna(trd.y.mean()).values
        out[fold] = float(np.median(
            metrics.spearman_per_allele(ted, pm, censored="tied")))
    return pd.Series(out)


def fold_medians(arm):
    """Per-fold ensemble Spearman -> Series indexed by fold_name."""
    if arm not in _ENS:
        _ENS[arm] = (lookup_table_folds() if arm == LOOKUP else
                     B.rescore_ensemble(arm, "tied").set_index("fold_name").spearman)
    return _ENS[arm]


def headline(arm):
    """(median, iqr_lo, iqr_hi, worst_fold, n_folds, n_folds_le_zero)."""
    v = fold_medians(arm).dropna()
    if not len(v):
        return (np.nan,) * 4 + (0, 0)
    return (float(np.median(v)), float(np.percentile(v, 25)),
            float(np.percentile(v, 75)), float(v.min()),
            len(v), int((v <= 0).sum()))


def verify():
    """Reproduce RESULTS.md section 2 before reporting any new number."""
    print("-- reproducing RESULTS.md section 2 under its mandated estimator --")
    ok_all = True
    for arm, (want, what) in PUBLISHED.items():
        try:
            got = headline(arm)[0]
        except FileNotFoundError:
            print(f"  {what:<34} results file missing -- cannot verify")
            ok_all = False
            continue
        hit = abs(got - want) < 6e-4
        ok_all &= hit
        print(f"  {what:<34} published {want:.3f}   recomputed {got:+.4f}   "
              f"{'MATCH' if hit else 'MISMATCH'}")
    print(f"  -> estimator {'confirmed' if ok_all else 'DOES NOT MATCH -- STOP'}"
          f"  (5-seed ensemble, tied, median over folds)\n")
    return ok_all


# ---------------------------------------------------------------------------
# feature sanity, printed once so the run log carries it
# ---------------------------------------------------------------------------

def sanity(df):
    """Prove the 650M tables are wired in and arm B's peptide-group trick still
    recovers peptide identity at the new dimensionality."""
    P150, _ = _EMB150_LOAD("peptides")
    P650, i650 = embed.load("peptides")
    S650, s650 = embed.load("pseudo")
    print(f"embeddings: 150M peptides {P150.shape} -> 650M peptides {P650.shape}, "
          f"650M pseudo {S650.shape}")
    print(f"  peptide index keys identical to 150M: "
          f"{i650 == _EMB150_LOAD('peptides')[1]}")
    print(f"  650M finite: {bool(np.isfinite(P650).all() and np.isfinite(S650).all())}")

    Xb = B.featurize(df)
    g = B._peptide_groups(Xb)
    n_rec, n_true = int(g.max() + 1), int(df.Pep.nunique())
    print(f"  arm-B features {Xb.shape}  peptide groups recovered {n_rec} "
          f"vs {n_true} distinct peptides  {'OK' if n_rec == n_true else 'MISMATCH'}")
    if n_rec != n_true:
        raise RuntimeError("peptide-group recovery broke at 1280 dims; the inner "
                           "split would leak. Refusing to run.")
    Xd = D.featurize(df)
    print(f"  arm-D features {Xd.shape}  (peptide only, no allele column read)")
    del Xb, Xd


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _line(tag, v):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    if not len(v):
        return f"{tag:<40} no folds scored"
    return (f"{tag:<40} median {np.median(v):+.4f}  "
            f"IQR [{np.percentile(v, 25):+.4f}, {np.percentile(v, 75):+.4f}]  "
            f"worst {v.min():+.4f}  n {len(v)}")


BG, FG, MUTED = "#0d0d0f", "#f2f2f2", "#8a8a93"
C150, C650, CA = "#6f7380", "#4ad3a0", "#e4b363"


def make_figure(pf, path="arm_G_650m_scale.png"):
    """Scaling 150M -> 650M against the two reference lines that matter.

    Bar = median over 21 folds of the 5-seed ensemble, whisker = IQR across
    folds. The two dashed lines are the controls: a model at or below them has
    shown no evidence of using the allele.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arm_a = headline("A_supervised_nn")[0]
    ctrl650 = headline(PEPONLY_ARM)[0]
    look = headline(LOOKUP)[0]
    labels = ["peptide + pseudo-seq\nridge head",
              "peptide + pseudo-seq\nMLP head",
              "peptide only\nCONTROL, allele never shown"]

    fig, ax = plt.subplots(figsize=(15, 9), facecolor=BG)
    ax.set_facecolor(BG)
    x, w = np.arange(3), 0.36

    for i, (scale, col) in enumerate((("150M", C150), ("650M", C650))):
        off = (i - 0.5) * w
        med, lo, hi = [], [], []
        for a650, a150, _ in PAIRS:
            m, q1, q3, _mn, _n, _z = headline(a650 if scale == "650M" else a150)
            med.append(m); lo.append(m - q1); hi.append(q3 - m)
        ax.bar(x + off, med, w, color=col, zorder=3, edgecolor="none",
               label=f"ESM-2 {scale}")
        ax.errorbar(x + off, med, yerr=[lo, hi], fmt="none", ecolor="#cfcfd6",
                    elinewidth=2.2, capsize=7, capthick=2.2, zorder=4)
        for xi, m, h in zip(x + off, med, hi):
            ax.text(xi, m + h + 0.010, f"{m:.3f}", ha="center", va="bottom",
                    color=FG, fontsize=21, fontweight="bold", zorder=5)

    for yv, col, ls, txt in (
            (arm_a, CA, "-", f"conventional supervised net, no FM   {arm_a:.3f}"),
            (look, "#d98fb0", ":", f"peptide-mean lookup table, no model   {look:.3f}"),
            (ctrl650, C650, "--", f"650M peptide-only control   {ctrl650:.3f}")):
        ax.axhline(yv, color=col, lw=2.5, ls=ls, zorder=2)
        ax.text(-0.46, yv + 0.007, txt, color=col, fontsize=15.5,
                va="bottom", ha="left", fontweight="bold", zorder=6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=19, color=FG)
    ax.tick_params(axis="x", length=0, pad=14)
    ax.tick_params(axis="y", labelsize=17, colors=FG, length=6, width=2)
    ax.set_ylabel("Spearman rho", fontsize=22, color=FG, labelpad=12)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, arm_a * 1.16)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.yaxis.grid(True, color="#26262c", lw=1.2, zorder=0)
    ax.set_axisbelow(True)
    leg = ax.legend(fontsize=18, frameon=False, loc="upper center", ncol=2,
                    handlelength=1.3)
    for t in leg.get_texts():
        t.set_color(FG)

    fig.suptitle("Scaling ESM-2 4.3x does not close the gap to a conventional net",
                 fontsize=27, color=FG, fontweight="bold", x=0.045, ha="left",
                 y=0.975)
    fig.text(0.045, 0.908,
             "Frozen embeddings, 150M vs 650M. Identical folds, splits, heads, "
             "hyperparameters and metrics; only the encoder changes.",
             fontsize=16.5, color=MUTED, ha="left")
    fig.text(0.045, 0.055,
             "5-seed ensemble mean, censored=tied, median per-allele Spearman over "
             "21 groove-cluster folds; whisker is the IQR across folds.",
             fontsize=14, color=MUTED, ha="left")
    fig.text(0.045, 0.018,
             "Supervised net - 650M MLP = +0.172, 19/21 folds, Wilcoxon p=1.3e-5. "
             "No 650M arm separates from the lookup table (p>=0.37).",
             fontsize=14, color=MUTED, ha="left")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.855, bottom=0.20)
    fig.savefig(path, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"\nwrote {path}")
    return path


def _ens_top10(arm):
    if arm == LOOKUP:
        return np.nan
    return R._nanmedian(B.rescore_ensemble(arm, "tied").top10_precision)


def summarise():
    """Build results_G_650m.csv and results_G_650m_perfold.csv, print both."""
    from scipy.stats import wilcoxon

    ref = [("A_supervised_nn", "supervised NN, no FM (ladder row 1)", "-"),
           (LOOKUP, "peptide-mean lookup CONTROL (row 2)", "-")]
    arms = ref + [(a150, f"{w} [150M]", "150M") for a650, a150, w in PAIRS] \
               + [(a650, f"{w} [650M]", "650M") for a650, a150, w in PAIRS]

    ctrl650 = headline(PEPONLY_ARM)[0]
    ctrl150 = headline("D_peptide_only")[0]
    arm_a = headline("A_supervised_nn")[0]
    look = headline(LOOKUP)[0]

    rows = []
    for arm, what, scale in arms:
        med, lo, hi, worst, n_f, n_le0 = headline(arm)
        ctrl = ctrl650 if scale == "650M" else ctrl150
        rows.append({
            "arm": arm, "model_scale": scale, "what": what,
            "spearman_median_5seed_ensemble_tied": round(med, 4),
            "iqr_lo": round(lo, 4), "iqr_hi": round(hi, 4),
            "worst_fold": round(worst, 4),
            "top10_precision": (None if np.isnan(_ens_top10(arm))
                                else round(_ens_top10(arm), 4)),
            "n_folds": n_f, "folds_le_zero": n_le0,
            "delta_vs_supervised_NN": round(med - arm_a, 4),
            "delta_vs_lookup_table": round(med - look, 4),
            "delta_vs_peptide_only_control_same_scale": (
                None if scale == "-" else round(med - ctrl, 4)),
        })
    out = pd.DataFrame(rows)
    out.to_csv("results_G_650m.csv", index=False)

    pf_rows = []
    for a650, a150, what in PAIRS:
        x, y = fold_medians(a150), fold_medians(a650)
        for f in x.index.intersection(y.index):
            pf_rows.append({"what": what, "fold_name": f,
                            "rho_150M": round(float(x[f]), 4),
                            "rho_650M": round(float(y[f]), 4),
                            "delta_650_minus_150": round(float(y[f] - x[f]), 4)})
    pf = pd.DataFrame(pf_rows)
    pf.to_csv("results_G_650m_perfold.csv", index=False)

    print("\n" + "=" * 78)
    print("ARM G -- ESM-2 650M vs 150M. Identical folds, splits, heads, metrics.")
    print("5-seed ensemble mean, censored='tied', median over 21 folds "
          "(RESULTS.md section 2 estimator)")
    print("=" * 78)
    print(out[["what", "spearman_median_5seed_ensemble_tied", "iqr_lo", "iqr_hi",
               "worst_fold", "top10_precision", "folds_le_zero"]]
          .to_string(index=False))

    def test(x, y, lab):
        c = fold_medians(x).index.intersection(fold_medians(y).index)
        d = (fold_medians(x)[c] - fold_medians(y)[c]).dropna()
        p = wilcoxon(d.values).pvalue
        print(f"| {lab:<50} | {d.median():+.3f} | {int((d > 0).sum())}/{len(d)} "
              f"| {p:.3g} |")

    print("\n-- paired over the 21 folds (ensemble, tied) --")
    print("| comparison | median d | folds won | Wilcoxon p |")
    print("|---|---:|---:|---:|")
    for a650, a150, what in PAIRS:
        test(a650, a150, f"650M - 150M, {what}")
    print("|---|---:|---:|---:|")
    test(RIDGE_ARM, PEPONLY_ARM, "650M ridge - 650M peptide-only CONTROL")
    test(MLP_ARM, PEPONLY_ARM, "650M MLP   - 650M peptide-only CONTROL")
    test(RIDGE_ARM, LOOKUP, "650M ridge - lookup table")
    test(MLP_ARM, LOOKUP, "650M MLP   - lookup table")
    test(PEPONLY_ARM, LOOKUP, "650M peptide-only CONTROL - lookup table")
    test("A_supervised_nn", RIDGE_ARM, "supervised NN - 650M ridge")
    test("A_supervised_nn", MLP_ARM, "supervised NN - 650M MLP")

    print("\n-- the test that matters: does 650M use the ALLELE? --")
    print(f"  650M peptide-only control (allele never shown): {ctrl650:+.4f}")
    for arm650, _, what in PAIRS[:2]:
        m = headline(arm650)[0]
        v = "beats" if m > ctrl650 else "does NOT beat"
        print(f"  650M {what:<34} {m:+.4f}   {v} its own control "
              f"({m - ctrl650:+.4f})")
    print("  NOTE: rows in the 0.06-0.11 band are not separable; read these as "
          "null results, not as one arm beating another.")

    print("\n-- paired per-fold table --")
    print(pf.to_string(index=False))
    make_figure(pf)
    return out, pf


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ridge", action="store_true")
    ap.add_argument("--mlp", action="store_true")
    ap.add_argument("--peponly", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--no-summary", action="store_true",
                    help="run the arm(s) and stop. Use when running sub-arms as "
                         "separate processes: summarise() reads all three result "
                         "files and would report a half-finished one as if it "
                         "were complete.")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--folds", type=int, default=0, help="first N folds only (debug)")
    a = ap.parse_args()
    pick = a.ridge or a.mlp or a.peponly or a.report
    do_r, do_m, do_p = (a.ridge or not pick), (a.mlp or not pick), (a.peponly or not pick)

    t_all = time.time()
    df = R.load_df()
    folds = splits.choose_held_out(df)
    if a.folds:
        folds = folds[: a.folds]

    verify()
    if not a.report:
        sanity(df)
        print(f"\n{len(folds)} folds x {a.seeds} seeds, {len(df)} rows\n", flush=True)

    if do_r:
        t0 = time.time()
        B.RidgeHead.chosen_alpha.clear()
        R.run_arm(RIDGE_ARM, B.featurize, B.RidgeHead, seeds=a.seeds,
                  censored="tied", folds=folds, df=df)
        print(f"[{RIDGE_ARM}] alpha chosen: "
              f"{dict(pd.Series(B.RidgeHead.chosen_alpha).value_counts())}  "
              f"wall {time.time() - t0:.1f}s", flush=True)

    if do_m:
        t0 = time.time()
        B.MLPHead.stopped_epoch.clear()
        R.run_arm(MLP_ARM, B.featurize, B.MLPHead, seeds=a.seeds,
                  censored="tied", folds=folds, df=df)
        e = pd.Series(B.MLPHead.stopped_epoch)
        print(f"[{MLP_ARM}] best epoch: median {e.median():.0f} min {e.min()} "
              f"max {e.max()}  wall {time.time() - t0:.1f}s", flush=True)

    if do_p:
        t0 = time.time()
        R.run_arm(PEPONLY_ARM, D.featurize, D.make_model, seeds=a.seeds,
                  censored="tied", folds=folds, df=df)
        print(f"[{PEPONLY_ARM}] wall {time.time() - t0:.1f}s", flush=True)

    if a.no_summary:
        print(f"\n--no-summary: arm(s) written, stopping. "
              f"total wall clock {time.time() - t_all:.1f}s")
        return

    for arm in (RIDGE_ARM, MLP_ARM, PEPONLY_ARM):
        R.ensemble_metrics(arm, censored="tied")

    summarise()
    print(f"\ntotal wall clock {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
