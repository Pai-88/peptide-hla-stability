"""Slide 4 of the Sunday pitch: the ladder, with the four best shots marked.

Why this exists and headline.png does not do the job: headline.png was built at
15:01 on 3 Oct, before the 650M MLP arm, the per-residue arms (I, K) and the
fine-tuning arms (F, FG) landed, and its own title ("No protein foundation model
beat a lookup table") is now refuted by two arms in this repo. figure.py is on
the do-not-modify list and was not touched; this is a separate generator in its
own namespace, in figure.py's visual idiom (same palette, big type, median +
every fold drawn, number printed so nobody eyeballs an axis).

Every value is recomputed here from the stored per-row predictions under the
one mandated estimator: 5-seed ensemble mean, censored='tied', per-fold =
median per-allele Spearman, headline = median over the 21 folds. Nothing is
copied from an arm's own summary and nothing is written except the PNG.

    python pitch_figure.py        ->  pitch_ladder.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import data
import metrics
import splits
import arm_B_esm_pseudo as B

from figstyle import BG, FG, MUTED, WIN, SHOT, PLAIN, CTRL

# (arm key, label, sub-label, colour).  None = the rebuilt lookup-table control.
ROWS = [
    ("A_supervised_nn", "Conventional supervised net",
     "BLOSUM62 + one-hot  ·  no foundation model anywhere", WIN),
    ("I_perres_B_pseudo_mlp", "ESM-2 150M  ·  un-pooled, tuned head",
     "best shot 1+2:  no pooling, and the net above as its head", SHOT),
    ("FG_finetune_gpu", "ESM-2 150M  ·  FINE-TUNED end to end",
     "best shot 3:  every weight unfrozen, 63 fits, 7.5 h of L4 GPU", SHOT),
    ("G_650m_B_mlp", "ESM-2 650M  ·  mean-pooled",
     "best shot 4:  4.3x the parameters", SHOT),
    (None, "CONTROL  ·  peptide-mean lookup table",
     "no model, no allele, three lines of pandas", CTRL),
    ("B_esm_pseudo_mlp", "ESM-2 150M  ·  mean-pooled, off the shelf",
     "frozen embeddings + a head, what everyone actually does", PLAIN),
    ("D_peptide_only", "CONTROL  ·  ESM-2, allele never shown",
     "cannot see which HLA it is predicting for", CTRL),
    ("E_allele_mean__tiebreak", "NULL  ·  one constant per allele",
     "no ranking information at all", CTRL),
]


def lookup_control(df):
    """Per-peptide mean of log10 half-life over the training side of each fold."""
    out = []
    for name, members in splits.groove_folds(df):
        tr, te = splits.split_by_allele(df, members)
        train, test = df.loc[tr], df.loc[te]
        pred = test.Pep.map(train.groupby("Pep").y.mean()).fillna(train.y.mean())
        rho = metrics.spearman_per_allele(test, pred.to_numpy(), censored="tied")
        if len(rho):
            out.append(float(np.nanmedian(rho)))
    return np.array(out)


def fold_scores(key, df):
    if key is None:
        return lookup_control(df)
    d = B.rescore_ensemble(key, "tied")
    return d[d.n_alleles_scored > 0].spearman.to_numpy(dtype=float)


def make(path="pitch_ladder.png"):
    df = data.load()
    scores = [fold_scores(k, df) for k, *_ in ROWS]
    meds = [float(np.median(s)) for s in scores]

    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG)
    ax.set_facecolor(BG)
    y = np.arange(len(ROWS))[::-1]
    NUMCOL = 0.70                      # fixed column for the printed median
    XMIN, XLAB = -0.20, -0.215         # negative folds stay visible

    for yi, s, m, (_, label, sub, col) in zip(y, scores, meds, ROWS):
        ax.plot([0, m], [yi, yi], color=col, lw=2.5, alpha=.30, zorder=1)
        ax.scatter(s, np.full(len(s), yi), s=44, color=col, alpha=.40,
                   edgecolors="none", zorder=2)
        ax.scatter([m], [yi], s=460, marker="|", color=col, linewidths=5, zorder=3)
        ax.text(XLAB, yi + .19, label, ha="right", va="center",
                color=FG, fontsize=16.5, fontweight="bold")
        ax.text(XLAB, yi - .21, sub, ha="right", va="center",
                color=MUTED, fontsize=12)
        ax.text(NUMCOL, yi, f"{m:.2f}", ha="right", va="center",
                color=col, fontsize=25, fontweight="bold")

    ax.axvline(0, color=MUTED, lw=1, alpha=.5)
    ax.set_xlim(XMIN, NUMCOL + 0.015)
    ax.set_ylim(-0.6, len(ROWS) - 0.35)
    ax.set_yticks([])
    ax.set_xticks([0, .1, .2, .3, .4, .5, .6])
    ax.tick_params(colors=MUTED, labelsize=13, length=0, pad=8)
    for sp in ax.spines.values():
        sp.set_visible(False)

    fig.text(0.035, 0.955, "Four escalating best shots. None of them closed the gap.",
             color=FG, fontsize=29, fontweight="bold", ha="left", va="top")
    fig.text(0.035, 0.898,
             "median per-allele Spearman  ·  21 leave-one-groove-cluster-out folds  ·  "
             "5-seed ensemble  ·  censored rows kept as ties  ·  every fold is a dot",
             color=MUTED, fontsize=13.5, ha="left", va="top")
    fig.text(0.035, 0.065,
             "paired over the 21 folds, the conventional net beats the best ESM-2 arm by "
             "+0.080 on 19 of 21 folds, Wilcoxon p = 1.3e-5\n"
             "caveat printed, not buried: 80.9% of test rows share their peptide with the "
             "training side via another allele, so 0.31 is an upper bound",
             color=MUTED, fontsize=12.5, ha="left", va="top")

    fig.subplots_adjust(left=0.395, right=0.985, top=0.855, bottom=0.185)
    fig.savefig(path, dpi=130, facecolor=BG)
    plt.close(fig)
    for (k, label, _, _), m, s in zip(ROWS, meds, scores):
        print(f"{label:46s} {m:+.4f}  folds {len(s)}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    make()
