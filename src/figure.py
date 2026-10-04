"""The headline figure: Spearman by split, ours vs NetMHCstabpan.

Three splits on x, two series, error bars across held-out alleles. Designed to
be read projected from the back of a room: big type, two colours, no gridlines
competing with the bars, every number also printed on the bar so nobody has to
eyeball it against an axis.

Usage from the eval script:

    from metrics import spearman_per_allele
    import figure
    results = {
        "Random split":   {"Ours": s_ours_rand, "NetMHCstabpan": s_net_rand},
        "Unseen peptide": {"Ours": s_ours_pep,  "NetMHCstabpan": s_net_pep},
        "Unseen allele":  {"Ours": s_ours_all,  "NetMHCstabpan": s_net_all},
    }
    figure.make(results, "headline.png")

Each value is a per-allele Series from metrics.spearman_per_allele. The bar is
the MEDIAN over held-out alleles and the whisker is the interquartile range,
not a standard error. Deliberate: with ~10 alleles the distribution is skewed
and an SEM whisker would imply a precision we do not have, and would also hide
the worst allele, which is the number a judge will ask for. The worst allele is
printed under each bar for the same reason.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG = "#0d0d0f"
FG = "#f2f2f2"
MUTED = "#8a8a93"
COLORS = {"Ours": "#4ad3a0", "NetMHCstabpan": "#6f7380"}

SPLIT_ORDER = ["Random split", "Unseen peptide", "Unseen allele"]
SUBTITLE = {
    "Random split":   "rows shuffled",
    "Unseen peptide": "no peptide in both",
    "Unseen allele":  "no allele in both",
}


def _stats(s):
    v = np.asarray(s, dtype=float)
    v = v[~np.isnan(v)]
    return (np.median(v), np.percentile(v, 25), np.percentile(v, 75),
            v.min(), len(v))


def make(results, path="headline.png", series=("Ours", "NetMHCstabpan"),
         title="Ranking peptide-HLA stability",
         subtitle="Spearman rho per held-out allele - bar is median, whisker is IQR",
         censored_note="censored rows (Thalf = 0) kept as a tie block"):
    splits = [s for s in SPLIT_ORDER if s in results] + \
             [s for s in results if s not in SPLIT_ORDER]

    fig, ax = plt.subplots(figsize=(14, 9.5), facecolor=BG)
    ax.set_facecolor(BG)
    x = np.arange(len(splits))
    w = 0.34

    for i, name in enumerate(series):
        off = (i - (len(series) - 1) / 2) * w
        med, lo, hi, worst = [], [], [], []
        for sp in splits:
            m, q1, q3, mn, n = _stats(results[sp][name])
            med.append(m); lo.append(m - q1); hi.append(q3 - m); worst.append(mn)
        ax.bar(x + off, med, w, color=COLORS.get(name, MUTED), label=name,
               zorder=3, edgecolor="none")
        ax.errorbar(x + off, med, yerr=[lo, hi], fmt="none", ecolor=FG,
                    elinewidth=3, capsize=10, capthick=3, zorder=4)
        for xi, m, h, mn in zip(x + off, med, hi, worst):
            ax.text(xi, m + h + 0.045, f"{m:.2f}", ha="center",
                    va="bottom", color=FG, fontsize=26, fontweight="bold",
                    zorder=5)
            ax.text(xi, -0.035, f"worst {mn:.2f}", ha="center", va="top",
                    color=MUTED, fontsize=15)

    ax.axhline(0, color=MUTED, lw=2, zorder=2)
    n_alleles = _stats(results[splits[-1]][series[0]])[4]

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n{SUBTITLE.get(s, '')}" for s in splits],
                       fontsize=26, color=FG)
    ax.tick_params(axis="x", length=0, pad=40)
    ax.tick_params(axis="y", labelsize=22, colors=FG, length=6, width=2)
    ax.set_ylabel("Spearman rho", fontsize=28, color=FG, labelpad=16)
    ax.set_ylim(-0.10, 1.10)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["left"].set_linewidth(2)
    ax.yaxis.grid(True, color="#26262c", lw=1.5, zorder=0)
    ax.set_axisbelow(True)

    leg = ax.legend(fontsize=25, frameon=False, loc="upper right",
                    ncol=2, handlelength=1.4, borderaxespad=0.2)
    for t in leg.get_texts():
        t.set_color(FG)

    fig.suptitle(title, fontsize=40, color=FG, fontweight="bold",
                 x=0.055, ha="left", y=0.98)
    fig.text(0.055, 0.893, subtitle, fontsize=21, color=MUTED, ha="left")
    fig.text(0.055, 0.018,
             f"{n_alleles} held-out alleles - {censored_note} - higher is better",
             fontsize=17, color=MUTED, ha="left")

    fig.subplots_adjust(left=0.085, right=0.975, top=0.845, bottom=0.24)
    fig.savefig(path, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Runs standalone on PLACEHOLDER numbers so the layout can be checked before the
# model exists. The placeholder shape (both methods degrade across the three
# splits, ours less so) is the hypothesis, NOT a result. Do not screenshot this.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd
    import data, metrics

    rng = np.random.default_rng(1)
    df = data.load()
    alleles = data.loao_eligible(df)[:10]
    test = df[df.HLA.isin(alleles)].copy()

    results = {}
    for split, (r_ours, r_net) in [
        ("Random split",   (0.80, 0.74)),
        ("Unseen peptide", (0.62, 0.58)),
        ("Unseen allele",  (0.49, 0.31)),
    ]:
        entry = {}
        for name, r in [("Ours", r_ours), ("NetMHCstabpan", r_net)]:
            pred = np.empty(len(test))
            for _, idx in test.groupby("HLA").indices.items():
                # per-allele jitter so the spread across alleles is realistic
                rr = np.clip(r + rng.normal(0, 0.09), 0, 0.98)
                pred[idx] = metrics._synth(test.y.values[idx], rr, rng)
            entry[name] = metrics.spearman_per_allele(test, pred)
        results[split] = entry

    for sp, e in results.items():
        for name, s in e.items():
            print(f"{sp:16s} {name:14s} {metrics.summarise(s, '')}")
    make(results, "headline.png",
         subtitle="PLACEHOLDER NUMBERS - layout check only, not a result")
