"""Dataset EDA figures for the Serova stability data.

Describes the DATA only: no model, no arm, nothing from runs.csv. Safe to show
before any result, and safe under the DO NOT SAY list because it quotes no
model number. House style from figure.py.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from collections import Counter

BG, FG, MUTED = "#0d0d0f", "#f2f2f2", "#8a8a93"
ACC, WARN, COOL = "#4ad3a0", "#e8704a", "#5aa9e6"
AA = "ACDEFGHIKLMNPQRSTVWY"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.edgecolor": MUTED, "font.size": 9,
    "axes.titlesize": 10, "axes.titlecolor": FG,
})

df = pd.read_csv(os.environ.get("SEROVA_RAW_CSV", "../data/stability.txt"), sep=None, engine="python")
df["censored"] = df.thalf_hours <= 0.0
LOG = np.log1p(df.thalf_hours)


def bare(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=MUTED, alpha=0.15, lw=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- figure 1
fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))

# A. the target distribution
a = ax[0, 0]
nz = df.loc[~df.censored, "thalf_hours"]
a.hist(np.log10(nz), bins=60, color=ACC, alpha=0.85)
a.bar([-2.35], [df.censored.sum()], width=0.12, color=WARN)
a.annotate(f"{df.censored.sum():,} rows at exactly 0 h\n({df.censored.mean():.0%}, below assay floor)",
           xy=(-2.3, df.censored.sum()), xytext=(-1.7, df.censored.sum() * 0.78),
           color=WARN, fontsize=8.5,
           arrowprops=dict(arrowstyle="-", color=WARN, lw=0.8))
a.set_xlabel("log10 half-life (hours)")
a.set_ylabel("peptide-HLA pairs")
a.set_title("Target is heavy-tailed and one fifth is censored", loc="left")
bare(a)

# B. censoring is not uniform across alleles
b = ax[0, 1]
cz = df.groupby("allele").agg(rate=("censored", "mean"), n=("censored", "size"))
cz = cz[cz.n >= 50].sort_values("rate")
b.bar(range(len(cz)), cz.rate * 100, color=[WARN if r > .3 else COOL for r in cz.rate])
b.set_xticks([0, len(cz) - 1])
b.set_xticklabels([cz.index[0].replace("HLA-", ""), cz.index[-1].replace("HLA-", "")], fontsize=8)
b.set_ylabel("% of rows censored")
b.set_xlabel(f"{len(cz)} alleles with >=50 rows, sorted")
b.set_title(f"Censoring runs {cz.rate.min():.0%} to {cz.rate.max():.0%} by allele", loc="left")
b.text(0, 84, "a censor-blind metric does not score these alleles\non comparable footing", color=MUTED, fontsize=8.5)
b.axhline(df.censored.mean() * 100, color=FG, lw=0.8, ls="--")
b.annotate(f"dataset mean {df.censored.mean():.0%}", xy=(2, df.censored.mean() * 100 + 1.5),
           color=FG, fontsize=8)
bare(b)

# C. rows per allele: a long tail
c = ax[1, 0]
n = df.allele.value_counts()
c.bar(range(len(n)), n.values, color=COOL, width=1.0)
c.set_yscale("log")
c.set_xlabel(f"{len(n)} alleles, sorted by row count")
c.set_ylabel("rows (log)")
c.set_title(f"Per-allele data spans {n.max():,} down to {n.min()} rows", loc="left")
c.annotate(f"{n.index[0].replace('HLA-','')} {n.iloc[0]:,} rows", xy=(0, n.iloc[0]),
           xytext=(5, n.iloc[0] * 0.42), color=FG, fontsize=8,
           arrowprops=dict(arrowstyle="->", color=FG, lw=.8))
c.annotate(f"{n.index[-1].replace('HLA-','')} {n.iloc[-1]} rows", xy=(len(n) - 1, n.iloc[-1]),
           xytext=(len(n) - 34, n.iloc[-1] * 2.6), color=WARN, fontsize=8,
           arrowprops=dict(arrowstyle="->", color=WARN, lw=.8))
bare(c)

# D. peptide reuse: the leakage hazard
d = ax[1, 1]
reuse = df.peptide.value_counts().value_counts().sort_index()
d.bar(reuse.index, reuse.values, color=[MUTED if k == 1 else WARN for k in reuse.index])
d.set_yscale("log")
d.set_xlabel("number of alleles the same peptide is measured against")
d.set_ylabel("peptides (log)")
share = (df.peptide.value_counts() > 1).mean()
d.set_title(f"{share:.0%} of peptides appear under more than one allele", loc="left")
d.annotate("a random row split puts\nthese peptides on both sides", xy=(6, 500),
           xytext=(17, 400), color=WARN, fontsize=8.5,
           arrowprops=dict(arrowstyle="->", color=WARN, lw=.8))
bare(d)

fig.suptitle("Serova peptide-HLA stability: what the data looks like before any model",
             color=FG, fontsize=12, x=0.012, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("eda_1_data.png", dpi=160)
print("wrote eda_1_data.png")


# ---------------------------------------------------------------- figure 2
df["y"] = LOG
fig2, ax2 = plt.subplots(2, 2, figsize=(12, 7.8),
                         gridspec_kw={"height_ratios": [1, 1.25]})

# A. where in the 9-mer does the signal live
keep = df[df.allele.map(df.allele.value_counts()) >= 200]
rows = []
for al, g in keep.groupby("allele"):
    tv = g.y.var()
    if tv == 0:
        continue
    for p in range(9):
        bv = g.groupby(g.peptide.str[p].values).y.mean().var()
        rows.append((al, p + 1, bv / tv))
eta = pd.DataFrame(rows, columns=["allele", "pos", "eta2"])
m = eta.groupby("pos").eta2.mean()
q1 = eta.groupby("pos").eta2.quantile(.25)
q3 = eta.groupby("pos").eta2.quantile(.75)

a = ax2[0, 0]
cols = [ACC if p in (2, 9) else (COOL if p in (1, 3) else MUTED) for p in m.index]
a.bar(m.index, m.values, color=cols)
a.vlines(m.index, q1, q3, color=FG, lw=1.2, alpha=.7)
a.set_xticks(range(1, 10))
a.set_xticklabels([f"P{i}" for i in range(1, 10)])
a.set_ylabel("variance in stability explained\nby the residue at that position")
a.set_title(f"P2 and P9 are the anchors, read from data alone ({eta.allele.nunique()} alleles)", loc="left")
a.annotate("solvent-exposed middle", xy=(5.5, 0.11), xytext=(4.1, 0.30), color=MUTED, fontsize=8.5,
           arrowprops=dict(arrowstyle="->", color=MUTED, lw=.8))
bare(a)

# B. variance decomposition: why a peptide-only lookup table works at all
b = ax2[0, 1]
big = df[df.peptide.map(df.peptide.value_counts()) >= 5]
tot = big.y.var()
pv = big.groupby("peptide").y.mean().var() / tot
av = big.groupby("allele").y.mean().var() / tot
parts = [("peptide identity", pv, ACC), ("allele identity", av, COOL),
         ("interaction + noise", max(0, 1 - pv - av), MUTED)]
left = 0
for lab, v, c in parts:
    b.barh([0], [v], left=left, color=c, height=.5)
    b.text(left + v / 2, 0, f"{lab}\n{v:.0%}", ha="center", va="center",
           color=BG if c is not MUTED else FG, fontsize=9, fontweight="bold")
    left += v
b.set_xlim(0, 1); b.set_ylim(-.6, .6); b.axis("off")
b.set_title("40% of stability is peptide-intrinsic, before the groove is known", loc="left")
b.text(0, -.45, "this is why a peptide-mean lookup table with no allele information\n"
                "is a serious control, not a straw man", color=MUTED, fontsize=8.5)

# C/D. anchor chemistry differs by allele
SHOW = ["HLA-A*02:01", "HLA-A*03:01", "HLA-A*11:01", "HLA-A*24:02",
        "HLA-B*07:02", "HLA-B*08:01", "HLA-B*15:01", "HLA-B*35:01"]
for ax_, pos in ((ax2[1, 0], 2), (ax2[1, 1], 9)):
    M = np.full((len(SHOW), 20), np.nan)
    for i, al in enumerate(SHOW):
        g = df[df.allele == al]
        z = (g.y - g.y.mean()) / g.y.std()
        for j, aa in enumerate(AA):
            sel = z[g.peptide.str[pos - 1] == aa]
            if len(sel) >= 10:   # 5 was letting single-digit cells read as anchors
                M[i, j] = sel.mean()
    # centre each allele on its own across-residue mean: 20% censored rows drag
    # every residue below the allele mean, which otherwise paints the whole map blue
    M = M - np.nanmean(M, axis=1, keepdims=True)
    im = ax_.imshow(M, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    ax_.set_xticks(range(20)); ax_.set_xticklabels(list(AA), fontsize=8)
    ax_.set_yticks(range(len(SHOW)))
    ax_.set_yticklabels([s.replace("HLA-", "") for s in SHOW], fontsize=8)
    ax_.set_xlabel(f"residue at P{pos}   (cells with <10 measurements left blank)")
    ax_.set_title(f"P{pos} preference is allele-specific (red = stabilising)", loc="left")
    for s in ax_.spines.values():
        s.set_visible(False)
    cb = fig2.colorbar(im, ax=ax_, fraction=.025, pad=.01)
    cb.set_label("stability vs this allele's own average", color=MUTED, fontsize=7.5)
    cb.ax.tick_params(colors=MUTED, labelsize=7)

fig2.suptitle("Where the signal is: anchors, allele specificity, and what a peptide alone already tells you",
              color=FG, fontsize=12, x=0.012, ha="left")
fig2.tight_layout(rect=[0, 0, 1, 0.955])
fig2.savefig("eda_2_signal.png", dpi=160)
print("wrote eda_2_signal.png")
