"""demo_scan.png: the anchor-optimisation scan, as a figure rather than a screenshot.

The old demo_scan.png was a screengrab of the app when it was dark-themed, so it
aged badly and sat at screenshot resolution. This draws the same content from the
precomputed scan, in the repo's palette, at print resolution.

    python demo_figure.py
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figstyle import BG, FG, MUTED, LINE, WIN, PLAIN, WARN

TOP = 14


def make(path="demo_scan.png", prefer=("VTTEVAFGL", "NLVPMVATV")):
    d = json.load(open("design_data.json"))
    key = next((k for p in prefer for k in d["scans"] if k.startswith(p + "|")),
               sorted(d["scans"])[0])
    scan = d["scans"][key]
    pep, allele = key.split("|")
    rows = scan["rows"][:TOP][::-1]          # best at the top of the axis

    fig, ax = plt.subplots(figsize=(13, 7.6), facecolor=BG)
    ax.set_facecolor(BG)
    ys = range(len(rows))
    # an anchor substitution is one at P2 or P9, which is what the scan is about
    anchor = [r[0][1] in ("2", "9") for r in rows]
    ax.barh(list(ys), [r[2] for r in rows],
            color=[WARN if a else PLAIN for a in anchor],
            height=.68, zorder=3)
    ax.axvline(scan["wt_pred"], color=FG, lw=1.4, ls="--", zorder=4)
    ax.text(scan["wt_pred"], len(rows) - .3, "  wild type, predicted",
            color=FG, fontsize=11, va="bottom")

    for y, r in zip(ys, rows):
        ax.text(-.4, y, r[0], color=WARN if anchor[y] else FG, fontsize=12.5,
                fontweight="bold" if anchor[y] else "normal", ha="right", va="center")
        ax.text(r[2], y, f"  {r[2]:.1f} h" + ("   anchor" if anchor[y] else ""),
                color=WARN if anchor[y] else MUTED, fontsize=11.5, va="center")

    ax.set_yticks([]); ax.set_ylim(-.8, len(rows) + .3)
    ax.set_xlim(0, max(r[2] for r in rows) * 1.22)
    ax.set_xlabel("predicted complex half-life, hours", color=MUTED, fontsize=12)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=11)
    ax.grid(axis="x", color=LINE, lw=.8, zorder=0)

    # both lines placed on the figure, so they cannot collide with each other
    fig.text(.013, .955, f"Every single mutant of {pep} on {allele}, ranked",
             color=FG, fontsize=20, fontweight="bold", va="top")
    fig.text(.013, .900,
             f"top {TOP} of all 171 single substitutions  "
             f"\u00b7  measured wild type {scan['measured']:.2f} h  "
             f"\u00b7  we enumerate, we do not search",
             color=MUTED, fontsize=12.5, va="top")
    fig.subplots_adjust(left=.16, right=.97, top=.845, bottom=.1)
    fig.savefig(path, dpi=160, facecolor=BG)
    print(f"wrote {path}  ({pep} on {allele})")


if __name__ == "__main__":
    make()
