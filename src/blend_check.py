"""Does blending the conventional net with a foundation-model arm help or hurt?

The README asserts it hurts. This recomputes that claim from the stored per-row
predictions so the number in the README is reproducible rather than asserted.

Protocol, identical to every other number in the repo: average the seeds of each arm,
rank-average the two arms within each fold (the metric is a rank correlation, and the
arms are not on a common scale), per-allele Spearman with censored='tied', mean within
a fold, median across the 21 folds.

    python src/blend_check.py [dir-with-predictions_*.parquet]
        ->  results/blend_conventional_plus_esm.csv

The prediction parquets are regenerable by run_experiment.py and are gitignored for
size, so pass the directory holding them if it is not the current one.
"""

import os
import numpy as np
import pandas as pd

import metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONVENTIONAL = "A_supervised_nn"
FOUNDATION = "I_perres_B_pseudo_mlp"      # the strongest foundation-model arm


def seed_mean(arm, where="."):
    d = pd.read_parquet(os.path.join(where, f"predictions_{arm}.parquet"))
    return d.groupby(["row_id", "fold_name", "HLA", "Pep", "y_true"],
                     as_index=False).y_pred.mean()


def per_fold(df, col):
    out = []
    for fold, g in df.groupby("fold_name"):
        sub = g.rename(columns={col: "p"})
        rho = metrics.spearman_per_allele(sub.assign(Thalf=10 ** sub.y_true),
                                          sub.p.to_numpy(), censored="tied")
        if len(rho):
            out.append((fold, float(np.nanmean(rho))))
    return pd.DataFrame(out, columns=["fold_name", "spearman"])


def main(where="."):
    a = seed_mean(CONVENTIONAL, where)
    e = seed_mean(FOUNDATION, where)
    m = a.merge(e, on=["row_id", "fold_name", "HLA", "Pep", "y_true"],
                suffixes=("_a", "_e"))
    m["blend"] = (m.groupby("fold_name").y_pred_a.rank(pct=True)
                  + m.groupby("fold_name").y_pred_e.rank(pct=True))

    rows = []
    for label, col in [("conventional", "y_pred_a"),
                       ("foundation", "y_pred_e"),
                       ("blend", "blend")]:
        pf = per_fold(m, col).assign(arm=label)
        rows.append(pf)
        print(f"  {label:14s} median over folds {pf.spearman.median():.3f}  "
              f"(n={len(pf)})")
    out = pd.concat(rows, ignore_index=True)[["arm", "fold_name", "spearman"]]
    dst = os.path.join(ROOT, "results", "blend_conventional_plus_esm.csv")
    out.to_csv(dst, index=False)
    print(f"wrote {dst}")
    return out


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
