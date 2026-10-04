"""Load and clean the NetMHCstabpan stability data.

Source: https://services.healthtech.dtu.dk/suppl/immunology/NetMHCstabpan-1.0/Stability.txt
28,166 rows, 75 HLA alleles, 5,633 distinct 9-mer peptides, half-life in hours.
"""

import numpy as np
import pandas as pd

PATH = "stability.txt"

# 5,679 rows (20.2%) sit at exactly 0.0 h. That is not a measurement of zero,
# it is "dissociated faster than the assay could resolve". Left-censored.
# log(0) is undefined, so we need a floor before any log transform. 0.05 is half
# the smallest non-zero value observed (0.1) which is the conventional choice.
CENSOR_FLOOR = 0.05


def load(path=PATH):
    df = pd.read_csv(path, sep=r"\s+")
    df["censored"] = df.Thalf <= 0.0
    df["y"] = np.log10(df.Thalf.clip(lower=CENSOR_FLOOR))
    return df


def allele_summary(df):
    """Per-allele stats. Use this to decide which alleles can be held out at all."""
    g = df.groupby("HLA")
    return pd.DataFrame({
        "n": g.size(),
        "censored_frac": g.censored.mean(),
        "y_std": g.y.std(),
        "median_thalf": g.Thalf.median(),
    }).sort_values("n", ascending=False)


def loao_eligible(df, min_n=100, max_censored=0.5):
    """Alleles we can meaningfully hold out.

    Two alleles have as few as 7 rows: a Spearman on 7 points is noise.
    Eight alleles are more than half censored: almost everything is a non-binder,
    so there is no ranking to recover. Both are excluded from the headline, and
    both exclusions get stated on the slide rather than hidden.
    """
    s = allele_summary(df)
    keep = s[(s.n >= min_n) & (s.censored_frac <= max_censored)]
    return list(keep.index)


if __name__ == "__main__":
    df = load()
    s = allele_summary(df)
    elig = loao_eligible(df)
    print(f"rows {len(df)}  alleles {df.HLA.nunique()}  peptides {df.Pep.nunique()}")
    print(f"censored at 0.0: {df.censored.sum()} ({100 * df.censored.mean():.1f}%)")
    print(f"rows per allele: min {s.n.min()}  median {int(s.n.median())}  max {s.n.max()}")
    print(f"LOAO-eligible alleles: {len(elig)} of {df.HLA.nunique()}")
    print()
    print("excluded:")
    print(s[~s.index.isin(elig)].to_string())
