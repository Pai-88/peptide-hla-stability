"""The three splits. This file decides what our headline number means."""

import numpy as np


def split_random(df, test_frac=0.2, seed=0):
    """Split 1. Flatters everything. Included only so we can show the contrast."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(len(df) * (1 - test_frac))
    return df.index[idx[:cut]], df.index[idx[cut:]]


def split_by_peptide(df, test_frac=0.2, seed=0):
    """Split 2. No peptide appears in both sides.

    Each peptide was measured against ~5 alleles, so a random row split leaks
    the same peptide across the boundary and inflates everything.
    """
    rng = np.random.default_rng(seed)
    peps = df.Pep.unique()
    rng.shuffle(peps)
    cut = int(len(peps) * (1 - test_frac))
    train_peps = set(peps[:cut])
    mask = df.Pep.isin(train_peps)
    return df.index[mask], df.index[~mask]


# ---------------------------------------------------------------------------
# Split 3: leave-one-groove-cluster-out.
#
# Leave-one-ALLELE-out is impossible on this dataset. Every allele has a near-twin
# in it (A*23:01 and A*24:02 share 97% of their 34 peptide-contact residues; the
# most isolated allele anywhere is still at 0.76). Hold out one allele and a nearly
# identical binding groove stays in training, so you measure interpolation.
#
# So the fold unit is a groove cluster. See supertypes.py.
#
# PAING: the ONE decision left is CUT, below. That is the threshold at which two
# alleles count as the same groove. Pick it, write one sentence saying why, and
# this is done. Everything else here is mechanical.
# ---------------------------------------------------------------------------

import supertypes

# The decision. 0.80 -> 22 folds, tight clusters, recovers known supertypes.
#                0.75 -> 15 folds, coarser, the A locus starts merging.
#                0.70 -> 11 folds, brutal, two huge blobs dominate.
CUT = 0.80

# WHY (Paing, write this, it becomes a sentence in the pitch):
#   ...


def split_by_allele(df, held_out):
    """(train_index, test_index) with every row for `held_out` alleles in test."""
    mask = df.HLA.isin(set(held_out))
    return df.index[~mask], df.index[mask]


def groove_folds(df, cut=CUT, min_test_rows=100):
    """Every usable held-out fold, as (name, [alleles]).

    One fold per groove cluster. Clusters with too few test rows are dropped, since
    a Spearman on a handful of points is noise, not a result.
    """
    out = []
    for members in supertypes.clusters(cut).values():
        if int(df[df.HLA.isin(members)].shape[0]) >= min_test_rows:
            name = min(members).replace("HLA-", "") + (f" +{len(members)-1}" if len(members) > 1 else "")
            out.append((name, members))
    return sorted(out, key=lambda t: t[0])


def choose_held_out(df, cut=CUT, min_test_rows=100):
    """The held-out groove clusters, biggest first so the headline runs first."""
    folds = groove_folds(df, cut, min_test_rows)
    return sorted(folds, key=lambda t: -int(df[df.HLA.isin(t[1])].shape[0]))
