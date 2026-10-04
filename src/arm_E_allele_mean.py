"""ARM E -- the honest null. No learning of any kind.

Three predictors, all of which ignore the PEPTIDE completely:

  E_allele_mean__global   predict mean y of the training rows. One number per fold.
  E_allele_mean           predict the mean y of the GROOVE-NEAREST TRAINING allele.
                          Nearest = highest fraction of the 34 peptide-contact
                          residues shared (supertypes.identity_matrix), among the
                          alleles actually seen during fit. One number per held-out
                          allele.
  E_allele_mean__tiebreak E_allele_mean plus a 1e-6 random jitter, seeded per seed.

WHY THE THIRD ONE EXISTS -- read this before quoting any number from here.
metrics.py scores PER HELD-OUT ALLELE. Both mandated nulls are constant WITHIN an
allele, so metrics.spearman_per_allele drops every allele as "no variance" and the
arm scores nothing at all. That is the correct behaviour and it is itself the
result: a predictor with no peptide-level information cannot even be ranked.
But "undefined" is not a floor a judge can compare against, so the third variant
breaks the ties at random, which makes the metric computable and measures the
EMPIRICAL chance level of this exact protocol -- same folds, same allele sizes,
same censoring -- instead of asserting it from theory.

The constant predictors DO still produce a top-10 precision number, because
top10_precision_per_allele has no variance guard: np.argsort on a constant vector
with kind="stable" returns rows 0..9 in dataframe order. That number measures the
row order of stability.txt, not the model. It is reported here labelled as an
artefact and must never be quoted as a baseline.

Run:  PYTHONPATH=. python arm_E_allele_mean.py
Writes only results_E_allele_mean*.csv / predictions_E_allele_mean*.parquet /
ensemble_E_allele_mean.csv -- everything inside this arm's own name space.
"""

import time

import numpy as np
import pandas as pd
from scipy import stats

import data
import metrics
import run_experiment as R
import splits
import supertypes

# Canonical allele ordering + the 34-residue groove identity matrix. Both come
# straight from supertypes.py so this arm and the fold definition cannot drift.
ALLELES, IDENT = supertypes.identity_matrix()
CODE = {a: i for i, a in enumerate(ALLELES)}

HEADLINE = "E_allele_mean"
GLOBAL = "E_allele_mean__global"
TIEBREAK = "E_allele_mean__tiebreak"


# ---------------------------------------------------------------------------
# featurizer: the allele, and nothing else
# ---------------------------------------------------------------------------

def allele_code(df_subset):
    """One column: the index of the row's allele in the canonical ordering.

    Deterministic, side-agnostic, carries no peptide information and no label
    information. The nearest-neighbour fallback lives in the model, where it can
    see which alleles were actually present at fit time -- so the featurizer never
    has to guess whether it is being handed the train or the test side.
    """
    return np.array([[CODE[a]] for a in df_subset.HLA.to_numpy()], dtype=np.float64)


# ---------------------------------------------------------------------------
# the three nulls
# ---------------------------------------------------------------------------

class GlobalMean:
    """Predict the mean y of the training rows. Ignores the allele too."""

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X, y):
        self.mu = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mu, dtype=np.float64)


class GrooveNearestAlleleMean:
    """Predict the mean y of the groove-nearest allele seen during fit.

    A held-out allele is never in the training rows (split_by_allele moves every
    one of its rows to test), so every test prediction goes through the
    nearest-neighbour path. .unseen_frac records that, rather than assuming it.
    """

    def __init__(self, seed=0, jitter=0.0):
        self.seed, self.jitter = seed, jitter

    def fit(self, X, y):
        c = X[:, 0].astype(int)
        self.mu = float(np.mean(y))
        self.group = {int(k): float(y[c == k].mean()) for k in np.unique(c)}
        self.seen = np.array(sorted(self.group), dtype=int)
        return self

    def predict(self, X):
        c = X[:, 0].astype(int)
        out = np.empty(len(c), dtype=np.float64)
        unseen = 0
        for i, code in enumerate(c):
            code = int(code)
            if code in self.group:
                out[i] = self.group[code]
            else:
                unseen += 1
                j = int(self.seen[np.argmax(IDENT[code, self.seen])])
                out[i] = self.group[j]
        self.unseen_frac = unseen / max(len(c), 1)
        if self.jitter:
            rng = np.random.default_rng(self.seed)
            out = out + rng.standard_normal(len(out)) * self.jitter
        return out


def make_global(seed):
    return GlobalMean(seed)


def make_allele(seed):
    return GrooveNearestAlleleMean(seed)


def make_tiebreak(seed):
    # 1e-6 is far below the spread of the allele means, so the between-allele
    # ordering is untouched; within an allele every value is identical, so the
    # jitter alone decides the order -- i.e. a uniformly random permutation.
    return GrooveNearestAlleleMean(seed, jitter=1e-6)


# ---------------------------------------------------------------------------
# diagnostics that do not need a model run
# ---------------------------------------------------------------------------

def nearest_training_identity(df, folds):
    """For each fold, how close is the closest TRAINING groove to each held-out
    allele? This is the thing the arm-E allele mean is leaning on, so it should be
    on the slide next to the number."""
    rows = []
    for name, held in folds:
        train = [a for a in ALLELES if a not in set(held)]
        tcode = np.array([CODE[a] for a in train])
        for a in held:
            sim = IDENT[CODE[a], tcode]
            j = int(np.argmax(sim))
            rows.append({"fold_name": name, "held_out": a,
                         "nearest_training_allele": train[j],
                         "groove_identity": float(sim[j]),
                         "n_test_rows": int((df.HLA == a).sum())})
    return pd.DataFrame(rows)


def pooled_vs_per_allele(name, df):
    """What a POOLED Spearman would have reported for this null, per fold.

    metrics.py refuses to pool, on the grounds that between-allele differences in
    mean stability can carry a pooled correlation while every allele is at chance.
    Arm E is the cleanest possible demonstration: it is exactly a table of
    per-allele means, so any pooled rho it earns is 100% between-allele.
    """
    pred = pd.read_parquet(f"predictions_{name}.parquet")
    pred = pred[pred.seed == pred.seed.min()]
    rows = []
    for fold, sub in pred.groupby("fold_name", sort=False):
        t = df.loc[sub.row_id.to_numpy()]
        if np.ptp(sub.y_pred.to_numpy()) == 0:
            rows.append({"fold_name": fold, "n_test": len(sub),
                         "pooled_spearman": np.nan,
                         "note": "prediction constant across the whole fold"})
            continue
        rows.append({"fold_name": fold, "n_test": len(sub),
                     "pooled_spearman": float(stats.spearmanr(
                         sub.y_pred.to_numpy(), t.Thalf.to_numpy()).statistic),
                     "note": ""})
    return pd.DataFrame(rows)


def seed_identical(name):
    """Confirm, not assume, that the 5 seeds of a deterministic null agree."""
    p = pd.read_parquet(f"predictions_{name}.parquet")
    g = p.groupby(["fold_name", "row_id"]).y_pred
    spread = float((g.max() - g.min()).max())
    return spread, int(p.seed.nunique())


def summarise_run(res, label):
    ok = res[res.status == "ok"]
    print(f"\n  {label}")
    print(f"    rows {len(res)}  ok {len(ok)}  skipped {(res.status=='skipped').sum()}"
          f"  failed {(res.status=='failed').sum()}")
    if not len(ok):
        why = res.error.dropna().astype(str)
        why = why[why != ""]
        print(f"    no scored rows. reason (first): {why.iloc[0][:150] if len(why) else '-'}")
        return
    for col in ("spearman", "top10_precision"):
        v = ok[col].astype(float).dropna().to_numpy()
        if len(v):
            print(f"    {metrics.summarise(v, col)}")


# ---------------------------------------------------------------------------

SHOW = ["fold_name", "n_held_out_alleles", "n_train", "n_test", "seed", "status",
        "spearman", "top10_precision", "n_alleles_scored", "n_alleles_skipped",
        "wall_clock_s", "error"]


def main():
    t0 = time.time()
    df = R.load_df()
    folds = splits.choose_held_out(df)
    print(f"ARM E -- the honest null. {len(folds)} groove folds, {len(df)} rows, "
          f"{df.HLA.nunique()} alleles.")

    print("\n" + "=" * 78)
    print("0. HOW FAR IS THE NEAREST TRAINING GROOVE? (no model involved)")
    print("=" * 78)
    nn = nearest_training_identity(df, folds)
    print(f"  {len(nn)} held-out alleles across {nn.fold_name.nunique()} folds")
    print(f"  groove identity to nearest TRAINING allele: "
          f"median {nn.groove_identity.median():.3f}  "
          f"min {nn.groove_identity.min():.3f}  max {nn.groove_identity.max():.3f}")
    print(f"  above the 0.80 clustering cut: "
          f"{int((nn.groove_identity >= 0.80).sum())}/{len(nn)}")
    worst = nn.nsmallest(5, "groove_identity")
    print("  five most isolated held-out alleles:")
    print(worst.to_string(index=False))

    out = {}
    for arm, maker, label in [
        (GLOBAL, make_global, "global training mean (ignores the allele too)"),
        (HEADLINE, make_allele, "groove-nearest training allele mean"),
        (TIEBREAK, make_tiebreak, "allele mean + 1e-6 random tie-break"),
    ]:
        for pol in ("tied", "drop"):
            name = arm if pol == "tied" else f"{arm}__drop"
            print("\n" + "=" * 78)
            print(f"{name}   [{label}]   censored={pol}")
            print("=" * 78)
            res = R.run_arm(name, allele_code, maker, seeds=5, censored=pol,
                            folds=folds, df=df, verbose=False)
            out[name] = res
            summarise_run(res, f"{name} (censored={pol})")

    print("\n" + "=" * 78)
    print("PER-FOLD DETAIL, headline arm (seed 0 only; all seeds identical)")
    print("=" * 78)
    h = out[HEADLINE]
    print(h[h.seed == 0][SHOW].to_string(index=False))

    print("\n" + "=" * 78)
    print("PER-FOLD DETAIL, tie-break arm, censored=tied (the measured floor)")
    print("=" * 78)
    tb = out[TIEBREAK]
    ok = tb[tb.status == "ok"]
    per_fold = ok.groupby("fold_name", sort=False).agg(
        n_test=("n_test", "first"), n_alleles=("n_alleles_scored", "first"),
        rho_median=("spearman", "median"), rho_min=("spearman", "min"),
        rho_max=("spearman", "max"), top10_median=("top10_precision", "median"))
    per_fold = per_fold.sort_values("rho_median")
    print(per_fold.to_string())

    print("\n" + "=" * 78)
    print("DETERMINISM: do the 5 seeds agree where they should?")
    print("=" * 78)
    for name in (GLOBAL, HEADLINE, TIEBREAK):
        spread, nseeds = seed_identical(name)
        print(f"  {name:<28} {nseeds} seeds, max across-seed spread "
              f"{spread:.3e}  ({'identical' if spread == 0 else 'stochastic'})")

    print("\n" + "=" * 78)
    print("ENSEMBLE PATH on the deterministic headline arm (sigma must be blanked)")
    print("=" * 78)
    ens = R.ensemble_metrics(HEADLINE)
    print(ens[["fold_name", "n_seeds", "n_test", "n_alleles_scored", "spearman",
               "top10_precision", "calibration_euc", "coverage68",
               "status", "error"]].head(4).to_string(index=False))

    print("\n" + "=" * 78)
    print("WHAT A POOLED SPEARMAN WOULD HAVE CLAIMED FOR THIS NULL")
    print("=" * 78)
    pooled = pooled_vs_per_allele(HEADLINE, df)
    v = pooled.pooled_spearman.dropna()
    print(f"  per-fold pooled rho: median {v.median():+.3f}  "
          f"min {v.min():+.3f}  max {v.max():+.3f}  ({len(v)}/{len(pooled)} folds defined)")
    print(pooled.to_string(index=False))
    print("\n  Per-allele rho for the SAME predictions: undefined, every allele")
    print("  dropped for 'no variance'. A pooled metric would have put a visible")
    print("  number on a model that has not looked at a single peptide.")

    print(f"\nTOTAL WALL CLOCK {time.time() - t0:.1f}s")
    return out


if __name__ == "__main__":
    main()
