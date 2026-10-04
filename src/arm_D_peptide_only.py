"""ARM D -- the null control. Peptide embedding ONLY, no allele information.

The model is handed the ESM-2 embedding of the 9-mer and nothing else. It cannot
know which HLA it is predicting for, so whatever it scores is the part of the
signal that is just "some peptides are sticky in general" (plus, on this
dataset, peptide-level memorisation -- see the caveat below).

Any arm that uses allele information has to beat THIS, not zero. If arm B or C
lands near arm D, the foundation model has learned almost nothing
allele-specific and that is the finding, whatever the headline Spearman is.

CAVEAT THAT HAS TO GO ON THE SLIDE
  The fold unit is a groove cluster, not a peptide. Each peptide was measured
  against ~5 alleles, so a peptide held out with its allele is usually still
  present in training paired with OTHER alleles. Measured: 80.9% of all test
  rows (and 100% in 8 of the 21 folds) have their peptide somewhere in the
  training side. So arm D is not a pure "peptide chemistry prior": it is also a
  peptide-level memorisation baseline, which makes it a STRONGER null than it
  looks. That is the conservative direction for a control, so it is kept, and
  main() measures the seen-peptide / unseen-peptide split explicitly.

MODEL
  Standardise the 640-dim ESM-2 peptide embedding, then a 1-hidden-layer MLP
  (256 units, Adam, early stopping on an internal 10% split of the TRAINING rows
  only). random_state = the run seed, so the 5 seeds genuinely differ and the
  across-seed spread is a real uncertainty estimate rather than a constant 0.
  Untuned beyond this: see main(), which also reports a plain ridge head and a
  peptide-mean lookup so the null cannot be accused of being underpowered.

CENSORING
  Headline is censored='tied'. The 'drop' robustness numbers are recomputed from
  the SAME stored predictions -- censoring is a metric-side policy only
  (run_experiment trains on train_df.y regardless), so re-scoring is exactly
  equivalent to a second run_arm and costs no extra fits.

Run:  python arm_D_peptide_only.py
Writes: results_D_peptide_only.csv, predictions_D_peptide_only.parquet
        (+ ensemble_D_peptide_only.csv, written by run_experiment.ensemble_metrics)
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import embed
import metrics
import run_experiment as R

ARM = "D_peptide_only"

_EMB = None


def _emb():
    """ESM-2 (esm2_t30_150M_UR50D) mean-pooled peptide embeddings, 640 dims."""
    global _EMB
    if _EMB is None:
        _EMB = embed.load("peptides")
    return _EMB


def featurize(df_subset):
    """Pure lookup, so it is deterministic and identical for train and test.

    No allele column is read. No statistic is fitted here -- standardisation
    lives inside the model so it only ever sees the training rows.
    """
    mat, ix = _emb()
    return mat[[ix[p] for p in df_subset.Pep]].astype(np.float64)


def make_model(seed):
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=300,
                     early_stopping=True, n_iter_no_change=15, random_state=seed),
    )


# ---------------------------------------------------------------------------
# reporting helpers -- all read the stored predictions, none refit anything
# ---------------------------------------------------------------------------

def rescore(name=ARM, censored="drop"):
    """Per-(fold, seed) metrics recomputed from predictions_<name>.parquet.

    Used for the 'drop' robustness row. Censoring only affects metrics.py, so
    this is exactly what run_arm(censored='drop') would have reported.
    """
    df = R.load_df()
    pred = pd.read_parquet(f"predictions_{name}.parquet")
    rows = []
    for (fold, seed), sub in pred.groupby(["fold_name", "seed"], sort=False):
        test_df = df.loc[sub.row_id.to_numpy()]
        p = sub.y_pred.to_numpy(dtype=np.float64)
        rho = metrics.spearman_per_allele(test_df, p, censored=censored)
        top = metrics.top10_precision_per_allele(test_df, p, censored=censored)
        rows.append({"fold_name": fold, "seed": int(seed),
                     "censored_policy": censored,
                     "spearman": np.median(rho) if len(rho) else np.nan,
                     "top10_precision": np.median(top) if len(top) else np.nan,
                     "n_alleles_scored": len(rho)})
    return pd.DataFrame(rows)


def rescore_ensemble(name=ARM, censored="drop"):
    """Same, for the across-seed MEAN prediction (one row per fold)."""
    df = R.load_df()
    pred = pd.read_parquet(f"predictions_{name}.parquet")
    agg = (pred.groupby(["fold_name", "row_id"], sort=False)
               .y_pred.mean().reset_index())
    rows = []
    for fold, sub in agg.groupby("fold_name", sort=False):
        test_df = df.loc[sub.row_id.to_numpy()]
        p = sub.y_pred.to_numpy(dtype=np.float64)
        rho = metrics.spearman_per_allele(test_df, p, censored=censored)
        top = metrics.top10_precision_per_allele(test_df, p, censored=censored)
        rows.append({"fold_name": fold, "censored_policy": censored,
                     "n_test": len(test_df),
                     "spearman": np.median(rho) if len(rho) else np.nan,
                     "top10_precision": np.median(top) if len(top) else np.nan,
                     "n_alleles_scored": len(rho)})
    return pd.DataFrame(rows)


def seen_vs_unseen_peptides(name=ARM, censored="tied"):
    """Split each fold's test rows by whether the peptide also appears in train.

    Separates "arm D remembers this peptide's other measurements" from "ESM-2
    generalises to a peptide it has not been measured on". Per-allele Spearman
    as everywhere else; an allele needs metrics.MIN_N rows on a side to score.
    """
    import splits
    df = R.load_df()
    pred = pd.read_parquet(f"predictions_{name}.parquet")
    agg = (pred.groupby(["fold_name", "row_id"], sort=False)
               .y_pred.mean().reset_index())
    folds = {n: a for n, a in splits.choose_held_out(df)}
    rows = []
    for fold, sub in agg.groupby("fold_name", sort=False):
        tr_idx, _ = splits.split_by_allele(df, folds[fold])
        train_peps = set(df.loc[tr_idx].Pep)
        test_df = df.loc[sub.row_id.to_numpy()]
        p = sub.y_pred.to_numpy(dtype=np.float64)
        seen = test_df.Pep.isin(train_peps).to_numpy()
        out = {"fold_name": fold, "n_test": len(test_df),
               "frac_peptide_seen_in_train": float(seen.mean())}
        for label, m in (("seen", seen), ("unseen", ~seen)):
            if m.sum() >= metrics.MIN_N:
                rho = metrics.spearman_per_allele(test_df[m], p[m], censored=censored)
                out[f"rho_{label}"] = float(np.median(rho)) if len(rho) else np.nan
                out[f"n_{label}"] = int(m.sum())
                out[f"alleles_{label}"] = len(rho)
            else:
                out[f"rho_{label}"] = np.nan
                out[f"n_{label}"] = int(m.sum())
                out[f"alleles_{label}"] = 0
        rows.append(out)
    return pd.DataFrame(rows)


def reference_nulls():
    """Two cheap reference points, so 'the null is underpowered' is answered.

    ridge     the same ESM-2 peptide embedding through a closed-form linear head
    pepmean   no embedding at all: predict the mean training y of that exact
              peptide (global mean if unseen). The memorisation ceiling.
    Single deterministic fit per fold; diagnostics only, nothing is written.
    """
    import splits
    from sklearn.linear_model import Ridge
    df = R.load_df()
    rows = []
    for fold, alle in splits.choose_held_out(df):
        tr, te = splits.split_by_allele(df, alle)
        trd, ted = df.loc[tr], df.loc[te]
        r = {"fold_name": fold}
        p = Ridge(alpha=1.0).fit(featurize(trd), trd.y.values).predict(featurize(ted))
        r["ridge_rho"] = float(np.median(metrics.spearman_per_allele(ted, p)))
        r["ridge_top10"] = float(np.median(metrics.top10_precision_per_allele(ted, p)))
        pm = ted.Pep.map(trd.groupby("Pep").y.mean()).fillna(trd.y.mean()).values
        r["pepmean_rho"] = float(np.median(metrics.spearman_per_allele(ted, pm)))
        r["pepmean_top10"] = float(np.median(metrics.top10_precision_per_allele(ted, pm)))
        rows.append(r)
    return pd.DataFrame(rows)


def _line(v, label):
    return metrics.summarise(np.asarray(v, dtype=float), label)


def main(seeds=5):
    t0 = time.time()
    print("=" * 78)
    print("ARM D -- peptide embedding ONLY (null control). No allele information.")
    print("=" * 78)

    res = R.run_arm(ARM, featurize, make_model, seeds=seeds, censored="tied")

    print("\n--- ensemble (5-seed mean), censored='tied'  [HEADLINE] ---")
    ens = R.ensemble_metrics(ARM, censored="tied")
    print(ens[["fold_name", "n_seeds", "n_test", "n_alleles_scored", "spearman",
               "top10_precision", "calibration_euc", "coverage68", "mean_sigma",
               "mean_abs_err", "status", "error"]].to_string(index=False))
    print()
    print(_line(ens.spearman, "ENSEMBLE tied  Spearman"))
    print(_line(ens.top10_precision, "ENSEMBLE tied  top-10 precision (chance 0.100)"))
    worst = ens.loc[ens.spearman.idxmin()]
    print(f"WORST FOLD (tied): {worst.fold_name}  rho {worst.spearman:+.4f}  "
          f"top10 {worst.top10_precision:.3f}  n_test {int(worst.n_test)}")

    print("\n--- ensemble, censored='drop'  [ROBUSTNESS, same predictions] ---")
    ed = rescore_ensemble(censored="drop")
    print(ed.to_string(index=False))
    print()
    print(_line(ed.spearman, "ENSEMBLE drop  Spearman"))
    print(_line(ed.top10_precision, "ENSEMBLE drop  top-10 precision"))
    w = ed.loc[ed.spearman.idxmin()]
    print(f"WORST FOLD (drop): {w.fold_name}  rho {w.spearman:+.4f}")

    print("\n--- per-(fold, seed), both policies ---")
    ok = res[res.status == "ok"]
    print(_line(ok.spearman, "per-(fold,seed) tied  Spearman"))
    rd = rescore(censored="drop")
    print(_line(rd.spearman, "per-(fold,seed) drop  Spearman"))

    print("\n--- seen vs unseen peptide (memorisation vs generalisation) ---")
    sv = seen_vs_unseen_peptides()
    print(sv.to_string(index=False))
    print()
    print(_line(sv.rho_seen.dropna(), "rho, peptide SEEN in train  "))
    print(_line(sv.rho_unseen.dropna(), "rho, peptide UNSEEN in train"))

    print("\n--- reference nulls (is arm D underpowered?) ---")
    rn = reference_nulls()
    print(rn.to_string(index=False))
    print()
    print(_line(rn.ridge_rho, "ESM-2 peptide emb + ridge   "))
    print(_line(rn.pepmean_rho, "peptide-mean lookup (no emb)"))

    print(f"\ntotal wall clock {time.time() - t0:.1f}s")
    return res, ens, ed, sv, rn


if __name__ == "__main__":
    main()
