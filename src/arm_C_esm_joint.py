"""ARM C -- ESM-2 joint (allele + linker + peptide) embedding + supervised head.

THE ENCODING (built by embed.py --joint all, read via embed.load("joint")):
    allele mature chain + "GGGGSGGGGS" + 9-mer peptide
    -> one ESM-2 forward pass (facebook/esm2_t30_150M_UR50D, 640-dim)
    -> mean-pool over the PEPTIDE token positions ONLY.
So the peptide representation is modulated by the HLA groove through attention,
which is the whole reason to pay for a foundation model here. 640 features/pair.

THE HEAD
A one-hidden-layer MLP, deliberately matched in spirit to the brief's
recommended comparator ("a simple supervised neural network trained on peptide
and HLA pairs"). Keeping the head simple is the point: if arm C wins it should
be because the REPRESENTATION carries allele-aware signal, not because this arm
got a bigger model than the baseline.

Scaling lives INSIDE the pipeline, so StandardScaler is fit on the fold's
training rows only and applied to the held-out rows. Doing it in featurize()
would scale each side by its own statistics, which is silently wrong.

The MLP is stochastic (random init + minibatch shuffling), so the 5 seeds give
a genuine across-seed spread and ensemble_metrics() has real sigma to calibrate
-- unlike a deterministic Ridge, whose sigma is identically 0.

CENSORING
Training always uses data.load()'s y = log10(Thalf clipped to 0.05), so the
censored rows are present as the floor in BOTH runs. 'tied' vs 'drop' changes
only WHICH held-out rows the metric scores. So the robustness run needs no
refit: it re-scores the stored predictions. Identical predictions, two scoring
policies, directly paired fold by fold.

Writes only: results_C_esm_joint.csv, predictions_C_esm_joint.parquet
             (and ensemble_C_esm_joint.csv via run_experiment.ensemble_metrics).
results_C_esm_joint.csv holds BOTH policies; filter on `censored_policy`.

    python arm_C_esm_joint.py             # full 21 folds x 5 seeds
    python arm_C_esm_joint.py --check     # embedding coverage only, no fitting
    python arm_C_esm_joint.py --time      # time one fit on the biggest fold
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import embed
import metrics
import run_experiment as R

ARM = "C_esm_joint"
SEEDS = 5

_MAT = _IDX = None


def emb():
    """(matrix, {'HLA|Pep': row}). Loaded once per process, not at import."""
    global _MAT, _IDX
    if _MAT is None:
        _MAT, _IDX = embed.load("joint")
    return _MAT, _IDX


def coverage(df):
    """(n_present, n_missing, [a few missing keys]). Call BEFORE trusting a run."""
    _, idx = emb()
    keys = [f"{h}|{p}" for h, p in zip(df.HLA, df.Pep)]
    missing = [k for k in keys if k not in idx]
    return len(keys) - len(missing), len(missing), missing[:5]


def featurize(sub):
    """One 640-dim joint embedding per row. KeyError here fails the fold loudly,
    which is what we want: a silently-subsetted arm is the worst outcome."""
    mat, idx = emb()
    return mat[[idx[f"{h}|{p}"] for h, p in zip(sub.HLA, sub.Pep)]]


def make_model(seed):
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(256,),
            activation="relu",
            alpha=1e-3,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=seed,
        ),
    )


# ---------------------------------------------------------------------------
# robustness: re-score the SAME stored predictions under censored='drop'
# ---------------------------------------------------------------------------

def rescore(policy="drop", name=ARM, df=None):
    """Re-score predictions_<name>.parquet under a different censoring policy.

    No refit: the model never saw the policy, it only changes which held-out
    rows the metric is allowed to look at. Returns rows in RESULT_COLUMNS shape
    so they can be appended to results_<name>.csv alongside the headline run.
    """
    df = R.load_df() if df is None else df
    pred = pd.read_parquet(f"predictions_{name}.parquet")
    base = pd.read_csv(f"results_{name}.csv")
    # geometry (n_train, fold_index, ...) comes from the rows we already wrote
    meta = {r.fold_name: r for r in base[base.status == "ok"].itertuples()}

    out = []
    for (fold_name, seed), sub in pred.groupby(["fold_name", "seed"], sort=False):
        test_df = df.loc[sub.row_id.to_numpy()]
        sc, _, _ = R._score(test_df, sub.y_pred.to_numpy(float), policy, None)
        m = meta.get(fold_name)
        row = {c: np.nan for c in R.RESULT_COLUMNS}
        row.update({
            "arm": name, "fold_name": fold_name,
            "fold_index": getattr(m, "fold_index", np.nan),
            "n_held_out_alleles": getattr(m, "n_held_out_alleles", np.nan),
            "n_train": getattr(m, "n_train", np.nan), "n_test": len(test_df),
            "seed": int(seed), "status": "ok", "censored_policy": policy,
            "calibration_source": "ensemble_only",
            "error": "re-scored from stored predictions, no refit",
        })
        row.update(sc)
        row["calibration_source"] = "ensemble_only"
        out.append(row)
    return pd.DataFrame(out, columns=R.RESULT_COLUMNS)


def report(res, label):
    ok = res[res.status == "ok"]
    if not len(ok):
        print(f"{label}: NO ok rows")
        return
    per_fold = ok.groupby("fold_name").spearman.mean().sort_values()
    t10 = ok.groupby("fold_name").top10_precision.mean()
    q1, q3 = per_fold.quantile(0.25), per_fold.quantile(0.75)
    print(f"\n--- {label} ({len(ok)} ok rows, {per_fold.size} folds) ---")
    print(f"  Spearman  median {per_fold.median():+.4f}   "
          f"IQR [{q1:+.4f}, {q3:+.4f}]   worst {per_fold.iloc[0]:+.4f} ({per_fold.index[0]})")
    print(f"  top-10    median {t10.median():.4f}  (chance 0.10)")
    print(per_fold.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="coverage only")
    ap.add_argument("--time", action="store_true", help="time one fit")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    args = ap.parse_args()

    df = R.load_df()
    mat, idx = emb()
    n_ok, n_miss, sample = coverage(df)
    print(f"emb_joint.npy {mat.shape}  index {len(idx)} keys")
    print(f"coverage: {n_ok}/{len(df)} rows embedded, {n_miss} missing {sample}")
    if n_miss:
        sys.exit(f"ABORT: {n_miss} rows have no joint embedding. "
                 f"Running on a subset would misreport the arm.")
    if args.check:
        return

    if args.time:
        import splits
        name, alle = splits.choose_held_out(df)[0]
        tr, te = splits.split_by_allele(df, alle)
        Xtr, Xte = featurize(df.loc[tr]), featurize(df.loc[te])
        t0 = time.time()
        m = make_model(0)
        m.fit(Xtr, df.loc[tr].y.to_numpy())
        p = m.predict(Xte)
        dt = time.time() - t0
        rho = metrics.spearman_per_allele(df.loc[te], p, censored="tied")
        print(f"fold {name}: fit+predict {dt:.1f}s, "
              f"{m[-1].n_iter_} iters, median rho {rho.median():+.4f}")
        print(f"estimate for 21 folds x {args.seeds} seeds: {dt * 21 * args.seeds / 60:.1f} min")
        return

    t0 = time.time()
    res = R.run_arm(ARM, featurize, make_model, seeds=args.seeds, censored="tied")
    print(f"\n[{ARM}] tied run wall clock {time.time() - t0:.1f}s")

    drop = rescore("drop", ARM, df)
    drop.to_csv(f"results_{ARM}.csv", mode="a", header=False, index=False)
    print(f"[{ARM}] appended {len(drop)} censored='drop' rows "
          f"(re-scored, no refit) to results_{ARM}.csv")

    report(res, "censored='tied'  (HEADLINE)")
    report(drop, "censored='drop'  (robustness)")

    ens = R.ensemble_metrics(ARM, censored="tied", df=df)
    ok = ens[ens.status == "ok"]
    if len(ok):
        print(f"\n--- 5-seed ensemble, censored='tied' ---")
        print(f"  Spearman median {ok.spearman.median():+.4f}   "
              f"worst {ok.spearman.min():+.4f} "
              f"({ok.loc[ok.spearman.idxmin(), 'fold_name']})")
        print(f"  top-10   median {ok.top10_precision.median():.4f}")
        print(f"  calib EUC median {ok.calibration_euc.median():+.4f}   "
              f"coverage68 median {ok.coverage68.median():.4f}")
    print(f"\nTOTAL wall clock {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
