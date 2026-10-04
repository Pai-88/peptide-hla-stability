"""One runner every experimental arm calls.

    import run_experiment as R
    res = R.run_arm("esm_pseudo_ridge", featurize, make_model)
    ens = R.ensemble_metrics("esm_pseudo_ridge")

Evaluation protocol (fixed here so every arm is comparable):
  * folds       = splits.choose_held_out(df), the 21 groove clusters. Leave-one-
                  ALLELE-out is impossible on this data (every allele has a near
                  twin), so the fold unit is a cluster of grooves. See splits.py.
  * repeats     = `seeds` independent fits per fold (default 5).
  * metrics     = metrics.py, per held-out ALLELE, then summarised by the MEDIAN
                  across the alleles of that fold. Never pooled over rows.
  * censoring   = passed through to metrics.py explicitly ('tied' or 'drop').

Outputs, written as the run proceeds so a crash never loses finished folds:
  results_<name>.csv            one row per (fold, seed)  -- see RESULT_COLUMNS
  predictions_<name>.parquet    one row per (fold, seed, test row)

Uncertainty: a single fit has no spread, so calibration_euc / coverage68 are
EMPTY in results_<name>.csv unless the model object exposes .predict_std(X).
The real uncertainty estimate is the across-seed spread; ensemble_metrics()
computes it from the predictions file. That is the arm's distinctive claim, so
it is deliberately not faked at the single-seed level.

Nothing here runs at import time.
"""

from __future__ import annotations

import os
import random
import time
import traceback

import numpy as np
import pandas as pd

import data
import metrics
import splits

try:
    import torch
except Exception:                                      # pragma: no cover
    torch = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:                                      # pragma: no cover
    pa = pq = None


# A fold needs at least this many test rows before it is worth fitting, and at
# least one held-out allele with metrics.MIN_N (20) rows or metrics.py will score
# nothing. Both guards are explicit and every skip is logged.
MIN_FOLD_TEST_ROWS = 100
MIN_FOLD_TRAIN_ROWS = 100

RESULT_COLUMNS = [
    "arm", "fold_name", "fold_index", "n_held_out_alleles", "n_train", "n_test",
    "seed", "status",
    "spearman", "spearman_mean", "spearman_min",
    "top10_precision", "top10_precision_mean",
    "calibration_euc", "coverage68", "calibration_source",
    "n_alleles_scored", "n_alleles_skipped", "censored_policy",
    "featurize_s", "fit_predict_s", "wall_clock_s", "error",
]

ENSEMBLE_COLUMNS = [
    "arm", "fold_name", "n_seeds", "n_test", "n_alleles_scored", "n_alleles_skipped",
    "spearman", "spearman_mean", "spearman_min",
    "top10_precision", "top10_precision_mean",
    "calibration_euc", "coverage68", "mean_sigma", "mean_abs_err",
    "censored_policy", "status", "error",
]

_DF = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_df():
    """data.load(), cached for the life of the process. Lazy: not import-time."""
    global _DF
    if _DF is None:
        _DF = data.load()
    return _DF


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU + MPS) so a (fold, seed) is reproducible.

    Called before featurizing a fold and again before each fit. PYTHONHASHSEED is
    deliberately NOT set here: it only takes effect at interpreter start, so
    setting it mid-process would be false reassurance.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        if torch.cuda.is_available():                  # pragma: no cover
            torch.cuda.manual_seed_all(seed)


def _check_X(X, n_rows, what):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] != n_rows:
        raise ValueError(f"featurize returned {X.shape[0]} rows for {what}, expected {n_rows}")
    if not np.isfinite(X).all():
        bad = int((~np.isfinite(X)).sum())
        raise ValueError(f"featurize returned {bad} non-finite values for {what}")
    return X


def _check_pred(p, n_rows):
    p = np.asarray(p, dtype=np.float64).ravel()
    if p.shape[0] != n_rows:
        raise ValueError(f"predict returned {p.shape[0]} values, expected {n_rows}")
    if not np.isfinite(p).all():
        bad = int((~np.isfinite(p)).sum())
        raise ValueError(f"predict returned {bad} non-finite values")
    return p


def _med(s):
    v = np.asarray(s, dtype=float)
    v = v[~np.isnan(v)]
    return (float(np.median(v)), float(v.mean()), float(v.min())) if len(v) else (np.nan,) * 3


def _nanmedian(a):
    """np.nanmedian without the all-NaN warning; all-NaN means 'not measured'."""
    v = np.asarray(a, dtype=float)
    v = v[~np.isnan(v)]
    return float(np.median(v)) if len(v) else np.nan


def sigma_usable(sigma):
    """(ok, reason). A degenerate sigma must not be scored.

    A deterministic model gives every seed the same prediction, so the
    across-seed spread is identically 0. metrics.py then returns EUC = NaN
    (no variance to rank) and coverage68 = 0.0 -- and that 0.0 is an artefact of
    determinism, not a miscalibrated model. Reporting it would read as a result,
    so we blank the calibration columns and say why instead.
    """
    if sigma is None:
        return False, "no sigma supplied"
    s = np.asarray(sigma, dtype=float)
    if not np.isfinite(s).all():
        return False, "sigma has non-finite values"
    if not np.any(s > 0):
        return False, ("across-seed sigma is identically 0 (deterministic model): "
                       "calibration undefined, not measured")
    return True, ""


def _score(test_df, pred, censored, sigma=None):
    """metrics.py -> the handful of numbers that go in one CSV row."""
    rho = metrics.spearman_per_allele(test_df, pred, censored=censored)
    top = metrics.top10_precision_per_allele(test_df, pred, censored=censored)
    r_med, r_mean, r_min = _med(rho)
    t_med, t_mean, _ = _med(top)
    out = {
        "spearman": r_med, "spearman_mean": r_mean, "spearman_min": r_min,
        "top10_precision": t_med, "top10_precision_mean": t_mean,
        "calibration_euc": np.nan, "coverage68": np.nan,
        "calibration_source": "",
        "n_alleles_scored": len(rho),
        "n_alleles_skipped": len(rho.attrs["skipped"]),
    }
    ok, _reason = sigma_usable(sigma)
    if ok:
        cal = metrics.calibration_per_allele(test_df, pred, sigma, censored=censored)
        if len(cal):
            out["calibration_euc"] = _nanmedian(cal.euc.values)
            out["coverage68"] = _nanmedian(cal.coverage68.values)
    return out, rho, top


class _PredWriter:
    """Stream per-fold predictions to parquet (csv.gz if pyarrow is missing)."""

    COLS = ["row_id", "fold_name", "seed", "HLA", "Pep", "y_true", "y_pred"]

    def __init__(self, name):
        self.parquet = pa is not None
        if self.parquet:
            self.path = f"predictions_{name}.parquet"
            self.schema = pa.schema([
                ("row_id", pa.int64()), ("fold_name", pa.string()), ("seed", pa.int32()),
                ("HLA", pa.string()), ("Pep", pa.string()),
                ("y_true", pa.float64()), ("y_pred", pa.float64()),
            ])
            self.writer = pq.ParquetWriter(self.path, self.schema)
        else:                                          # pragma: no cover
            self.path = f"predictions_{name}.csv.gz"
            if os.path.exists(self.path):
                os.remove(self.path)
            self.first = True
        self.n = 0

    def write(self, test_df, fold_name, seed, pred):
        d = {
            "row_id": test_df.index.to_numpy(dtype=np.int64),
            "fold_name": [fold_name] * len(test_df),
            "seed": np.full(len(test_df), int(seed), dtype=np.int32),
            "HLA": test_df.HLA.astype(str).tolist(),
            "Pep": test_df.Pep.astype(str).tolist(),
            "y_true": test_df.y.to_numpy(dtype=np.float64),
            "y_pred": np.asarray(pred, dtype=np.float64),
        }
        if self.parquet:
            self.writer.write_table(pa.Table.from_pydict(d, schema=self.schema))
        else:                                          # pragma: no cover
            pd.DataFrame(d)[self.COLS].to_csv(
                self.path, mode="w" if self.first else "a",
                header=self.first, index=False, compression="gzip")
            self.first = False
        self.n += len(test_df)

    def close(self):
        if self.parquet:
            self.writer.close()


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------

def run_arm(name, featurize, make_model, seeds=5, censored="tied", folds=None,
            df=None, verbose=True):
    """Run one experimental arm over the groove-cluster folds.

    name        str, used for results_<name>.csv and predictions_<name>.parquet.
    featurize   callable(df_subset) -> X, shape (len(df_subset), d). df_subset is
                a slice of data.load() keeping its original index and all columns
                (HLA, Pep, Thalf, censored, y). Called ONCE per fold per side
                (train, test) and reused across seeds, so it must be
                deterministic and must not depend on the seed.
    make_model  callable(seed) -> object with .fit(X, y) and .predict(X).
                Optionally .predict_std(X) for single-fit uncertainty; without it
                the calibration columns stay empty and ensemble_metrics() is the
                source of uncertainty.
    seeds       int (0..seeds-1) or an explicit iterable of ints.
    censored    'tied' or 'drop', passed straight to metrics.py.
    folds       [(fold_name, [allele, ...]), ...]; default splits.choose_held_out.

    Returns the results DataFrame (also written to results_<name>.csv).
    Rows with status != 'ok' have EMPTY metric cells and a reason in `error`;
    they are never silently a zero or a NaN that reads as a result.
    """
    if censored not in ("tied", "drop"):
        raise ValueError("censored must be 'tied' or 'drop'")
    df = load_df() if df is None else df
    folds = splits.choose_held_out(df) if folds is None else folds
    seed_list = list(range(seeds)) if isinstance(seeds, int) else [int(s) for s in seeds]

    csv_path = f"results_{name}.csv"
    pd.DataFrame(columns=RESULT_COLUMNS).to_csv(csv_path, index=False)
    writer = _PredWriter(name)
    rows = []

    def emit(r):
        rows.append(r)
        pd.DataFrame([r])[RESULT_COLUMNS].to_csv(csv_path, mode="a", header=False, index=False)

    def blank(fold_name, i, alleles, n_tr, n_te, seed, status, err):
        return {**{c: np.nan for c in RESULT_COLUMNS},
                "arm": name, "fold_name": fold_name, "fold_index": i,
                "n_held_out_alleles": len(alleles), "n_train": n_tr, "n_test": n_te,
                "seed": seed, "status": status, "censored_policy": censored,
                "calibration_source": "", "error": err}

    t_all = time.time()
    if verbose:
        print(f"[{name}] {len(folds)} folds x {len(seed_list)} seeds, "
              f"censored={censored}, {len(df)} rows", flush=True)

    try:
        for i, (fold_name, alleles) in enumerate(folds, 1):
            try:
                tr_idx, te_idx = splits.split_by_allele(df, alleles)
                train_df, test_df = df.loc[tr_idx], df.loc[te_idx]
                n_tr, n_te = len(train_df), len(test_df)

                # --- explicit fold-level guards, every skip logged -------------
                skip = None
                if n_te < MIN_FOLD_TEST_ROWS:
                    skip = f"fold guard: n_test={n_te} < {MIN_FOLD_TEST_ROWS}"
                elif n_tr < MIN_FOLD_TRAIN_ROWS:
                    skip = f"fold guard: n_train={n_tr} < {MIN_FOLD_TRAIN_ROWS}"
                else:
                    per = test_df.HLA.value_counts()
                    if not (per >= metrics.MIN_N).any():
                        skip = (f"fold guard: no held-out allele has >= {metrics.MIN_N} "
                                f"test rows (max {int(per.max())}); metrics.py would score none")
                if skip:
                    emit(blank(fold_name, i, alleles, n_tr, n_te, -1, "skipped", skip))
                    if verbose:
                        print(f"[{name}] {i:2d}/{len(folds)} {fold_name:<18} SKIPPED  {skip}",
                              flush=True)
                    continue

                t0 = time.time()
                set_seed(seed_list[0])
                Xtr = _check_X(featurize(train_df), n_tr, "train")
                Xte = _check_X(featurize(test_df), n_te, "test")
                if Xtr.shape[1] != Xte.shape[1]:
                    raise ValueError(f"featurize gave {Xtr.shape[1]} train cols vs "
                                     f"{Xte.shape[1]} test cols")
                feat_s = time.time() - t0
                ytr = train_df.y.to_numpy(dtype=np.float64)

            except Exception as e:                     # featurize / split failed
                emit(blank(fold_name, i, alleles, np.nan, np.nan, -1, "failed",
                           f"{type(e).__name__}: {e}"))
                if verbose:
                    print(f"[{name}] {i:2d}/{len(folds)} {fold_name:<18} FAILED   "
                          f"{type(e).__name__}: {e}", flush=True)
                    traceback.print_exc()
                continue

            for j, seed in enumerate(seed_list):
                t1 = time.time()
                try:
                    set_seed(seed)
                    model = make_model(seed)
                    model.fit(Xtr, ytr)
                    pred = _check_pred(model.predict(Xte), n_te)
                    fit_s = time.time() - t1

                    sigma, source = None, ""
                    if hasattr(model, "predict_std"):
                        sigma = np.asarray(model.predict_std(Xte), dtype=np.float64).ravel()
                        usable, why = sigma_usable(sigma)
                        if sigma.shape[0] != n_te:
                            sigma, source = None, "predict_std wrong length, ignored"
                        elif not usable:
                            sigma, source = None, f"predict_std unusable: {why}"
                        else:
                            source = "model.predict_std"

                    sc, rho, _ = _score(test_df, pred, censored, sigma)
                    sc["calibration_source"] = source or "ensemble_only"
                    writer.write(test_df, fold_name, seed, pred)

                    status, err = "ok", ""
                    if sc["n_alleles_scored"] == 0:
                        status = "skipped"
                        err = (f"metrics scored 0 alleles "
                               f"(skipped: {list(rho.attrs['skipped'].items())[:3]})")
                        for k in ("spearman", "spearman_mean", "spearman_min",
                                  "top10_precision", "top10_precision_mean"):
                            sc[k] = np.nan

                    r = blank(fold_name, i, alleles, n_tr, n_te, seed, status, err)
                    r.update(sc)
                    r["featurize_s"] = round(feat_s if j == 0 else 0.0, 3)
                    r["fit_predict_s"] = round(fit_s, 3)
                    r["wall_clock_s"] = round(time.time() - t1 + (feat_s if j == 0 else 0.0), 3)
                    emit(r)
                    if verbose:
                        print(f"[{name}] {i:2d}/{len(folds)} {fold_name:<18} "
                              f"s{seed} tr{n_tr:<6} te{n_te:<5} "
                              f"rho {r['spearman']:+.3f} top10 {r['top10_precision']:.3f} "
                              f"({r['n_alleles_scored']}a) {r['wall_clock_s']:.1f}s"
                              + (f"  [{status}] {err}" if status != "ok" else ""), flush=True)

                except Exception as e:
                    r = blank(fold_name, i, alleles, n_tr, n_te, seed, "failed",
                              f"{type(e).__name__}: {e}")
                    r["wall_clock_s"] = round(time.time() - t1, 3)
                    emit(r)
                    if verbose:
                        print(f"[{name}] {i:2d}/{len(folds)} {fold_name:<18} s{seed} "
                              f"FAILED  {type(e).__name__}: {e}", flush=True)
                        traceback.print_exc()
    finally:
        writer.close()

    res = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    ok = res[res.status == "ok"]
    if verbose:
        print(f"[{name}] done in {time.time() - t_all:.1f}s  "
              f"{len(ok)} ok / {(res.status == 'skipped').sum()} skipped / "
              f"{(res.status == 'failed').sum()} failed  "
              f"-> {csv_path}, {writer.path} ({writer.n} prediction rows)", flush=True)
        if len(ok):
            print(f"[{name}] {metrics.summarise(ok.spearman, 'per-fold median Spearman')}",
                  flush=True)
    return res


# ---------------------------------------------------------------------------
# ensemble
# ---------------------------------------------------------------------------

def ensemble_metrics(name, censored="tied", df=None, per_allele=False):
    """Per-fold metrics for the 5-seed MEAN prediction, plus the calibration of
    the across-seed STANDARD DEVIATION against the absolute error.

    The across-seed spread is this project's uncertainty estimate, so it is
    computed from the stored per-row predictions rather than from any summary:
    rows are matched across seeds by their original dataframe index (row_id),
    so alignment cannot silently drift.

    sigma      = std across seeds, ddof=1 (undefined, and reported empty, for a
                 single seed).
    calibration_euc / coverage68 = medians over held-out alleles of
                 metrics.calibration_per_allele(test, mean_pred, sigma).

    Returns one row per fold (also written to ensemble_<name>.csv). With
    per_allele=True returns the per-allele frame instead.
    """
    df = load_df() if df is None else df
    pq_path, csv_path = f"predictions_{name}.parquet", f"predictions_{name}.csv.gz"
    if os.path.exists(pq_path):
        pred = pd.read_parquet(pq_path)
    elif os.path.exists(csv_path):                     # pragma: no cover
        pred = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"no predictions file for arm '{name}' "
                                f"(looked for {pq_path} and {csv_path})")

    g = pred.groupby(["fold_name", "row_id"], sort=False)
    agg = g.y_pred.agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["fold_name", "row_id", "pred_mean", "pred_std", "n_seeds"]

    rows, pa_rows = [], []
    for fold_name, sub in agg.groupby("fold_name", sort=False):
        n_seeds = int(sub.n_seeds.max())
        test_df = df.loc[sub.row_id.to_numpy()]
        mean_pred = sub.pred_mean.to_numpy(dtype=np.float64)
        sigma = sub.pred_std.to_numpy(dtype=np.float64) if n_seeds > 1 else None

        base = {c: np.nan for c in ENSEMBLE_COLUMNS}
        base.update({"arm": name, "fold_name": fold_name, "n_seeds": n_seeds,
                     "n_test": len(test_df), "censored_policy": censored,
                     "status": "ok", "error": ""})
        if n_seeds < 2:
            sigma = None
            base["error"] = "single seed: across-seed sigma undefined"
        else:
            usable, why = sigma_usable(sigma)
            if not usable:
                base["error"] = why           # calibration columns stay empty
        try:
            sc, rho, _ = _score(test_df, mean_pred, censored, sigma)
            base.update({k: sc[k] for k in
                         ("spearman", "spearman_mean", "spearman_min",
                          "top10_precision", "top10_precision_mean",
                          "calibration_euc", "coverage68",
                          "n_alleles_scored", "n_alleles_skipped")})
            if sigma is not None:
                err = np.abs(mean_pred - test_df.y.to_numpy(dtype=np.float64))
                base["mean_sigma"] = float(np.nanmean(sigma))
                base["mean_abs_err"] = float(err.mean())
            if sc["n_alleles_scored"] == 0:
                base["status"], base["error"] = "skipped", "metrics scored 0 alleles"
            if per_allele:
                cal = (metrics.calibration_per_allele(test_df, mean_pred, sigma,
                                                      censored=censored)
                       if sigma_usable(sigma)[0] else None)
                for allele in rho.index:
                    pa_rows.append({
                        "arm": name, "fold_name": fold_name, "HLA": allele,
                        "n_seeds": n_seeds,
                        "spearman": float(rho[allele]),
                        "euc": (float(cal.euc[allele]) if cal is not None
                                and allele in cal.index else np.nan),
                        "coverage68": (float(cal.coverage68[allele]) if cal is not None
                                       and allele in cal.index else np.nan)})
        except Exception as e:
            base["status"], base["error"] = "failed", f"{type(e).__name__}: {e}"
        rows.append(base)

    if per_allele:
        return pd.DataFrame(pa_rows)
    out = pd.DataFrame(rows, columns=ENSEMBLE_COLUMNS)
    out.to_csv(f"ensemble_{name}.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# smoke test: trivial featurizer + sklearn Ridge
# ---------------------------------------------------------------------------

AA = "ACDEFGHIKLMNPQRSTVWY"


def aa_composition(df_subset):
    """20 amino-acid counts + peptide length = 21 dims. No allele information at
    all, so this is a floor, not a baseline."""
    X = np.zeros((len(df_subset), 21), dtype=np.float64)
    for i, p in enumerate(df_subset.Pep.to_numpy()):
        for ch in p:
            j = AA.find(ch)
            if j >= 0:
                X[i, j] += 1.0
        X[i, 20] = len(p)
    return X


class _BootstrapRidge:
    """Ridge on a seed-dependent bootstrap sample. Only exists so the smoke test
    exercises the across-seed-spread path with a genuinely stochastic model;
    plain Ridge is deterministic and its spread is identically 0."""

    def __init__(self, seed, alpha=1.0):
        from sklearn.linear_model import Ridge
        self.seed, self.m = seed, Ridge(alpha=alpha)

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        i = rng.integers(0, len(X), len(X))
        self.m.fit(X[i], y[i])
        return self

    def predict(self, X):
        return self.m.predict(X)


SHOW = ["fold_name", "n_held_out_alleles", "n_train", "n_test", "seed", "status",
        "spearman", "top10_precision", "calibration_euc", "coverage68",
        "calibration_source", "n_alleles_scored", "wall_clock_s"]
SHOW_ENS = ["fold_name", "n_seeds", "n_test", "n_alleles_scored", "spearman",
            "top10_precision", "calibration_euc", "coverage68", "mean_sigma",
            "mean_abs_err", "error"]


def _smoke():
    from sklearn.linear_model import Ridge

    df = load_df()
    folds = splits.choose_held_out(df)[:3]
    out = {}

    print("=" * 78)
    print("SMOKE 1: aa_composition (21 dims) + sklearn Ridge, 3 folds x 2 seeds")
    print("=" * 78)
    res = run_arm("smoke", aa_composition, lambda s: Ridge(alpha=1.0), seeds=2, folds=folds)
    print("\n--- results_smoke.csv ---")
    print(res[SHOW].to_string(index=False))
    print("\n--- ensemble_metrics('smoke') ---")
    ens = ensemble_metrics("smoke")
    print(ens[SHOW_ENS].to_string(index=False))
    out["smoke"] = (res, ens)

    print("\n" + "=" * 78)
    print("SMOKE 2: same featurizer, STOCHASTIC model (bootstrap Ridge), so the")
    print("         across-seed spread is non-zero and calibration is measurable")
    print("=" * 78)
    res2 = run_arm("smoke_stoch", aa_composition, _BootstrapRidge, seeds=2, folds=folds)
    print("\n--- ensemble_metrics('smoke_stoch') ---")
    ens2 = ensemble_metrics("smoke_stoch")
    print(ens2[SHOW_ENS].to_string(index=False))
    out["smoke_stoch"] = (res2, ens2)

    print("\n" + "=" * 78)
    print("SMOKE 3: guards -- a failing featurizer, a failing model, a tiny fold")
    print("=" * 78)

    def bad_featurize(d):
        if d.HLA.iloc[0].startswith("HLA-B*15"):
            raise RuntimeError("deliberate featurizer explosion")
        return aa_composition(d)

    class BadModel(Ridge):
        def fit(self, X, y):
            raise RuntimeError("deliberate fit explosion")

    extra = [("tiny B*13:02", ["HLA-B*13:02"])]            # 7 rows -> skip guard 1
    res3 = run_arm("smoke_guard", bad_featurize,
                   lambda s: BadModel() if s == 1 else Ridge(alpha=1.0),
                   seeds=2, folds=folds + extra)
    print("\n--- results_smoke_guard.csv (status / error) ---")
    print(res3[["fold_name", "seed", "status", "spearman", "error"]].to_string(index=False))
    out["smoke_guard"] = (res3, None)

    # Guard 2 (no held-out allele reaches metrics.MIN_N rows) needs a fold that
    # clears the row floor but has only tiny alleles, which no real cluster does.
    # Lower the floor to reach it, so both branches are shown rather than assumed.
    global MIN_FOLD_TEST_ROWS
    keep, MIN_FOLD_TEST_ROWS = MIN_FOLD_TEST_ROWS, 10
    try:
        print()
        run_arm("smoke_guard2", aa_composition, lambda s: Ridge(alpha=1.0), seeds=1,
                folds=[("4 tiny alleles", ["HLA-B*40:02", "HLA-A*68:02",
                                           "HLA-A*69:01", "HLA-B*13:02"])])
    finally:
        MIN_FOLD_TEST_ROWS = keep

    print("\n" + "=" * 78)
    print("FILE CHECKS")
    print("=" * 78)
    for arm in ("smoke", "smoke_stoch", "smoke_guard"):
        c = pd.read_csv(f"results_{arm}.csv")
        p = pd.read_parquet(f"predictions_{arm}.parquet")
        print(f"results_{arm}.csv        {c.shape[0]} rows x {c.shape[1]} cols  "
              f"schema_ok={list(c.columns) == RESULT_COLUMNS}  "
              f"status={dict(c.status.value_counts())}")
        print(f"predictions_{arm}.parquet {len(p)} rows  cols={list(p.columns)}  "
              f"folds={p.fold_name.nunique()} seeds={sorted(p.seed.unique())}  "
              f"dupes={int(p.duplicated(['fold_name', 'seed', 'row_id']).sum())}  "
              f"nulls={int(p.isna().sum().sum())}")
    p = pd.read_parquet("predictions_smoke.parquet")
    j = p.merge(df[["HLA", "Pep", "y"]], left_on="row_id", right_index=True,
                suffixes=("", "_df"))
    print(f"\nrow_id join back to data.load(): {len(j)}/{len(p)} matched, "
          f"HLA agrees={bool((j.HLA == j.HLA_df).all())}, "
          f"Pep agrees={bool((j.Pep == j.Pep_df).all())}, "
          f"y_true agrees={bool(np.allclose(j.y_true, j.y))}")
    print(p.head(3).to_string(index=False))
    print(p.dtypes.to_string())
    return out


if __name__ == "__main__":
    _smoke()
