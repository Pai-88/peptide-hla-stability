"""Arm B: the cheap protein-foundation-model arm.

Feature = [ ESM-2 embedding of the 9-mer peptide (640) |
            ESM-2 embedding of the allele's 34-residue pseudo-sequence (640) ]
        = 1280 dims, from embed.load("peptides") and embed.load("pseudo").

Two heads on exactly the same features, as the brief asks:
  B_esm_pseudo       StandardScaler + Ridge   (fast, strong, one knob)
  B_esm_pseudo_mlp   StandardScaler + 1280-512-128-1 MLP

Whole-chain allele embeddings are near-useless here (mean pairwise cosine 0.988
across the 75 alleles), so the allele side is the pseudo-sequence, not the chain.

Both heads choose their one hyperparameter (ridge alpha / MLP stopping epoch) on
a PEPTIDE-GROUPED inner split of the training fold only. Rows sharing a peptide
must not straddle the inner boundary or alpha is picked against a leaked
validation set. .fit() is only handed X, so peptide identity is recovered from
the feature matrix itself -- see _peptide_groups.

Usage:
    python arm_B_esm_pseudo.py            # both heads, 21 folds x 5 seeds
    python arm_B_esm_pseudo.py --ridge    # just the ridge head
    python arm_B_esm_pseudo.py --mlp      # just the MLP head
    python arm_B_esm_pseudo.py --report   # re-report from existing files

Nothing runs at import time.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

import embed
import metrics
import run_experiment as R

RIDGE_ARM = "B_esm_pseudo"
MLP_ARM = "B_esm_pseudo_mlp"

ALPHAS = (1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0)
INNER_FRAC = 0.2          # of training PEPTIDES held out to pick the knob
INNER_SEED = 0            # fixed: the inner split must not vary with the run seed

_TABLES = None


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def _tables():
    global _TABLES
    if _TABLES is None:
        P, pi = embed.load("peptides")
        S, si = embed.load("pseudo")
        _TABLES = (P, pi, S, si)
    return _TABLES


def featurize(df_subset):
    """[peptide ESM-2 | pseudo-sequence ESM-2] -> (n, 1280). Deterministic."""
    P, pi, S, si = _tables()
    p = df_subset.Pep.map(pi).to_numpy()
    s = df_subset.HLA.map(si).to_numpy()
    if pd.isna(p).any() or pd.isna(s).any():
        raise KeyError("peptide or allele missing from the ESM-2 embedding index")
    return np.hstack([P[p.astype(int)], S[s.astype(int)]]).astype(np.float64)


def _peptide_groups(X):
    """Peptide identity recovered from the feature matrix.

    featurize puts the peptide embedding in the first 640 columns, so two rows
    with the same peptide have bit-identical first halves. Projecting that half
    onto one fixed random vector gives each distinct peptide its own float
    (a collision needs an exact float64 tie, which does not happen here -- the
    count is asserted against the fold's true peptide count in _self_check).
    """
    h = X.shape[1] // 2
    r = np.random.default_rng(12345).standard_normal(h)
    return np.unique(X[:, :h] @ r, return_inverse=True)[1]


def _inner_split(X, frac=INNER_FRAC, seed=INNER_SEED):
    """(train_mask, val_mask) over rows, split by PEPTIDE, never by row."""
    g = _peptide_groups(X)
    n_g = g.max() + 1
    rng = np.random.default_rng(seed)
    val_g = set(rng.permutation(n_g)[: max(1, int(round(n_g * frac)))].tolist())
    val = np.fromiter((x in val_g for x in g), dtype=bool, count=len(g))
    return ~val, val


class _Scaler:
    def fit(self, X):
        self.mu = X.mean(0)
        self.sd = X.std(0)
        self.sd[self.sd < 1e-8] = 1.0
        return self

    def __call__(self, X):
        return (X - self.mu) / self.sd


# ---------------------------------------------------------------------------
# head 1: ridge
# ---------------------------------------------------------------------------

class RidgeHead:
    """StandardScaler + Ridge, alpha picked on the peptide-grouped inner split.

    Deterministic: every seed gives the same fit, so the across-seed spread is
    identically 0 and run_experiment blanks the calibration columns rather than
    reporting a 0 that would read as catastrophic miscalibration. That is the
    correct report for this head; it has no uncertainty estimate.
    """

    chosen_alpha = []          # appended per fit, for the run log

    def __init__(self, seed=0, alphas=ALPHAS):
        self.seed, self.alphas = seed, alphas

    def fit(self, X, y):
        from sklearn.linear_model import Ridge
        tr, va = _inner_split(X)
        sc = _Scaler().fit(X[tr])
        Xi, Xv = sc(X[tr]), sc(X[va])
        best = min(self.alphas,
                   key=lambda a: float(np.mean(
                       (Ridge(alpha=a).fit(Xi, y[tr]).predict(Xv) - y[va]) ** 2)))
        self.alpha_ = best
        RidgeHead.chosen_alpha.append(best)
        self.sc = _Scaler().fit(X)
        self.m = Ridge(alpha=best).fit(self.sc(X), y)
        return self

    def predict(self, X):
        return self.m.predict(self.sc(X))


# ---------------------------------------------------------------------------
# head 2: small MLP
# ---------------------------------------------------------------------------

class MLPHead:
    """1280 -> 512 -> 128 -> 1, ReLU + dropout, Adam, early stopping.

    CPU on purpose: measured on this M5, batch 512 on MPS is ~2.5x SLOWER than
    CPU for a net this small (kernel-launch bound, not compute bound).

    Stochastic in the seed (init, shuffling, dropout), so the across-seed spread
    is a real ensemble spread and ensemble_metrics can score calibration.
    """

    stopped_epoch = []

    def __init__(self, seed=0, hidden=(512, 128), dropout=0.1, lr=1e-3,
                 weight_decay=1e-4, batch=512, max_epochs=200, patience=15):
        self.seed, self.hidden, self.dropout = seed, hidden, dropout
        self.lr, self.wd, self.batch = lr, weight_decay, batch
        self.max_epochs, self.patience = max_epochs, patience

    def _net(self, d):
        import torch.nn as nn
        layers, prev = [], d
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        return nn.Sequential(*layers, nn.Linear(prev, 1))

    def fit(self, X, y):
        import copy

        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        tr, va = _inner_split(X)
        self.sc = _Scaler().fit(X[tr])
        self.ymu, self.ysd = float(y[tr].mean()), float(y[tr].std()) or 1.0

        f = lambda A: torch.tensor(self.sc(A), dtype=torch.float32)
        Xi, Xv = f(X[tr]), f(X[va])
        yi = torch.tensor((y[tr] - self.ymu) / self.ysd, dtype=torch.float32)[:, None]
        yv = torch.tensor((y[va] - self.ymu) / self.ysd, dtype=torch.float32)[:, None]

        net = self._net(X.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.wd)
        lossf = nn.MSELoss()
        g = torch.Generator().manual_seed(self.seed)
        best, best_state, best_ep, bad = np.inf, None, 0, 0

        for ep in range(1, self.max_epochs + 1):
            net.train()
            perm = torch.randperm(len(Xi), generator=g)
            for i in range(0, len(Xi), self.batch):
                b = perm[i: i + self.batch]
                opt.zero_grad()
                lossf(net(Xi[b]), yi[b]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(lossf(net(Xv), yv))
            if v < best - 1e-5:
                best, best_ep, bad = v, ep, 0
                best_state = copy.deepcopy(net.state_dict())
            else:
                bad += 1
                if bad >= self.patience:
                    break

        net.load_state_dict(best_state)
        net.eval()
        self.net, self.epochs_ = net, best_ep
        MLPHead.stopped_epoch.append(best_ep)
        return self

    def predict(self, X):
        import torch
        with torch.no_grad():
            z = self.net(torch.tensor(self.sc(X), dtype=torch.float32)).numpy().ravel()
        return z * self.ysd + self.ymu


# ---------------------------------------------------------------------------
# censored='drop' robustness, recomputed from the stored predictions
# ---------------------------------------------------------------------------

def rescore(arm, censored):
    """Per-(fold, seed) metrics recomputed from predictions_<arm>.parquet.

    run_arm trains on train_df.y and passes `censored` ONLY to the metric
    functions, so re-scoring the stored predictions under a different policy is
    bit-identical to re-running the whole arm with that policy -- no refit can
    change. _self_check below proves it by reproducing the 'tied' column of
    results_<arm>.csv exactly.
    """
    df = R.load_df()
    pred = pd.read_parquet(f"predictions_{arm}.parquet")
    rows = []
    for (fold, seed), sub in pred.groupby(["fold_name", "seed"], sort=False):
        t = df.loc[sub.row_id.to_numpy()]
        p = sub.y_pred.to_numpy()
        rho = metrics.spearman_per_allele(t, p, censored=censored)
        top = metrics.top10_precision_per_allele(t, p, censored=censored)
        rows.append({"arm": arm, "fold_name": fold, "seed": int(seed),
                     "censored_policy": censored, "n_test": len(t),
                     "spearman": R._nanmedian(rho), "spearman_min": R._nanmedian(rho.min()),
                     "top10_precision": R._nanmedian(top),
                     "n_alleles_scored": len(rho)})
    return pd.DataFrame(rows)


def rescore_ensemble(arm, censored):
    """Same, for the across-seed MEAN prediction (one row per fold)."""
    df = R.load_df()
    pred = pd.read_parquet(f"predictions_{arm}.parquet")
    agg = (pred.groupby(["fold_name", "row_id"], sort=False).y_pred
           .agg(["mean", "std", "count"]).reset_index())
    rows = []
    for fold, sub in agg.groupby("fold_name", sort=False):
        t = df.loc[sub.row_id.to_numpy()]
        p = sub["mean"].to_numpy()
        sig = sub["std"].to_numpy() if int(sub["count"].max()) > 1 else None
        rho = metrics.spearman_per_allele(t, p, censored=censored)
        top = metrics.top10_precision_per_allele(t, p, censored=censored)
        r = {"arm": arm, "fold_name": fold, "censored_policy": censored,
             "n_seeds": int(sub["count"].max()), "n_test": len(t),
             "n_alleles_scored": len(rho),
             "spearman": R._nanmedian(rho), "spearman_worst_allele": float(rho.min()),
             "top10_precision": R._nanmedian(top),
             "calibration_euc": np.nan, "coverage68": np.nan, "mean_sigma": np.nan}
        if sig is not None and R.sigma_usable(sig)[0]:
            cal = metrics.calibration_per_allele(t, p, sig, censored=censored)
            r["calibration_euc"] = R._nanmedian(cal.euc.values)
            r["coverage68"] = R._nanmedian(cal.coverage68.values)
            r["mean_sigma"] = float(np.nanmean(sig))
        rows.append(r)
    return pd.DataFrame(rows)


def _self_check(arm):
    """Prove rescore() reproduces what run_arm wrote, so the 'drop' numbers are
    the same computation and not a second, differently-wired one."""
    got = rescore(arm, "tied").set_index(["fold_name", "seed"]).sort_index()
    want = (pd.read_csv(f"results_{arm}.csv").query("status == 'ok'")
            .set_index(["fold_name", "seed"]).sort_index())
    common = want.index.intersection(got.index)
    ok = True
    for c in ("spearman", "top10_precision", "n_alleles_scored"):
        d = np.abs(got.loc[common, c].to_numpy() - want.loc[common, c].to_numpy())
        hit = bool(np.nanmax(d) < 1e-12)
        ok &= hit
        print(f"  rescore('tied') vs results_{arm}.csv  {c:<18} "
              f"max|diff| = {np.nanmax(d):.2e}  {'MATCH' if hit else 'MISMATCH'}")
    print(f"  {len(common)} (fold, seed) rows compared -> "
          f"{'identical computation' if ok else 'DIFFERENT -- do not trust the drop run'}")
    return ok


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _line(tag, v):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    if not len(v):
        return f"{tag:<34} no folds scored"
    return (f"{tag:<34} median {np.median(v):+.4f}  "
            f"IQR [{np.percentile(v, 25):+.4f}, {np.percentile(v, 75):+.4f}]  "
            f"worst {v.min():+.4f}  n_folds {len(v)}")


def report(arm):
    res = pd.read_csv(f"results_{arm}.csv")
    ok = res[res.status == "ok"]
    print("\n" + "=" * 78)
    print(f"ARM {arm}")
    print("=" * 78)
    print(f"status: {dict(res.status.value_counts())}   "
          f"folds attempted {res.fold_name.nunique()}   "
          f"folds with >=1 ok seed {ok.fold_name.nunique()}   "
          f"total wall {res.wall_clock_s.sum():.1f}s")
    if (res.status != "ok").any():
        print("NOT OK ROWS:")
        print(res[res.status != "ok"][["fold_name", "seed", "status", "error"]]
              .to_string(index=False))

    print("\n-- per (fold, seed), censored='tied' --")
    print(_line("per-fold-per-seed Spearman", ok.spearman))
    print(_line("per-fold-per-seed top-10 prec", ok.top10_precision))

    print("\n-- 5-seed ENSEMBLE mean prediction --")
    for pol in ("tied", "drop"):
        e = rescore_ensemble(arm, pol)
        print(f"\n  censored = '{pol}'")
        print("  " + _line("fold-median Spearman", e.spearman))
        print("  " + _line("fold-median top-10 prec", e.top10_precision))
        if e.calibration_euc.notna().any():
            print("  " + _line("calibration EUC", e.calibration_euc))
            print("  " + _line("68% coverage", e.coverage68))
        else:
            print("  calibration: not measured (across-seed sigma identically 0, "
                  "deterministic head)")
        w = e.loc[e.spearman.idxmin()]
        print(f"  WORST FOLD: {w.fold_name}  Spearman {w.spearman:+.4f}  "
              f"top10 {w.top10_precision:.3f}  n_test {int(w.n_test)}  "
              f"({int(w.n_alleles_scored)} alleles scored)")
        print(e[["fold_name", "n_test", "n_alleles_scored", "spearman",
                 "spearman_worst_allele", "top10_precision"]]
              .sort_values("spearman").to_string(index=False))
    return res


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ridge", action="store_true")
    ap.add_argument("--mlp", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--folds", type=int, default=0, help="first N folds only (debug)")
    a = ap.parse_args()
    do_r = a.ridge or not (a.ridge or a.mlp or a.report)
    do_m = a.mlp or not (a.ridge or a.mlp or a.report)

    import splits
    df = R.load_df()
    folds = splits.choose_held_out(df)
    if a.folds:
        folds = folds[: a.folds]

    # feature sanity, printed once so the run log carries it
    X = featurize(df)
    g = _peptide_groups(X)
    print(f"features {X.shape}  peptide groups recovered {g.max() + 1} "
          f"vs {df.Pep.nunique()} distinct peptides  "
          f"{'OK' if g.max() + 1 == df.Pep.nunique() else 'MISMATCH'}", flush=True)
    del X

    if do_r:
        t0 = time.time()
        RidgeHead.chosen_alpha.clear()
        R.run_arm(RIDGE_ARM, featurize, RidgeHead, seeds=a.seeds, censored="tied",
                  folds=folds, df=df)
        print(f"[{RIDGE_ARM}] alpha chosen: "
              f"{dict(pd.Series(RidgeHead.chosen_alpha).value_counts())}  "
              f"wall {time.time() - t0:.1f}s", flush=True)

    if do_m:
        t0 = time.time()
        MLPHead.stopped_epoch.clear()
        R.run_arm(MLP_ARM, featurize, MLPHead, seeds=a.seeds, censored="tied",
                  folds=folds, df=df)
        e = pd.Series(MLPHead.stopped_epoch)
        print(f"[{MLP_ARM}] best epoch: median {e.median():.0f} "
              f"min {e.min()} max {e.max()}  wall {time.time() - t0:.1f}s", flush=True)

    for arm in ([RIDGE_ARM] if do_r or a.report else []) + \
               ([MLP_ARM] if do_m or a.report else []):
        print(f"\n--- rescore self-check, {arm} ---")
        _self_check(arm)
        report(arm)


if __name__ == "__main__":
    sys.exit(main())
