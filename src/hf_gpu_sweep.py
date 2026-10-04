# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.6.0",
#   "numpy<2.3",
#   "pandas",
#   "scipy",
#   "scikit-learn",
#   "xgboost",
#   "pyarrow",
#   "huggingface_hub",
# ]
# ///
"""Wide architecture / loss / head sweep for the conventional supervised net.

WHAT THIS IS
------------
arm_A_supervised_nn.py found its hyperparameters with `tune`: 198 fits over 3
stages, 4 outer folds x 3 seeds, scored ONLY on inner splits of TRAIN. It
reported the winner sits in a flat region, with four candidates effectively
tied. A laptop could not search much wider than that. A GPU can.

This file searches the same space far more widely, under the IDENTICAL
protocol, and adds three things arm A never tried:

  * censored-aware losses. 20.2% of rows sit at the left-censoring floor
    (Thalf == 0, floored to 0.05 h). Arm A regresses them with plain squared
    error, i.e. it tells the network those rows are exactly log10(0.05) when
    all the assay actually says is "at most that". A Tobit (left-censored
    Gaussian) likelihood says the truthful thing instead. A one-sided hinge is
    the cheap version of the same idea.
  * gradient boosting as an alternative head on the same features.
  * architectures, widths, dropout, learning rates, batch sizes and patience
    well outside arm A's grid.

PROTOCOL (this is the part that makes the numbers mean anything)
----------------------------------------------------------------
Copied from arm_A_supervised_nn.tune(). For each of the first N outer folds of
splits.choose_held_out(df) we take its TRAIN side ONLY, hold out a further
groove cluster from inside that train side as an inner validation set, fit each
candidate on what is left, and score the median per-allele Spearman on the
inner validation alleles. The outer test rows are never loaded, so no number
here can leak into the headline. The headline itself is produced later and
locally, by run_experiment.run_arm, with the winner's configuration frozen.

Every candidate is run on the SAME (outer, inner, seed) cells as the arm A
baseline candidate, so candidates can be compared PAIRED rather than on their
noisy absolute means. Arm A's own BEST is included as a candidate in every
stage; if the sweep cannot reproduce it, nothing else in the output is
trustworthy.

LOCAL IMPORT
------------
This file is also imported locally by hf_gpu_score.py, which needs the exact
same SweepMLP class to score the winner through the normal harness. The PEP 723
header above is a comment and the heavy imports are lazy, so importing it on
the laptop is free.

USAGE
    hf jobs uv run --flavor a10g-small --secrets HF_TOKEN \
        -v hf://datasets/<user>/phla-inputs:/inputs \
        hf_gpu_sweep.py --out-repo <user>/phla-emb650 --stage all
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

# --------------------------------------------------------------------------
# features: BLOSUM62 / one-hot, transcribed from arm_A_supervised_nn.py
# --------------------------------------------------------------------------
# Transcribed rather than imported because the GPU container has no copy of the
# project. hf_gpu_score.py asserts, on the laptop, that this featurizer is
# bit-identical to arm_A_supervised_nn.featurizer() on real rows -- if it ever
# drifts, the winner found here would not be the winner scored there.

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}
PEP_LEN = 9
PSEUDO_LEN = 34
BLOSUM_SCALE = 5.0

_B62_ORDER = "ARNDCQEGHILKMFPSTWYV"
_B62_ROWS = """
 4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
-1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
-2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
-2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
 0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
-1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
 0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
-2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
-1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
-1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
-1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
-2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
 1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
 0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
-2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
 0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""


def _blosum62():
    m = np.array([[float(x) for x in r.split()] for r in _B62_ROWS.strip().splitlines()])
    assert m.shape == (20, 20) and (m == m.T).all(), "BLOSUM62 is not symmetric"
    p = [_B62_ORDER.index(a) for a in AA]
    m = m[np.ix_(p, p)]
    for a, v in [("W", 11), ("C", 9), ("H", 8), ("P", 7), ("Y", 7), ("G", 6)]:
        assert m[AA_IDX[a], AA_IDX[a]] == v, f"BLOSUM62 diagonal wrong at {a}"
    return m / BLOSUM_SCALE


B62 = _blosum62()
EYE = np.eye(20)
TABLES = {"blosum": [B62], "onehot": [EYE], "both": [B62, EYE]}

_PSEUDO = None
_CACHE: dict = {}


def pseudo_map(path="alleles.json"):
    global _PSEUDO
    if _PSEUDO is None:
        d = json.load(open(path))
        _PSEUDO = {k: v["pseudo"] for k, v in d.items()}
        bad = [k for k, v in _PSEUDO.items() if not v or len(v) != PSEUDO_LEN]
        if bad:
            raise ValueError(f"alleles without a 34-mer pseudo-sequence: {bad}")
    return _PSEUDO


def _encode(seq, tables):
    idx = [AA_IDX[c] for c in seq]
    return np.concatenate([t[idx].ravel() for t in tables])


def _lookup(keys, seqs, tables, tag):
    cache = _CACHE.setdefault(tag, {})
    for k, s in zip(keys, seqs):
        if k not in cache:
            cache[k] = _encode(s, tables)
    return cache


def featurizer(mode="both"):
    """featurize(df_subset) -> X. Deterministic, no fitted state, no seed."""
    tables = TABLES[mode]

    def featurize(df_subset):
        pm = pseudo_map()
        peps = df_subset.Pep.to_numpy()
        hlas = df_subset.HLA.to_numpy()
        pc = _lookup(np.unique(peps), np.unique(peps), tables, f"pep/{mode}")
        au = np.unique(hlas)
        ac = _lookup(au, [pm[a] for a in au], tables, f"hla/{mode}")
        return np.hstack([np.stack([pc[p] for p in peps]),
                          np.stack([ac[a] for a in hlas])])

    featurize.mode = mode
    featurize.n_pep_cols = PEP_LEN * 20 * len(tables)
    return featurize


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------
# SweepMLP is arm_A_supervised_nn.TorchMLP with three additions:
#   loss       'mse' (arm A's), 'tobit', 'hinge', 'huber'
#   act        'relu' (arm A's) or 'gelu'
#   norm       None (arm A's) or 'batch'
# With loss='mse', act='relu', norm=None it is arm A's network, which
# hf_gpu_score.py checks by fitting both on the same data and comparing
# predictions.
#
# CENSORING, AND HOW THE MODEL LEARNS WHICH ROWS ARE CENSORED
#   .fit() only ever sees (X, y) -- that is run_experiment's contract -- so the
#   censoring flag has to come out of y. data.load() sets
#   y = log10(Thalf.clip(lower=0.05)) and censored = Thalf <= 0, and the
#   smallest NON-zero Thalf in the file is exactly 0.05, so y <= log10(0.05)
#   recovers the censored set with ONE false positive out of 28,166 (the single
#   genuine Thalf == 0.05 measurement). That row is treated as "at most 0.05 h"
#   instead of "exactly 0.05 h", which is a distinction the assay cannot make
#   anyway. Nothing about the TEST rows is used: this is read off y_train only.

CENSOR_FLOOR_Y = float(np.log10(0.05))


def censored_from_y(y):
    return np.asarray(y, dtype=np.float64) <= CENSOR_FLOOR_Y + 1e-9


def _allele_groups(X, n_pep_cols):
    """Group row indices by allele, read back out of X. Verbatim from arm A."""
    blk = X[:, n_pep_cols:]
    v = np.random.default_rng(20261003).standard_normal(blk.shape[1])
    _, inv = np.unique(np.round(blk @ v, 6), return_inverse=True)
    n = int(inv.max()) + 1
    assert 2 <= n <= 80, f"allele grouping found {n} groups, expected 2..80"
    return inv, n


def _torch():
    import torch
    return torch


def _device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_net(d_in, hidden, dropout, act, norm):
    import torch.nn as nn
    acts = {"relu": nn.ReLU, "gelu": nn.GELU}
    layers, d = [], d_in
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if norm == "batch":
            layers.append(nn.BatchNorm1d(h))
        layers += [acts[act](), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


class SweepMLP:
    """sklearn-shaped wrapper: .fit(X, y) / .predict(X)."""

    def __init__(self, seed=0, hidden=(256, 128), dropout=0.1, lr=3e-4,
                 weight_decay=1e-5, batch=512, max_epochs=300, patience=25,
                 val_frac=0.15, val_mode="allele", val_metric="rho",
                 loss="mse", act="relu", norm=None, huber_delta=0.5,
                 n_pep_cols=360, device=None):
        self.__dict__.update(locals())
        del self.self
        self.device = device
        self.best_epoch_ = None

    # -- the three losses arm A does not have ------------------------------
    def _loss(self, pred, yb, cb, log_sigma):
        import torch
        import torch.nn.functional as F
        if self.loss == "mse":
            return F.mse_loss(pred, yb)
        if self.loss == "huber":
            return F.huber_loss(pred, yb, delta=self.huber_delta)
        if self.loss == "hinge":
            # Uncensored: squared error. Censored: penalise only predictions
            # ABOVE the floor -- "at most c" is satisfied by anything below it.
            unc = (~cb).float()
            sq = (pred - yb) ** 2
            over = torch.clamp(pred - CENSOR_FLOOR_Y, min=0.0) ** 2
            return (unc * sq + (1 - unc) * over).mean()
        if self.loss == "tobit":
            # Left-censored Gaussian likelihood with one learned global sigma.
            # Uncensored rows: Gaussian NLL. Censored rows: -log Phi((c-mu)/s).
            sigma = torch.exp(log_sigma).clamp(1e-3, 10.0)
            z = (yb - pred) / sigma
            nll_obs = 0.5 * z ** 2 + log_sigma
            zc = (CENSOR_FLOOR_Y - pred) / sigma
            nll_cen = -torch.special.log_ndtr(zc)
            return torch.where(cb, nll_cen, nll_obs).mean()
        raise ValueError(f"unknown loss {self.loss!r}")

    def _split(self, X, n):
        rng = np.random.default_rng(1000 + self.seed)
        if self.val_mode == "allele":
            g, ng = _allele_groups(X, self.n_pep_cols)
            order = rng.permutation(ng)
            k0 = min(max(1, int(round(self.val_frac * ng))), ng - 1)
            m = np.isin(g, order[:k0])
            for k in range(k0 + 1, ng):
                if m.sum() >= 300:
                    break
                m = np.isin(g, order[:k])
            return ~m, m
        m = np.zeros(n, dtype=bool)
        m[rng.choice(n, max(1, int(self.val_frac * n)), replace=False)] = True
        return ~m, m

    def _val_groups(self, X, va, min_n=20):
        g, _ = _allele_groups(X, self.n_pep_cols)
        g = g[va]
        return [np.where(g == u)[0] for u in np.unique(g) if (g == u).sum() >= min_n]

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        from scipy import stats
        dev = self.device or _device()
        self.device = dev

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        cen = censored_from_y(y)
        tr, va = self._split(X, len(X))

        Xtr = torch.from_numpy(X[tr]).to(dev)
        ytr = torch.from_numpy(y[tr]).to(dev)
        ctr = torch.from_numpy(cen[tr]).to(dev)
        Xva = torch.from_numpy(X[va]).to(dev)
        yva = torch.from_numpy(y[va]).to(dev)
        yva_np = y[va].astype(np.float64)
        vgroups = self._val_groups(X, va) if self.val_metric == "rho" else []

        torch.manual_seed(self.seed)
        self.model = _make_net(X.shape[1], self.hidden, self.dropout,
                               self.act, self.norm).to(dev)
        params = list(self.model.parameters())
        self.log_sigma = nn.Parameter(torch.zeros((), device=dev))
        if self.loss == "tobit":
            params = params + [self.log_sigma]
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        gen = torch.Generator(device="cpu").manual_seed(self.seed)

        best, best_state, bad, n = np.inf, None, 0, len(Xtr)
        ep = -1
        for ep in range(self.max_epochs):
            self.model.train()
            perm = torch.randperm(n, generator=gen).to(dev)
            for i in range(0, n, self.batch):
                b = perm[i:i + self.batch]
                if self.norm == "batch" and len(b) < 2:
                    continue
                opt.zero_grad(set_to_none=True)
                out = self.model(Xtr[b]).squeeze(-1)
                self._loss(out, ytr[b], ctr[b], self.log_sigma).backward()
                opt.step()
            self.model.eval()
            with torch.no_grad():
                pv_t = self.model(Xva).squeeze(-1)
                if self.val_metric == "rho":
                    pv = pv_t.cpu().numpy().astype(np.float64)
                    r = [stats.spearmanr(pv[i], yva_np[i]).statistic for i in vgroups]
                    r = [x for x in r if np.isfinite(x)]
                    v = -float(np.median(r)) if r else np.inf
                else:
                    # Early stopping always uses plain MSE here when
                    # val_metric='mse', regardless of the TRAINING loss, so the
                    # stopping rule is comparable across loss variants.
                    v = float(((pv_t - yva) ** 2).mean())
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = {k: t.detach().clone()
                              for k, t in self.model.state_dict().items()}
                self.best_epoch_ = ep
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.val_score_ = best
        self.n_val_ = int(va.sum())
        self.epochs_run_ = ep + 1
        return self

    def predict(self, X):
        import torch
        self.model.eval()
        X = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 4096):
                out.append(self.model(X[i:i + 4096]).squeeze(-1).cpu().numpy())
        return np.concatenate(out).astype(np.float64)


class XGBHead:
    """Gradient boosting on the same BLOSUM/one-hot features.

    Early stopping uses the SAME allele-grouped inner split as SweepMLP, so the
    two heads are stopped on the same information and the comparison is about
    the head, not the stopping rule.
    """

    def __init__(self, seed=0, n_estimators=3000, max_depth=6, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.5, min_child_weight=5,
                 reg_lambda=1.0, val_frac=0.15, n_pep_cols=360, early=100):
        self.__dict__.update(locals())
        del self.self

    def fit(self, X, y):
        import xgboost as xgb
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        tr, va = SweepMLP(seed=self.seed, val_frac=self.val_frac,
                          n_pep_cols=self.n_pep_cols)._split(X, len(X))
        try:
            dev = "cuda"
            import torch
            if not torch.cuda.is_available():
                dev = "cpu"
        except Exception:
            dev = "cpu"
        self.m = xgb.XGBRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight, reg_lambda=self.reg_lambda,
            tree_method="hist", device=dev, random_state=self.seed,
            early_stopping_rounds=self.early, eval_metric="rmse")
        self.m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        self.best_epoch_ = int(getattr(self.m, "best_iteration", -1))
        self.epochs_run_ = self.best_epoch_ + 1
        return self

    def predict(self, X):
        return np.asarray(self.m.predict(np.asarray(X, dtype=np.float32)),
                          dtype=np.float64)


# --------------------------------------------------------------------------
# the candidate grid
# --------------------------------------------------------------------------

BASE = dict(hidden=(256, 128), dropout=0.1, lr=3e-4, weight_decay=1e-5,
            batch=512, max_epochs=300, patience=25, val_frac=0.15,
            val_mode="allele", val_metric="rho", loss="mse", act="relu",
            norm=None)

# (label, encoding mode, kwargs-diff-from-BASE). "ARM A BEST" must come first:
# every other candidate is reported as a paired delta against it.
STAGE1 = [
    ("ARM A BEST (baseline)",      "both",   {}),
    ("arch 128",                   "both",   dict(hidden=(128,))),
    ("arch 512-256",               "both",   dict(hidden=(512, 256))),
    ("arch 512-256-128",           "both",   dict(hidden=(512, 256, 128))),
    ("arch 1024-512-256",          "both",   dict(hidden=(1024, 512, 256))),
    ("arch 1024-1024-512-256",     "both",   dict(hidden=(1024, 1024, 512, 256))),
    ("arch 256-128-64",            "both",   dict(hidden=(256, 128, 64))),
    ("arch 2048-512",              "both",   dict(hidden=(2048, 512))),
    ("dropout 0.0",                "both",   dict(dropout=0.0)),
    ("dropout 0.05",               "both",   dict(dropout=0.05)),
    ("dropout 0.2",                "both",   dict(dropout=0.2)),
    ("dropout 0.3",                "both",   dict(dropout=0.3)),
    ("lr 1e-4",                    "both",   dict(lr=1e-4)),
    ("lr 1e-3",                    "both",   dict(lr=1e-3)),
    ("lr 3e-3",                    "both",   dict(lr=3e-3)),
    ("wd 0",                       "both",   dict(weight_decay=0.0)),
    ("wd 1e-4",                    "both",   dict(weight_decay=1e-4)),
    ("wd 1e-3",                    "both",   dict(weight_decay=1e-3)),
    ("batch 128",                  "both",   dict(batch=128)),
    ("batch 256",                  "both",   dict(batch=256)),
    ("batch 1024",                 "both",   dict(batch=1024)),
    ("patience 60 / 800 epochs",   "both",   dict(patience=60, max_epochs=800)),
    ("gelu",                       "both",   dict(act="gelu")),
    ("batchnorm",                  "both",   dict(norm="batch")),
    ("LOSS tobit",                 "both",   dict(loss="tobit")),
    ("LOSS hinge (one-sided)",     "both",   dict(loss="hinge")),
    ("LOSS huber 0.5",             "both",   dict(loss="huber")),
    ("stop on MSE not rho",        "both",   dict(val_metric="mse")),
    ("encoding blosum only",       "blosum", {}),
    ("encoding onehot only",       "onehot", {}),
]

STAGE_GBM = [
    ("GBM depth6 lr0.05",   "both", dict(max_depth=6,  learning_rate=0.05)),
    ("GBM depth4 lr0.05",   "both", dict(max_depth=4,  learning_rate=0.05)),
    ("GBM depth8 lr0.05",   "both", dict(max_depth=8,  learning_rate=0.05)),
    ("GBM depth6 lr0.02",   "both", dict(max_depth=6,  learning_rate=0.02)),
    ("GBM depth10 lr0.03",  "both", dict(max_depth=10, learning_rate=0.03,
                                         min_child_weight=10)),
    ("GBM depth6 colsub0.2", "both", dict(max_depth=6, learning_rate=0.05,
                                          colsample_bytree=0.2)),
]


# --------------------------------------------------------------------------
# the sweep itself
# --------------------------------------------------------------------------

def inner_cells(df, splits_mod, n_outer):
    """The (outer, inner) inner-validation cells. Verbatim from arm_A.tune()."""
    outer = splits_mod.choose_held_out(df)
    cells = []
    for oi in range(n_outer):
        o_name, o_alleles = outer[oi]
        tr_idx, _ = splits_mod.split_by_allele(df, o_alleles)
        train = df.loc[tr_idx]
        elig = [(n, a) for n, a in outer
                if n != o_name and train.HLA.isin(a).sum() >= 300]
        i_name, i_alleles = elig[oi % len(elig)]
        i_tr, i_te = splits_mod.split_by_allele(train, i_alleles)
        cells.append((o_name, i_name, train.loc[i_tr], train.loc[i_te]))
    return cells


def run_sweep(cells, cands, seeds, metrics_mod, kind="mlp", tag=""):
    import pandas as pd
    rows = []
    for ci, (label, mode, kw) in enumerate(cands, 1):
        f = featurizer(mode)
        t_c = time.time()
        for o_name, i_name, fit_df, val_df in cells:
            Xf, Xv = f(fit_df), f(val_df)
            yf = fit_df.y.to_numpy()
            for s in seeds:
                t0 = time.time()
                np.random.seed(s)
                if kind == "mlp":
                    m = SweepMLP(seed=s, n_pep_cols=f.n_pep_cols, **{**BASE, **kw})
                else:
                    m = XGBHead(seed=s, n_pep_cols=f.n_pep_cols, **kw)
                m.fit(Xf, yf)
                p = m.predict(Xv)
                rho = metrics_mod.spearman_per_allele(val_df, p)
                rows.append({
                    "stage": tag, "cand": label, "mode": mode,
                    "outer": o_name, "inner": i_name, "seed": int(s),
                    "rho": float(np.median(rho)) if len(rho) else np.nan,
                    "n_val_alleles": len(rho),
                    "epochs": int(getattr(m, "epochs_run_", -1)),
                    "fit_s": round(time.time() - t0, 1),
                    "cfg": json.dumps({k: (list(v) if isinstance(v, tuple) else v)
                                       for k, v in kw.items()}),
                })
                print(f"  [{tag}] {label:<26} {o_name:<14} s{s} "
                      f"rho={rows[-1]['rho']:+.4f} ep={rows[-1]['epochs']:>3} "
                      f"{rows[-1]['fit_s']:.1f}s", flush=True)
        done = pd.DataFrame(rows)
        sub = done[done.cand == label]
        print(f"[{tag}] {ci}/{len(cands)} {label:<26} MEAN rho={sub.rho.mean():+.4f} "
              f"({time.time() - t_c:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def main():
    import argparse
    import shutil
    import sys

    import pandas as pd
    from huggingface_hub import HfApi

    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/inputs")
    ap.add_argument("--out-repo", required=True)
    ap.add_argument("--n-outer", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--stage", default="all", choices=("all", "1", "gbm", "refine"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    t_container = time.time()
    work = "/tmp/work"
    os.makedirs(work, exist_ok=True)
    for f in ("stability.txt", "alleles.json", "data.py", "splits.py",
              "supertypes.py", "metrics.py"):
        src = os.path.join(a.inputs, "harness", f)
        if not os.path.exists(src):
            src = os.path.join(a.inputs, f)
        shutil.copy(src, os.path.join(work, f))
    os.chdir(work)
    sys.path.insert(0, work)

    import data
    import metrics as metrics_mod
    import splits as splits_mod

    import torch
    print(f"device {_device()} | cuda {torch.cuda.is_available()} | "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
          flush=True)

    df = data.load()
    folds = splits_mod.choose_held_out(df)
    print(f"data {df.shape}  folds {len(folds)}  "
          f"(fold 1 = {folds[0][0]}, {len(folds[0][1])} alleles)", flush=True)
    assert len(folds) == 21, f"expected 21 folds, got {len(folds)} -- protocol changed"

    cells = inner_cells(df, splits_mod, a.n_outer)
    for o, i, ftr, val in cells:
        print(f"cell outer={o:<16} inner-val={i:<16} "
              f"{len(ftr)} fit / {len(val)} val rows, "
              f"{val.HLA.nunique()} val alleles", flush=True)

    seeds = list(range(a.seeds))
    api = HfApi()

    def push(dfr, name):
        p = os.path.join(work, name)
        dfr.to_csv(p, index=False)
        try:
            api.upload_file(path_or_fileobj=p, path_in_repo=f"sweep/{name}",
                            repo_id=a.out_repo, repo_type="dataset")
            print(f"  uploaded sweep/{name} ({len(dfr)} rows)", flush=True)
        except Exception as e:
            print(f"  upload failed ({type(e).__name__}: {e})", flush=True)

    if a.smoke:
        out = run_sweep(cells[:1], STAGE1[:2], [0], metrics_mod, "mlp", "smoke")
        out = pd.concat([out, run_sweep(cells[:1], STAGE_GBM[:1], [0],
                                        metrics_mod, "gbm", "smoke")])
        push(out, "smoke.csv")
        print(out.to_string(index=False))
        return

    allr = []
    if a.stage in ("all", "1"):
        r1 = run_sweep(cells, STAGE1, seeds, metrics_mod, "mlp", "stage1")
        push(r1, "stage1.csv")
        allr.append(r1)
    if a.stage in ("all", "gbm"):
        rg = run_sweep(cells, STAGE_GBM, seeds, metrics_mod, "gbm", "gbm")
        push(rg, "gbm.csv")
        allr.append(rg)

    full = pd.concat(allr) if allr else pd.DataFrame(columns=["stage", "cand"])
    mlp = full[full.stage == "stage1"]
    if a.stage in ("all", "refine") and len(mlp):
        base = mlp[mlp.cand == STAGE1[0][0]]
        piv = mlp.pivot_table(index="cand", values="rho", aggfunc="mean")
        # Paired delta against the arm A baseline on the SAME cells.
        key = ["outer", "inner", "seed"]
        b = base.set_index(key).rho
        deltas = {}
        for c, sub in mlp.groupby("cand"):
            d = sub.set_index(key).rho - b
            deltas[c] = (float(d.mean()), int((d > 0).sum()), int(d.notna().sum()))
        print("\n=== STAGE 1: paired vs ARM A BEST on identical inner cells ===")
        for c in sorted(deltas, key=lambda c: -deltas[c][0]):
            m, w, n = deltas[c]
            print(f"  {c:<28} mean rho {float(piv.loc[c, 'rho']):+.4f}   "
                  f"paired delta {m:+.4f}  wins {w}/{n}")
        top = [c for c in sorted(deltas, key=lambda c: -deltas[c][0])
               if c != STAGE1[0][0]][:3]
        print(f"top 3: {top}", flush=True)

        # Combine the best three one-at-a-time changes, pairwise and all three.
        by_label = {lab: (mode, kw) for lab, mode, kw in STAGE1}
        refine = [("ARM A BEST (baseline)", "both", {})]
        import itertools
        for r in (2, 3):
            for combo in itertools.combinations(top, r):
                kw, mode = {}, "both"
                ok = True
                for c in combo:
                    m2, k2 = by_label[c]
                    if m2 != "both":
                        mode = m2
                    if any(k in kw and kw[k] != v for k, v in k2.items()):
                        ok = False
                    kw.update(k2)
                if ok:
                    refine.append((" + ".join(combo)[:40], mode, kw))
        if len(refine) > 1:
            r3 = run_sweep(cells, refine, list(range(max(3, a.seeds))),
                           metrics_mod, "mlp", "refine")
            push(r3, "refine.csv")
            allr.append(r3)

    out = pd.concat(allr) if allr else pd.DataFrame()
    push(out, "all.csv")
    print(f"\nCONTAINER WALL {time.time() - t_container:.1f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
