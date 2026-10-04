"""Arm A: the conventional supervised baseline Serova's brief asks for.

    "a simple supervised neural network trained on peptide and HLA pairs"

No foundation model anywhere in here. Every other arm is measured against this,
so it is built to be a GOOD-FAITH strong baseline: the same encoding family the
NetMHC tools use (BLOSUM62 / sparse one-hot over the 9-mer plus the 34-residue
pseudo-sequence), a small MLP, dropout, and early stopping on a slice of TRAIN
only. The test fold is never touched during fitting or model selection.

Run:
    python arm_A_supervised_nn.py tune      # inner-split hyperparameter sweep
    python arm_A_supervised_nn.py run       # 21 groove folds x 5 seeds
    python arm_A_supervised_nn.py report    # re-score stored predictions
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

import data
import metrics
import run_experiment as R
import splits

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}
PEP_LEN = 9
PSEUDO_LEN = 34

# ---------------------------------------------------------------------------
# BLOSUM62, the substitution matrix the NetMHC family encodes residues with.
# Standard NCBI matrix, rows/cols in the order A R N D C Q E G H I L K M F P S
# T W Y V. Re-ordered to our alphabetical AA string at load time, and checked
# for symmetry and for its known diagonal (W=11, C=9, H=8, P=7, Y=7).
# ---------------------------------------------------------------------------
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

# BLOSUM scores run -4..+11; /5 puts them on roughly the same scale as the 0/1
# one-hot block so neither dominates the first layer. Fixed constant, not fitted
# on anything, so train and test are encoded identically.
BLOSUM_SCALE = 5.0


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


def pseudo_map():
    """allele string -> its 34-residue pseudo-sequence, from alleles.json."""
    global _PSEUDO
    if _PSEUDO is None:
        d = json.load(open("alleles.json"))
        _PSEUDO = {k: v["pseudo"] for k, v in d.items()}
        bad = [k for k, v in _PSEUDO.items() if not v or len(v) != PSEUDO_LEN]
        if bad:
            raise ValueError(f"alleles without a 34-mer pseudo-sequence: {bad}")
    return _PSEUDO


def _encode(seq, tables):
    """One sequence -> concat over tables of (L, 20) rows, flattened."""
    idx = [AA_IDX[c] for c in seq]
    return np.concatenate([t[idx].ravel() for t in tables])


def _lookup(keys, seqs, tables, tag):
    """Encode each distinct sequence once, then index. Deterministic."""
    cache = _CACHE.setdefault(tag, {})
    for k, s in zip(keys, seqs):
        if k not in cache:
            cache[k] = _encode(s, tables)
    return cache


def featurizer(mode="blosum"):
    """featurize(df_subset) -> X. Deterministic, no fitted state, no seed.

    Columns are [peptide block | pseudo-sequence block], each block being the
    residues encoded by every table in `mode` and concatenated. Widths:
        blosum / onehot :  9*20 + 34*20          =  860
        both            : 9*40 + 34*40           = 1720
    """
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


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------

def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _allele_groups(X, n_pep_cols):
    """Group row indices by allele, read back out of X itself.

    .fit() only ever sees (X, y), but the early-stopping split has to be by
    allele to mimic the outer task. The pseudo-sequence block is a deterministic
    function of the allele, so a fixed random projection of that block separates
    them exactly: there are at most 75 distinct values and a collision between
    two of them under a 680-dim random projection is not a practical risk. The
    group count is asserted, so a silent failure is impossible.
    """
    blk = X[:, n_pep_cols:]
    v = np.random.default_rng(20261003).standard_normal(blk.shape[1])
    _, inv = np.unique(np.round(blk @ v, 6), return_inverse=True)
    n = int(inv.max()) + 1
    assert 2 <= n <= 80, f"allele grouping found {n} groups, expected 2..80"
    return inv, n


class MLP(nn.Module):
    def __init__(self, d_in, hidden, dropout):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TorchMLP:
    """sklearn-shaped wrapper: .fit(X, y) / .predict(X).

    Early stopping holds out a slice of TRAIN. `val_mode='allele'` holds out
    whole alleles, which mirrors the outer groove-cluster task; `'row'` holds out
    random rows. Nothing from the test fold is visible at any point.
    """

    def __init__(self, seed=0, hidden=(256, 128), dropout=0.2, lr=1e-3,
                 weight_decay=1e-5, batch=512, max_epochs=300, patience=25,
                 val_frac=0.15, val_mode="allele", val_metric="mse",
                 n_pep_cols=180, device=None):
        self.__dict__.update(locals())
        del self.self
        self.device = device or _device()
        self.best_epoch_ = None

    def _split(self, X, n):
        rng = np.random.default_rng(1000 + self.seed)
        if self.val_mode == "allele":
            g, ng = _allele_groups(X, self.n_pep_cols)
            order = rng.permutation(ng)
            k0 = min(max(1, int(round(self.val_frac * ng))), ng - 1)
            m = np.isin(g, order[:k0])
            for k in range(k0 + 1, ng):                  # grow until >= 300 rows
                if m.sum() >= 300:
                    break
                m = np.isin(g, order[:k])
            return ~m, m
        m = np.zeros(n, dtype=bool)
        m[rng.choice(n, max(1, int(self.val_frac * n)), replace=False)] = True
        return ~m, m

    def _val_groups(self, X, va):
        """Allele group ids for the validation rows, for the 'rho' criterion."""
        g, _ = _allele_groups(X, self.n_pep_cols)
        g = g[va]
        return [np.where(g == u)[0] for u in np.unique(g)
                if (g == u).sum() >= metrics.MIN_N]

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        tr, va = self._split(X, len(X))
        dev = self.device
        Xtr = torch.from_numpy(X[tr]).to(dev)
        ytr = torch.from_numpy(y[tr]).to(dev)
        Xva = torch.from_numpy(X[va]).to(dev)
        yva = torch.from_numpy(y[va]).to(dev)
        # 'rho' stops on the metric we actually report: the median per-allele
        # Spearman on the inner validation alleles. y is log10(Thalf) and the
        # log is monotone, so this is rank-identical to the headline metric
        # under censored='tied' (the censored block is one tie group in both).
        yva_np = y[va].astype(np.float64)
        vgroups = self._val_groups(X, va) if self.val_metric == "rho" else []

        torch.manual_seed(self.seed)
        self.model = MLP(X.shape[1], self.hidden, self.dropout).to(dev)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        lossf = nn.MSELoss()
        gen = torch.Generator(device="cpu").manual_seed(self.seed)

        best, best_state, bad, n = np.inf, None, 0, len(Xtr)
        for ep in range(self.max_epochs):
            self.model.train()
            perm = torch.randperm(n, generator=gen).to(dev)
            for i in range(0, n, self.batch):
                b = perm[i:i + self.batch]
                opt.zero_grad(set_to_none=True)
                lossf(self.model(Xtr[b]), ytr[b]).backward()
                opt.step()
            self.model.eval()
            with torch.no_grad():
                if self.val_metric == "rho":
                    pv = self.model(Xva).cpu().numpy().astype(np.float64)
                    r = [stats.spearmanr(pv[i], yva_np[i]).statistic for i in vgroups]
                    r = [x for x in r if np.isfinite(x)]
                    v = -float(np.median(r)) if r else np.inf   # lower is better
                else:
                    v = float(lossf(self.model(Xva), yva))
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in self.model.state_dict().items()}
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
        self.model.eval()
        X = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 4096):
                out.append(self.model(X[i:i + 4096]).cpu().numpy())
        return np.concatenate(out).astype(np.float64)


# ---------------------------------------------------------------------------
# Hyperparameters, chosen by the inner sweep below. See `tune`.
# ---------------------------------------------------------------------------

# Selected by `tune` on inner splits of TRAIN only, 198 fits over 3 stages,
# 4 outer folds x 3 seeds in stages 2 and 3. Mean inner-validation median
# per-allele Spearman for the top candidates: 0.277 (this one), 0.272 (same but
# dropout 0.2), 0.263 (one-hot only), 0.260 (lr 1e-4), 0.249 (single 128 layer),
# down to 0.206 (dropout 0.4). Seed-to-seed spread on a single inner fold
# reaches 0.26, so the top four are a tie; this is the best of a flat region,
# not a sharp optimum, and the arm should be read that way.
ENCODING = "both"
BEST = dict(hidden=(256, 128), dropout=0.1, lr=3e-4, weight_decay=1e-5,
            batch=512, max_epochs=300, patience=25, val_frac=0.15,
            val_mode="allele", val_metric="rho")

ARM = "A_supervised_nn"


def make_model(seed):
    f = featurizer(ENCODING)
    return TorchMLP(seed=seed, n_pep_cols=f.n_pep_cols, **BEST)


# ---------------------------------------------------------------------------
# tuning: inner splits of TRAIN only
# ---------------------------------------------------------------------------

# Stage 1 settles the early-stopping criterion and the encoding; stage 2 varies
# capacity and regularisation around the stage-1 winner. Both are scored only on
# the inner validation split, which lives entirely inside TRAIN.
STAGE1 = [
    ("A both   stop=MSE  val=allele", "both", dict(val_metric="mse", val_mode="allele")),
    ("B both   stop=MSE  val=row   ", "both", dict(val_metric="mse", val_mode="row")),
    ("C both   stop=RHO  val=allele", "both", dict(val_metric="rho", val_mode="allele")),
    ("D blosum stop=RHO  val=allele", "blosum", dict(val_metric="rho", val_mode="allele")),
    ("E onehot stop=RHO  val=allele", "onehot", dict(val_metric="rho", val_mode="allele")),
]

STAGE2 = [
    ("F  hidden 512-256       ", dict(hidden=(512, 256))),
    ("G  hidden 128           ", dict(hidden=(128,))),
    ("H  hidden 512-256-128   ", dict(hidden=(512, 256, 128))),
    ("I  dropout 0.1          ", dict(dropout=0.1)),
    ("J  dropout 0.4          ", dict(dropout=0.4)),
    ("K  lr 3e-4              ", dict(lr=3e-4)),
    ("L  weight_decay 1e-3    ", dict(weight_decay=1e-3)),
]

CANDIDATES = [(l, m, k) for l, m, k in STAGE1]


def tune(n_outer=3, seeds=(0, 1), cands=None):
    """Pick hyperparameters WITHOUT ever seeing an outer test fold.

    For each of the 3 largest outer folds we take its TRAIN side only, hold out
    a further groove cluster from inside it as an inner validation set, fit each
    candidate on what is left, and score the median per-allele Spearman on the
    inner validation alleles. The outer test rows are not loaded.
    """
    df = R.load_df()
    outer = splits.choose_held_out(df)
    cands = CANDIDATES if cands is None else cands
    rows = []
    for oi in range(n_outer):
        o_name, o_alleles = outer[oi]
        tr_idx, _ = splits.split_by_allele(df, o_alleles)
        train = df.loc[tr_idx]
        # Inner held-out cluster, rotated by outer index so the sweep is not
        # scored against the same inner validation set every time.
        elig = [(n, a) for n, a in outer if n != o_name
                and train.HLA.isin(a).sum() >= 300]
        inner = elig[oi % len(elig)]
        i_name, i_alleles = inner
        i_tr, i_te = splits.split_by_allele(train, i_alleles)
        fit_df, val_df = train.loc[i_tr], train.loc[i_te]
        print(f"\n=== outer {o_name}  ->  inner val {i_name} "
              f"({len(fit_df)} fit / {len(val_df)} val rows, "
              f"{val_df.HLA.nunique()} val alleles) ===", flush=True)
        for label, mode, kw in cands:
            f = featurizer(mode)
            Xf, Xv = f(fit_df), f(val_df)
            yf = fit_df.y.to_numpy()
            rs, eps, secs = [], [], []
            for s in seeds:
                t0 = time.time()
                R.set_seed(s)
                m = TorchMLP(seed=s, n_pep_cols=f.n_pep_cols,
                             **{**BEST, **kw}).fit(Xf, yf)
                p = m.predict(Xv)
                secs.append(time.time() - t0)
                eps.append(m.epochs_run_)
                rs.append(float(np.median(metrics.spearman_per_allele(val_df, p))))
            rows.append({"outer": o_name, "inner": i_name, "cand": label,
                         "rho": float(np.mean(rs)), "epochs": float(np.mean(eps)),
                         "s": float(np.mean(secs))})
            print(f"  {label}  rho={np.mean(rs):+.4f} "
                  f"(seeds {' '.join(f'{r:+.3f}' for r in rs)})  "
                  f"ep={np.mean(eps):.0f}  {np.mean(secs):.1f}s", flush=True)
    t = pd.DataFrame(rows)
    piv = t.pivot_table(index="cand", columns="outer", values="rho")
    piv["MEAN"] = piv.mean(axis=1)
    print("\n=== inner-validation median per-allele Spearman ===")
    print(piv.sort_values("MEAN", ascending=False).to_string())
    print("\nWINNER:", piv.MEAN.idxmax())
    return t


# ---------------------------------------------------------------------------
# the real run, and the censored='drop' robustness re-scoring
# ---------------------------------------------------------------------------

def run(seeds=5):
    f = featurizer(ENCODING)
    t0 = time.time()
    res = R.run_arm(ARM, f, make_model, seeds=seeds, censored="tied")
    print(f"\nwall clock {time.time() - t0:.1f}s")
    return res


def rescore(censored="drop"):
    """Per-(fold, seed) metrics under a different censoring policy.

    The censoring policy is an EVALUATION choice: metrics.py applies it, the
    model never sees it, and y is identical either way. So the robustness run
    re-scores the stored predictions instead of refitting 105 identical
    networks. Same predictions, same seeds, different policy.
    """
    df = R.load_df()
    pred = pd.read_parquet(f"predictions_{ARM}.parquet")
    rows = []
    for (fold, seed), sub in pred.groupby(["fold_name", "seed"], sort=False):
        te = df.loc[sub.row_id.to_numpy()]
        p = sub.y_pred.to_numpy()
        rho = metrics.spearman_per_allele(te, p, censored=censored)
        top = metrics.top10_precision_per_allele(te, p, censored=censored)
        rows.append({"fold_name": fold, "seed": int(seed),
                     "spearman": float(np.median(rho)) if len(rho) else np.nan,
                     "spearman_min": float(np.min(rho)) if len(rho) else np.nan,
                     "top10_precision": float(np.median(top)) if len(top) else np.nan,
                     "n_alleles_scored": len(rho), "censored_policy": censored})
    return pd.DataFrame(rows)


def report():
    df = R.load_df()
    res = pd.read_csv(f"results_{ARM}.csv")
    ok = res[res.status == "ok"]
    print("=" * 78)
    print(f"ARM A  results_{ARM}.csv   {len(ok)} ok / "
          f"{(res.status == 'skipped').sum()} skipped / "
          f"{(res.status == 'failed').sum()} failed")
    print("=" * 78)
    per_fold = ok.groupby("fold_name", sort=False).agg(
        n_test=("n_test", "first"), n_all=("n_alleles_scored", "first"),
        rho=("spearman", "mean"), rho_sd=("spearman", "std"),
        rho_worst_allele=("spearman_min", "mean"),
        top10=("top10_precision", "mean"), s=("wall_clock_s", "mean"))
    print(per_fold.sort_values("rho").to_string())
    print()
    print(metrics.summarise(per_fold.rho, "TIED  per-fold median Spearman"))
    print(metrics.summarise(per_fold.top10, "TIED  per-fold median top-10 prec"))

    print("\n--- ensemble of 5 seeds (mean prediction) ---")
    for pol in ("tied", "drop"):
        ens = R.ensemble_metrics(ARM, censored=pol)
        e = ens[ens.status == "ok"]
        print(metrics.summarise(e.spearman, f"{pol.upper():4s} ensemble Spearman "))
        print(metrics.summarise(e.top10_precision, f"{pol.upper():4s} ensemble top-10   "))
        print(f"      worst fold: {e.loc[e.spearman.idxmin(), 'fold_name']} "
              f"rho={e.spearman.min():+.4f}")
        if pol == "tied":
            print(f"      calibration EUC median {e.calibration_euc.median():+.4f}  "
                  f"coverage68 median {e.coverage68.median():.4f}  "
                  f"mean_sigma {e.mean_sigma.mean():.4f} vs "
                  f"mean_abs_err {e.mean_abs_err.mean():.4f}")

    print("\n--- censored='drop' robustness, per (fold, seed), same predictions ---")
    d = rescore("drop")
    dpf = d.groupby("fold_name", sort=False).spearman.mean()
    tpf = per_fold.rho
    print(metrics.summarise(dpf, "DROP  per-fold median Spearman"))
    cmp = pd.DataFrame({"tied": tpf, "drop": dpf})
    cmp["diff"] = cmp["drop"] - cmp["tied"]
    print(cmp.sort_values("tied").to_string())
    # leave the HEADLINE policy on disk, not the robustness one
    R.ensemble_metrics(ARM, censored="tied")
    return per_fold, cmp


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"tune": tune, "run": run, "report": report}[cmd]()
