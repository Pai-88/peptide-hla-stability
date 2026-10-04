"""Regression heads on cached ESM-2 embeddings, with censored-aware losses.

WHAT IS HERE
  encodings   'concat' = pep-alone || HLA-alone      (no cross-talk possible)
              'joint'  = pep-in-context || HLA-in-context, one ESM-2 pass
  heads       'ridge'  closed form / linear-torch    'mlp' 1 hidden layer
  losses      'naive'  plain squared error on the floored y      <- WRONG, kept
              'tobit'  left-censored Gaussian likelihood          <- default
              'twopart' classify above/below floor, regress above it
  ensemble    5 seeds. mean = prediction, std across seeds = uncertainty.
  logging     every run appends one row to runs.csv.

THE CENSORING DECISION (documented, not silent)
5,679 of 28,166 rows (20.2%) sit at Thalf == 0.0, i.e. "dissociated faster than
the assay could resolve". data.load() floors them at 0.05 h so log10 is defined,
which makes y = -1.301 for all of them. Three ways to treat that:

  naive   Train as if -1.301 were a measurement. It is not: the true value is
          somewhere in (-inf, -1.301]. Squared error then PUNISHES the model for
          predicting -2.5 on a peptide that really does fall off instantly, so
          the fit is dragged upward at the bottom of the range and the whole
          calibration tilts. Reported only as the control.
  tobit   Observed rows get the Gaussian density, censored rows get the Gaussian
          CDF below the floor: -log P(y <= c). The model is free to predict
          anything below the floor at no cost, which is exactly what the data
          says. One extra learned scalar (log sigma). This is the default.
  twopart P(above floor) from a logistic head, plus a regression fit on the
          above-floor rows only. Cleaner if the floor is a different physical
          process rather than the tail of one; costs you a combination rule.

Interface note: this module only ever sees (train_index, test_index), so it
works with splits.split_random, split_by_peptide, and split_by_allele the moment
that one is written. Nothing here imports the held-out-allele logic.
"""

from __future__ import annotations

import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

import data

FLOOR = float(np.log10(data.CENSOR_FLOOR))   # -1.301
SEEDS = (0, 1, 2, 3, 4)
RUNS_CSV = "runs.csv"


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------

def _embed_module():
    """Prefer the shared embed.py; fall back to embed_local.py."""
    try:
        import embed
        if hasattr(embed, "get"):
            return embed
    except Exception:
        pass
    import embed_local
    return embed_local


def features(df, encoding):
    """[n_rows, D] float32 aligned to df.index order.

    Encoding names are "<kind>" or "<kind>_<region>", where region in
    {groove, pseudo} selects which part of the HLA chain was embedded (see
    embed_local). Plain "concat"/"joint" mean the whole mature chain.
    """
    E = _embed_module()
    kind, _, region = encoding.partition("_")
    sfx = "_" + region if region in ("groove", "pseudo") else ""
    if kind == "joint":
        arr, ix = E.get("joint" + sfx)
        rows = [ix[(p, h)] for p, h in zip(df.Pep, df.HLA)]
        return arr[rows]
    if kind == "concat":
        pa, pi = E.get("pep")
        ha, hi = E.get("hla" + sfx)
        return np.hstack([pa[[pi[p] for p in df.Pep]],
                          ha[[hi[h] for h in df.HLA]]]).astype(np.float32)
    if kind == "pep" or encoding == "pep_only":      # ablation: can it work with no allele at all?
        pa, pi = E.get("pep")
        return pa[[pi[p] for p in df.Pep]]
    raise ValueError(encoding)


def _standardise(xtr, xte):
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    return (xtr - mu) / sd, (xte - mu) / sd


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------

def tobit_nll(mu, logsig, y, censored):
    """Left-censored Gaussian NLL. censored rows contribute log Phi((c-mu)/sig)."""
    sig = logsig.exp()
    z = (y - mu) / sig
    obs = 0.5 * z ** 2 + logsig
    cen = -torch.special.log_ndtr((FLOOR - mu) / sig)
    return torch.where(censored, cen, obs).mean()


# --------------------------------------------------------------------------
# heads. All must fit in seconds so the split ladder is cheap to rerun.
# --------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, d, hidden=256, drop=0.1, heads=1):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(drop))
        self.out = nn.Linear(hidden, heads)

    def forward(self, x):
        return self.out(self.body(x))


def _torch_fit(xtr, ytr, ctr, xte, seed, head, loss, epochs=60, lr=3e-3,
               wd=1e-4, batch=1024, device="cpu"):
    """One torch fit. Returns test prediction (point estimate in log10 hours)."""
    torch.manual_seed(seed)
    d = xtr.shape[1]
    Xtr = torch.tensor(xtr, device=device)
    Xte = torch.tensor(xte, device=device)
    Y = torch.tensor(ytr, dtype=torch.float32, device=device)
    C = torch.tensor(ctr, device=device)

    nout = 2 if loss == "twopart" else 1
    net = (nn.Linear(d, nout) if head == "ridge" else MLP(d, heads=nout)).to(device)
    logsig = nn.Parameter(torch.zeros((), device=device) - 0.5)
    opt = torch.optim.AdamW(list(net.parameters()) + [logsig], lr=lr, weight_decay=wd)

    n = len(Y)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            o = net(Xtr[b])
            y, c = Y[b], C[b]
            if loss == "naive":
                l = ((o[:, 0] - y) ** 2).mean()
            elif loss == "tobit":
                l = tobit_nll(o[:, 0], logsig, y, c)
            elif loss == "twopart":
                # head 0: mean of y among above-floor rows. head 1: logit above-floor.
                above = ~c
                reg = (((o[above, 0] - y[above]) ** 2).mean()
                       if above.any() else torch.zeros((), device=device))
                cls = nn.functional.binary_cross_entropy_with_logits(
                    o[:, 1], above.float())
                l = reg + cls
            else:
                raise ValueError(loss)
            opt.zero_grad(); l.backward(); opt.step()

    net.eval()
    with torch.no_grad():
        o = net(Xte)
        if loss == "twopart":
            # COMBINATION RULE (see module docstring): expected value under the
            # two-part model, with the below-floor branch pinned at the floor.
            # It is monotone in both p and mu, which is what the ranking needs.
            p = torch.sigmoid(o[:, 1])
            pred = p * o[:, 0] + (1 - p) * FLOOR
            # mixture variance: between-branch spread + within-branch noise
            sreg = float(((net(Xtr)[:, 0][~C] - Y[~C]) ** 2).mean().sqrt()) if (~C).any() else 0.0
            var = (p * (o[:, 0] - pred) ** 2 + (1 - p) * (FLOOR - pred) ** 2
                   + p * sreg ** 2)
            alea = var.clamp_min(1e-8).sqrt()
        else:
            pred = o[:, 0]
            if loss == "tobit":
                alea = logsig.exp().expand_as(pred)          # learned, per-model
            else:
                alea = ((net(Xtr)[:, 0] - Y) ** 2).mean().sqrt().expand_as(pred)
    return pred.cpu().numpy(), alea.cpu().numpy()


def _ridge_closed(xtr, ytr, xte, lam=10.0):
    """Closed-form ridge. Naive loss only -- there is no closed form under
    censoring. This is the sub-second baseline everything else must beat."""
    X = np.hstack([xtr, np.ones((len(xtr), 1), np.float32)])
    A = X.T @ X + lam * np.eye(X.shape[1], dtype=np.float32)
    w = np.linalg.solve(A, X.T @ ytr)
    resid = float(np.sqrt(((X @ w - ytr) ** 2).mean()))
    pred = np.hstack([xte, np.ones((len(xte), 1), np.float32)]) @ w
    return pred, np.full(len(pred), resid, np.float32)


# --------------------------------------------------------------------------
# ensemble
# --------------------------------------------------------------------------

def fit_predict(df, train_index, test_index, encoding="joint", head="mlp",
                loss="tobit", seeds=SEEDS, device="cpu", **kw):
    """5-seed ensemble. Returns (mean_pred, sigma, fit_seconds).

    TWO SOURCES OF UNCERTAINTY, REPORTED SEPARATELY.
      sigma_seed  std of the 5 seed predictions. EPISTEMIC: how much the answer
                  depends on where the optimiser started. This is the quantity
                  the project proposed as its contribution. Measured on its own
                  it turns out to be badly over-confident (see report), because
                  5 seeds of the same architecture on the same data agree far
                  more than they should.
      sigma_alea  the model's own claimed noise level. Under 'tobit' this is the
                  LEARNED scalar sigma of the censored likelihood, which is free
                  and principled -- the likelihood had to estimate it anyway.
                  Under 'naive'/'ridge' it is the train residual RMSE, and under
                  'twopart' the two-branch mixture std.
      sigma_total sqrt(seed^2 + alea^2), the standard decomposition. This is the
                  number to quote as a predictive interval.
    Returns (mean_pred, sigma_seed, sigma_total, fit_seconds).
    """
    tr, te = df.loc[train_index], df.loc[test_index]
    X = features(df, encoding)
    pos = {ix: i for i, ix in enumerate(df.index)}
    xtr = X[[pos[i] for i in train_index]]
    xte = X[[pos[i] for i in test_index]]
    xtr, xte = _standardise(xtr, xte)
    ytr = tr.y.values.astype(np.float32)
    ctr = tr.censored.values

    t0 = time.time()
    preds, aleas = [], []
    for s in seeds:
        if head == "ridge" and loss == "naive":
            # still seed-varied: bootstrap the rows, else all 5 seeds are identical
            rng = np.random.default_rng(s)
            b = rng.integers(0, len(xtr), len(xtr))
            p, a = _ridge_closed(xtr[b], ytr[b], xte)
        else:
            p, a = _torch_fit(xtr, ytr, ctr, xte, s, head, loss,
                              device=device, **kw)
        preds.append(p); aleas.append(a)
    P, A = np.vstack(preds), np.vstack(aleas)
    seed_sd = P.std(0, ddof=1)
    total = np.sqrt(seed_sd ** 2 + (A ** 2).mean(0))
    return P.mean(0), seed_sd, total, time.time() - t0


# --------------------------------------------------------------------------
# logging. By Sunday we want a table, not scrollback.
# --------------------------------------------------------------------------

FIELDS = ["ts", "split", "encoding", "head", "loss", "n_seeds", "n_train", "n_test",
          "fit_s", "censored_policy", "rho_median", "rho_iqr_lo", "rho_iqr_hi",
          "rho_worst", "n_alleles", "top10_median", "euc_median", "coverage68_median",
          "rmse_uncensored", "sigma_mean", "euc_seed_median",
          "coverage68_seed_median", "sigma_seed_mean", "held_out", "note"]


def log_run(row, path=RUNS_CSV):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def evaluate(df, test_index, pred, sigma, sigma_seed=None, censored_policy="tied"):
    """Everything metrics.py offers, reduced to the numbers that go on a slide."""
    import metrics
    te = df.loc[test_index]
    rho = metrics.spearman_per_allele(te, pred, censored=censored_policy)
    top = metrics.top10_precision_per_allele(te, pred, censored=censored_policy)
    cal = metrics.calibration_per_allele(te, pred, sigma, censored=censored_policy)
    cs = (metrics.calibration_per_allele(te, pred, sigma_seed, censored=censored_policy)
          if sigma_seed is not None else None)
    v = np.asarray(rho, float); v = v[~np.isnan(v)]
    obs = ~te.censored.values
    return {
        "censored_policy": censored_policy,
        "rho_median": round(float(np.median(v)), 4) if len(v) else "",
        "rho_iqr_lo": round(float(np.percentile(v, 25)), 4) if len(v) else "",
        "rho_iqr_hi": round(float(np.percentile(v, 75)), 4) if len(v) else "",
        "rho_worst": round(float(v.min()), 4) if len(v) else "",
        "n_alleles": len(v),
        "top10_median": round(float(np.nanmedian(np.asarray(top, float))), 4) if len(top) else "",
        "euc_median": round(float(np.nanmedian(cal.euc.values)), 4) if len(cal) else "",
        "coverage68_median": round(float(np.nanmedian(cal.coverage68.values)), 4) if len(cal) else "",
        "rmse_uncensored": round(float(np.sqrt(((pred[obs] - te.y.values[obs]) ** 2).mean())), 4),
        "sigma_mean": round(float(sigma.mean()), 4),
        "euc_seed_median": ("" if cs is None else
                            round(float(np.nanmedian(cs.euc.values)), 4)),
        "coverage68_seed_median": ("" if cs is None else
                            round(float(np.nanmedian(cs.coverage68.values)), 4)),
        "sigma_seed_mean": ("" if sigma_seed is None else round(float(sigma_seed.mean()), 4)),
        "_rho": rho, "_cal": cal,
    }


PER_ALLELE_CSV = "per_allele.csv"


def log_per_allele(row, ev, path=PER_ALLELE_CSV):
    """One line per (config, held-out allele). This is the evidence table:
    every number in the figure traces back to a row here."""
    import pandas as pd
    rho, cal = ev["_rho"], ev["_cal"]
    recs = []
    for allele in rho.index:
        recs.append({"split": row["split"], "held_out": row["held_out"],
                     "encoding": row["encoding"], "head": row["head"],
                     "loss": row["loss"], "allele": allele,
                     "rho": float(rho[allele]),
                     "euc": (float(cal.euc[allele]) if allele in cal.index else ""),
                     "coverage68": (float(cal.coverage68[allele])
                                    if allele in cal.index else ""),
                     "n": (int(cal.n[allele]) if allele in cal.index else "")})
    pd.DataFrame(recs).to_csv(path, mode="a", header=not os.path.exists(path),
                              index=False)


def run(df, split_name, train_index, test_index, encoding, head, loss,
        seeds=SEEDS, held_out="", note="", device="cpu", **kw):
    pred, sd_seed, sigma, secs = fit_predict(df, train_index, test_index, encoding,
                                             head, loss, seeds, device=device, **kw)
    ev = evaluate(df, test_index, pred, sigma, sigma_seed=sd_seed)
    row = dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), split=split_name,
               encoding=encoding, head=head, loss=loss, n_seeds=len(seeds),
               n_train=len(train_index), n_test=len(test_index),
               fit_s=round(secs, 2), held_out=held_out, note=note,
               **{k: v for k, v in ev.items() if not k.startswith("_")})
    log_run(row)
    log_per_allele(row, ev)
    print(f"{split_name:22s} {encoding:8s} {head:5s} {loss:8s} "
          f"rho={row['rho_median']} top10={row['top10_median']} "
          f"euc={row['euc_median']} cov={row['coverage68_median']} "
          f"cov_seed={row['coverage68_seed_median']} "
          f"rmse={row['rmse_uncensored']} {secs:.1f}s", flush=True)
    return row, pred, sigma, ev


def cheap_splits(df):
    """The two splits that flatter us. Included for contrast, not for the claim."""
    import splits
    return [("random", "", *splits.split_random(df)),
            ("by_peptide", "", *splits.split_by_peptide(df))]


def allele_folds(df):
    """Leave-one-groove-cluster-out folds, from splits.choose_held_out().

    choose_held_out returns [(fold_name, [alleles])], i.e. a whole ladder rather
    than one held-out set, so this is a cross-validation over GROOVE CLUSTERS:
    every allele sharing a binding groove with a test allele is removed from
    training too. That is the only version of the split where "unseen allele"
    cannot quietly mean "interpolated between two near-identical alleles".
    """
    import splits
    try:
        folds = splits.choose_held_out(df)
    except (NotImplementedError, AttributeError):
        print("note: splits.choose_held_out not implemented yet -- "
              "allele-generalisation rows are MISSING, not zero.", flush=True)
        return []
    out = []
    for name, members in folds:
        tr, te = splits.split_by_allele(df, members)
        out.append((f"groove:{name}", ";".join(sorted(members)), tr, te))
    return out


def table(path=RUNS_CSV, cols=None):
    """Print runs.csv as a readable table. This is the Sunday deliverable:
    every experiment in one place, so nothing has to be reconstructed from
    scrollback at 2am."""
    import pandas as pd
    d = pd.read_csv(path)
    cols = cols or ["split", "encoding", "head", "loss", "rho_median",
                    "rho_iqr_lo", "rho_iqr_hi", "rho_worst", "n_alleles",
                    "top10_median", "euc_median", "coverage68_median",
                    "coverage68_seed_median", "rmse_uncensored", "fit_s", "note"]
    cols = [c for c in cols if c in d.columns]
    print(d[cols].to_string(index=False))
    return d


# Full grid on the cheap splits; on the 21 groove folds only the comparisons the
# claim actually rests on, because 11 configs x 21 folds is an hour we do not need
# to spend. ridge+tobit is kept as the linear reference since it costs ~3s.
FULL_GRID = [(e, h, l)
             for e in ("concat", "joint")
             for h, l in (("ridge", "naive"), ("ridge", "tobit"),
                          ("mlp", "naive"), ("mlp", "tobit"), ("mlp", "twopart"))]
FOLD_GRID = [(e, h, l)
             for e in ("concat", "joint")
             for h, l in (("ridge", "tobit"), ("mlp", "naive"), ("mlp", "tobit"))]


def main():
    df = data.load()
    for sname, held, tr, te in cheap_splits(df):
        for e, h, l in FULL_GRID:
            run(df, sname, tr, te, e, h, l, held_out=held)
        run(df, sname, tr, te, "pep_only", "mlp", "tobit", held_out=held,
            note="ablation: no allele input")
    folds = allele_folds(df)
    print(f"\n{len(folds)} groove-cluster folds x {len(FOLD_GRID)} configs", flush=True)
    for sname, held, tr, te in folds:
        for e, h, l in FOLD_GRID:
            run(df, sname, tr, te, e, h, l, held_out=held)
    print(f"\nwrote {RUNS_CSV} and {PER_ALLELE_CSV}")
    aggregate()


# The region comparison: does embedding only the binding groove (or only the 34
# pseudo-sequence contact residues) fix the cross-groove collapse that the
# whole-chain encoding shows? Same heads, same losses, same folds -- only the
# HLA region changes, so the comparison is clean.
REGION_ENCODINGS = ["concat_groove", "joint_groove", "concat_pseudo", "joint_pseudo"]


def main_regions():
    df = data.load()
    for sname, held, tr, te in cheap_splits(df):
        for e in REGION_ENCODINGS:
            for h, l in (("ridge", "tobit"), ("mlp", "naive"), ("mlp", "tobit")):
                run(df, sname, tr, te, e, h, l, held_out=held)
    folds = allele_folds(df)
    print(f"\n{len(folds)} folds x {len(REGION_ENCODINGS)} region encodings", flush=True)
    for sname, held, tr, te in folds:
        for e in REGION_ENCODINGS:
            for h, l in (("mlp", "naive"), ("mlp", "tobit")):
                run(df, sname, tr, te, e, h, l, held_out=held)
    aggregate()


def aggregate(path=PER_ALLELE_CSV):
    """Pool the per-allele rhos ACROSS folds, which is the headline number.

    Pooling the raw per-allele values, not averaging each fold's median: folds
    hold out between 1 and 9 alleles, so a mean of medians would weight a
    single-allele fold the same as a nine-allele one.
    """
    import pandas as pd
    d = pd.read_csv(path)
    g = d[d.split.str.startswith("groove:")]
    if not len(g):
        return
    print("\nPOOLED OVER ALL HELD-OUT GROOVE CLUSTERS (per-allele rho):")
    t = (g.groupby(["encoding", "head", "loss"])
          .rho.agg(["median", lambda v: v.quantile(.25),
                    lambda v: v.quantile(.75), "min", "count"]))
    t.columns = ["median", "q25", "q75", "worst", "n_alleles"]
    print(t.round(4).to_string())
    return t


if __name__ == "__main__":
    import sys
    if "regions" in sys.argv:
        main_regions()
    elif "aggregate" in sys.argv:
        table(); aggregate()
    else:
        main()
