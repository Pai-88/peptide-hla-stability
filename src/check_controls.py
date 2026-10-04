"""Adversarial control check: do the nulls kill the arms?

Read-only. Recomputes every per-fold number from the stored per-row predictions
rather than trusting results_*.csv, then runs paired tests against the controls.

Usage: python check_controls.py [stage]
   stages: ladder seen decomp null all
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

import data
import splits
import metrics

ARMS = {
    "A_supervised_nn": "A  supervised NN (BLOSUM+onehot, no FM)",
    "B_esm_pseudo": "B  ESM pep|pseudo + ridge",
    "B_esm_pseudo_mlp": "B' ESM pep|pseudo + MLP",
    "C_esm_joint": "C  ESM joint (allele-conditioned)",
    "D_peptide_only": "D  ESM peptide only  [NULL: no allele]",
    "E_allele_mean__tiebreak": "E  allele-mean + jitter [NULL: no peptide]",
}

DF = data.load()
FOLDS = splits.groove_folds(DF)
FOLD_NAMES = [n for n, _ in FOLDS]


def ensemble(arm):
    """(fold_name, row_id) -> mean prediction over the 5 seeds."""
    p = pd.read_parquet(f"predictions_{arm}.parquet")
    return p.groupby(["fold_name", "row_id"], as_index=False).y_pred.mean()


def fold_scores(pred_long, censored="tied"):
    """Per-fold median per-allele Spearman + top10, and the per-allele series."""
    rho, top, per_allele = {}, {}, {}
    for fold, g in pred_long.groupby("fold_name"):
        sub = DF.loc[g.row_id.values]
        s = metrics.spearman_per_allele(sub, g.y_pred.values, censored=censored)
        t = metrics.top10_precision_per_allele(sub, g.y_pred.values, censored=censored)
        if len(s):
            rho[fold] = float(np.median(s))
            per_allele[fold] = s
        if len(t):
            top[fold] = float(np.median(t))
    return pd.Series(rho), pd.Series(top), per_allele


def pooled_rho(pred_long, censored="tied"):
    out = {}
    for fold, g in pred_long.groupby("fold_name"):
        sub = DF.loc[g.row_id.values]
        pr = g.y_pred.values
        if censored == "drop":
            m = ~sub.censored.values
            sub, pr = sub[m], pr[m]
        if np.ptp(pr) == 0:
            continue
        out[fold] = stats.spearmanr(pr, sub.Thalf.values).statistic
    return pd.Series(out)


def lookup_table_baseline():
    """Peptide-mean lookup: predict mean train y of that exact peptide.

    Deterministic, uses no allele and no model. This is the bar arm D claims.
    """
    rows = []
    for name, alleles in FOLDS:
        tr_idx, te_idx = splits.split_by_allele(DF, alleles)
        tr, te = DF.loc[tr_idx], DF.loc[te_idx]
        m = tr.groupby("Pep").y.mean()
        pred = te.Pep.map(m).fillna(tr.y.mean()).values
        rows.append(pd.DataFrame({"fold_name": name, "row_id": te.index.values,
                                  "y_pred": pred}))
    return pd.concat(rows, ignore_index=True)


def seen_mask_per_fold():
    """row_id -> was this peptide present anywhere in that fold's TRAIN side."""
    out = {}
    for name, alleles in FOLDS:
        tr_idx, te_idx = splits.split_by_allele(DF, alleles)
        train_peps = set(DF.loc[tr_idx].Pep)
        te = DF.loc[te_idx]
        out[name] = pd.Series(te.Pep.isin(train_peps).values, index=te.index)
    return out


# ---------------------------------------------------------------------------
def stage_ladder():
    print("=" * 78)
    print("1. THE LADDER: per-fold median per-allele Spearman, 5-seed ensemble")
    print("=" * 78)
    store = {}
    for arm, label in ARMS.items():
        e = ensemble(arm)
        for pol in ("tied", "drop"):
            r, t, pa = fold_scores(e, pol)
            store[(arm, pol)] = (r, t, pa)
        store[(arm, "ens")] = e
    lut = lookup_table_baseline()
    for pol in ("tied", "drop"):
        r, t, pa = fold_scores(lut, pol)
        store[("LUT_peptide_mean", pol)] = (r, t, pa)
    store[("LUT_peptide_mean", "ens")] = lut
    ARMS["LUT_peptide_mean"] = "L  peptide-mean lookup [NULL: no allele, no model]"

    for pol in ("tied", "drop"):
        print(f"\n-- censored='{pol}' --")
        print(f"{'arm':46s} {'rho_med':>8s} {'IQR':>16s} {'worst':>8s} "
              f"{'<=0':>4s} {'top10':>6s}")
        for arm, label in ARMS.items():
            r, t, _ = store[(arm, pol)]
            print(f"{label:46s} {np.median(r):8.4f} "
                  f"[{np.percentile(r,25):6.3f},{np.percentile(r,75):6.3f}] "
                  f"{r.min():8.4f} {int((r<=0).sum()):4d} {np.median(t):6.3f}")

    print("\n" + "=" * 78)
    print("2. PAIRED MARGINS vs the controls (per fold, 21 folds, tied)")
    print("=" * 78)
    for pol in ("tied", "drop"):
        print(f"\n-- censored='{pol}' --")
        for ctrl in ("D_peptide_only", "E_allele_mean__tiebreak", "LUT_peptide_mean"):
            cr = store[(ctrl, pol)][0]
            print(f"\n  vs {ARMS[ctrl]}  (median {np.median(cr):.4f})")
            for arm in ARMS:
                if arm == ctrl:
                    continue
                ar = store[(arm, pol)][0]
                common = ar.index.intersection(cr.index)
                d = (ar[common] - cr[common]).astype(float)
                w = stats.wilcoxon(d) if d.abs().sum() > 0 else None
                p = f"{w.pvalue:.4f}" if w else "n/a"
                print(f"    {ARMS[arm]:46s} dmed {d.median():+.4f}  "
                      f"wins {int((d>0).sum()):2d}/{len(d)}  p={p}  "
                      f"worst fold loss {d.min():+.4f} ({d.idxmin()})")
    return store


def stage_null_wins(store):
    print("\n" + "=" * 78)
    print("3. FOLDS WHERE A CONTROL BEATS THE ARM (tied). Named.")
    print("=" * 78)
    tab = pd.DataFrame({arm: store[(arm, "tied")][0] for arm in ARMS})
    tab = tab.loc[FOLD_NAMES]
    nrows = {n: int(DF.HLA.isin(a).sum()) for n, a in FOLDS}
    nall = {n: len(a) for n, a in FOLDS}
    tab.insert(0, "n_test", [nrows[f] for f in tab.index])
    tab.insert(1, "n_all", [nall[f] for f in tab.index])
    pd.set_option("display.width", 250)
    print(tab.round(4).to_string())
    print("\n  best arm per fold:")
    arms_only = tab[list(ARMS)]
    for f in tab.index:
        row = arms_only.loc[f]
        print(f"    {f:16s} n={tab.n_test[f]:5d}  winner {row.idxmax():26s} "
              f"{row.max():+.4f}   A={row['A_supervised_nn']:+.4f} "
              f"D={row['D_peptide_only']:+.4f} L={row['LUT_peptide_mean']:+.4f}")
    return tab


def stage_seen(store):
    print("\n" + "=" * 78)
    print("4. SEEN vs UNSEEN PEPTIDES. The fold unit is a groove, not a peptide.")
    print("=" * 78)
    seen = seen_mask_per_fold()
    frac = {f: float(s.mean()) for f, s in seen.items()}
    print(f"  fraction of test rows whose peptide IS in that fold's train side:")
    print(f"    overall {np.mean([frac[f] for f in FOLD_NAMES]):.3f}  "
          f"min {min(frac.values()):.3f}  max {max(frac.values()):.3f}  "
          f"folds at 100%: {sum(v>0.999 for v in frac.values())}/21")
    print(f"\n{'arm':46s} {'all':>8s} {'SEEN':>8s} {'UNSEEN':>8s} {'n_folds_unseen':>15s}")
    for arm in ARMS:
        e = store[(arm, "ens")]
        allr, seenr, unseenr = [], [], []
        for fold, g in e.groupby("fold_name"):
            sm = seen[fold].loc[g.row_id.values].values
            for lab, mask, acc in (("a", np.ones(len(g), bool), allr),
                                   ("s", sm, seenr), ("u", ~sm, unseenr)):
                gg = g[mask]
                if len(gg) < 20:
                    continue
                sub = DF.loc[gg.row_id.values]
                s = metrics.spearman_per_allele(sub, gg.y_pred.values)
                if len(s):
                    acc.append(float(np.median(s)))
        print(f"{ARMS[arm]:46s} {np.median(allr):8.4f} {np.median(seenr):8.4f} "
              f"{np.median(unseenr):8.4f} {len(unseenr):15d}")
    print("\n  (UNSEEN column = genuinely novel peptide for that allele's groove)")


def stage_decomp(store):
    print("\n" + "=" * 78)
    print("5. BETWEEN-ALLELE OFFSET vs WITHIN-ALLELE RANKING, per fold")
    print("=" * 78)
    print("  frac_between = variance of the per-allele MEAN prediction / total")
    print("  pooled rho   = what a pooled metric would have reported")
    print(f"\n{'fold':16s} {'n_all':>5s} {'y_betw':>7s} ", end="")
    for arm in ARMS:
        print(f"{arm.split('_')[0]:>7s}", end="")
    print("   | pooled rho (tied)")
    for name, alleles in FOLDS:
        sub = DF[DF.HLA.isin(alleles)]
        if sub.HLA.nunique() < 2:
            ybet = np.nan
        else:
            ybet = sub.groupby("HLA").y.mean().var(ddof=0) / sub.y.var(ddof=0)
        print(f"{name:16s} {sub.HLA.nunique():5d} {ybet:7.3f} ", end="")
        pooled = []
        for arm in ARMS:
            e = store[(arm, "ens")]
            g = e[e.fold_name == name]
            s2 = DF.loc[g.row_id.values]
            if s2.HLA.nunique() < 2:
                f = np.nan
            else:
                mu = pd.Series(g.y_pred.values).groupby(s2.HLA.values).mean()
                cnt = s2.HLA.value_counts().reindex(mu.index)
                f = np.average((mu - np.average(mu, weights=cnt))**2,
                               weights=cnt) / np.var(g.y_pred.values)
            print(f"{f:7.3f}", end="")
            pooled.append(stats.spearmanr(g.y_pred.values, s2.Thalf.values).statistic
                          if np.ptp(g.y_pred.values) > 0 else np.nan)
        print("   | " + " ".join(f"{v:+.3f}" for v in pooled))
    print("\n  columns after y_betw are frac of each ARM's prediction variance that")
    print("  is between-allele. High = the model is mostly predicting an allele offset.")

    print("\n  POOLED vs PER-ALLELE median, over folds (tied):")
    for arm in ARMS:
        e = store[(arm, "ens")]
        pr = pooled_rho(e)
        pa = store[(arm, "tied")][0]
        print(f"    {ARMS[arm]:46s} pooled med {np.median(pr):+.4f}   "
              f"per-allele med {np.median(pa):+.4f}   "
              f"inflation {np.median(pr)-np.median(pa):+.4f}")


def stage_null_band(store):
    print("\n" + "=" * 78)
    print("6. IS THE MARGIN BIGGER THAN THE NULL BAND? (arm E tiebreak = pure noise)")
    print("=" * 78)
    p = pd.read_parquet("predictions_E_allele_mean__tiebreak.parquet")
    per_seed = []
    for (fold, seed), g in p.groupby(["fold_name", "seed"]):
        sub = DF.loc[g.row_id.values]
        s = metrics.spearman_per_allele(sub, g.y_pred.values)
        if len(s):
            per_seed.append(float(np.median(s)))
    v = np.array(per_seed)
    print(f"  105 (fold,seed) null draws of the per-fold median: "
          f"mean {v.mean():+.4f} sd {v.std(ddof=1):.4f} "
          f"p2.5 {np.percentile(v,2.5):+.4f} p97.5 {np.percentile(v,97.5):+.4f}")
    print(f"  => a single fold's median rho is inside the null unless |rho| > "
          f"~{max(abs(np.percentile(v,2.5)), np.percentile(v,97.5)):.3f}")
    for arm in ARMS:
        r = store[(arm, "tied")][0]
        inside = int((r.abs() <= np.percentile(v, 97.5)).sum())
        print(f"    {ARMS[arm]:46s} {inside:2d}/21 folds inside the single-fold null band")


def stage_weighting(store):
    print("\n" + "=" * 78)
    print("7. IS THE HEADLINE INFLATED BY SMALL SINGLE-ALLELE FOLDS?")
    print("=" * 78)
    nrows = {n: int(DF.HLA.isin(a).sum()) for n, a in FOLDS}
    nall = {n: len(a) for n, a in FOLDS}
    singles = [n for n in FOLD_NAMES if nall[n] == 1]
    print(f"  {len(singles)}/21 folds hold exactly ONE allele "
          f"({sum(nrows[n] for n in singles)} of {sum(nrows.values())} test rows, "
          f"{100*sum(nrows[n] for n in singles)/sum(nrows.values()):.0f}%).")
    print("  The reported statistic is median-over-folds of median-over-alleles,")
    print("  so each singleton fold = one allele's rho, equal weight to a 9-allele fold.")
    print(f"\n{'arm':46s} {'fold-med':>9s} {'1-allele':>9s} {'multi':>9s} "
          f"{'allele-med':>11s} {'row-wtd':>9s}")
    for arm in ARMS:
        _, _, pa = store[(arm, "tied")]
        fr = pd.Series({f: float(np.median(s)) for f, s in pa.items()})
        allele_level = pd.concat([s for s in pa.values()])
        wts = np.array([DF[DF.HLA == a].shape[0] for a in allele_level.index], float)
        order = np.argsort(allele_level.values)
        v, w = allele_level.values[order], wts[order]
        rowmed = v[np.searchsorted(np.cumsum(w), 0.5 * w.sum())]
        print(f"{ARMS[arm]:46s} {np.median(fr):9.4f} "
              f"{np.median(fr[[f for f in fr.index if nall[f]==1]]):9.4f} "
              f"{np.median(fr[[f for f in fr.index if nall[f]>1]]):9.4f} "
              f"{np.median(allele_level):11.4f} {rowmed:9.4f}")
    print("\n  allele-med = median over all scored (fold, allele) pairs directly.")
    print("  row-wtd    = the same, weighted by each allele's row count.")


def stage_residual(store):
    print("\n" + "=" * 78)
    print("8. DO THE FOUNDATION-MODEL ARMS CARRY ANYTHING ARM A MISSES?")
    print("=" * 78)
    print("  Per allele: Spearman(arm X prediction, arm A RESIDUAL y_true - y_pred_A),")
    print("  censored rows dropped. >0 means X knows something A got wrong.")
    a = store[("A_supervised_nn", "ens")].set_index(["fold_name", "row_id"]).y_pred
    for arm in ARMS:
        if arm == "A_supervised_nn":
            continue
        x = store[(arm, "ens")].set_index(["fold_name", "row_id"]).y_pred
        per_fold = []
        for fold in FOLD_NAMES:
            ids = a.loc[fold].index
            sub = DF.loc[ids]
            m = ~sub.censored.values
            resid = (sub.y.values - a.loc[fold].values)[m]
            xp = x.loc[fold].reindex(ids).values[m]
            hl = sub.HLA.values[m]
            rr = [stats.spearmanr(xp[hl == h], resid[hl == h]).statistic
                  for h in np.unique(hl) if (hl == h).sum() >= 20
                  and np.ptp(xp[hl == h]) > 0]
            if rr:
                per_fold.append(float(np.median(rr)))
        pf = np.array(per_fold)
        w = stats.wilcoxon(pf)
        print(f"    {ARMS[arm]:46s} median {np.median(pf):+.4f}  "
              f">0 in {int((pf>0).sum()):2d}/{len(pf)} folds  p={w.pvalue:.4f}")


def stage_pool_spread(store):
    print("\n" + "=" * 78)
    print("9. WOULD POOLING HAVE INFLATED THE NUMBER? (checking a project claim)")
    print("=" * 78)
    for arm in ARMS:
        e = store[(arm, "ens")]
        pr = pooled_rho(e)
        pa = store[(arm, "tied")][0]
        multi = [n for n, a in FOLDS if len(a) > 1]
        d = (pr - pa).reindex(multi).dropna()
        print(f"    {ARMS[arm]:46s} pooled-minus-perallele (14 multi-allele folds): "
              f"median {d.median():+.4f} max {d.max():+.4f} ({d.idxmax()})  "
              f"min {d.min():+.4f} ({d.idxmin()})")


def stage_stack(store):
    """Out-of-fold stack: does arm A + arm X beat arm A alone?"""
    print("\n" + "=" * 78)
    print("10. OUT-OF-FOLD STACK: does adding a foundation-model arm to arm A help?")
    print("=" * 78)
    print("  Within each allele, z-score both arms' predictions (kills the allele")
    print("  offset, so only ranking information can contribute). Fit w on the")
    print("  OTHER 20 folds' stored predictions, apply to this fold, rescore.")

    def znorm(pred, hla):
        out = np.zeros(len(pred))
        for h in np.unique(hla):
            m = hla == h
            s = pred[m].std()
            out[m] = (pred[m] - pred[m].mean()) / (s if s > 0 else 1.0)
        return out

    a = store[("A_supervised_nn", "ens")].set_index(["fold_name", "row_id"]).y_pred
    base = store[("A_supervised_nn", "tied")][0]
    for arm in ARMS:
        if arm == "A_supervised_nn":
            continue
        x = store[(arm, "ens")].set_index(["fold_name", "row_id"]).y_pred
        cols = {}
        for fold in FOLD_NAMES:
            ids = a.loc[fold].index
            sub = DF.loc[ids]
            cols[fold] = (np.column_stack([znorm(a.loc[fold].values, sub.HLA.values),
                                           znorm(x.loc[fold].reindex(ids).values,
                                                 sub.HLA.values)]),
                          stats.zscore(stats.rankdata(sub.y.values)), ids)
        new = {}
        for fold in FOLD_NAMES:
            Xtr = np.vstack([cols[f][0] for f in FOLD_NAMES if f != fold])
            ytr = np.concatenate([cols[f][1] for f in FOLD_NAMES if f != fold])
            w = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
            Xte, _, ids = cols[fold]
            s = metrics.spearman_per_allele(DF.loc[ids], Xte @ w)
            if len(s):
                new[fold] = float(np.median(s))
        new = pd.Series(new)
        d = (new - base).reindex(FOLD_NAMES).dropna()
        print(f"    A + {ARMS[arm]:42s} {np.median(new):.4f} vs A alone "
              f"{np.median(base):.4f}   delta med {d.median():+.4f}  "
              f"helps {int((d>0).sum()):2d}/21  p={stats.wilcoxon(d).pvalue:.4f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    store = stage_ladder()
    if which in ("all", "null"):
        stage_null_wins(store)
        stage_null_band(store)
    if which in ("all", "seen"):
        stage_seen(store)
    if which in ("all", "decomp"):
        stage_decomp(store)
        stage_pool_spread(store)
        stage_stack(store)
    if which in ("all", "weight"):
        stage_weighting(store)
        stage_residual(store)


# ---------------------------------------------------------------------------
# Added in the adversarial controls pass. Everything below is read-only.
# ---------------------------------------------------------------------------
QUOTED = {  # the numbers circulated in the summary, for audit against the files
    "A_supervised_nn": 0.277, "G_650m_B_mlp": 0.109, "B_esm_pseudo_mlp": 0.096,
    "G_650m_B_ridge": 0.090, "D_peptide_only": 0.074, "C_esm_joint": 0.071,
    "B_esm_pseudo": 0.060, "E_allele_mean__tiebreak": 0.005,
}


def stage_aggregation():
    """Which aggregation do the quoted numbers correspond to, and does the
    arm-vs-control ORDERING survive changing it?

    results_*.csv has one row per (fold, seed). A median over those 105 rows
    gives seed noise the same weight as fold structure; a median over the 21
    per-fold medians does not. The quoted ladder uses the former.
    """
    print("\n" + "=" * 78)
    print("11. AGGREGATION AUDIT: row-median vs fold-median vs seed-ensemble")
    print("=" * 78)
    print(f"{'arm':26s} {'quoted':>7s} {'rowmed':>8s} {'foldmed':>8s} {'ens':>8s} "
          f"{'n_rows':>6s} {'n_folds':>7s}  matches")
    for arm, q in QUOTED.items():
        try:
            d = pd.read_csv(f"results_{arm}.csv")
        except FileNotFoundError:
            print(f"{arm:26s} {q:7.3f}   FILE MISSING")
            continue
        d = d[d.status == "ok"] if "status" in d else d
        if not len(d):
            print(f"{arm:26s} {q:7.3f}   EMPTY (header only) -- nothing to quote")
            continue
        rowmed, nf = d.spearman.median(), d.fold_name.nunique()
        foldmed = d.groupby("fold_name").spearman.median().median()
        try:
            ens = np.median(fold_scores(ensemble(arm))[0])
        except Exception:
            ens = np.nan
        near = [n for n, v in (("row", rowmed), ("fold", foldmed), ("ens", ens))
                if abs(v - q) < 0.002]
        print(f"{arm:26s} {q:7.3f} {rowmed:8.4f} {foldmed:8.4f} {ens:8.4f} "
              f"{len(d):6d} {nf:7d}  {','.join(near) or 'NONE'}")

    print("\n  ORDERING vs the no-allele control D, under each aggregation:")
    for lab in ("row", "fold", "ens"):
        bits = []
        for arm in ("B_esm_pseudo", "B_esm_pseudo_mlp", "C_esm_joint"):
            d = pd.read_csv(f"results_{arm}.csv")
            dd = pd.read_csv("results_D_peptide_only.csv")
            if lab == "row":
                a, c = d.spearman.median(), dd.spearman.median()
            elif lab == "fold":
                a = d.groupby("fold_name").spearman.median().median()
                c = dd.groupby("fold_name").spearman.median().median()
            else:
                a = np.median(fold_scores(ensemble(arm))[0])
                c = np.median(fold_scores(ensemble("D_peptide_only"))[0])
            bits.append(f"{arm.split('_')[0]+arm[-4:]:12s}{a - c:+.4f}")
        print(f"    {lab:5s} " + "  ".join(bits))


def stage_margin_ci(store, n_boot=10000, seed=0):
    """Bootstrap CI on the per-fold margin against each control.

    Resample the 21 FOLDS with replacement (the fold is the independent unit),
    recompute the median paired difference. If the CI straddles 0 the margin is
    not distinguishable from noise at 21 folds, whatever the point estimate.
    """
    print("\n" + "=" * 78)
    print("12. BOOTSTRAP CI ON THE MARGIN (resample the 21 folds, 10k draws)")
    print("=" * 78)
    rng = np.random.default_rng(seed)
    for ctrl in ("D_peptide_only", "E_allele_mean__tiebreak", "LUT_peptide_mean"):
        cr = store[(ctrl, "tied")][0]
        print(f"\n  vs {ARMS[ctrl]}")
        for arm in ARMS:
            if arm == ctrl:
                continue
            ar = store[(arm, "tied")][0]
            common = sorted(ar.index.intersection(cr.index))
            d = (ar[common] - cr[common]).astype(float).values
            idx = rng.integers(0, len(d), size=(n_boot, len(d)))
            bs = np.median(d[idx], axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            flag = "straddles 0" if lo <= 0 <= hi else "excludes 0"
            print(f"    {ARMS[arm]:46s} {np.median(d):+.4f} "
                  f"[{lo:+.4f},{hi:+.4f}]  {flag}")


def stage_within_allele(store):
    """Strip the between-allele offset, then ask whether anything is left.

    Within each allele, z-score the prediction. Spearman is already invariant to
    that, so this does NOT change rho; the point is to report how much of each
    arm's prediction variance the offset was eating, next to the rho it earned.
    A model whose variance is nearly all offset has not done the ranking task.
    """
    print("\n" + "=" * 78)
    print("13. OFFSET vs RANKING, pooled over the 14 multi-allele folds")
    print("=" * 78)
    multi = [n for n, a in FOLDS if len(a) > 1]
    print(f"{'arm':46s} {'frac_between (median over folds)':>32s} {'rho':>8s}")
    for arm in ARMS:
        e = store[(arm, "ens")]
        fr = []
        for name in multi:
            g = e[e.fold_name == name]
            s2 = DF.loc[g.row_id.values]
            mu = pd.Series(g.y_pred.values).groupby(s2.HLA.values).mean()
            cnt = s2.HLA.value_counts().reindex(mu.index)
            v = np.var(g.y_pred.values)
            if v > 0:
                fr.append(np.average((mu - np.average(mu, weights=cnt)) ** 2,
                                     weights=cnt) / v)
        r = store[(arm, "tied")][0]
        print(f"{ARMS[arm]:46s} {np.median(fr):32.3f} {np.median(r):8.4f}")
    print("\n  Read this as: the arms that spent their variance on the allele")
    print("  offset are exactly the arms with the lowest per-allele rho.")
