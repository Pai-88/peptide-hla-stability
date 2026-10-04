"""Adversarial leakage / split-integrity audit. Read-only: writes nothing.

Run:  python check_leakage.py [section ...]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

import data
import metrics
import splits
import supertypes

ARMS = ["A_supervised_nn", "B_esm_pseudo", "B_esm_pseudo_mlp", "C_esm_joint",
        "D_peptide_only", "E_allele_mean__tiebreak"]

df = data.load()
FOLDS = splits.choose_held_out(df)
FOLD_ALLELES = dict(FOLDS)


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


# ---------------------------------------------------------------------------
def sec1_allele_integrity():
    hr("1. ALLELE INTEGRITY: does any held-out allele appear in its own TRAIN?")
    all_rows = 0
    bad = 0
    seen_test_rows = set()
    for name, alleles in FOLDS:
        tr, te = splits.split_by_allele(df, alleles)
        tr_al = set(df.loc[tr].HLA.unique())
        te_al = set(df.loc[te].HLA.unique())
        overlap = tr_al & te_al
        row_overlap = set(tr) & set(te)
        if overlap or row_overlap:
            bad += 1
            print(f"  LEAK {name}: alleles {overlap} rows {len(row_overlap)}")
        all_rows += len(te)
        dup = seen_test_rows & set(te)
        if dup:
            print(f"  OVERLAPPING TEST FOLDS {name}: {len(dup)} rows also in an earlier fold")
        seen_test_rows |= set(te)
    print(f"  folds {len(FOLDS)}  allele-overlap folds {bad}  "
          f"total test rows {all_rows}  distinct rows covered {len(seen_test_rows)} "
          f"of {len(df)}")
    missing = sorted(set(df.index) - seen_test_rows)
    print(f"  rows never held out: {len(missing)} "
          f"({sorted(df.loc[missing].HLA.unique()) if missing else '-'})")

    # the same check at the artifact level: every stored prediction row's HLA
    # must be in that fold's held-out allele list.
    for arm in ARMS:
        try:
            p = pd.read_parquet(f"predictions_{arm}.parquet",
                                columns=["row_id", "fold_name", "HLA"])
        except Exception as e:
            print(f"  {arm}: no predictions ({e})")
            continue
        bad_rows = 0
        for fold, sub in p.groupby("fold_name"):
            allowed = set(FOLD_ALLELES.get(fold, []))
            bad_rows += int((~sub.HLA.isin(allowed)).sum())
        print(f"  {arm:<26} prediction rows with an HLA not in their fold: {bad_rows}")


# ---------------------------------------------------------------------------
def sec2_peptide_overlap():
    hr("2. PEPTIDE OVERLAP ACROSS THE FOLD BOUNDARY")
    print(f"  peptides {df.Pep.nunique()}  rows {len(df)}  "
          f"median alleles per peptide {int(df.groupby('Pep').HLA.nunique().median())}")
    rows = []
    for name, alleles in FOLDS:
        tr, te = splits.split_by_allele(df, alleles)
        trp = set(df.loc[tr].Pep)
        ted = df.loc[te]
        seen = ted.Pep.isin(trp)
        rows.append({"fold": name, "n_test": len(ted), "frac_pep_seen": seen.mean(),
                     "n_unseen": int((~seen).sum()),
                     "n_test_alleles": ted.HLA.nunique()})
    o = pd.DataFrame(rows)
    print(o.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    w = np.average(o.frac_pep_seen, weights=o.n_test)
    print(f"\n  ROW-WEIGHTED fraction of test rows whose peptide is in train: {w:.4f}")
    print(f"  folds at 100%: {(o.frac_pep_seen > 0.9999).sum()} of {len(o)}")

    # how much of the label is peptide-level at all?
    hr("2b. HOW MUCH OF y IS A PEPTIDE EFFECT? (does peptide sharing hand over y?)")
    y = df.y.values
    tot = ((y - y.mean()) ** 2).sum()
    for key in ["Pep", "HLA"]:
        gm = df.groupby(key).y.transform("mean").values
        print(f"  variance of y explained by {key:>3} identity: "
              f"{1 - ((y - gm) ** 2).sum() / tot:.4f}")
    # correlation between a peptide's measurement on one allele and on another
    piv = df.pivot_table(index="Pep", columns="HLA", values="y")
    cnt = piv.notna().sum()
    big = cnt.sort_values(ascending=False).index[:12]
    cc = piv[big].corr(min_periods=50).values
    iu = np.triu_indices_from(cc, 1)
    v = cc[iu]
    v = v[~np.isnan(v)]
    print(f"  cross-allele correlation of the SAME peptide's y, 12 largest alleles: "
          f"median {np.median(v):.3f}  range [{v.min():.3f}, {v.max():.3f}]  n_pairs {len(v)}")


# ---------------------------------------------------------------------------
def _fold_frames(arm):
    """yield (fold, test_df, ensemble_pred, per_seed dict)"""
    p = pd.read_parquet(f"predictions_{arm}.parquet")
    for fold, sub in p.groupby("fold_name", sort=False):
        agg = sub.groupby("row_id", sort=False).y_pred.agg(["mean", "count"])
        test_df = df.loc[agg.index.to_numpy()]
        yield fold, test_df, agg["mean"].to_numpy(), sub


def sec3_memorisation():
    hr("3. IS THE SIGNAL PEPTIDE MEMORISATION? seen vs unseen peptides, per arm")
    print("  Per-allele Spearman (censored='tied'), 5-seed ensemble, recomputed")
    print("  from the stored predictions. 'unseen' = that peptide appears nowhere")
    print("  in the fold's training rows. metrics.MIN_N=20 applies to each subset.\n")
    summary = []
    for arm in ARMS:
        try:
            gen = list(_fold_frames(arm))
        except Exception as e:
            print(f"  {arm}: skipped ({e})")
            continue
        full, seen_r, unseen_r, nun = [], [], [], 0
        for fold, test_df, pred, _ in gen:
            trp = set(df.loc[splits.split_by_allele(df, FOLD_ALLELES[fold])[0]].Pep)
            m = test_df.Pep.isin(trp).values
            r = metrics.spearman_per_allele(test_df, pred)
            if len(r):
                full.append(np.median(r))
            if m.sum():
                r = metrics.spearman_per_allele(test_df[m], pred[m])
                if len(r):
                    seen_r.append(np.median(r))
            if (~m).sum() >= 20:
                nun += 1
                r = metrics.spearman_per_allele(test_df[~m], pred[~m])
                if len(r):
                    unseen_r.append(np.median(r))
        summary.append({
            "arm": arm, "folds": len(full),
            "rho_all": np.median(full) if full else np.nan,
            "rho_seen_pep": np.median(seen_r) if seen_r else np.nan,
            "n_folds_unseen": len(unseen_r),
            "rho_unseen_pep": np.median(unseen_r) if unseen_r else np.nan,
        })
    s = pd.DataFrame(summary)
    print(s.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return s


# ---------------------------------------------------------------------------
def sec4_peptide_lookup_control():
    hr("4. CONTROL: the peptide-mean lookup table, and arm A against it, per fold")
    print("  lookup = mean training y of that exact peptide (global mean if unseen).")
    print("  It uses ONLY information that peptide-sharing hands across the boundary,")
    print("  and no allele information at all.\n")
    try:
        A = {f: (t, p) for f, t, p, _ in _fold_frames("A_supervised_nn")}
    except Exception as e:
        print(f"  arm A predictions unavailable: {e}")
        A = {}
    rows = []
    for name, alleles in FOLDS:
        tr, te = splits.split_by_allele(df, alleles)
        train_df, test_df = df.loc[tr], df.loc[te]
        pm = train_df.groupby("Pep").y.mean()
        gm = train_df.y.mean()
        look = test_df.Pep.map(pm).fillna(gm).to_numpy()
        rl = metrics.spearman_per_allele(test_df, look)
        row = {"fold": name, "n_test": len(test_df),
               "rho_lookup": np.median(rl) if len(rl) else np.nan}
        if name in A:
            t2, p2 = A[name]
            assert (t2.index == test_df.index).all()
            ra = metrics.spearman_per_allele(test_df, p2)
            row["rho_armA_ens"] = np.median(ra) if len(ra) else np.nan
            row["A_minus_lookup"] = row["rho_armA_ens"] - row["rho_lookup"]
        rows.append(row)
    o = pd.DataFrame(rows)
    print(o.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n  median rho_lookup      {o.rho_lookup.median():.4f}")
    if "rho_armA_ens" in o:
        print(f"  median rho_armA_ens    {o.rho_armA_ens.median():.4f}")
        d = o.A_minus_lookup.dropna()
        print(f"  median A - lookup      {d.median():+.4f}   A wins {int((d>0).sum())}/{len(d)} folds"
              f"   Wilcoxon p={stats.wilcoxon(d).pvalue:.4f}")
    return o


# ---------------------------------------------------------------------------
def sec5_groove_proximity():
    hr("5. IS 'UNSEEN GROOVE' REALLY UNSEEN? nearest training allele per fold")
    names, M = supertypes.identity_matrix()
    pos = {a: i for i, a in enumerate(names)}
    have = set(df.HLA.unique())
    rows = []
    for name, alleles in FOLDS:
        tr, te = splits.split_by_allele(df, alleles)
        train_al = sorted(set(df.loc[tr].HLA.unique()) & have)
        test_al = sorted(set(df.loc[te].HLA.unique()))
        best = []
        for a in test_al:
            ids = [M[pos[a], pos[b]] for b in train_al if b in pos]
            best.append(max(ids))
        rows.append({"fold": name, "n_test": len(df.loc[te]),
                     "max_ident_to_train": max(best), "median_ident": float(np.median(best))})
    o = pd.DataFrame(rows)
    print(o.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n  across folds: max {o.max_ident_to_train.max():.3f}  "
          f"median of per-fold max {o.max_ident_to_train.median():.3f}  "
          f"min {o.max_ident_to_train.min():.3f}")
    print(f"  folds whose closest held-out/train pair exceeds the 0.80 cut: "
          f"{(o.max_ident_to_train > 0.80).sum()} of {len(o)}")
    # does fold score track groove proximity?
    for arm in ["A_supervised_nn", "C_esm_joint", "D_peptide_only"]:
        try:
            sc = {f: np.median(metrics.spearman_per_allele(t, p))
                  for f, t, p, _ in _fold_frames(arm)}
        except Exception:
            continue
        v = o.assign(rho=o.fold.map(sc)).dropna(subset=["rho"])
        r = stats.spearmanr(v.max_ident_to_train, v.rho)
        print(f"  {arm:<20} rho(fold score, max groove identity to train) = "
              f"{r.statistic:+.3f}  p={r.pvalue:.3f}  n={len(v)}")
    return o


# ---------------------------------------------------------------------------
def sec6_headline_reproduce():
    hr("6. DO THE REPORTED HEADLINES REPRODUCE FROM THE STORED PREDICTIONS?")
    for arm in ARMS:
        try:
            p = pd.read_parquet(f"predictions_{arm}.parquet")
        except Exception as e:
            print(f"  {arm}: {e}")
            continue
        per_cell, ens = [], []
        for fold, sub in p.groupby("fold_name", sort=False):
            cell = []
            for seed, s2 in sub.groupby("seed", sort=False):
                t = df.loc[s2.row_id.to_numpy()]
                r = metrics.spearman_per_allele(t, s2.y_pred.to_numpy())
                if len(r):
                    cell.append(np.median(r))
            if cell:
                per_cell.append((fold, np.mean(cell), cell))
            agg = sub.groupby("row_id", sort=False).y_pred.mean()
            t = df.loc[agg.index.to_numpy()]
            r = metrics.spearman_per_allele(t, agg.to_numpy())
            if len(r):
                ens.append((fold, np.median(r)))
        pf = np.array([x[1] for x in per_cell])
        allcells = np.array([v for x in per_cell for v in x[2]])
        e = np.array([x[1] for x in ens])
        print(f"  {arm:<26} folds {len(pf):2d}  "
              f"median-of-fold-means {np.median(pf):+.4f}  "
              f"median-of-all-cells {np.median(allcells):+.4f}  "
              f"ensemble median {np.median(e):+.4f}  "
              f"worst ens fold {ens[int(np.argmin(e))][0]} {e.min():+.4f}")


# ---------------------------------------------------------------------------
def sec7_embedding_sanity():
    hr("7. ESM-2 EMBEDDINGS: could the cache carry label information?")
    import embed
    for nm in ["peptides", "pseudo", "joint"]:
        try:
            M, ix = embed.load(nm)
        except Exception as e:
            print(f"  {nm}: {e}")
            continue
        print(f"  emb_{nm}: {M.shape}, index {len(ix)} keys, "
              f"keys are {'<HLA>|<Pep>' if nm == 'joint' else 'sequence/allele names'}")
    M, ix = embed.load("peptides")
    print(f"  distinct peptides in data {df.Pep.nunique()}, rows in peptide table {M.shape[0]}: "
          f"one row per sequence, so the table cannot encode a per-measurement label.")
    # a label-leaking embedding would make y predictable from the peptide vector
    # far better than chance ACROSS alleles; the arm-D result already bounds that.
    j, jx = embed.load("joint")
    keys = [f"{h}|{p}" for h, p in zip(df.HLA, df.Pep)]
    print(f"  joint coverage: {sum(k in jx for k in keys)}/{len(keys)} rows")
    # are two rows with the same peptide but different allele actually different?
    same = df.groupby("Pep").filter(lambda g: len(g) > 1).groupby("Pep").head(2)
    pairs = [g for _, g in same.groupby("Pep") if len(g) == 2][:200]
    d = [np.abs(j[jx[f"{g.HLA.iloc[0]}|{g.Pep.iloc[0]}"]] -
                j[jx[f"{g.HLA.iloc[1]}|{g.Pep.iloc[1]}"]]).max() for g in pairs]
    print(f"  joint vectors for the same peptide under 2 alleles differ by "
          f"max-abs median {np.median(d):.4g} over {len(d)} pairs "
          f"(0 would mean the allele half is inert)")


# ---------------------------------------------------------------------------
def sec8_csv_hazards():
    hr("8. RESULT-FILE HAZARDS (a wrong number a reader could take off the file)")
    for arm in ARMS + ["B_esm_pseudo_mlp"]:
        try:
            c = pd.read_csv(f"results_{arm}.csv")
        except Exception:
            continue
        pol = c.censored_policy.dropna().unique().tolist()
        ok = c[c.status == "ok"]
        naive = ok.spearman.median()
        line = (f"  results_{arm}.csv  rows {len(c):3d}  status "
                f"{dict(c.status.value_counts())}  policies {pol}  "
                f"naive median(spearman|ok) {naive:+.4f}")
        if len(pol) > 1:
            tied = ok[ok.censored_policy == "tied"].spearman.median()
            line += f"   <-- MIXED POLICIES; tied-only median {tied:+.4f}"
        print(line)


def sec9_nmatched():
    hr("9. SEEN vs UNSEEN PEPTIDES, SAME FOLDS AND SAME n (the fair comparison)")
    print("  The unseen subsets are smaller, and a smaller n makes a per-allele rho")
    print("  noisier. So the 'seen' side is SUBSAMPLED to the same per-allele n,")
    print("  20 draws, and both sides use only the folds/alleles where the unseen")
    print("  side has >= metrics.MIN_N rows. Same ensemble predictions throughout.\n")
    rng = np.random.default_rng(0)
    out = []
    for arm in ARMS:
        try:
            gen = list(_fold_frames(arm))
        except Exception:
            continue
        seen_m, unseen_m, folds_used, n_al = [], [], 0, 0
        for fold, test_df, pred, _ in gen:
            trp = set(df.loc[splits.split_by_allele(df, FOLD_ALLELES[fold])[0]].Pep)
            m = test_df.Pep.isin(trp).values
            su, ss = [], []
            for allele, idx in test_df.groupby("HLA").indices.items():
                iu = idx[~m[idx]]
                isn = idx[m[idx]]
                if len(iu) < metrics.MIN_N or len(isn) < len(iu):
                    continue
                t = test_df.iloc[iu]
                r = metrics.spearman_per_allele(t, pred[iu], min_n=metrics.MIN_N)
                if not len(r):
                    continue
                su.append(float(r.iloc[0]))
                draws = []
                for _ in range(20):
                    pick = rng.choice(isn, len(iu), replace=False)
                    t2 = test_df.iloc[pick]
                    r2 = metrics.spearman_per_allele(t2, pred[pick], min_n=metrics.MIN_N)
                    if len(r2):
                        draws.append(float(r2.iloc[0]))
                if draws:
                    ss.append(float(np.mean(draws)))
            if su:
                folds_used += 1
                n_al += len(su)
                unseen_m.append(np.median(su))
                seen_m.append(np.median(ss))
        if unseen_m:
            w = stats.wilcoxon(np.array(seen_m) - np.array(unseen_m))
            out.append({"arm": arm, "folds": folds_used, "alleles": n_al,
                        "rho_seen_nmatched": np.median(seen_m),
                        "rho_unseen": np.median(unseen_m),
                        "delta": np.median(np.array(seen_m) - np.array(unseen_m)),
                        "wilcoxon_p": w.pvalue})
    print(pd.DataFrame(out).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def sec10_featurizer_state():
    hr("10. DOES ANY FEATURIZER CARRY STATE FROM THE ROWS AROUND IT?")
    print("  A transductive leak would show up as featurize(X) depending on which")
    print("  other rows are in the subset. Each featurizer is called on the full")
    print("  frame, on a train slice, and on a 50-row test slice; the vectors for")
    print("  the SAME row must be bit-identical in all three.\n")
    import importlib
    probes = [("A_supervised_nn", "arm_A_supervised_nn", None),
              ("B_esm_pseudo", "arm_B_esm_pseudo", None),
              ("C_esm_joint", "arm_C_esm_joint", None),
              ("D_peptide_only", "arm_D_peptide_only", None)]
    fold, alleles = FOLDS[0]
    tr, te = splits.split_by_allele(df, alleles)
    sample = list(df.loc[te].index[:50])
    for label, mod, _ in probes:
        try:
            m = importlib.import_module(mod)
            f = m.featurizer(m.ENCODING) if hasattr(m, "featurizer") else m.featurize
            a = f(df.loc[sample])
            b = f(df.loc[te])
            c = f(df)
            pos_b = {r: i for i, r in enumerate(df.loc[te].index)}
            pos_c = {r: i for i, r in enumerate(df.index)}
            d1 = max(np.abs(a[i] - b[pos_b[r]]).max() for i, r in enumerate(sample))
            d2 = max(np.abs(a[i] - c[pos_c[r]]).max() for i, r in enumerate(sample))
            print(f"  {label:<18} dims {a.shape[1]:5d}  max|slice - foldwise| {d1:.3g}  "
                  f"max|slice - wholedata| {d2:.3g}")
        except Exception as e:
            print(f"  {label:<18} FAILED: {type(e).__name__}: {e}")

    hr("10b. DUPLICATE MEASUREMENTS IN stability.txt")
    d = df.duplicated(["HLA", "Pep"], keep=False)
    print(f"  rows sharing an (HLA, Pep) key: {int(d.sum())}")
    if d.sum():
        g = df[d].groupby(["HLA", "Pep"]).Thalf.agg(["count", "nunique", "min", "max"])
        print(g.head(10).to_string())
    print(f"  exact duplicate rows (HLA, Pep, Thalf): "
          f"{int(df.duplicated(['HLA', 'Pep', 'Thalf']).sum())}")


SECTIONS = {
    "1": sec1_allele_integrity, "2": sec2_peptide_overlap, "3": sec3_memorisation,
    "9": sec9_nmatched, "10": sec10_featurizer_state,
    "4": sec4_peptide_lookup_control, "5": sec5_groove_proximity,
    "6": sec6_headline_reproduce, "7": sec7_embedding_sanity, "8": sec8_csv_hazards,
}

# ===========================================================================
# ADVERSARIAL PASS 2 (leakage / split-integrity lens). Appended, nothing above
# modified. All three sections are fit-free: they read stored predictions and
# the pseudo-sequences only, so they can run while arms F and G are still going.
# ===========================================================================

LOOKUP_ARMS = ARMS + ["G_650m_B_ridge", "G_650m_B_mlp"]


def _lookup_per_fold():
    """Median per-allele rho of the peptide-mean lookup table, per fold.

    The lookup table is the strongest predictor buildable from ONLY the
    information that peptide-sharing hands across the groove boundary: the mean
    training y of that exact peptide. It contains no allele information and no
    model. Anything that cannot beat it has not been shown to use the groove.
    """
    out = {}
    for name, alleles in FOLDS:
        tr, te = splits.split_by_allele(df, alleles)
        train_df, test_df = df.loc[tr], df.loc[te]
        look = test_df.Pep.map(train_df.groupby("Pep").y.mean()) \
                      .fillna(train_df.y.mean()).to_numpy()
        r = metrics.spearman_per_allele(test_df, look)
        out[name] = float(np.median(r)) if len(r) else np.nan
    return out


def sec11_vs_lookup():
    hr("11. DOES ANY ARM BEAT THE PEPTIDE-MEAN LOOKUP TABLE? (paired, per fold)")
    print("  Paired over the folds each arm actually scored. The lookup table uses")
    print("  no allele information and no model, so it is the floor that a claim of")
    print("  'the model reads the groove' has to clear. Ensemble predictions.\n")
    look = _lookup_per_fold()
    rows = []
    for arm in LOOKUP_ARMS:
        try:
            sc = {f: float(np.median(metrics.spearman_per_allele(t, p)))
                  for f, t, p, _ in _fold_frames(arm)}
        except Exception as e:
            print(f"  {arm}: skipped ({e})")
            continue
        common = [f for f in sc if f in look and not np.isnan(look[f])]
        d = np.array([sc[f] - look[f] for f in common])
        w = stats.wilcoxon(d).pvalue if len(d) >= 6 else np.nan
        rows.append({"arm": arm, "folds": len(d),
                     "rho_arm": np.median([sc[f] for f in common]),
                     "rho_lookup_same_folds": np.median([look[f] for f in common]),
                     "delta": np.median(d), "arm_wins": int((d > 0).sum()),
                     "wilcoxon_p": w})
    o = pd.DataFrame(rows)
    print(o.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n  PARTIAL RUNS: any arm whose 'folds' is below 21 is still filling in;")
    print("  its delta is not comparable to a 21-fold delta. Do not quote those.")
    return o


def sec12_cut_violation():
    hr("12. DOES THE 0.80 GROOVE CUT ACTUALLY HOLD ACROSS THE FOLD BOUNDARY?")
    print("  supertypes.clusters uses AVERAGE linkage at distance 1-0.80. Average")
    print("  linkage bounds the MEAN identity between clusters, not the MAXIMUM")
    print("  pairwise identity. So a held-out allele can still sit above 0.80 to an")
    print("  allele left in training. This counts how often, and by how much.\n")
    names, M = supertypes.identity_matrix()
    pos = {a: i for i, a in enumerate(names)}
    have = set(df.HLA.unique())
    pairs, per_fold = [], []
    for name, alleles in FOLDS:
        tr, te = splits.split_by_allele(df, alleles)
        tral = [b for b in sorted(set(df.loc[tr].HLA.unique()) & have) if b in pos]
        mx = 0.0
        for a in sorted(set(df.loc[te].HLA.unique())):
            for b in tral:
                v = M[pos[a], pos[b]]
                mx = max(mx, v)
                if v > splits.CUT:
                    pairs.append((v, name, a, b))
        per_fold.append({"fold": name, "n_test": len(df.loc[te]),
                         "max_ident_to_train": mx, "violates_cut": mx > splits.CUT})
    o = pd.DataFrame(per_fold)
    print(o.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n  folds whose held-out groove has a TRAINING allele above the "
          f"{splits.CUT:.2f} cut: {int(o.violates_cut.sum())} of {len(o)}")
    print(f"  offending held-out/train allele pairs: {len(pairs)}")
    pairs.sort(reverse=True)
    print("  worst five:")
    for v, f, a, b in pairs[:5]:
        print(f"    {v:.3f} ({round(v * 34)}/34 residues) fold {f:<18} "
              f"held-out {a.replace('HLA-', '')} vs TRAIN {b.replace('HLA-', '')}")
    print("\n  So 'held out at <=80% groove identity' is FALSE as a description of")
    print("  these folds. The defensible wording is 'held-out groove CLUSTER, with")
    print("  the nearest training groove as close as {:.0%} on 13 of 21 folds'."
          .format(o.max_ident_to_train.max()))
    return o


def sec13_live_files():
    hr("13. LIVE / PARTIAL FILES: what must NOT be quoted right now")
    import glob
    import os
    full = None
    for f in sorted(glob.glob("results_*.csv")):
        try:
            c = pd.read_csv(f)
        except Exception as e:
            print(f"  {f:<42} UNREADABLE ({e})")
            continue
        if "status" not in c.columns:
            print(f"  {f:<42} not a per-fold results file (summary table)")
            continue
        ok = c[c.status == "ok"]
        nf = ok.fold_name.nunique() if len(ok) else 0
        full = full or 21
        flag = "" if nf == 21 else f"  <-- PARTIAL ({nf}/21 folds) DO NOT QUOTE"
        med = f"{ok.spearman.median():+.4f}" if len(ok) else "   n/a"
        print(f"  {f:<42} rows {len(c):4d} folds {nf:2d} median {med}"
              f"  mtime {time.strftime('%H:%M:%S', time.localtime(os.path.getmtime(f)))}{flag}")
    print("\n  results_G_650m.csv is a hand-rolled SUMMARY table, not regenerated by")
    print("  the arms. Its G_650m_B_ridge row says 0.1338 from 2 folds; the 21-fold")
    print("  file now says a different number. A reader taking a figure off that")
    print("  summary would quote the partial. Check it against results_<arm>.csv.")


SECTIONS.update({"11": sec11_vs_lookup, "12": sec12_cut_violation,
                 "13": sec13_live_files})


if __name__ == "__main__":
    want = sys.argv[1:] or list(SECTIONS)
    t0 = time.time()
    for k in want:
        SECTIONS[k]()
    print(f"\n[check_leakage] {time.time() - t0:.1f}s", flush=True)
