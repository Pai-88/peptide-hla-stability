"""Adversarial statistical audit of the arm comparison. Writes nothing."""
import itertools
import sys

import numpy as np
import pandas as pd
from scipy import stats

ARMS = ["A_supervised_nn", "B_esm_pseudo", "B_esm_pseudo_mlp",
        "C_esm_joint", "D_peptide_only", "E_allele_mean__tiebreak"]


def load_results(arm, policy="tied"):
    d = pd.read_csv(f"results_{arm}.csv")
    if "censored_policy" in d:
        d = d[d.censored_policy == policy]
    return d[d.status == "ok"].copy()


def fold_table(arm, policy="tied", col="spearman"):
    """per-fold mean over seeds of the per-fold (median-over-allele) metric."""
    d = load_results(arm, policy)
    g = d.groupby("fold_name")[col]
    return pd.DataFrame({"mean": g.mean(), "sd": g.std(ddof=1), "n": g.size()})


def banner(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ---------------------------------------------------------------- 1. reproduce
banner("1. CAN THE REPORTED HEADLINES BE REPRODUCED FROM THE FILES?")
claimed = {"A_supervised_nn": 0.265, "B_esm_pseudo": 0.0605,
           "B_esm_pseudo_mlp": None, "C_esm_joint": 0.0772,
           "D_peptide_only": 0.0643, "E_allele_mean__tiebreak": 0.0041}
for arm in ARMS:
    f = fold_table(arm)
    d = load_results(arm)
    print(f"{arm:26s} nfolds={len(f):3d} "
          f"median(per-fold mean over seeds)={np.median(f['mean']):+.4f}  "
          f"median(all fold,seed rows)={np.median(d.spearman):+.4f}  "
          f"claimed={claimed[arm]}")

# ensemble files
banner("1b. ENSEMBLE FILES (what the A headline 0.307 / B 0.1059 / D 0.0643 cite)")
import os
for arm in ARMS:
    p = f"ensemble_{arm}.csv"
    if not os.path.exists(p):
        print(f"{arm:26s} NO ensemble file")
        continue
    e = pd.read_csv(p)
    e = e[e.status == "ok"] if "status" in e else e
    print(f"{arm:26s} nrows={len(e):3d} median spearman={np.nanmedian(e.spearman):+.4f} "
          f"IQR [{np.nanpercentile(e.spearman,25):+.4f},{np.nanpercentile(e.spearman,75):+.4f}]")

# ---------------------------------------------------------- 2. paired A vs FM
banner("2. PAIRED ACROSS-FOLD COMPARISON: arm A vs each foundation-model arm")
A = fold_table("A_supervised_nn")["mean"]
for arm in ["B_esm_pseudo", "B_esm_pseudo_mlp", "C_esm_joint", "D_peptide_only"]:
    B = fold_table(arm)["mean"]
    common = A.index.intersection(B.index)
    a, b = A[common].values, B[common].values
    d = a - b
    w = stats.wilcoxon(a, b)
    t = stats.ttest_rel(a, b)
    dz = d.mean() / d.std(ddof=1)
    rng = np.random.default_rng(0)
    boot = np.array([np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(20000)])
    print(f"A - {arm:20s} n={len(common)} meandiff={d.mean():+.4f} "
          f"mediandiff={np.median(d):+.4f} "
          f"boot95=[{np.percentile(boot,2.5):+.4f},{np.percentile(boot,97.5):+.4f}] "
          f"wins={int((d>0).sum())}/{len(d)} wilcoxon p={w.pvalue:.2e} "
          f"t p={t.pvalue:.2e} Cohen dz={dz:+.2f}")

# ------------------------------------------------- 3. seed variance vs the gap
banner("3. SEED NOISE WITHIN A FOLD vs THE ARM GAP")
for arm in ARMS:
    f = fold_table(arm)
    print(f"{arm:26s} across-seed sd of per-fold rho: median={np.nanmedian(f['sd']):.4f} "
          f"max={np.nanmax(f['sd']):.4f}  across-FOLD sd of the mean={f['mean'].std(ddof=1):.4f}")

print("\nVariance decomposition on the per-(fold,seed) rho (two-way: fold + seed):")
for arm in ARMS:
    d = load_results(arm)
    piv = d.pivot_table(index="fold_name", columns="seed", values="spearman")
    gm = piv.values.mean()
    fold_eff = piv.mean(axis=1) - gm
    seed_eff = piv.mean(axis=0) - gm
    resid = piv.values - gm - fold_eff.values[:, None] - seed_eff.values[None, :]
    tot = piv.values.var()
    print(f"{arm:26s} var(total)={tot:.5f}  fold={fold_eff.var():.5f} "
          f"({100*fold_eff.var()/tot:4.1f}%)  seed={seed_eff.var():.5f} "
          f"({100*seed_eff.var()/tot:4.1f}%)  resid={resid.var():.5f} "
          f"({100*resid.var()/tot:4.1f}%)")

# ---------------------------------------- 3b. does the ARM RANKING flip by seed
banner("3b. IF YOU RE-RAN WITH ONE SEED, WOULD THE RANKING HOLD? (per-seed medians)")
rows = {}
for arm in ARMS:
    d = load_results(arm)
    rows[arm] = d.groupby("seed").spearman.median()
print(pd.DataFrame(rows).round(4).to_string())

# single-seed paired test A vs best FM
banner("3c. PAIRED TEST USING ONE SEED AT A TIME (A vs B_mlp, A vs C)")
for arm in ["B_esm_pseudo_mlp", "C_esm_joint"]:
    for seed in sorted(load_results("A_supervised_nn").seed.unique()):
        a = load_results("A_supervised_nn").query("seed==@seed").set_index("fold_name").spearman
        b = load_results(arm).query("seed==@seed").set_index("fold_name").spearman
        common = a.index.intersection(b.index)
        d = (a[common] - b[common]).values
        print(f"A - {arm:18s} seed={seed} meandiff={d.mean():+.4f} "
              f"wins={int((d>0).sum())}/{len(d)} p={stats.wilcoxon(d).pvalue:.2e}")

# ------------------------------------------------ 4. per-fold distribution shape
banner("4. IS THE MEDIAN HIDING A BIMODAL / SKEWED DISTRIBUTION?")
for arm in ARMS:
    v = np.sort(fold_table(arm)["mean"].values)
    print(f"\n{arm}  (21 per-fold means, sorted)")
    print("  " + " ".join(f"{x:+.3f}" for x in v))
    print(f"  median={np.median(v):+.4f} mean={v.mean():+.4f} "
          f"IQR=[{np.percentile(v,25):+.4f},{np.percentile(v,75):+.4f}] "
          f"skew={stats.skew(v):+.2f} kurt={stats.kurtosis(v):+.2f} "
          f"shapiro p={stats.shapiro(v).pvalue:.3f} "
          f"dip-ish gapmax={np.max(np.diff(v)):.3f}")
    # Hartigan-style crude check: largest gap relative to range
    print(f"  largest gap {np.max(np.diff(v)):.3f} at position "
          f"{int(np.argmax(np.diff(v)))+1}/21, range {np.ptp(v):.3f} "
          f"-> gap/range = {np.max(np.diff(v))/np.ptp(v):.2f}")

# --------------------------------------------- 5. are the error bars what they say
banner("5. ERROR BARS: IQR over folds vs SEM, and what a bootstrap CI on the median is")
for arm in ARMS:
    v = fold_table(arm)["mean"].values
    rng = np.random.default_rng(1)
    boot = np.array([np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(20000)])
    sem = v.std(ddof=1) / np.sqrt(len(v))
    print(f"{arm:26s} median={np.median(v):+.4f} "
          f"IQR=[{np.percentile(v,25):+.4f},{np.percentile(v,75):+.4f}] "
          f"(halfwidth {(np.percentile(v,75)-np.percentile(v,25))/2:.4f})  "
          f"SEM={sem:.4f}  boot95 median=[{np.percentile(boot,2.5):+.4f},"
          f"{np.percentile(boot,97.5):+.4f}]")

# ---------------------------------------------------- 6. censoring: does it flip?
banner("6. CENSORED tied vs drop: RE-SCORE EVERY ARM FROM ITS OWN PREDICTIONS")
sys.path.insert(0, ".")
import data as D
import metrics as M

df_all = D.load()

PRED = {"A_supervised_nn": "predictions_A_supervised_nn.parquet",
        "B_esm_pseudo": "predictions_B_esm_pseudo.parquet",
        "B_esm_pseudo_mlp": "predictions_B_esm_pseudo_mlp.parquet",
        "C_esm_joint": "predictions_C_esm_joint.parquet",
        "D_peptide_only": "predictions_D_peptide_only.parquet",
        "E_allele_mean__tiebreak": "predictions_E_allele_mean__tiebreak.parquet"}

ens_fold = {}   # arm -> policy -> Series fold -> rho (5-seed ensemble)
single_fold = {}
for arm, path in PRED.items():
    p = pd.read_parquet(path)
    # ensemble = mean prediction across seeds
    ens = p.groupby(["fold_name", "row_id"], sort=False).y_pred.mean().reset_index()
    ens = ens.merge(df_all[["HLA", "Thalf", "y", "censored"]],
                    left_on="row_id", right_index=True, how="left", validate="1:1")
    ens_fold[arm] = {}
    single_fold[arm] = {}
    for pol in ["tied", "drop"]:
        out = {}
        for fname, sub in ens.groupby("fold_name"):
            r = M.spearman_per_allele(sub, sub.y_pred.values, censored=pol)
            out[fname] = np.median(r) if len(r) else np.nan
        ens_fold[arm][pol] = pd.Series(out)
    # single-fit: per (fold,seed) then mean over seeds
    for pol in ["tied", "drop"]:
        rows = []
        pm = p.drop(columns=["HLA", "Pep", "y_true"]).merge(
            df_all[["HLA", "Thalf", "y", "censored"]],
            left_on="row_id", right_index=True, how="left")
        for (fname, seed), sub in pm.groupby(["fold_name", "seed"]):
            r = M.spearman_per_allele(sub, sub.y_pred.values, censored=pol)
            rows.append((fname, seed, np.median(r) if len(r) else np.nan))
        t = pd.DataFrame(rows, columns=["fold", "seed", "rho"])
        single_fold[arm][pol] = t.groupby("fold").rho.mean()

print("\nENSEMBLE per-fold medians, recomputed:")
for arm in PRED:
    for pol in ["tied", "drop"]:
        v = ens_fold[arm][pol].dropna().values
        print(f"  {arm:26s} {pol:5s} median={np.median(v):+.4f} "
              f"IQR=[{np.percentile(v,25):+.4f},{np.percentile(v,75):+.4f}] "
              f"worst={v.min():+.4f} ({ens_fold[arm][pol].idxmin()})")

print("\nPAIRED A vs FM under BOTH policies (ensemble predictions):")
for pol in ["tied", "drop"]:
    a = ens_fold["A_supervised_nn"][pol]
    for arm in ["B_esm_pseudo", "B_esm_pseudo_mlp", "C_esm_joint", "D_peptide_only"]:
        b = ens_fold[arm][pol]
        common = a.dropna().index.intersection(b.dropna().index)
        d = (a[common] - b[common]).values
        print(f"  {pol:5s} A - {arm:20s} meandiff={d.mean():+.4f} "
              f"wins={int((d>0).sum())}/{len(d)} p={stats.wilcoxon(d).pvalue:.2e}")

print("\nDOES THE FM-ARM ORDERING FLIP BETWEEN POLICIES?")
for pol in ["tied", "drop"]:
    order = sorted(((np.nanmedian(ens_fold[a][pol]), a) for a in PRED), reverse=True)
    print(f"  {pol:5s}: " + "  >  ".join(f"{a}({v:+.3f})" for v, a in order))

# -------------------------------------- 7. FM arms against each other / the null
banner("7. ARE THE FM ARMS DISTINGUISHABLE FROM EACH OTHER AND FROM THE NULL?")
for pol in ["tied"]:
    for x, y in itertools.combinations(
            ["B_esm_pseudo", "B_esm_pseudo_mlp", "C_esm_joint", "D_peptide_only",
             "E_allele_mean__tiebreak"], 2):
        a, b = ens_fold[x][pol], ens_fold[y][pol]
        common = a.dropna().index.intersection(b.dropna().index)
        d = (a[common] - b[common]).values
        print(f"  {x:24s} - {y:24s} meandiff={d.mean():+.4f} "
              f"wins={int((d>0).sum())}/{len(d)} p={stats.wilcoxon(d).pvalue:.2e}")

# ------------------------------------- 8. multiplicity: how many tests are we doing
banner("8. MULTIPLICITY CHECK on the pairwise FM comparisons (Holm)")
pairs = list(itertools.combinations(
    ["A_supervised_nn", "B_esm_pseudo", "B_esm_pseudo_mlp", "C_esm_joint",
     "D_peptide_only", "E_allele_mean__tiebreak"], 2))
ps = []
for x, y in pairs:
    a, b = ens_fold[x]["tied"], ens_fold[y]["tied"]
    common = a.dropna().index.intersection(b.dropna().index)
    ps.append(stats.wilcoxon((a[common] - b[common]).values).pvalue)
order = np.argsort(ps)
m = len(ps)
holm = np.empty(m)
prev = 0
for rank, i in enumerate(order):
    adj = min(1.0, (m - rank) * ps[i])
    prev = max(prev, adj)
    holm[i] = prev
for (x, y), p, h in zip(pairs, ps, holm):
    print(f"  {x:24s} vs {y:24s} raw p={p:.2e} Holm p={h:.2e} "
          f"{'SIG' if h < 0.05 else 'ns'}")

# ---------------------------------------- 9. fold weighting / allele count sanity
banner("9. FOLD SIZES AND HOW MANY ALLELES EACH FOLD'S MEDIAN IS TAKEN OVER")
d = load_results("A_supervised_nn").drop_duplicates("fold_name")
t = d[["fold_name", "n_test", "n_held_out_alleles", "n_alleles_scored"]].sort_values("n_test")
print(t.to_string(index=False))
print(f"\nfolds whose 'median over alleles' is a median over ONE allele: "
      f"{int((t.n_alleles_scored==1).sum())} of {len(t)}")
print(f"test rows: min {t.n_test.min()} max {t.n_test.max()} "
      f"-> the 21 folds are weighted EQUALLY despite a {t.n_test.max()/t.n_test.min():.0f}x size range")

# how much does the headline move if you weight by fold size / pool alleles?
banner("9b. SENSITIVITY OF THE HEADLINE TO THE SUMMARY CHOICE")
for arm in PRED:
    p = pd.read_parquet(PRED[arm])
    ens = p.groupby(["fold_name", "row_id"], sort=False).y_pred.mean().reset_index()
    ens = ens.merge(df_all[["HLA", "Thalf", "y", "censored"]],
                    left_on="row_id", right_index=True, how="left")
    per_allele = {}
    for fname, sub in ens.groupby("fold_name"):
        r = M.spearman_per_allele(sub, sub.y_pred.values, censored="tied")
        for al, v in r.items():
            per_allele[al] = v
    pa = pd.Series(per_allele)
    foldmed = ens_fold[arm]["tied"].dropna()
    # size-weighted mean of fold medians
    sizes = ens.groupby("fold_name").size()
    wm = np.average(foldmed.values, weights=sizes[foldmed.index].values)
    print(f"{arm:26s} median-over-folds={np.median(foldmed):+.4f}  "
          f"mean-over-folds={foldmed.mean():+.4f}  "
          f"size-weighted-mean-over-folds={wm:+.4f}  "
          f"median-over-ALL-{len(pa)}-alleles={np.median(pa):+.4f}  "
          f"mean-over-alleles={pa.mean():+.4f}")

print("\nDONE")

# =========================================================================
# 10. SINGLETON-FOLD SENSITIVITY: 7 of 21 fold scores ARE one allele's rho,
#     and arm E measured the per-allele null band at +/-0.11.
# =========================================================================
banner("10. DOES THE HEADLINE SURVIVE DROPPING THE 7 SINGLE-ALLELE FOLDS?")
meta = (load_results("A_supervised_nn").drop_duplicates("fold_name")
        .set_index("fold_name")[["n_test", "n_held_out_alleles", "n_alleles_scored"]])
single = meta.index[meta.n_alleles_scored == 1]
multi = meta.index[meta.n_alleles_scored > 1]
for arm in PRED:
    v = ens_fold[arm]["tied"]
    print(f"{arm:26s} all21={np.nanmedian(v):+.4f}  "
          f"7 singleton folds={np.nanmedian(v[single]):+.4f}  "
          f"14 multi-allele folds={np.nanmedian(v[multi]):+.4f}")
print("\nPaired A - FM restricted to the 14 multi-allele folds:")
a = ens_fold["A_supervised_nn"]["tied"]
for arm in ["B_esm_pseudo", "B_esm_pseudo_mlp", "C_esm_joint", "D_peptide_only"]:
    b = ens_fold[arm]["tied"]
    d = (a[multi] - b[multi]).dropna().values
    print(f"  A - {arm:20s} meandiff={d.mean():+.4f} wins={int((d>0).sum())}/{len(d)} "
          f"p={stats.wilcoxon(d).pvalue:.2e}")

banner("10b. IS FOLD DIFFICULTY DRIVEN BY FOLD SIZE? (rho vs n_test, n_alleles)")
for arm in PRED:
    v = ens_fold[arm]["tied"].dropna()
    c1 = stats.spearmanr(meta.loc[v.index, "n_test"], v)
    c2 = stats.spearmanr(meta.loc[v.index, "n_alleles_scored"], v)
    print(f"{arm:26s} rho vs n_test={c1.statistic:+.3f} (p={c1.pvalue:.3f})  "
          f"rho vs n_alleles={c2.statistic:+.3f} (p={c2.pvalue:.3f})")

# =========================================================================
# 11. PER-ALLELE PAIRED COMPARISON (71 alleles, the finer unit)
# =========================================================================
banner("11. PAIRED AT THE ALLELE LEVEL (n=71, ensemble predictions, tied)")
per_allele = {}
for arm, path in PRED.items():
    p = pd.read_parquet(path)
    ens = p.groupby(["fold_name", "row_id"], sort=False).y_pred.mean().reset_index()
    ens = ens.merge(df_all[["HLA", "Thalf", "y", "censored"]],
                    left_on="row_id", right_index=True, how="left")
    acc = {}
    for fname, sub in ens.groupby("fold_name"):
        r = M.spearman_per_allele(sub, sub.y_pred.values, censored="tied")
        acc.update(r.to_dict())
    per_allele[arm] = pd.Series(acc)
PA = pd.DataFrame(per_allele)
print(f"alleles scored in every arm: {PA.dropna().shape[0]}")
base = PA.dropna()
for arm in ["B_esm_pseudo", "B_esm_pseudo_mlp", "C_esm_joint", "D_peptide_only",
            "E_allele_mean__tiebreak"]:
    d = (base["A_supervised_nn"] - base[arm]).values
    print(f"  A - {arm:24s} meandiff={d.mean():+.4f} "
          f"wins={int((d>0).sum())}/{len(d)} p={stats.wilcoxon(d).pvalue:.2e}")
print("\n  FM arms against the measured null, allele level:")
for arm in ["B_esm_pseudo", "B_esm_pseudo_mlp", "C_esm_joint", "D_peptide_only"]:
    d = (base[arm] - base["E_allele_mean__tiebreak"]).values
    print(f"  {arm:24s} - null  meandiff={d.mean():+.4f} "
          f"wins={int((d>0).sum())}/{len(d)} p={stats.wilcoxon(d).pvalue:.2e}")
print("\n  negative alleles per arm (of 71):")
print("  " + "  ".join(f"{a}={int((base[a]<0).sum())}" for a in base.columns))

# =========================================================================
# 12. HEAD-MATCHED CONTROL: is the A-vs-FM gap features, or head+stopping?
#     Same ridge head, three featurisations, same 21 folds, no early stopping.
# =========================================================================
banner("12. HEAD-MATCHED: RIDGE ON BLOSUM+ONEHOT vs RIDGE ON ESM, 21 FOLDS")
import time as _t
import json
import splits as S
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import alleles as AL

BL = {}
try:
    from Bio.Align import substitution_matrices
    _b = substitution_matrices.load("BLOSUM62")
    AAs = "ACDEFGHIKLMNPQRSTVWY"
    for x in AAs:
        BL[x] = np.array([_b[x][y] for y in AAs], dtype=float) / 5.0
except Exception as e:                                   # fall back to arm A's own table
    print("  biopython unavailable, importing arm A's encoder:", e)
import importlib.util
spec = importlib.util.spec_from_file_location("armA", "arm_A_supervised_nn.py")
armA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(armA)
print("  arm A module loaded; encoder fns:",
      [n for n in dir(armA) if "encode" in n.lower() or "featur" in n.lower()])

feat_both = armA.featurizer("both")

Epep, Ipep = np.load("emb_peptides.npy"), json.load(open("emb_peptides_index.json"))
Epse, Ipse = np.load("emb_pseudo.npy"), json.load(open("emb_pseudo_index.json"))
Ejoint, Ijoint = np.load("emb_joint.npy"), json.load(open("emb_joint_index.json"))
pm = armA.pseudo_map()


def F_blosum(sub):
    return feat_both(sub)


def F_esm_pair(sub):
    a = Epep[[Ipep[p] for p in sub.Pep]]
    b = Epse[[Ipse[h] for h in sub.HLA]]
    return np.hstack([a, b])


def F_esm_joint(sub):
    return Ejoint[[Ijoint[f"{h}|{p}"] for h, p in zip(sub.HLA, sub.Pep)]]


folds = S.groove_folds(df_all)
ALPHAS = np.logspace(0, 5, 11)


def run_ridge(F, label):
    out = {}
    t0 = _t.time()
    for fname, alle in folds:
        tr, te = S.split_by_allele(df_all, alle)
        Xtr, Xte = F(df_all.loc[tr]), F(df_all.loc[te])
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        m.fit(Xtr, df_all.loc[tr].y.values)
        p = m.predict(Xte)
        r = M.spearman_per_allele(df_all.loc[te], p, censored="tied")
        out[fname] = np.median(r) if len(r) else np.nan
    s = pd.Series(out)
    print(f"  {label:34s} median={np.nanmedian(s):+.4f} "
          f"IQR=[{np.nanpercentile(s,25):+.4f},{np.nanpercentile(s,75):+.4f}] "
          f"worst={np.nanmin(s):+.4f}  ({_t.time()-t0:.0f}s)")
    return s


try:
    Ijoint_keys = list(Ijoint)[:2]
    print("  joint index key format:", Ijoint_keys)
except Exception as e:
    print(e)

R = {}
R["blosum1720"] = run_ridge(F_blosum, "ridge on BLOSUM+onehot (arm A feats)")
R["esm_pair1280"] = run_ridge(F_esm_pair, "ridge on ESM pep+pseudo (arm B feats)")
R["esm_joint640"] = run_ridge(F_esm_joint, "ridge on ESM joint (arm C feats)")

print("\n  PAIRED, head held constant (ridge), 21 folds:")
for k in ["esm_pair1280", "esm_joint640"]:
    d = (R["blosum1720"] - R[k]).dropna().values
    print(f"    blosum - {k:14s} meandiff={d.mean():+.4f} "
          f"wins={int((d>0).sum())}/{len(d)} p={stats.wilcoxon(d).pvalue:.2e}")
print("\nDONE PART 12")

# =========================================================================
# 13. THE MISSING CELL: ESM features through ARM A's EXACT head and
#     early-stopping protocol. Nobody ran this, and it is the only way to
#     separate "the features are worse" from "the training recipe is better".
# =========================================================================
banner("13. ESM FEATURES THROUGH ARM A's OWN HEAD (same MLP, same rho-stopping)")
SEEDS = (0, 1, 2)


def run_armA_head(F, n_pep_cols, label):
    acc = {}
    t0 = _t.time()
    for fname, alle in folds:
        tr, te = S.split_by_allele(df_all, alle)
        Xtr, Xte = F(df_all.loc[tr]), F(df_all.loc[te])
        ytr = df_all.loc[tr].y.values
        preds = []
        for sd in SEEDS:
            m = armA.TorchMLP(seed=sd, n_pep_cols=n_pep_cols, **armA.BEST)
            m.fit(Xtr, ytr)
            preds.append(m.predict(Xte))
        p = np.mean(preds, axis=0)
        r = M.spearman_per_allele(df_all.loc[te], p, censored="tied")
        acc[fname] = np.median(r) if len(r) else np.nan
    s = pd.Series(acc)
    print(f"  {label:38s} median={np.nanmedian(s):+.4f} "
          f"IQR=[{np.nanpercentile(s,25):+.4f},{np.nanpercentile(s,75):+.4f}] "
          f"worst={np.nanmin(s):+.4f}  ({_t.time()-t0:.0f}s, {len(folds)*len(SEEDS)} fits)")
    return s


H = {}
H["esm_pair"] = run_armA_head(F_esm_pair, 640, "arm A head on ESM pep+pseudo (1280d)")
H["esm_joint_cat"] = run_armA_head(
    lambda sub: np.hstack([F_esm_joint(sub), Epse[[Ipse[h] for h in sub.HLA]]]),
    640, "arm A head on ESM joint+pseudo (1280d)")

# arm A's own 3-seed ensemble on the SAME seeds, for a like-for-like baseline
pA = pd.read_parquet("predictions_A_supervised_nn.parquet")
pA = pA[pA.seed.isin(SEEDS)]
eA = pA.groupby(["fold_name", "row_id"], sort=False).y_pred.mean().reset_index()
eA = eA.merge(df_all[["HLA", "Thalf", "y", "censored"]],
              left_on="row_id", right_index=True, how="left")
acc = {}
for fname, sub in eA.groupby("fold_name"):
    r = M.spearman_per_allele(sub, sub.y_pred.values, censored="tied")
    acc[fname] = np.median(r) if len(r) else np.nan
H["blosum_armA"] = pd.Series(acc)
print(f"  {'arm A head on BLOSUM+onehot (stored)':38s} "
      f"median={np.nanmedian(H['blosum_armA']):+.4f} "
      f"IQR=[{np.nanpercentile(H['blosum_armA'],25):+.4f},"
      f"{np.nanpercentile(H['blosum_armA'],75):+.4f}] "
      f"worst={np.nanmin(H['blosum_armA']):+.4f}  (3-seed ensemble, same seeds)")

print("\n  HEAD HELD CONSTANT at arm A's MLP, paired over 21 folds:")
for k in ["esm_pair", "esm_joint_cat"]:
    d = (H["blosum_armA"] - H[k]).dropna().values
    print(f"    blosum - {k:14s} meandiff={d.mean():+.4f} "
          f"mediandiff={np.median(d):+.4f} wins={int((d>0).sum())}/{len(d)} "
          f"p={stats.wilcoxon(d).pvalue:.2e}")

print("\n  AND: ESM features under arm A's head vs the SAME ESM features under")
print("       the arm's own published head (how much was the recipe worth?):")
for k, pub in [("esm_pair", "B_esm_pseudo_mlp"), ("esm_joint_cat", "C_esm_joint")]:
    b = ens_fold[pub]["tied"]
    common = H[k].dropna().index.intersection(b.dropna().index)
    d = (H[k][common] - b[common]).values
    print(f"    {k:14s} (arm A head) - {pub:18s} meandiff={d.mean():+.4f} "
          f"wins={int((d>0).sum())}/{len(d)} p={stats.wilcoxon(d).pvalue:.2e}")

banner("13b. GAP DECOMPOSITION, all on the same 21 folds, censored='tied'")
print(f"  published A (5-seed ens, BLOSUM, arm A head)      {np.nanmedian(ens_fold['A_supervised_nn']['tied']):+.4f}")
print(f"  A head + BLOSUM, 3 seeds                          {np.nanmedian(H['blosum_armA']):+.4f}")
print(f"  A head + ESM pep|pseudo, 3 seeds                  {np.nanmedian(H['esm_pair']):+.4f}")
print(f"  A head + ESM joint|pseudo, 3 seeds                {np.nanmedian(H['esm_joint_cat']):+.4f}")
print(f"  published B_mlp (sklearn MLP, ESM pep|pseudo)     {np.nanmedian(ens_fold['B_esm_pseudo_mlp']['tied']):+.4f}")
print(f"  published C (sklearn MLP, ESM joint)              {np.nanmedian(ens_fold['C_esm_joint']['tied']):+.4f}")
print(f"  ridge + BLOSUM                                    {np.nanmedian(R['blosum1720']):+.4f}")
print(f"  ridge + ESM pep|pseudo                            {np.nanmedian(R['esm_pair1280']):+.4f}")
print(f"  ridge + ESM joint                                 {np.nanmedian(R['esm_joint640']):+.4f}")
print(f"  measured null (arm E tie-break)                   {np.nanmedian(ens_fold['E_allele_mean__tiebreak']['tied']):+.4f}")
print("\nALL DONE")


# =========================================================================
# 14. THE LADDER NUMBERS vs THE FILES. The summary passed around quotes a
#     number and an "n" per arm. Neither matches the fold unit of analysis.
# =========================================================================
banner("14. DO THE QUOTED LADDER NUMBERS MATCH THE FILES, AND WHAT IS 'n'?")
LADDER = {"A_supervised_nn": (105, +0.277), "G_650m_B_mlp": (43, +0.109),
          "B_esm_pseudo_mlp": (105, +0.096), "G_650m_B_ridge": (105, +0.090),
          "D_peptide_only": (105, +0.074), "C_esm_joint": (210, +0.071),
          "B_esm_pseudo": (105, +0.060),
          "E_allele_mean__tiebreak": (105, +0.005)}
print(f"{'arm':26s} {'quoted':>8s} {'quoted n':>9s} | "
      f"{'rowmed(tied)':>12s} {'foldmed':>8s} {'ens':>8s} | {'rows':>5s} {'folds':>5s} {'policies'}")
for arm, (qn, qv) in LADDER.items():
    raw = pd.read_csv(f"results_{arm}.csv")
    ok = raw[raw.status == "ok"]
    tied = ok[ok.censored_policy == "tied"] if "censored_policy" in ok else ok
    fm = tied.groupby("fold_name").spearman.mean()
    ens_p = f"ensemble_{arm}.csv"
    ensv = np.nan
    if os.path.exists(ens_p):
        e = pd.read_csv(ens_p)
        ensv = np.nanmedian(e[e.status == "ok"].spearman) if "status" in e else np.nanmedian(e.spearman)
    print(f"{arm:26s} {qv:+8.4f} {qn:9d} | {np.median(tied.spearman):+12.4f} "
          f"{np.median(fm):+8.4f} {ensv:+8.4f} | {len(raw):5d} {fm.size:5d} "
          f"{sorted(ok.censored_policy.unique())}")
print("""
  Reading of the above:
  - the quoted value is the median over (fold,seed) ROWS, not over folds. Seeds are
    not independent experiments, so 'n=105' is 21 real units counted five times.
  - C_esm_joint's 'n=210' is 105 tied rows + 105 drop rows in one file. The quoted
    +0.071 is the tied-row median, so the VALUE is fine, but the n is two
    censoring policies stacked and is not comparable to any other arm's n.
  - row-median, fold-median and the 5-seed-ensemble median disagree by up to
    0.04 on the same arm (A: 0.2775 / 0.2651 / 0.3069). Pick one and label it.""")

# =========================================================================
# 15. THE PARTIAL ARM. G_650m_B_mlp is still being written. Is its number
#     quotable? Three separate reasons it is not.
# =========================================================================
banner("15. IS THE PARTIAL G_650m_B_mlp NUMBER SAFE TO QUOTE?")
meta21 = (load_results("A_supervised_nn").drop_duplicates("fold_name")
          .set_index("fold_name")[["n_test", "n_alleles_scored"]])
g = load_results("G_650m_B_mlp")
gf = g.groupby("fold_name").spearman.agg(["mean", "size"])
done, rem = list(gf.index), [f for f in meta21.index if f not in gf.index]

print(f"15a. COVERAGE: {len(g)} of 105 rows, {len(done)} of 21 folds "
      f"({int((gf['size'] == 5).sum())} folds with all 5 seeds).")
print(f"     rows per completed fold: {sorted(gf['size'].tolist())}")

print("\n15b. RUN ORDER IS DESCENDING FOLD SIZE, so the unrun folds are not random:")
print(f"     completed folds n_test: {sorted(meta21.loc[done,'n_test'])}")
print(f"     unrun     folds n_test: {sorted(meta21.loc[rem,'n_test'])}")
print(f"     single-allele folds completed: "
      f"{int((meta21.loc[done,'n_alleles_scored']==1).sum())} of 7; "
      f"unrun: {int((meta21.loc[rem,'n_alleles_scored']==1).sum())} of 7")
print("     -> the easy/large grooves are done and every hard singleton is pending.")

print("\n15c. THE SAME SPLIT, ON ARMS THAT FINISHED: big folds vs small folds")
print(f"     {'arm':24s} {'G-completed folds':>18s} {'unrun folds':>12s} {'all 21':>8s}")
for arm in ["A_supervised_nn", "B_esm_pseudo", "B_esm_pseudo_mlp", "C_esm_joint",
            "D_peptide_only", "G_650m_B_ridge", "E_allele_mean__tiebreak"]:
    f = load_results(arm).groupby("fold_name").spearman.mean()
    print(f"     {arm:24s} {np.median(f[done]):+18.4f} "
          f"{np.median(f[rem]):+12.4f} {np.median(f):+8.4f}")
print("     Both ESM-pseudo arms (150M ridge, 650M ridge) score LOWER on exactly")
print("     the folds G has not reached yet. G_650m_B_mlp shares those features.")

print("\n15d. PROJECTED final value (A PROJECTION, NOT A MEASUREMENT): transfer each")
print("     reference arm's own big-fold -> unrun-fold shift onto G's completed folds.")
gbig = np.median(gf["mean"])
for ref in ["B_esm_pseudo", "G_650m_B_ridge", "B_esm_pseudo_mlp", "C_esm_joint",
            "D_peptide_only", "A_supervised_nn"]:
    r = load_results(ref).groupby("fold_name").spearman.mean()
    proj = np.concatenate([gf["mean"].values, gbig + (r[rem].values - np.median(r[done]))])
    print(f"     from {ref:24s} -> {np.median(proj):+.4f}")
print(f"     G on its completed folds alone: {gbig:+.4f}")
print("     -> the plausible landing zone overlaps B_esm_pseudo_mlp and G_650m_B_ridge")
print("        entirely. 650M + an MLP head buys nothing that can be demonstrated.")

print("\n15e. MEASURED DRIFT of a running median, file order, finished arms:")
for arm in ["G_650m_B_ridge", "B_esm_pseudo_mlp", "A_supervised_nn"]:
    v = load_results(arm).spearman.values
    print(f"     {arm:20s} " + "  ".join(
        f"n={k}:{np.median(v[:k]):+.4f}" for k in (4, 10, 20, 43, 60, 105) if k <= len(v)))
v = g.spearman.values
print(f"     {'G_650m_B_mlp (LIVE)':20s} " + "  ".join(
    f"n={k}:{np.median(v[:k]):+.4f}" for k in (4, 10, 20, 30, 43, len(v))))
print(f"     G_650m_B_ridge swung {0.1338 - 0.0901:+.4f} between n=20 and n=105.")
print("     That single-arm drift is LARGER than the whole spread between the five")
print("     ESM arms (0.060 to 0.106 = 0.046). A partial number cannot rank anything.")

# =========================================================================
# 16. THE ERROR BARS. figure.py takes a per-ALLELE series: 71 alleles, but
#     only 21 independent model fits. Does that overstate precision?
# =========================================================================
banner("16. ERROR BARS: 21 INDEPENDENT FOLDS vs 71 NON-INDEPENDENT ALLELES")
print("figure.py bars median over the series it is handed and whiskers its IQR,")
print("and its docstring says IQR, not SEM. That label is honest. The question is n.")
rng = np.random.default_rng(7)
print(f"\n{'arm':20s} {'unit':8s} {'n':>4s} {'median':>8s} {'IQR halfwidth':>14s} {'boot95 on median':>26s}")
for arm in ["A_supervised_nn", "B_esm_pseudo_mlp", "C_esm_joint", "D_peptide_only"]:
    for unit, v in (("folds", ens_fold[arm]["tied"].dropna().values),
                    ("alleles", PA[arm].dropna().values)):
        b = np.array([np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(20000)])
        print(f"{arm:20s} {unit:8s} {len(v):4d} {np.median(v):+8.4f} "
              f"{(np.percentile(v,75)-np.percentile(v,25))/2:14.4f} "
              f"   [{np.percentile(b,2.5):+.4f},{np.percentile(b,97.5):+.4f}]"
              f"  width {np.percentile(b,97.5)-np.percentile(b,2.5):.4f}")
print("""
  The IQR is WIDER on alleles than on folds, so the whisker does not understate
  spread. But the bootstrap CI narrows on the allele unit, because alleles inside
  one fold share a single fit and a single groove cluster. Quote any interval or
  any n from the 21 folds, never from the 71 alleles.""")

# =========================================================================
# 17. THE ONE-LINE ANSWER TO THE LENS: is the A-vs-ESM gap noise?
# =========================================================================
banner("17. IS THE A-vs-ESM GAP NOISE? EVERY CUT, ONE TABLE")
A5 = fold_table("A_supervised_nn")["mean"]
print(f"seed sd of a per-fold rho: A median {np.nanmedian(fold_table('A_supervised_nn')['sd']):.4f}, "
      f"B_mlp median {np.nanmedian(fold_table('B_esm_pseudo_mlp')['sd']):.4f}")
print(f"A - B_esm_pseudo_mlp fold-mean gap: "
      f"{(A5 - fold_table('B_esm_pseudo_mlp')['mean']).mean():+.4f}")
print("-> the gap is roughly 4x the within-fold seed sd, and seed explains "
      "<2% of variance (sec 3).")
print("\nSurvival table for 'the supervised baseline beats every ESM arm':")
rows = [
    ("paired over 21 folds, tied",        "A - B_mlp +0.1994, 21/21 folds, p=9.5e-07"),
    ("paired over 21 folds, drop",        "A - B_mlp +0.1695, 20/21 folds, p=2.9e-06"),
    ("paired over 71 alleles",            "A - B_mlp +0.1762, 68/71, p=1.2e-11"),
    ("14 multi-allele folds only",         "A - B_mlp +0.2048, 14/14, p=1.2e-04"),
    ("any single seed alone",              "A - B_mlp +0.153 to +0.198, all p<2e-05"),
    ("Holm-corrected across 15 pairs",     "all five A-vs-X comparisons stay significant"),
    ("size-weighted instead of median",    "A +0.3012 vs B_mlp +0.1112"),
]
for k, v in rows:
    print(f"  {k:34s} {v}")
print("""
  VERDICT on the lens: the A-vs-ESM gap is NOT noise. It survives the censoring
  switch, the fold/allele unit switch, the singleton-fold drop, every single seed,
  and Holm correction. The headline direction is safe.

  What is NOT safe, and all of it is reported above:
  - the specific ladder values and their 'n' labels (sec 14)
  - G_650m_B_mlp's partial number, in any form (sec 15)
  - any ranking AMONG the ESM arms: Holm leaves none of them distinguishable
    from each other (sec 8), and their order flips with the censoring policy
    (sec 6). B_esm_pseudo vs the measured null is itself ns after Holm.
  - 'ESM uses the allele': C_esm_joint - D_peptide_only is +0.0018, p=0.95.
    The allele-blind control is statistically indistinguishable from the
    allele-aware ESM arms.""")
print("\nALL DONE (14-17)")
