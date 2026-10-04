"""Adversarial biological-plausibility audit. Read-only: writes nothing but stdout.

Stages (run with: python check_biology.py <stage>, or `all`):
  motif    anchor-residue check on top-predicted peptides
  spec     is the top-10 list allele-specific at all?
  source   source-organism confound
  groove   per-fold performance vs groove similarity to training
  c67s     the three C67S lab constructs
"""
import json
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

import data
import splits
import supertypes
import sources

AA = "ACDEFGHIKLMNPQRSTVWY"

ARMS = {
    "A": "predictions_A_supervised_nn.parquet",
    "B": "predictions_B_esm_pseudo.parquet",
    "Bmlp": "predictions_B_esm_pseudo_mlp.parquet",
    "C": "predictions_C_esm_joint.parquet",
    "D": "predictions_D_peptide_only.parquet",
}


def load_ens(arm):
    """5-seed ensemble mean prediction per (fold, row)."""
    p = pd.read_parquet(ARMS[arm])
    g = p.groupby(["fold_name", "row_id"], as_index=False).agg(
        y_pred=("y_pred", "mean"), HLA=("HLA", "first"),
        Pep=("Pep", "first"), y_true=("y_true", "first"))
    return g


# ---------------------------------------------------------------- 1. motif

def pwm(peps, pseudocount=0.5):
    """9x20 position frequency matrix."""
    m = np.full((9, 20), pseudocount)
    for p in peps:
        for i, c in enumerate(p):
            if c in AA:
                m[i, AA.index(c)] += 1
    return m / m.sum(1, keepdims=True)


def motif_check(arm, alleles, top_frac=0.10, k=20):
    df = data.load()
    ens = load_ens(arm)
    print(f"\n=== MOTIF, arm {arm} (ensemble mean) ===")
    print("For each allele: the motif is derived from that allele's OWN")
    print("high-stability rows (ground truth, top decile by Thalf).")
    print("Then: do the model's top-%d predicted peptides carry those anchors?" % k)
    rows = []
    for al in alleles:
        sub = ens[ens.HLA == al]
        if len(sub) < 50:
            print(f"  {al}: only {len(sub)} test rows, skipped")
            continue
        thr = sub.y_true.quantile(1 - top_frac)
        true_top = sub[sub.y_true >= thr]
        bg = pwm(sub.Pep.tolist())
        fg = pwm(true_top.Pep.tolist())
        # strongest anchor positions = largest KL from the allele's own candidate pool
        kl = (fg * np.log(fg / bg)).sum(1)
        anchors = np.argsort(-kl)[:3]
        picked = sub.nlargest(k, "y_pred")
        print(f"\n  {al}  (n={len(sub)}, true top-decile n={len(true_top)})")
        for pos in sorted(anchors):
            # the residues the TRUE high-stability set prefers at this position
            pref = [AA[j] for j in np.argsort(-fg[pos])[:3]]
            def frq(peps, rs):
                return np.mean([p[pos] in rs for p in peps])
            f_bg = frq(sub.Pep.tolist(), pref)
            f_true = frq(true_top.Pep.tolist(), pref)
            f_pred = frq(picked.Pep.tolist(), pref)
            # one-sided binomial: are the model's picks enriched over background?
            n_hit = int(round(f_pred * k))
            pval = stats.binomtest(n_hit, k, f_bg, alternative="greater").pvalue
            print(f"    P{pos+1}  true-motif prefers {''.join(pref)}  "
                  f"| pool {f_bg:.2f}  true-top {f_true:.2f}  model-top{k} {f_pred:.2f}"
                  f"  (binom p={pval:.3f})")
            rows.append(dict(arm=arm, allele=al, pos=pos + 1, pref="".join(pref),
                             pool=f_bg, true=f_true, pred=f_pred, p=pval))
    return pd.DataFrame(rows)


# ------------------------------------------------- 2. allele-specific picks?

def specificity(arms=("A", "B", "Bmlp", "C", "D")):
    print("\n=== ARE THE TOP-10 PICKS ALLELE-SPECIFIC? ===")
    print("Within one fold, several alleles are held out together and share the")
    print("same candidate peptide pool. If the model has learned allele-specific")
    print("binding, the top-10 should DIFFER between those alleles.")
    print("Measured on peptides each pair of alleles actually has in common.\n")
    for arm in arms:
        ens = load_ens(arm)
        jac, pairs = [], 0
        for fold, g in ens.groupby("fold_name"):
            als = sorted(g.HLA.unique())
            if len(als) < 2:
                continue
            for i in range(len(als)):
                for j in range(i + 1, len(als)):
                    a, b = g[g.HLA == als[i]], g[g.HLA == als[j]]
                    shared = set(a.Pep) & set(b.Pep)
                    if len(shared) < 50:
                        continue
                    ta = set(a[a.Pep.isin(shared)].nlargest(10, "y_pred").Pep)
                    tb = set(b[b.Pep.isin(shared)].nlargest(10, "y_pred").Pep)
                    jac.append(len(ta & tb) / len(ta | tb))
                    pairs += 1
        print(f"  arm {arm:5s}  {pairs:4d} allele pairs  "
              f"median Jaccard of top-10 = {np.median(jac):.3f}  "
              f"(1.000 = identical picks for every allele)")


# ---------------------------------------------------------------- 3. source

def source_confound(arm="A"):
    print(f"\n=== SOURCE-ORGANISM CONFOUND, arm {arm} ===")
    df = data.load()
    src = json.load(open("peptide_sources.json"))

    def org(p):
        hits = src.get(p)
        if not hits:
            return "unmatched"
        c = Counter(h["source"] for h in hits)
        return c.most_common(1)[0][0]

    df["src"] = df.Pep.map(org)
    print("\n  panel composition (rows):")
    print(df.src.value_counts().head(12).to_string())

    # Does the LABEL itself depend on source organism?
    print("\n  measured y (log10 Thalf) by source, all rows:")
    g = df.groupby("src").y.agg(["mean", "count"]).sort_values("mean", ascending=False)
    print(g[g["count"] >= 100].to_string())
    big = [s for s, n in df.src.value_counts().items() if n >= 200]
    k = stats.kruskal(*[df[df.src == s].y.values for s in big])
    print(f"  Kruskal-Wallis across {len(big)} sources with n>=200: "
          f"H={k.statistic:.1f}  p={k.pvalue:.3g}")

    # Within an allele, does the model's prediction depend on source?
    ens = load_ens(arm).merge(df[["Pep", "src"]].drop_duplicates(), on="Pep", how="left")
    print(f"\n  per-allele: variance of model prediction explained by source organism")
    r2s, r2t = [], []
    for al, s in ens.groupby("HLA"):
        if len(s) < 100 or s.src.nunique() < 3:
            continue
        for col, acc in (("y_pred", r2s), ("y_true", r2t)):
            tot = s[col].var()
            if tot == 0:
                continue
            within = s.groupby("src")[col].transform("mean")
            acc.append(1 - (s[col] - within).var() / tot)
    print(f"    R^2(prediction ~ source) median {np.median(r2s):.3f} over {len(r2s)} alleles")
    print(f"    R^2(truth      ~ source) median {np.median(r2t):.3f} over {len(r2t)} alleles")

    # do top-10 picks over-represent an organism?
    print("\n  source of the top-10 predicted peptides, pooled over alleles:")
    tops = ens.groupby("HLA", group_keys=False).apply(
        lambda s: s.nlargest(10, "y_pred"), include_groups=False)
    a = tops.src.value_counts(normalize=True)
    b = ens.src.value_counts(normalize=True)
    comp = pd.DataFrame({"top10": a, "pool": b}).fillna(0)
    comp["ratio"] = comp.top10 / comp.pool
    print(comp.sort_values("ratio", ascending=False).head(8).to_string())


# ---------------------------------------------------------------- 4. groove

def groove_vs_perf():
    print("\n=== DOES PERFORMANCE TRACK GROOVE SIMILARITY TO TRAINING? ===")
    df = data.load()
    ps = {a: v["pseudo"] for a, v in json.load(open("alleles.json")).items()}
    folds = splits.groove_folds(df)
    rows = []
    for name, members in folds:
        train_als = [a for a in ps if a not in members]
        sims = []
        for h in members:
            sims.append(max(sum(x == y for x, y in zip(ps[h], ps[t])) / 34
                            for t in train_als))
        rows.append(dict(fold_name=name, max_id=max(sims), mean_id=float(np.mean(sims)),
                         n_test=int(df.HLA.isin(members).sum())))
    gf = pd.DataFrame(rows)
    for arm, path in [("A", "results_A_supervised_nn.csv"),
                      ("B", "results_B_esm_pseudo.csv"),
                      ("Bmlp", "results_B_esm_pseudo_mlp.csv"),
                      ("C", "results_C_esm_joint.csv"),
                      ("D", "results_D_peptide_only.csv")]:
        r = pd.read_csv(path)
        r = r[r.censored_policy == "tied"] if "censored_policy" in r else r
        r = r[r.status == "ok"]
        per = r.groupby("fold_name").spearman.mean().rename("rho")
        m = gf.merge(per, on="fold_name")
        rp = stats.spearmanr(m.mean_id, m.rho)
        rp2 = stats.spearmanr(m.n_test, m.rho)
        print(f"  arm {arm:5s} n={len(m):2d} folds  "
              f"rho(groove_identity, fold_spearman) = {rp.statistic:+.3f} p={rp.pvalue:.3f}"
              f"   | rho(n_test, fold_spearman) = {rp2.statistic:+.3f} p={rp2.pvalue:.3f}")
    print("\n  per-fold nearest-training-allele groove identity:")
    print(gf.sort_values("mean_id").to_string(index=False))


# ---------------------------------------------------------------- 5. C67S

def c67s():
    print("\n=== THE THREE C67S LAB CONSTRUCTS ===")
    df = data.load()
    al = json.load(open("alleles.json"))
    muts = [a for a, v in al.items() if v.get("mutation")]
    print(f"  constructs: {muts}")
    print(f"  position 67 in PSEUDO_POS? {'YES' if 67 in __import__('alleles').PSEUDO_POS else 'no'}"
          f"  -> index {__import__('alleles').PSEUDO_POS.index(67)} of the 34-mer")
    for a in muts:
        n = int((df.HLA == a).sum())
        print(f"  {a:24s} {n:5d} rows   pseudo[8]={al[a]['pseudo'][8]}")

    # which folds do they land in, and are those folds scored?
    folds = splits.groove_folds(df)
    for name, members in folds:
        hit = [m for m in members if m in muts]
        if hit:
            print(f"  fold {name!r}: {len(members)} alleles, contains {hit}, "
                  f"n_test={int(df.HLA.isin(members).sum())}")
    allf = [m for cl in supertypes.clusters(splits.CUT).values() for m in cl]
    for a in muts:
        infold = any(a in mem for _, mem in folds)
        print(f"  {a}: in a scored fold? {infold}")

    # counterfactual: what if the mutation were reverted (C at 67)?
    print("\n  COUNTERFACTUAL: revert C67S in the pseudo-sequence and re-cluster.")
    ps = {a: v["pseudo"] for a, v in al.items()}
    i67 = __import__('alleles').PSEUDO_POS.index(67)
    wt = dict(ps)
    for a in muts:
        s = list(wt[a]); s[i67] = "C"; wt[a] = "".join(s)
    for tag, p in (("as-built (S67)", ps), ("reverted (C67)", wt)):
        names, m = supertypes.identity_matrix(p)
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        z = linkage(squareform(1 - m, checks=False), method="average")
        lab = fcluster(z, t=1 - splits.CUT, criterion="distance")
        cl = {}
        for n, g in zip(names, lab):
            cl.setdefault(int(g), []).append(n)
        nf = sum(1 for mem in cl.values() if int(df.HLA.isin(mem).sum()) >= 100)
        print(f"    {tag}: {len(cl)} clusters, {nf} scored folds")
        for mem in cl.values():
            if any(a in mem for a in muts):
                print(f"      cluster containing a construct: "
                      f"{[x.replace('HLA-','') for x in mem]}")


# ------------------------------------------------- 6. motif-swap (decisive)

def motif_swap(arm="A", k=10):
    """For each allele, score the model's top-k peptides under that allele's OWN
    ground-truth motif and under every OTHER allele's motif. If the model has
    learned allele-specific immunology, the own-motif score should win."""
    print(f"\n=== MOTIF-SWAP TEST, arm {arm} ===")
    print("Each allele's true motif = PWM of its own top-decile (by Thalf) peptides.")
    print(f"Score the model's top-{k} picks for allele X under X's motif, and under")
    print("every other allele's motif. Rank of the own-motif score, 1 = best.\n")
    ens = load_ens(arm)
    als = [a for a, s in ens.groupby("HLA") if len(s) >= 100]
    mot, bgm = {}, {}
    for a in als:
        s = ens[ens.HLA == a]
        thr = s.y_true.quantile(0.9)
        mot[a] = np.log(pwm(s[s.y_true >= thr].Pep.tolist()))
        bgm[a] = np.log(pwm(s.Pep.tolist()))

    def score(peps, a):
        return float(np.mean([sum(mot[a][i, AA.index(c)] - bgm[a][i, AA.index(c)]
                                  for i, c in enumerate(p) if c in AA) for p in peps]))

    ranks, own_z = [], []
    for a in als:
        picks = ens[ens.HLA == a].nlargest(k, "y_pred").Pep.tolist()
        sc = {b: score(picks, b) for b in als}
        order = sorted(als, key=lambda b: -sc[b])
        r = order.index(a) + 1
        others = [sc[b] for b in als if b != a]
        z = (sc[a] - np.mean(others)) / (np.std(others) + 1e-9)
        ranks.append(r); own_z.append(z)
    ranks = np.array(ranks)
    print(f"  {len(als)} alleles. Own-motif rank: median {np.median(ranks):.0f} of {len(als)}, "
          f"best-of-all in {(ranks == 1).sum()}, top-5 in {(ranks <= 5).sum()}")
    print(f"  Expected by chance: median {(len(als)+1)/2:.0f}, rank-1 in {len(als)/len(als):.0f}, "
          f"top-5 in {5:.0f}")
    print(f"  Own-motif z vs the other alleles' motifs: median {np.median(own_z):+.2f}")
    w = stats.wilcoxon(ranks - (len(als) + 1) / 2, alternative="less")
    print(f"  Wilcoxon own-rank better than chance: p={w.pvalue:.2e}")
    worst = sorted(zip(ranks, als))[-6:]
    print("  worst alleles (own motif ranked low):")
    for r, a in worst:
        print(f"    {a:24s} rank {r} of {len(als)}")


# --------------------------------- 7. memorisation vs chemistry (decisive)

def memorisation(arms=("A", "Bmlp", "C", "D")):
    print("\n=== HOW MUCH IS PEPTIDE MEMORISATION? ===")
    print("Fold unit is a groove cluster, so a test peptide is usually still in")
    print("training paired with OTHER alleles. Split each fold's test rows on that.\n")
    df = data.load()
    folds = splits.groove_folds(df)
    seen_map = {}
    for name, members in folds:
        tr = set(df[~df.HLA.isin(members)].Pep)
        seen_map[name] = tr
    # peptide-mean lookup table reference, same folds
    print(f"  {'arm':6s} {'all':>8s} {'seen-pep':>9s} {'unseen-pep':>11s} {'n folds unseen':>15s}")
    rows = []
    for arm in list(arms) + ["LOOKUP"]:
        if arm == "LOOKUP":
            recs = []
            for name, members in folds:
                tr = df[~df.HLA.isin(members)]
                te = df[df.HLA.isin(members)].copy()
                mu = tr.groupby("Pep").y.mean()
                te["y_pred"] = te.Pep.map(mu).fillna(tr.y.mean())
                te["fold_name"] = name
                recs.append(te[["fold_name", "HLA", "Pep", "y_pred"]].assign(
                    y_true=te.y.values))
            ens = pd.concat(recs)
        else:
            ens = load_ens(arm)
        a_all, a_seen, a_uns, nf = [], [], [], 0
        for name, g in ens.groupby("fold_name"):
            tr = seen_map[name]
            g = g.copy(); g["seen"] = g.Pep.isin(tr)
            sub = df.loc[:, ["HLA", "Pep", "Thalf"]]
            gm = g.merge(sub.drop_duplicates(["HLA", "Pep"]), on=["HLA", "Pep"], how="left")
            def med(h):
                v = []
                for al, s in h.groupby("HLA"):
                    if len(s) >= 20 and s.Thalf.nunique() > 1 and s.y_pred.nunique() > 1:
                        v.append(stats.spearmanr(s.y_pred, s.Thalf).statistic)
                return np.median(v) if v else np.nan
            a_all.append(med(gm))
            a_seen.append(med(gm[gm.seen]))
            u = med(gm[~gm.seen])
            if not np.isnan(u):
                a_uns.append(u); nf += 1
        rows.append((arm, np.nanmedian(a_all), np.nanmedian(a_seen), np.median(a_uns), nf))
        print(f"  {arm:6s} {rows[-1][1]:8.3f} {rows[-1][2]:9.3f} {rows[-1][3]:11.3f} {nf:15d}")
    print("\n  (seen = that exact 9-mer appears in the fold's TRAIN side with a")
    print("   different allele; unseen = genuinely novel peptide)")


# -------------------------------------- 8. allele-level groove correlation

def groove_allele(arms=("A", "Bmlp", "C", "D")):
    print("\n=== GROOVE SIMILARITY vs PERFORMANCE, PER ALLELE (71 points, more power) ===")
    df = data.load()
    ps = {a: v["pseudo"] for a, v in json.load(open("alleles.json")).items()}
    folds = dict(splits.groove_folds(df))
    for arm in arms:
        ens = load_ens(arm)
        xs, ys = [], []
        for name, g in ens.groupby("fold_name"):
            members = folds[name]
            tr = [a for a in ps if a not in members]
            for al, s in g.groupby("HLA"):
                if len(s) < 20 or s.y_pred.nunique() < 2:
                    continue
                sub = df[df.HLA == al].set_index("Pep").Thalf
                t = s.Pep.map(sub).values
                if np.ptp(t) == 0:
                    continue
                xs.append(max(sum(x == y for x, y in zip(ps[al], ps[b])) / 34 for b in tr))
                ys.append(stats.spearmanr(s.y_pred.values, t).statistic)
        r = stats.spearmanr(xs, ys)
        pr = stats.pearsonr(xs, ys)
        print(f"  arm {arm:5s} n={len(xs):3d} alleles  spearman {r.statistic:+.3f} p={r.pvalue:.4f}"
              f"   pearson {pr.statistic:+.3f} p={pr.pvalue:.4f}")
    print("\n  Expectation: STRONGLY POSITIVE. A held-out allele whose groove is")
    print("  nearly identical to a training allele should be easy.")


# ------------------------- 9. motif-swap v2: common background + ceiling

def motif_swap2(arm="A", k=10, seed=0):
    """Fixed version of motif_swap.

    Two defects in v1: (a) each allele's log-odds used its OWN candidate pool as
    the background, so scores were not comparable across alleles; (b) no ceiling,
    so a bad rank could just mean the motif representation is weak.
    Here: ONE common background (all 5,633 peptides), and a positive control that
    builds each motif from half the allele's true top-decile and scores the other
    half -- the best any peptide-picker could do under this scoring.
    """
    print(f"\n=== MOTIF-SWAP v2, arm {arm} (common background + ceiling) ===")
    rng = np.random.default_rng(seed)
    df = data.load()
    ens = load_ens(arm)
    bg = np.log(pwm(sorted(df.Pep.unique())))
    als = [a for a, s in ens.groupby("HLA") if len(s) >= 100]
    mot, held = {}, {}
    for a in als:
        s = ens[ens.HLA == a]
        top = s[s.y_true >= s.y_true.quantile(0.9)].Pep.tolist()
        rng.shuffle(top)
        h = max(5, len(top) // 2)
        mot[a] = np.log(pwm(top[h:])) - bg      # motif from half
        held[a] = top[:h]                        # the other half, for the ceiling

    def score(peps, a):
        return float(np.mean([sum(mot[a][i, AA.index(c)] for i, c in enumerate(p)
                                  if c in AA) for p in peps]))

    def ranks_for(getpicks, tag):
        rs, zs = [], []
        for a in als:
            picks = getpicks(a)
            if not picks:
                continue
            sc = {b: score(picks, b) for b in als}
            rs.append(sorted(als, key=lambda b: -sc[b]).index(a) + 1)
            o = [sc[b] for b in als if b != a]
            zs.append((sc[a] - np.mean(o)) / (np.std(o) + 1e-9))
        rs = np.array(rs)
        w = stats.wilcoxon(rs - (len(als) + 1) / 2, alternative="less")
        print(f"  {tag:34s} n={len(rs):3d}  own-motif rank median {np.median(rs):5.1f}"
              f" of {len(als)}  rank-1 {(rs==1).sum():3d}  top-5 {(rs<=5).sum():3d}"
              f"  z {np.median(zs):+6.2f}  p(better than chance)={w.pvalue:.2e}")
        return rs

    print(f"  chance: median rank {(len(als)+1)/2:.1f}, rank-1 in ~1, top-5 in ~5\n")
    ranks_for(lambda a: held[a][:k], "CEILING: true top-decile peptides")
    ranks_for(lambda a: list(rng.choice(ens[ens.HLA == a].Pep.unique(),
                                        min(k, ens[ens.HLA == a].Pep.nunique()),
                                        replace=False)), "FLOOR: random peptides from pool")
    r = ranks_for(lambda a: ens[ens.HLA == a].nlargest(k, "y_pred").Pep.tolist(),
                  f"arm {arm}: model's top-{k} picks")
    for other in ("Bmlp", "C", "D"):
        e2 = load_ens(other)
        ranks_for(lambda a, e=e2: e[e.HLA == a].nlargest(k, "y_pred").Pep.tolist(),
                  f"arm {other}: model's top-{k} picks")
    return r


# --------------------- 10. canonical P2/P9 anchors, stated explicitly

def anchors(arm="A", k=10):
    """P2 and P9 (the two canonical class-I anchor positions) for every allele."""
    print(f"\n=== CANONICAL P2 / P9 ANCHOR ENRICHMENT, arm {arm} ===")
    print("Preferred residues at P2 and P9 are read off each allele's OWN")
    print("top-decile peptides (data-derived, not from memory). Then: what")
    print("fraction of the model's top-%d picks carry them, vs the pool?\n" % k)
    ens = load_ens(arm)
    rows = []
    for a, s in ens.groupby("HLA"):
        if len(s) < 100:
            continue
        top = s[s.y_true >= s.y_true.quantile(0.9)]
        picks = s.nlargest(k, "y_pred").Pep.tolist()
        rec = {"allele": a.replace("HLA-", ""), "n": len(s)}
        for pos, tag in ((1, "P2"), (8, "P9")):
            fg = pwm(top.Pep.tolist())[pos]
            pref = {AA[j] for j in np.argsort(-fg)[:2]}
            rec[tag + "_pref"] = "".join(sorted(pref))
            rec[tag + "_pool"] = np.mean([p[pos] in pref for p in s.Pep])
            rec[tag + "_top"] = np.mean([p[pos] in pref for p in top.Pep])
            rec[tag + "_mdl"] = np.mean([p[pos] in pref for p in picks])
        rows.append(rec)
    t = pd.DataFrame(rows)
    for tag in ("P2", "P9"):
        d = t[tag + "_mdl"] - t[tag + "_pool"]
        w = stats.wilcoxon(d, alternative="greater")
        print(f"  {tag}: model-top{k} enrichment over pool, median {d.median():+.3f}, "
              f"positive in {(d > 0).sum()}/{len(d)} alleles, Wilcoxon p={w.pvalue:.4f}")
        dt = t[tag + "_top"] - t[tag + "_pool"]
        print(f"      (ceiling: TRUE top-decile enrichment median {dt.median():+.3f})")
    print("\n  sanity: the data-derived anchors for textbook alleles")
    for a in ("A*02:01", "A*03:01", "B*07:02", "B*27:05", "B*57:01", "B*44:05", "A*01:01"):
        r = t[t.allele == a]
        if len(r):
            r = r.iloc[0]
            print(f"    {a:9s} P2 {r.P2_pref:3s} (pool {r.P2_pool:.2f} top {r.P2_top:.2f} "
                  f"model {r.P2_mdl:.2f}) | P9 {r.P9_pref:3s} (pool {r.P9_pool:.2f} "
                  f"top {r.P9_top:.2f} model {r.P9_mdl:.2f})")
    return t


# ------------- 11. paired: model picks vs random draws from the SAME pool

def paired_floor(arms=("A", "Bmlp", "C", "D"), k=10, reps=200, seed=0):
    """Per allele, compare the motif-typicality of the model's top-k picks with
    `reps` random draws of k peptides from that allele's OWN candidate pool.
    This removes the pool-composition confound that makes the swap test look
    good for every arm, including the allele-blind one."""
    print("\n=== PAIRED: MODEL PICKS vs RANDOM DRAWS FROM THE SAME POOL ===")
    print("Motif = each allele's own top-decile PWM, log-odds vs the global PWM.")
    print("Reported: percentile of the model's top-%d among %d random draws of %d.\n"
          % (k, reps, k))
    rng = np.random.default_rng(seed)
    df = data.load()
    bg = np.log(pwm(sorted(df.Pep.unique())))
    base = load_ens("A")
    als = [a for a, s in base.groupby("HLA") if len(s) >= 100]
    mot = {}
    for a in als:
        s = base[base.HLA == a]
        mot[a] = np.log(pwm(s[s.y_true >= s.y_true.quantile(0.9)].Pep.tolist())) - bg

    def sc(peps, a):
        return float(np.mean([sum(mot[a][i, AA.index(c)] for i, c in enumerate(p)
                                  if c in AA) for p in peps]))

    # true top-decile percentile = the ceiling, same scale
    cel = []
    for a in als:
        s = base[base.HLA == a]
        pool = s.Pep.unique()
        null = np.array([sc(rng.choice(pool, k, replace=False), a) for _ in range(reps)])
        top = s.nlargest(k, "y_true").Pep.tolist()
        cel.append((null < sc(top, a)).mean())
    print(f"  CEILING true top-{k} by Thalf : median percentile {np.median(cel)*100:5.1f}%"
          f"   above 50% in {sum(c > .5 for c in cel)}/{len(cel)} alleles")

    for arm in arms:
        ens = load_ens(arm)
        pct = []
        for a in als:
            s = ens[ens.HLA == a]
            pool = s.Pep.unique()
            null = np.array([sc(rng.choice(pool, k, replace=False), a)
                             for _ in range(reps)])
            pct.append((null < sc(s.nlargest(k, "y_pred").Pep.tolist(), a)).mean())
        pct = np.array(pct)
        w = stats.wilcoxon(pct - 0.5)
        print(f"  arm {arm:5s} model top-{k}        : median percentile {np.median(pct)*100:5.1f}%"
              f"   above 50% in {(pct > .5).sum()}/{len(pct)} alleles   Wilcoxon vs 50% p={w.pvalue:.4f}")
    print("\n  50% = the model's picks are no more motif-typical for this allele")
    print("  than peptides drawn at random from the very same candidate pool.")


# ---- 12. paired floor, de-circularised: motif excludes the arm's own picks

def paired_floor2(arms=("A", "Bmlp", "C", "D"), k=10, reps=200, seed=0):
    print("\n=== PAIRED FLOOR, DE-CIRCULARISED ===")
    print("Same as the paired floor, but for each (allele, arm) the motif is built")
    print("from the allele's top decile MINUS that arm's own top-%d picks, so a" % k)
    print("correct pick can no longer have contributed to the motif it is scored on.\n")
    rng = np.random.default_rng(seed)
    df = data.load()
    bg = np.log(pwm(sorted(df.Pep.unique())))
    base = load_ens("A")
    als = [a for a, s in base.groupby("HLA") if len(s) >= 100]
    for arm in arms:
        ens = load_ens(arm)
        pct = []
        for a in als:
            s = ens[ens.HLA == a]
            picks = s.nlargest(k, "y_pred").Pep.tolist()
            top = s[s.y_true >= s.y_true.quantile(0.9)]
            top = top[~top.Pep.isin(picks)].Pep.tolist()
            if len(top) < 10:
                continue
            m = np.log(pwm(top)) - bg
            f = lambda ps: float(np.mean([sum(m[i, AA.index(c)] for i, c in enumerate(p)
                                              if c in AA) for p in ps]))
            pool = s[~s.Pep.isin(picks)].Pep.unique()
            null = np.array([f(rng.choice(pool, k, replace=False)) for _ in range(reps)])
            pct.append((null < f(picks)).mean())
        pct = np.array(pct)
        w = stats.wilcoxon(pct - 0.5)
        print(f"  arm {arm:5s} n={len(pct):3d}  median percentile {np.median(pct)*100:5.1f}%"
              f"   above 50% in {(pct > .5).sum():2d}/{len(pct)}   Wilcoxon vs 50% p={w.pvalue:.5f}")


# ------------------------------- 13. verify the reported headline numbers

def verify_headline():
    print("\n=== RE-DERIVING THE REPORTED HEADLINE NUMBERS FROM THE FILES ===")
    for arm, path in [("A", "results_A_supervised_nn.csv"),
                      ("B", "results_B_esm_pseudo.csv"),
                      ("Bmlp", "results_B_esm_pseudo_mlp.csv"),
                      ("C", "results_C_esm_joint.csv"),
                      ("D", "results_D_peptide_only.csv")]:
        r = pd.read_csv(path)
        if "censored_policy" in r:
            r = r[r.censored_policy == "tied"]
        ok = r[r.status == "ok"]
        perfold = ok.groupby("fold_name").spearman.mean()
        print(f"  arm {arm:5s} rows={len(r):3d} ok={len(ok):3d}  "
              f"median-over-folds-of-seed-mean {perfold.median():.4f}  "
              f"median-over-all-rows {ok.spearman.median():.4f}  "
              f"top10 median {ok.groupby('fold_name').top10_precision.mean().median():.4f}")
    # top-10 significance vs chance for arm A ensemble
    print("\n  arm A ensemble, per-allele top-10 precision vs chance 0.10:")
    ens = load_ens("A")
    df = data.load()
    vals = []
    for a, s in ens.groupby("HLA"):
        if len(s) < 20:
            continue
        thr = s.y_true.quantile(0.9)
        tt = (s.y_true >= thr).values
        if tt.all():
            continue
        vals.append(tt[np.argsort(-s.y_pred.values, kind="stable")[:10]].mean())
    vals = np.array(vals)
    w = stats.wilcoxon(vals - 0.10, alternative="greater")
    print(f"    n={len(vals)} alleles, median {np.median(vals):.3f}, mean {vals.mean():.3f}, "
          f"at or below chance in {(vals <= 0.10).sum()}, Wilcoxon p={w.pvalue:.2e}")


# ---------- 14. clean split-half motif test (no circularity, no anti-bias)

def split_half_motif(arms=("A", "Bmlp", "C", "D"), k=10, reps=200, nsplit=20, seed=0):
    """The unbiased version.

    Split each allele's test rows at random into halves. Build the motif from
    half 1's top decile. Take the model's top-k from half 2 ONLY, and compare
    with random draws of k from half 2. The motif can never have seen the
    peptides being scored, and nothing is removed from the model's side.
    Averaged over `nsplit` random splits.
    """
    print("\n=== SPLIT-HALF MOTIF TEST (unbiased) ===")
    print("Motif from half 1's top decile; model picks and the random null both")
    print("drawn from half 2. %d splits x %d null draws.\n" % (nsplit, reps))
    rng = np.random.default_rng(seed)
    df = data.load()
    bg = np.log(pwm(sorted(df.Pep.unique())))
    base = load_ens("A")
    als = [a for a, s in base.groupby("HLA") if len(s) >= 100]
    ens = {a: load_ens(a) for a in arms}
    res = {a: [] for a in arms}
    cel = []
    for al in als:
        s0 = base[base.HLA == al].reset_index(drop=True)
        n = len(s0)
        for _ in range(nsplit):
            perm = rng.permutation(n)
            h1, h2 = set(s0.Pep.values[perm[:n // 2]]), s0.Pep.values[perm[n // 2:]]
            a1 = s0[s0.Pep.isin(h1)]
            top1 = a1[a1.y_true >= a1.y_true.quantile(0.9)].Pep.tolist()
            if len(top1) < 10 or len(h2) < 3 * k:
                continue
            m = np.log(pwm(top1)) - bg
            f = lambda ps: float(np.mean([sum(m[i, AA.index(c)] for i, c in enumerate(p)
                                              if c in AA) for p in ps]))
            h2s = set(h2)
            null = np.array([f(rng.choice(list(h2s), k, replace=False)) for _ in range(reps)])
            b2 = s0[s0.Pep.isin(h2s)]
            cel.append((null < f(b2.nlargest(k, "y_true").Pep.tolist())).mean())
            for arm in arms:
                e = ens[arm]
                e2 = e[(e.HLA == al) & (e.Pep.isin(h2s))]
                if len(e2) < k:
                    continue
                res[arm].append((null < f(e2.nlargest(k, "y_pred").Pep.tolist())).mean())
    print(f"  CEILING true top-{k}: median percentile {np.median(cel)*100:5.1f}%")
    for arm in arms:
        v = np.array(res[arm])
        w = stats.wilcoxon(v - 0.5)
        print(f"  arm {arm:5s} n={len(v):5d} (allele x split)  median percentile "
              f"{np.median(v)*100:5.1f}%   above 50% in {(v > .5).mean()*100:4.1f}%"
              f"   Wilcoxon vs 50% p={w.pvalue:.2e}  [NOT independent, see below]")
    # Correct inference: collapse to one number per allele (68 independent points)
    print("\n  --- same thing with the 20 splits averaged per allele (n=68 independent) ---")
    per = {arm: np.array([np.mean(res[arm][i * nsplit:(i + 1) * nsplit])
                          for i in range(len(res[arm]) // nsplit)]) for arm in arms}
    for arm in arms:
        v = per[arm]
        w = stats.wilcoxon(v - 0.5)
        print(f"  arm {arm:5s} n={len(v):3d} alleles  mean percentile {v.mean()*100:5.1f}%"
              f"   above 50% in {(v > .5).sum():2d}/{len(v)}   Wilcoxon vs 50% p={w.pvalue:.2e}")
    # The decisive comparison: arm A vs the ALLELE-BLIND arm D, paired per allele
    print("\n  --- DECISIVE: paired vs the allele-blind null (arm D) ---")
    print("  arm D cannot see the allele at all, so whatever it scores is the")
    print("  pool-composition + generic-stickiness floor, not allele knowledge.")
    for arm in arms:
        if arm == "D":
            continue
        n = min(len(per[arm]), len(per["D"]))
        d = per[arm][:n] - per["D"][:n]
        w = stats.wilcoxon(d)
        print(f"  arm {arm:5s} minus arm D: mean {d.mean()*100:+5.1f} pts  "
              f"positive in {(d > 0).sum():2d}/{n}   Wilcoxon p={w.pvalue:.4f}")


# --------------------------------------------- 15. loose ends / cross-checks

def loose_ends():
    df = data.load()
    print("\n=== PEPTIDE-LEVEL SOURCE COUNTS (cross-check the brief's figures) ===")
    src = json.load(open("peptide_sources.json"))
    peps = sorted(df.Pep.unique())
    c = Counter()
    for p in peps:
        for s in {h["source"] for h in src.get(p, [])}:
            c[s] += 1
    traceable = sum(1 for p in peps if src.get(p))
    print(f"  distinct peptides {len(peps)}, traceable {traceable} "
          f"({100*traceable/len(peps):.1f}%)")
    for k, v in c.most_common():
        print(f"    {k:14s} {v:5d}")

    print("\n=== THE C67S FOLD's PERFORMANCE ===")
    for arm, path in [("A", "results_A_supervised_nn.csv"), ("C", "results_C_esm_joint.csv")]:
        r = pd.read_csv(path)
        if "censored_policy" in r:
            r = r[r.censored_policy == "tied"]
        pf = r[r.status == "ok"].groupby("fold_name").spearman.mean().sort_values()
        rank = list(pf.index).index("B*14:01(C67S) +5") + 1
        print(f"  arm {arm}: fold 'B*14:01(C67S) +5' rho={pf['B*14:01(C67S) +5']:.3f}, "
              f"rank {rank} of {len(pf)} (1 = worst)")
    print("  constructs are NOT excluded anywhere: 1135 of 28166 rows (4.0%),")
    print("  3 of the 71 scored alleles, inside a 6-allele fold with 3 WT members.")

    print("\n=== PER-ALLELE: arm A rho vs that allele's censored fraction ===")
    ens = load_ens("A")
    xs, ys, ns = [], [], []
    for a, s in ens.groupby("HLA"):
        if len(s) < 20:
            continue
        sub = df[df.HLA == a]
        t = s.Pep.map(sub.set_index("Pep").Thalf).values
        if np.ptp(t) == 0:
            continue
        xs.append(sub.censored.mean())
        ys.append(stats.spearmanr(s.y_pred.values, t).statistic)
        ns.append(len(s))
    r = stats.spearmanr(xs, ys)
    print(f"  n={len(xs)}  rho(censored_fraction, per-allele rho) = {r.statistic:+.3f} "
          f"p={r.pvalue:.4f}")
    r2 = stats.spearmanr(ns, ys)
    print(f"  n={len(xs)}  rho(n_test_rows,      per-allele rho) = {r2.statistic:+.3f} "
          f"p={r2.pvalue:.4f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("loose", "all"):
        loose_ends()
    if which in ("sh", "all"):
        split_half_motif()
    if which in ("floor", "all"):
        paired_floor()
    if which in ("floor2", "all"):
        paired_floor2()
    if which in ("verify", "all"):
        verify_headline()
    if which in ("swap", "all"):
        motif_swap("A")
    if which in ("swap2", "all"):
        motif_swap2("A")
    if which in ("anchors", "all"):
        anchors("A")
    if which in ("mem", "all"):
        memorisation()
    if which in ("ga", "all"):
        groove_allele()
    if which in ("motif", "all"):
        motif_check("A", ["HLA-A*02:01", "HLA-B*07:02", "HLA-B*27:05",
                          "HLA-A*03:01", "HLA-B*57:01", "HLA-A*01:01"])
    if which in ("spec", "all"):
        specificity()
    if which in ("source", "all"):
        source_confound("A")
    if which in ("groove", "all"):
        groove_vs_perf()
    if which in ("c67s", "all"):
        c67s()


# ---------------------------------------------------------------------------
# EXTENSION (adversarial biology pass 2). Appended, nothing above modified.
# Stages: elig  blind  copy
# ---------------------------------------------------------------------------

EXT_ARMS = {
    "A": "predictions_A_supervised_nn.parquet",
    "B_ridge": "predictions_B_esm_pseudo.parquet",
    "B_mlp": "predictions_B_esm_pseudo_mlp.parquet",
    "C": "predictions_C_esm_joint.parquet",
    "G650_ridge": "predictions_G_650m_B_ridge.parquet",
    "D": "predictions_D_peptide_only.parquet",
}


def _per_allele_rho(path, df=None):
    """(fold, HLA) -> Spearman of the 5-seed ensemble mean vs Thalf."""
    df = data.load() if df is None else df
    thal = df.set_index(["HLA", "Pep"]).Thalf
    p = pd.read_parquet(path)
    ens = p.groupby(["fold_name", "row_id"], as_index=False).agg(
        y_pred=("y_pred", "mean"), HLA=("HLA", "first"), Pep=("Pep", "first"))
    rows = []
    for (f, a), s in ens.groupby(["fold_name", "HLA"]):
        if len(s) < 20:   # same floor as metrics.MIN_N
            continue
        t = np.asarray(s.set_index(["HLA", "Pep"]).index.map(thal), dtype=float)
        if np.ptp(t) == 0 or s.y_pred.nunique() < 2:
            continue
        rows.append(dict(fold=f, HLA=a, rho=stats.spearmanr(s.y_pred.values, t).statistic))
    return pd.DataFrame(rows).set_index(["fold", "HLA"]).rho


def elig_sensitivity():
    """brief.html says 'Usable for leave-one-allele-out: 61 of 75' and lists the
    14 exclusions (7 >50% censored, 7 too small, 3 C67S constructs). data.py's
    docstring says they are 'excluded from the headline'. They are NOT: nothing
    in splits.groove_folds or run_experiment calls data.loao_eligible, so all 71
    scorable alleles enter every per-fold median. This measures what the stated
    exclusion would have cost."""
    print("\n=== STATED EXCLUSIONS vs WHAT THE CODE DOES ===")
    df = data.load()
    elig = set(data.loao_eligible(df))
    muts = {a for a, v in json.load(open("alleles.json")).items() if v.get("mutation")}
    print(f"  loao_eligible(): {len(elig)} of {df.HLA.nunique()} alleles")
    print("  call sites of loao_eligible: metrics._verify (synthetic self-test),")
    print("  supertypes.__main__ (printing), figure.py (demo). NOT splits/run_experiment.")
    print(f"  C67S constructs: {sorted(a.replace('HLA-','') for a in muts)}")
    print(f"  construct rows: {int(df.HLA.isin(muts).sum())} of {len(df)} "
          f"({100*df.HLA.isin(muts).mean():.1f}%); they are 75-92% censored.\n")
    print(f"  {'arm':12s} {'as-run':>8s} {'-ineligible':>12s} {'-C67S only':>11s}  "
          f"{'n_al':>5s} {'n_al_el':>8s}")
    for arm, path in EXT_ARMS.items():
        try:
            r = _per_allele_rho(path, df)
        except Exception as e:
            print(f"  {arm:12s} unreadable ({type(e).__name__}) - still being written?")
            continue
        al = r.index.get_level_values(1)
        def med(mask):
            s = r[mask]
            return s.groupby(level=0).median().median() if len(s) else np.nan
        print(f"  {arm:12s} {med(np.ones(len(r), bool)):8.3f} "
              f"{med(al.isin(elig)):12.3f} {med(~al.isin(muts)):11.3f}  "
              f"{al.nunique():5d} {al[al.isin(elig)].nunique():8d}")
    print("\n  Direction matters: the stated exclusion would LOWER the ESM arms more")
    print("  than arm A, so running without it is conservative for the negative")
    print("  headline. The brief's sentence is still wrong as written.")


def blind_paired():
    """Is any foundation-model arm distinguishable from the ALLELE-BLIND control?
    Arm D sees no allele information at all. Pair per allele AND per fold: alleles
    inside one groove cluster are not independent, so the per-allele p is
    pseudo-replicated and the fold-level p is the honest one."""
    print("\n=== EVERY ARM MINUS THE ALLELE-BLIND CONTROL (arm D) ===")
    df = data.load()
    R = {}
    for arm, path in EXT_ARMS.items():
        try:
            R[arm] = _per_allele_rho(path, df)
        except Exception:
            print(f"  {arm}: unreadable, skipped")
    D = R["D"]
    for arm, v in R.items():
        if arm == "D":
            continue
        d = (v - D).dropna()
        fd = d.groupby(level=0).median()
        wa = stats.wilcoxon(d.values)
        wf = stats.wilcoxon(fd.values)
        print(f"  {arm:12s} per-allele mean {d.mean():+.3f} (n={len(d)}, "
              f"p={wa.pvalue:.4f}, PSEUDO-REPLICATED)   | per-fold median "
              f"{fd.median():+.3f} (n={len(fd)}, p={wf.pvalue:.4f})")
    print("\n  The per-fold column is the one to quote. On it, no ESM arm separates")
    print("  from an allele-blind predictor.")


def copy_nearest(arm="A", k=10, reps=200, seed=0):
    """Generalisation or interpolation? Score each held-out allele's top-k picks
    under its OWN top-decile motif and under the motif of its nearest TRAINING
    allele by groove identity. If the model is copying the nearest training
    groove, the neighbour's motif should fit the picks at least as well."""
    print(f"\n=== OWN MOTIF vs NEAREST TRAINING ALLELE'S MOTIF, arm {arm} ===")
    rng = np.random.default_rng(seed)
    df = data.load()
    ps = {a: v["pseudo"] for a, v in json.load(open("alleles.json")).items()}
    folds = dict(splits.groove_folds(df))
    bg = np.log(pwm(sorted(df.Pep.unique())))
    # motif per allele from ALL its rows' top decile (train-side alleles included)
    mot = {}
    for a, s in df.groupby("HLA"):
        top = s[s.Thalf >= s.Thalf.quantile(0.9)]
        if len(top) >= 10:
            mot[a] = np.log(pwm(top.Pep.tolist())) - bg
    ens = load_ens(arm) if arm in ARMS else None
    p = pd.read_parquet(EXT_ARMS[arm])
    ens = p.groupby(["fold_name", "row_id"], as_index=False).agg(
        y_pred=("y_pred", "mean"), HLA=("HLA", "first"), Pep=("Pep", "first"))

    def sc(peps, m):
        return float(np.mean([sum(m[i, AA.index(c)] for i, c in enumerate(q) if c in AA)
                              for q in peps]))

    thal = df.set_index(["HLA", "Pep"]).Thalf
    own, nb, cown, cnb, ident = [], [], [], [], []
    for (f, a), s in ens.groupby(["fold_name", "HLA"]):
        if len(s) < 100 or a not in mot:
            continue
        tr = [b for b in ps if b not in folds[f] and b in mot]
        best = max(tr, key=lambda b: sum(x == y for x, y in zip(ps[a], ps[b])))
        s = s.copy()
        s["Thalf"] = np.asarray(s.set_index(["HLA", "Pep"]).index.map(thal), dtype=float)
        picks = s.nlargest(k, "y_pred").Pep.tolist()
        truth = s.nlargest(k, "Thalf").Pep.tolist()   # CEILING, same motifs
        pool = s.Pep.unique()
        for m, ap, ac in ((mot[a], own, cown), (mot[best], nb, cnb)):
            null = np.array([sc(rng.choice(pool, k, replace=False), m) for _ in range(reps)])
            ap.append((null < sc(picks, m)).mean())
            ac.append((null < sc(truth, m)).mean())
        ident.append(sum(x == y for x, y in zip(ps[a], ps[best])) / 34)
    own, nb, cown, cnb = map(np.array, (own, nb, cown, cnb))
    w = stats.wilcoxon(own - nb)
    wc = stats.wilcoxon(cown - cnb)
    print(f"  n={len(own)} held-out alleles, mean groove identity to the nearest")
    print(f"  TRAINING allele {np.mean(ident):.3f}")
    print(f"  percentile of the top-{k} vs {reps} same-pool random draws:")
    print(f"    {'':26s} {'OWN motif':>10s} {'NEIGHBOUR motif':>16s}  own>nb   p")
    print(f"    {'CEILING true top-'+str(k)+' by Thalf':26s} {np.median(cown)*100:9.1f}% "
          f"{np.median(cnb)*100:15.1f}%  {int((cown > cnb).sum()):3d}/{len(cown)}  "
          f"{wc.pvalue:.4f}")
    print(f"    {'arm '+arm+' top-'+str(k)+' picks':26s} {np.median(own)*100:9.1f}% "
          f"{np.median(nb)*100:15.1f}%  {int((own > nb).sum()):3d}/{len(own)}  "
          f"{w.pvalue:.4f}")
    print("\n  The CEILING row is the control: if the TRUE best peptides also prefer")
    print("  the neighbour's motif, the gap is a motif-estimation artefact, not the")
    print("  model copying. If only the model's row prefers the neighbour, it is.")
    print("\n  If own >> neighbour, the picks are allele-specific. If they are equal,")
    print("  the model cannot be shown to do more than reuse the nearest known groove.")


if __name__ == "__main__":
    # second dispatcher: the extension stages, defined after the original
    # __main__ block above, so they need their own entry point.
    _w = sys.argv[1] if len(sys.argv) > 1 else "all"
    if _w in ("elig", "all"):
        elig_sensitivity()
    if _w in ("blind", "all"):
        blind_paired()
    if _w in ("copy", "all"):
        copy_nearest("A")
