"""Does arm M leak? Seven checks, each one a thing that could have gone wrong.

A previous audit of this project found that 80.9% of test rows share their peptide with
a training row (paired against a different allele), and that this overlap carries
essentially all of the weaker arms' apparent signal. That overlap is a property of the
dataset and arm A sits on it too, so it is not something arm M introduces. What arm M
COULD introduce, and what this file is here to rule out, is:

  1  a pocket specification fitted with the held-out alleles' rows in it
  2  the two bookkeeping columns (allele id, censored flag) acting as features
  3  the censored flag reaching the model at prediction time
  4  a different fold set from splits.choose_held_out
  5  features that differ from arm A's when the interaction block is switched off
  6  real-looking scores from a model trained on permuted labels
  7  per-peptide target information smuggled in through the features

    python arm_M_leakcheck.py          # 1-5 and 7, all cheap
    python arm_M_leakcheck.py full     # also 6, the permutation control (fits models)
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import arm_A_supervised_nn as A
import arm_M_interact as M
import arm_M_pockets as PK
import data
import metrics
import run_experiment as R
import splits

PASS, FAIL = "PASS", "FAIL"
_results = []


def check(name, ok, detail=""):
    _results.append((name, PASS if ok else FAIL, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  -- {detail}" if detail else ""),
          flush=True)
    return ok


# ---------------------------------------------------------------------------

def c1_pocket_guard(df, folds):
    """The pocket spec must be derived from train rows only, and the guard must fire
    if it is not."""
    print("\n1. POCKET SPEC IS DERIVED FROM TRAIN ROWS ONLY")
    cfg = M.Cfg(inter="pca6", pocket_src="label", k=6)
    f = M.Featurizer(cfg, all_alleles=sorted(df.HLA.unique()))
    name, alleles = folds[0]
    tr, te = splits.split_by_allele(df, alleles)
    f(df.loc[tr])
    spec = f._spec
    check("spec derived from a training side excludes every held-out allele",
          not (set(alleles) & set(spec.train_alleles)),
          f"{spec.n_train_alleles} train alleles, {len(alleles)} held out")
    f(df.loc[te])
    check("featurizing the test side does not re-derive the spec",
          f.n_derivations == 1, f"derivations={f.n_derivations}")

    # now try to break it on purpose
    f2 = M.Featurizer(cfg, all_alleles=sorted(df.HLA.unique()))
    f2(df)                                   # "train" on everything, including fold 0
    fired = False
    try:
        f2(df.loc[te])
    except AssertionError as e:
        fired = "LEAK GUARD" in str(e)
    check("the guard FIRES when a held-out allele was in the derivation set", fired)

    # the derivation must not depend on the test rows' labels in any other way
    g = M.Featurizer(cfg, all_alleles=sorted(df.HLA.unique()))
    shuffled = df.copy()
    rng = np.random.default_rng(0)
    m = shuffled.HLA.isin(alleles)
    shuffled.loc[m, "y"] = rng.permutation(shuffled.loc[m, "y"].values)
    g(shuffled.loc[tr])
    check("spec is identical when the HELD-OUT rows' labels are destroyed",
          g._spec.pockets == spec.pockets, g._spec.describe())


def c2_bookkeeping_inert(df, folds):
    """The last two columns must never reach the model."""
    print("\n2. BOOKKEEPING COLUMNS ARE STRIPPED, NOT USED")
    cfg = M.Cfg(inter="pca6")
    featurize, make_model, f = M.build(cfg, df=df)
    name, alleles = folds[0]
    tr, te = splits.split_by_allele(df, alleles)
    Xtr, Xte = featurize(df.loc[tr]), featurize(df.loc[te])
    check("featurize appends exactly N_BOOK bookkeeping columns",
          Xtr.shape[1] == M.width(cfg) + M.N_BOOK,
          f"{Xtr.shape[1]} = {M.width(cfg)} features + {M.N_BOOK}")
    feats, gid, cens = M._strip(Xtr)
    check("the censored bookkeeping column matches data.censored exactly",
          bool((cens == df.loc[tr].censored.to_numpy()).all()))
    check("the allele-id bookkeeping column is constant within an allele",
          df.loc[tr].assign(g=gid).groupby("HLA").g.nunique().max() == 1)

    m = make_model(0).fit(Xtr, df.loc[tr].y.to_numpy())
    p0 = m.predict(Xte)
    rng = np.random.default_rng(1)
    Xte2 = Xte.copy()
    Xte2[:, -1] = rng.integers(0, 2, len(Xte2))        # scramble the censored flag
    Xte2[:, -2] = rng.integers(0, 75, len(Xte2))       # scramble the allele id
    p1 = m.predict(Xte2)
    check("predictions are bit-identical when both bookkeeping columns are scrambled",
          bool(np.array_equal(p0, p1)),
          f"max |diff| = {np.abs(p0 - p1).max():.3g}")
    return Xtr, Xte, df.loc[tr], df.loc[te]


def c3_censor_only_in_loss(df, folds):
    """The Tobit loss may read the TRAIN censoring. It must not read the TEST one."""
    print("\n3. THE CENSORING INDICATOR DOES NOT REACH TEST-TIME PREDICTION")
    cfg = M.Cfg(inter="pca6", loss="tobit")
    featurize, make_model, f = M.build(cfg, df=df)
    name, alleles = folds[0]
    tr, te = splits.split_by_allele(df, alleles)
    Xtr, Xte = featurize(df.loc[tr]), featurize(df.loc[te])
    m = make_model(0).fit(Xtr, df.loc[tr].y.to_numpy())
    p0 = m.predict(Xte)
    Xte2 = Xte.copy()
    Xte2[:, -1] = 1.0 - Xte2[:, -1]
    check("tobit predictions unchanged when every test censoring flag is flipped",
          bool(np.array_equal(p0, m.predict(Xte2))))
    # and the censored rows must not simply be predicted at the floor
    te_df = df.loc[te]
    check("the model is not just reproducing the censoring indicator",
          abs(float(pd.Series(p0).corr(pd.Series(te_df.censored.astype(float).values)))) < 0.95,
          f"corr(pred, censored) = "
          f"{float(pd.Series(p0).corr(pd.Series(te_df.censored.astype(float).values))):+.3f}")


def c4_same_folds(df):
    print("\n4. THE FOLDS ARE THE PROJECT'S OWN")
    folds = splits.choose_held_out(df)
    check("splits.choose_held_out gives 21 folds", len(folds) == 21, str(len(folds)))
    stored = pd.read_csv("results_A_supervised_nn.csv")
    check("fold names match arm A's stored run exactly",
          sorted(n for n, _ in folds) == sorted(stored.fold_name.unique().tolist()))
    allele_union = set().union(*[set(a) for _, a in folds])
    overlaps = [(n1, n2) for i, (n1, a1) in enumerate(folds)
                for n2, a2 in folds[i + 1:] if set(a1) & set(a2)]
    check("no allele appears in two folds", not overlaps, str(overlaps[:3]))
    check("every fold's test rows are disjoint from its train rows",
          all(not (set(df.loc[splits.split_by_allele(df, a)[0]].HLA)
                   & set(df.loc[splits.split_by_allele(df, a)[1]].HLA))
              for _, a in folds))
    return folds


def c5_base_identical(df):
    print("\n5. WITH THE INTERACTION BLOCK OFF, THE FEATURES ARE ARM A's")
    fa = A.featurizer("both")(df)
    fm = M.Featurizer(M.Cfg(inter="none"), all_alleles=sorted(df.HLA.unique()))(df)
    check("arm M base block is byte-identical to arm A's featurizer",
          bool(np.array_equal(fa, fm[:, :-M.N_BOOK])),
          f"{fa.shape} vs {fm[:, :-M.N_BOOK].shape}")


def c7_no_peptide_target_info(df, folds):
    """Nothing in the feature matrix may be a function of the labels of OTHER rows.

    The dataset's real leak surface is that 80.9% of test peptides also appear in
    training against a different allele. A model is allowed to learn from that; what it
    must not do is receive a feature built out of those rows' measured half-lives (a
    per-peptide target encoding). This checks that no feature column changes when every
    training label is destroyed.
    """
    print("\n7. NO FEATURE IS A FUNCTION OF ANY ROW'S LABEL")
    cfg = M.Cfg(inter="pca6", pocket_src="structure")
    name, alleles = folds[0]
    tr, te = splits.split_by_allele(df, alleles)
    f1 = M.Featurizer(cfg, all_alleles=sorted(df.HLA.unique()))
    X1tr, X1te = f1(df.loc[tr]), f1(df.loc[te])
    bad = df.copy()
    rng = np.random.default_rng(7)
    bad["y"] = rng.permutation(bad.y.values)
    bad["Thalf"] = rng.permutation(bad.Thalf.values)
    f2 = M.Featurizer(cfg, all_alleles=sorted(df.HLA.unique()))
    X2tr, X2te = f2(bad.loc[tr]), f2(bad.loc[te])
    check("structural-pocket features are unchanged when ALL labels are permuted",
          bool(np.array_equal(X1tr[:, :-1], X2tr[:, :-1])
               and np.array_equal(X1te[:, :-1], X2te[:, :-1])))
    # and the label-derived variant must change ONLY through the pocket choice
    cfgL = M.Cfg(inter="pca6", pocket_src="label")
    g1 = M.Featurizer(cfgL, all_alleles=sorted(df.HLA.unique()))
    g1(df.loc[tr])
    g2 = M.Featurizer(cfgL, all_alleles=sorted(df.HLA.unique()))
    g2(bad.loc[tr])
    check("the label-derived pocket spec IS a function of the training labels "
          "(so it is refit per fold)", g1._spec.pockets != g2._spec.pockets,
          f"{g1._spec.describe()}  vs  {g2._spec.describe()}")


def c6_permutation_control(df, folds, n_folds=3, device=None):
    """Train on permuted labels. If the protocol is clean the score must collapse."""
    print("\n6. PERMUTATION CONTROL: train on shuffled labels, score on real ones")
    cfg = M.Cfg(inter="pca6", pocket_src="structure")
    rng = np.random.default_rng(0)
    rows = []
    for name, alleles in folds[:n_folds]:
        tr, te = splits.split_by_allele(df, alleles)
        train, test = df.loc[tr], df.loc[te]
        featurize, make_model, _ = M.build(cfg, df=df, device=device)
        Xtr, Xte = featurize(train), featurize(test)
        for label, y in [("real labels", train.y.to_numpy()),
                         ("permuted within allele",
                          train.groupby("HLA").y.transform(
                              lambda s: rng.permutation(s.values)).to_numpy()),
                         ("permuted globally", rng.permutation(train.y.to_numpy()))]:
            R.set_seed(0)
            m = make_model(0).fit(Xtr, y)
            rho = float(np.median(metrics.spearman_per_allele(test, m.predict(Xte))))
            rows.append({"fold": name, "labels": label, "rho": rho})
            print(f"    {name:<18} {label:<24} rho={rho:+.4f}", flush=True)
    t = pd.DataFrame(rows).pivot(index="fold", columns="labels", values="rho")
    t.to_csv("results_M_permutation_control.csv")
    print(t.round(4).to_string())
    check("globally permuted labels score near zero",
          abs(t["permuted globally"].median()) < 0.06,
          f"median {t['permuted globally'].median():+.4f}")
    check("within-allele permuted labels score near zero",
          abs(t["permuted within allele"].median()) < 0.06,
          f"median {t['permuted within allele'].median():+.4f}")
    check("real labels score far above both",
          t["real labels"].median() > t["permuted globally"].median() + 0.15,
          f"real {t['real labels'].median():+.4f}")


def main(full=False, device=None):
    df = data.load()
    print("=" * 78)
    print("ARM M LEAK CHECK")
    print("=" * 78)
    folds = c4_same_folds(df)
    c1_pocket_guard(df, folds)
    c2_bookkeeping_inert(df, folds)
    c3_censor_only_in_loss(df, folds)
    c5_base_identical(df)
    c7_no_peptide_target_info(df, folds)
    if full:
        c6_permutation_control(df, folds, device=device)
    print("\n" + "=" * 78)
    n_fail = sum(1 for _, s, _ in _results if s == FAIL)
    print(f"{len(_results) - n_fail}/{len(_results)} checks passed")
    for n, s, d in _results:
        if s == FAIL:
            print(f"  FAILED: {n}  {d}")
    print("=" * 78)
    pd.DataFrame(_results, columns=["check", "status", "detail"]).to_csv(
        "results_M_leakcheck.csv", index=False)
    return n_fail == 0


if __name__ == "__main__":
    ok = main(full="full" in sys.argv)
    sys.exit(0 if ok else 1)
