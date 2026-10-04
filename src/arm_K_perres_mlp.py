"""Arm K: per-residue ESM-2 through arm A's EXACT tuned MLP. The fair fight.

WHY THIS ARM EXISTS
-------------------
The results table up to now cannot separate "features" from "head":

    tuned MLP head, conventional features   0.277   (arm A, 5 seeds)
    ---- everything below is an UNTUNED RIDGE head ----
    per-residue ESM-2                       0.125   (arm I)
    conventional                            0.115   (arm J_conv)
    conv + ESM concatenated                 0.097   (arm J_hybrid)
    mean-pooled ESM-2                       0.060   (arm J_esm)

Arm A is the only tuned MLP in that table. Every ESM number is a ridge. So the
gap between 0.277 and 0.125 is a head effect, a feature effect, or both, and
nothing already run can tell them apart. Under a MATCHED ridge head the ordering
actually REVERSES: per-residue ESM 0.125 edges conventional 0.115.

This arm changes ONE thing relative to arm A: the featurizer. Same folds, same
seeds, same TorchMLP class, same BEST hyperparameters, same censored='tied',
same run_experiment.run_arm. It fills the missing cell.

TWO ARMS, because the control was also missing
----------------------------------------------
  K_perres_mlp   [per-residue ESM-2 peptide 5760 | ESM-2 pseudo 640]   6400 dims
  K_pooled_mlp   [mean-pooled ESM-2 peptide 640 | ESM-2 pseudo 640]    1280 dims

K_pooled_mlp is NOT a duplicate of the existing B_esm_pseudo_mlp. Arm B's MLP
head is its own class (arm_B_esm_pseudo.MLPHead) and differs from arm A's tuned
TorchMLP in at least eight ways:

    arm B MLPHead                    arm A TorchMLP (this arm)
    StandardScaler on X              no scaler
    y standardised                   raw y
    hidden (512, 128)                hidden (256, 128)
    lr 1e-3                          lr 3e-4
    weight_decay 1e-4                weight_decay 1e-5
    max_epochs 200, patience 15      max_epochs 300, patience 25
    PEPTIDE-grouped 20% inner split  ALLELE-grouped 15% inner split
    early stop on validation MSE     early stop on median per-allele Spearman
    CPU                              MPS

So B_esm_pseudo_mlp = 0.096 is "mean-pooled ESM through *a* tuned MLP", not
"through arm A's tuned MLP". K_pooled_mlp supplies the genuinely matched cell so
the 2x3 table is head-controlled on BOTH rows, not just the one this arm was
asked for.

n_pep_cols: THE SILENT-FAILURE RISK
-----------------------------------
arm A's TorchMLP uses n_pep_cols to locate the ALLELE block -- it slices
X[:, n_pep_cols:] and recovers allele identity by a random projection, in order
to build the allele-grouped early-stopping split. Point it at the wrong column
and the split silently stops being allele-grouped: no exception, no warning, a
number that looks fine and means nothing. `--verify` checks, on real fold data,
that the recovered grouping is EXACTLY the true allele partition for both arms.
Run it before trusting anything here.

Usage:
    python arm_K_perres_mlp.py --verify              # n_pep_cols proof, no fitting
    python arm_K_perres_mlp.py --which perres --seeds 5
    python arm_K_perres_mlp.py --which pooled --seeds 5
    python arm_K_perres_mlp.py --report
    python arm_K_perres_mlp.py --drop                # robustness re-score

Nothing runs at import time.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import arm_A_supervised_nn as A
import arm_B_esm_pseudo as B
import arm_I_perresidue as I
import metrics
import run_experiment as R
import splits

PEP_LEN = 9
ESM_DIM = 640

ARM_PERRES = "K_perres_mlp"
ARM_POOLED = "K_pooled_mlp"

# Width of the leading PEPTIDE block in each featurizer. Everything after it is
# the allele block, which is what arm A's MLP groups on. Verified by --verify.
NPC_PERRES = PEP_LEN * ESM_DIM          # 5760
NPC_POOLED = ESM_DIM                    # 640


# ---------------------------------------------------------------------------
# features: reused from arms I and B, not rebuilt. The embeddings on disk are
# the same arrays those arms scored, so the only moving part is the head.
# ---------------------------------------------------------------------------

def featurize_perres(df_subset):
    """[per-residue ESM-2 peptide (9x640) | mean-pooled ESM-2 pseudo (640)].

    This is arm_I_perresidue.featurize_B verbatim -- the SAME function object
    that produced the 0.125 ridge number, so the ridge/MLP comparison on this
    row differs in the head and nothing else.
    """
    X = I.featurize_B(df_subset)
    assert X.shape[1] == NPC_PERRES + ESM_DIM, X.shape
    return X


def featurize_pooled(df_subset):
    """[mean-pooled ESM-2 peptide (640) | mean-pooled ESM-2 pseudo (640)].

    arm_B_esm_pseudo.featurize verbatim -- the features behind the 0.060 ridge
    number and behind B_esm_pseudo_mlp's 0.096.
    """
    X = B.featurize(df_subset)
    assert X.shape[1] == NPC_POOLED + ESM_DIM, X.shape
    return X


SPEC = {
    # key      -> (arm name, featurizer, n_pep_cols, description)
    "perres": (ARM_PERRES, featurize_perres, NPC_PERRES,
               "per-residue ESM-2 peptide | ESM-2 pseudo, arm A's tuned MLP"),
    "pooled": (ARM_POOLED, featurize_pooled, NPC_POOLED,
               "mean-pooled ESM-2 peptide | ESM-2 pseudo, arm A's tuned MLP"),
}


# ---------------------------------------------------------------------------
# the head: arm A's, untouched
# ---------------------------------------------------------------------------

def make_model(seed, n_pep_cols):
    """arm A's TorchMLP with arm A's BEST dict. Nothing re-tuned, nothing added.

    A.BEST is read live from arm_A_supervised_nn rather than copied, so this
    cannot drift away from the arm it is supposed to match.
    """
    return A.TorchMLP(seed=seed, n_pep_cols=n_pep_cols, **A.BEST)


# ---------------------------------------------------------------------------
# verification of n_pep_cols -- the thing that fails silently
# ---------------------------------------------------------------------------

def verify(n_folds=2):
    """Prove, on real fold data, that n_pep_cols points at the allele boundary.

    Four checks per arm:
      1. the column count is exactly what the featurizer produced, and the
         LEADING block is not allele-constant (n_pep_cols not too small);
      2. the trailing block X[:, n_pep_cols:] is CONSTANT within an allele --
         i.e. it really is the allele block;
      3. A._allele_groups(X, n_pep_cols), the function arm A's MLP actually
         calls, recovers exactly the partition induced by that allele block,
         and no allele straddles the resulting early-stopping split;
      4. that partition is IDENTICAL to the one arm A itself builds from its
         conventional features on the same rows.

    ON THE EXPECTED 65-of-66 MERGE. Two alleles in alleles.json,
    HLA-B*14:01(C67S) and HLA-B*14:02(C67S), share the same 34-residue
    pseudo-sequence: they differ only outside the groove-contact positions.
    Every arm that encodes the allele by its pseudo-sequence therefore sees one
    group where there are two alleles -- arm A included, which recovers 65
    groups from 66 alleles on fold 1 and 68 from 69 on fold 2, exactly as this
    arm does. So the right criterion is NOT "groups == alleles" but "groups ==
    pseudo-sequence classes, and the same ones arm A uses". Check 4 is the
    guarantee that matters: it means arm K's early-stopping split is literally
    the split arm A would make on the same rows.
    """
    df = R.load_df()
    folds = splits.choose_held_out(df)
    ok = True

    for key in ("perres", "pooled"):
        arm, f, npc, desc = SPEC[key]
        print(f"\n{'=' * 74}\n{arm}: {desc}\n  n_pep_cols = {npc}\n{'=' * 74}")

        for i, (fold_name, alleles) in enumerate(folds[:n_folds], 1):
            tr_idx, _ = splits.split_by_allele(df, alleles)
            train = df.loc[tr_idx]
            X = f(train)
            hla = train.HLA.to_numpy()
            n_true = len(np.unique(hla))

            # 1. shape
            print(f"  [{fold_name}] X {X.shape}  "
                  f"peptide block 0:{npc}  allele block {npc}:{X.shape[1]}")
            assert X.shape[0] == len(train)

            # 2. the trailing block is a pure function of the allele
            blk = X[:, npc:]
            per_allele_unique = {}
            for a in np.unique(hla):
                rows = blk[hla == a]
                # every row for this allele must be bit-identical
                if not np.all(rows == rows[0]):
                    print(f"    FAIL allele block varies WITHIN allele {a}")
                    ok = False
                per_allele_unique[a] = rows[0]
            stacked = np.stack(list(per_allele_unique.values()))
            # the TRUE target: distinct allele blocks = distinct pseudo-seqs,
            # which is <= n_true because two alleles share a pseudo-sequence.
            n_blocks = len(np.unique(stacked, axis=0))
            merged = n_true - n_blocks
            print(f"    allele block constant within allele: yes | "
                  f"{n_true} alleles -> {n_blocks} distinct blocks "
                  f"({merged} merged by shared pseudo-sequence)")

            # also confirm the LEADING block is NOT allele-constant, i.e. we
            # have not accidentally pointed npc past the end of the peptide part
            pep_blk = X[:, :npc]
            a0 = np.unique(hla)[0]
            r0 = pep_blk[hla == a0]
            if len(r0) > 1 and np.all(r0 == r0[0]):
                print("    FAIL peptide block is constant within an allele "
                      "-- n_pep_cols looks too small")
                ok = False

            # 3. arm A's OWN grouping function must recover exactly the
            #    partition induced by the allele block -- no more, no less.
            g, n_groups = A._allele_groups(X, npc)
            blk_inv = np.unique(blk, axis=0, return_inverse=True)[1]
            same = (n_groups == n_blocks
                    and len(set(zip(g.tolist(), blk_inv.tolist()))) == n_blocks)
            print(f"    A._allele_groups -> {n_groups} groups | "
                  f"allele-block classes {n_blocks} | "
                  f"{'MATCH' if same else 'MISMATCH'}")
            if not same:
                print("    FAIL recovered grouping is not the allele-block "
                      "partition -- n_pep_cols is wrong")
                ok = False

            # 4. the partition must be the SAME ONE arm A builds from its own
            #    conventional features on these exact rows.
            fa = A.featurizer(A.ENCODING)
            Xa = fa(train)
            ga, nga = A._allele_groups(Xa, fa.n_pep_cols)
            identical = (nga == n_groups
                         and len(set(zip(g.tolist(), ga.tolist()))) == n_groups)
            print(f"    vs arm A's own grouping: arm A -> {nga} groups | "
                  f"{'IDENTICAL PARTITION' if identical else 'DIFFERENT'}")
            if not identical:
                print("    FAIL arm K would early-stop on a different split "
                      "than arm A")
                ok = False

            # and show that the early-stopping split really splits by allele
            m = make_model(0, npc)
            tr_m, va_m = m._split(X, len(X))
            tr_al, va_al = set(hla[tr_m]), set(hla[va_m])
            overlap = tr_al & va_al
            print(f"    early-stop split: {tr_m.sum()} fit / {va_m.sum()} val "
                  f"rows, {len(tr_al)} fit alleles / {len(va_al)} val alleles, "
                  f"overlap {len(overlap)}")
            if overlap:
                print(f"    FAIL alleles on both sides of the inner split: "
                      f"{sorted(overlap)[:5]}")
                ok = False

    print(f"\n{'=' * 74}")
    print("n_pep_cols VERIFICATION:", "PASS" if ok else "FAIL")
    print("=" * 74)
    if not ok:
        raise SystemExit(1)
    return ok


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def run(which="perres", seeds=5, censored="tied"):
    keys = list(SPEC) if which == "all" else [which]
    out = {}
    for k in keys:
        arm, f, npc, desc = SPEC[k]
        print(f"\n[{arm}] {desc}\n[{arm}] n_pep_cols={npc}  head=arm A TorchMLP "
              f"{A.BEST}", flush=True)
        t0 = time.time()
        out[arm] = R.run_arm(arm, f, lambda s, n=npc: make_model(s, n),
                             seeds=seeds, censored=censored)
        print(f"[{arm}] wall clock {time.time() - t0:.1f}s\n", flush=True)
    return out


def rescore(arm, censored="drop"):
    """Re-score stored predictions under the other censoring policy.

    Censoring is an EVALUATION choice; the model never sees it and y is
    identical either way, so this re-scores rather than refits.
    """
    return I.rescore(arm, censored)


# ---------------------------------------------------------------------------
# reporting: the estimator the rest of the project quotes
# ---------------------------------------------------------------------------

def headline(arm, seeds=None):
    """Median over all (fold, seed) rows of the per-fold median-per-allele
    Spearman.

    This is the estimator behind every number in the project's comparison table
    (arm A 0.277, B_esm_pseudo_mlp 0.096, J_conv 0.115, I_perres 0.125), so arm
    K is quoted the same way -- verified against results_*.csv on disk, not
    copied from a write-up.

    seeds: restrict to these seed ids, so an arm run with fewer seeds can be
    compared against arm A on the SAME seeds and the seed count stops being a
    difference between the cells.

    Returns (median, n_folds, n_seeds, per_fold mean-over-seeds).
    """
    r = pd.read_csv(f"results_{arm}.csv")
    ok = r[r.status == "ok"]
    if seeds is not None:
        ok = ok[ok.seed.isin(list(seeds))]
    if not len(ok):
        return np.nan, 0, 0, pd.Series(dtype=float)
    per_fold = ok.groupby("fold_name", sort=False).spearman.mean()
    return (float(ok.spearman.median()), int(ok.fold_name.nunique()),
            int(ok.seed.nunique()), per_fold)


TABLE_ROWS = [
    # label,                 ridge arm,            mlp arm
    ("conventional",         "J_conv",             "A_supervised_nn"),
    ("mean-pooled ESM",      "J_esm",              ARM_POOLED),
    ("per-residue ESM",      "I_perres_B_pseudo",  ARM_PERRES),
]


def report():
    print("=" * 78)
    print("ARM K -- head vs features, 21 groove-held-out folds, censored='tied'")
    print("estimator: median over all (fold, seed) rows")
    print("=" * 78)

    for key in ("perres", "pooled"):
        arm = SPEC[key][0]
        try:
            m, nf, ns, pf = headline(arm)
        except FileNotFoundError:
            print(f"\n{arm}: not run yet")
            continue
        print(f"\n{arm}: median {m:+.4f}  ({nf} folds x {ns} seeds)")
        if len(pf):
            print(f"  per-fold mean-over-seeds: median {pf.median():+.4f}  "
                  f"mean {pf.mean():+.4f}  worst {pf.min():+.4f} "
                  f"({pf.idxmin()})  best {pf.max():+.4f}")

    # Seeds actually completed by this arm, so arm A can be quoted on the same
    # ones and the seed count stops being a difference between the cells.
    try:
        kr = pd.read_csv(f"results_{ARM_PERRES}.csv")
        k_seeds = sorted(kr[kr.status == "ok"].seed.unique().tolist())
    except FileNotFoundError:
        k_seeds = []

    print("\n" + "=" * 78)
    print("THE 2x3 TABLE -- median over all (fold, seed) rows, censored='tied'")
    print("=" * 78)
    print(f"{'':22s} {'Ridge head':>14s} {'Tuned MLP head':>16s}   {'n (folds x seeds)':>20s}")
    for label, ridge_arm, mlp_arm in TABLE_ROWS:
        cells, ns_txt = [], []
        for a in (ridge_arm, mlp_arm):
            try:
                m, nf, ns, _ = headline(a)
                cells.append(f"{m:+.3f}" if nf else "pending")
                ns_txt.append(f"{nf}x{ns}" if nf else "-")
            except FileNotFoundError:
                cells.append("pending")
                ns_txt.append("-")
        print(f"{label:22s} {cells[0]:>14s} {cells[1]:>16s}   "
              f"{ns_txt[0]:>8s} / {ns_txt[1]:<8s}")

    if k_seeds:
        print(f"\nSeed-matched check -- arm A restricted to seeds {k_seeds}, "
              f"the ones arm K ran:")
        for a in ("A_supervised_nn",):
            m_all, nf_a, ns_a, _ = headline(a)
            m_m, nf_m, ns_m, _ = headline(a, seeds=k_seeds)
            print(f"  {a}: all {ns_a} seeds {m_all:+.4f}  |  "
                  f"seeds {k_seeds} {m_m:+.4f}  ({nf_m} folds)")

    print("\nNote: B_esm_pseudo_mlp (0.096) is arm B's OWN MLPHead, not arm A's "
          "tuned\nMLP -- see this module's docstring. K_pooled_mlp is the "
          "matched cell for\nthat row.")


def paired(arm=ARM_PERRES, ref="A_supervised_nn", censored="tied"):
    """Paired per-fold comparison against arm A, the conventional/MLP cell."""
    from scipy import stats
    a = pd.read_csv(f"results_{arm}.csv")
    b = pd.read_csv(f"results_{ref}.csv")
    pa = a[a.status == "ok"].groupby("fold_name").spearman.mean()
    pb = b[b.status == "ok"].groupby("fold_name").spearman.mean()
    common = pa.index.intersection(pb.index)
    d = (pa[common] - pb[common]).to_numpy()
    w = stats.wilcoxon(d) if len(d) > 5 else None
    t = stats.ttest_rel(pa[common], pb[common])
    row = {"arm": arm, "ref": ref, "censored": censored, "n_folds": len(common),
           "median_rho": float(np.median(pa[common])),
           "ref_median_rho": float(np.median(pb[common])),
           "median_delta": float(np.median(d)), "mean_delta": float(d.mean()),
           "folds_improved": int((d > 0).sum()),
           "wilcoxon_p": float(w.pvalue) if w is not None else np.nan,
           "paired_t_p": float(t.pvalue)}
    out = pd.DataFrame([row])
    path = f"results_K_paired__{censored}.csv"
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    print(f"-> {path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="prove n_pep_cols is right, no fitting")
    ap.add_argument("--which", default="perres",
                    choices=("perres", "pooled", "all"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--drop", action="store_true")
    ap.add_argument("--paired", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    if a.verify:
        verify()
        return
    if a.report:
        report()
        return
    if a.paired:
        paired()
        return
    if a.drop:
        for k in (list(SPEC) if a.which == "all" else [a.which]):
            rescore(SPEC[k][0], "drop")
        return
    run(which=a.which, seeds=a.seeds)
    report()


if __name__ == "__main__":
    main()
