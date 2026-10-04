"""Local side of the GPU run: verify the port, fetch the artefacts, score them.

Everything that produces a NUMBER FOR THE SLIDE happens here, on the laptop,
through the project's own run_experiment.ensemble_metrics over the stored
per-row predictions. The GPU only ever did the fitting.

    python hf_gpu_score.py verify     # is the GPU port the same computation?
    python hf_gpu_score.py fetch      # assemble emb650_joint / emb650_perres
    python hf_gpu_score.py report     # headline + drop robustness + paired tests
    python hf_gpu_score.py sweep      # read the sweep CSVs, pick the winner

`verify` is the gate. It checks four things, and if any fails the GPU numbers
are not comparable with the existing ones and must not be quoted:

  1. the GPU-computed 650M mean-pooled peptide/pseudo matrices reproduce the
     existing CPU-computed emb650_peptides.npy / emb650_pseudo.npy;
  2. the un-pooled matrix averages over its 9 position blocks back to the
     mean-pooled one (the same assertion arm_I makes at 150M);
  3. hf_gpu_sweep.featurizer() is bit-identical to
     arm_A_supervised_nn.featurizer() on real rows;
  4. hf_gpu_sweep.SweepMLP with loss='mse' reproduces
     arm_A_supervised_nn.TorchMLP on the same data and seed.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

EMB_REPO = "Paithealchemist88/phla-emb650"
HERE = os.path.dirname(os.path.abspath(__file__))

# The published reference numbers this run has to be read against. Source:
# RESULTS.md section 2 estimator (5-seed ensemble mean prediction, per-fold
# median per-allele Spearman, median over the 21 groove folds, censored='tied'),
# as reproduced in OVERNIGHT_RESULTS.md.
REFERENCE = {
    "A_supervised_nn":        (0.307, "conventional supervised net  <- the one to beat"),
    "I_perres_B_pseudo_mlp":  (0.195, "ESM-2 150M un-pooled, tuned head"),
    "G_650m_B_mlp":           (0.144, "ESM-2 650M mean-pooled, tuned head"),
    "lookup_table":           (0.130, "peptide-mean lookup CONTROL"),
    "B_esm_pseudo_mlp":       (0.106, "ESM-2 150M mean-pooled, tuned head"),
    "C_esm_joint":            (0.077, "ESM-2 150M joint encoding"),
    "D_peptide_only":         (0.064, "ESM-2 150M peptide-only CONTROL"),
}

# arms produced by hf_gpu_arms.py, and the existing arm each one pairs with
NEW_ARMS = [
    ("N_A_gpu_control",         "A_supervised_nn",        "CONTROL: arm A on CUDA, must reproduce 0.307"),
    ("N_650m_joint_C",          "C_esm_joint",            "650M joint, arm C head"),
    ("N_650m_joint_Amlp",       "C_esm_joint",            "650M joint, tuned MLP head"),
    ("N_650m_joint_pseudo_mlp", "C_esm_joint",            "650M joint | pseudo, tuned MLP"),
    ("N_650m_perres_B_mlp",     "I_perres_B_pseudo_mlp",  "650M un-pooled | pseudo, tuned MLP"),
    ("N_650m_perres_B_ridge",   "I_perres_B_pseudo",      "650M un-pooled | pseudo, ridge"),
]

LEAK_CEILING = 0.50   # anything above this is a bug, not a result


def _api():
    from huggingface_hub import HfApi
    return HfApi()


def _dl(fn):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(EMB_REPO, fn, repo_type="dataset")


# ---------------------------------------------------------------------------
# 1. verify
# ---------------------------------------------------------------------------

def verify():
    ok = True
    print("=" * 74)
    print("1. GPU 650M pooled vs the existing CPU-computed 650M pooled files")
    print("=" * 74)
    for gpu_f, cpu_f in [("emb650_gpu_peptides.npy", "emb650_peptides.npy"),
                         ("emb650_gpu_pseudo.npy", "emb650_pseudo.npy")]:
        if not os.path.exists(os.path.join(HERE, cpu_f)):
            print(f"  {cpu_f} missing locally, skipped")
            continue
        g = np.load(_dl(gpu_f))
        c = np.load(os.path.join(HERE, cpu_f))
        if g.shape != c.shape:
            print(f"  {gpu_f}: SHAPE MISMATCH {g.shape} vs {c.shape}")
            ok = False
            continue
        d = float(np.abs(g - c).max())
        rel = d / float(np.abs(c).max())
        good = d < 2e-3
        ok &= good
        print(f"  {gpu_f:<26} {g.shape}  max|diff| {d:.2e}  "
              f"(rel {rel:.1e})  {'OK' if good else 'MISMATCH'}")
    print("  (fp32 on A10G vs fp32 on Apple MPS, so bit-equality is not")
    print("   expected; the bar is agreement to float32 round-off.)")

    print()
    print("=" * 74)
    print("2. un-pooled averages back to mean-pooled (arm_I's assertion at 650M)")
    print("=" * 74)
    pr = np.load(_dl("emb650_perres.npy"))
    pri = json.load(open(_dl("perres_index.json")))
    gp = np.load(_dl("emb650_gpu_peptides.npy"))
    err = float(np.abs(pr.reshape(len(pri), 9, -1).mean(1) - gp).max())
    good = err < 1e-3
    ok &= good
    print(f"  max |mean(per-residue) - mean-pooled| = {err:.3e}  "
          f"{'OK' if good else 'MISMATCH'}")

    print()
    print("=" * 74)
    print("3. hf_gpu_sweep.featurizer == arm_A_supervised_nn.featurizer")
    print("=" * 74)
    sys.path.insert(0, HERE)
    import arm_A_supervised_nn as A
    import hf_gpu_sweep as S
    import run_experiment as R
    df = R.load_df()
    sub = df.sample(500, random_state=0)
    for mode in ("both", "blosum", "onehot"):
        a_x = A.featurizer(mode)(sub)
        s_x = S.featurizer(mode)(sub)
        d = float(np.abs(a_x - s_x).max())
        good = (a_x.shape == s_x.shape) and d == 0.0
        ok &= good
        print(f"  mode={mode:<7} shape {a_x.shape}  max|diff| {d:.1e}  "
              f"{'IDENTICAL' if good else 'MISMATCH'}")

    print()
    print("=" * 74)
    print("4. SweepMLP(loss='mse') == arm_A TorchMLP, same data, same seed")
    print("=" * 74)
    import splits
    folds = splits.choose_held_out(df)
    tr, te = splits.split_by_allele(df, folds[0][1])
    f = A.featurizer(A.ENCODING)
    Xtr, Xte = f(df.loc[tr]), f(df.loc[te])
    ytr = df.loc[tr].y.to_numpy()
    R.set_seed(0)
    ma = A.TorchMLP(seed=0, n_pep_cols=f.n_pep_cols, **A.BEST).fit(Xtr, ytr)
    pa = ma.predict(Xte)
    R.set_seed(0)
    ms = S.SweepMLP(seed=0, n_pep_cols=f.n_pep_cols, **S.BASE).fit(Xtr, ytr)
    ps = ms.predict(Xte)
    d = float(np.abs(pa - ps).max())
    r = float(np.corrcoef(pa, ps)[0, 1])
    good = d < 1e-4
    ok &= good
    print(f"  epochs  armA {ma.epochs_run_}  sweep {ms.epochs_run_}")
    print(f"  max|pred diff| {d:.3e}   pearson {r:.6f}   "
          f"{'IDENTICAL' if good else 'DIFFERENT'}")
    if not good:
        print("  -> SweepMLP is NOT arm A's network. Any sweep winner would be")
        print("     measuring the reimplementation, not the hyperparameters.")

    print()
    print("VERIFY", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# 2. fetch
# ---------------------------------------------------------------------------

def fetch():
    """Assemble emb650_joint.npy / emb650_perres.npy + indexes in this folder."""
    files = set(_api().list_repo_files(EMB_REPO, repo_type="dataset"))
    shards = sorted(f for f in files if f.startswith("joint/shard_"))
    print(f"{len(files)} files in {EMB_REPO}, {len(shards)} joint shards")

    idx = json.load(open(_dl("joint_index.json")))
    mats = [np.load(_dl(s)) for s in shards]
    joint = np.vstack(mats)
    print(f"joint {joint.shape}  index {len(idx)}")
    assert joint.shape[0] == len(idx), "shards do not cover the index"
    assert np.isfinite(joint).all()
    np.save(os.path.join(HERE, "emb650_joint.npy"), joint)
    json.dump(idx, open(os.path.join(HERE, "emb650_joint_index.json"), "w"))

    pr = np.load(_dl("emb650_perres.npy"))
    pri = json.load(open(_dl("perres_index.json")))
    np.save(os.path.join(HERE, "emb650_perres.npy"), pr)
    json.dump(pri, open(os.path.join(HERE, "emb650_perres_index.json"), "w"))
    print(f"perres {pr.shape}  index {len(pri)}")

    n = 0
    for f in sorted(files):
        if f.startswith("arms/"):
            dest = os.path.join(HERE, os.path.basename(f))
            src = _dl(f)
            with open(src, "rb") as a, open(dest, "wb") as b:
                b.write(a.read())
            n += 1
    print(f"copied {n} arm result/prediction files into {HERE}")


def fetch_sweep():
    files = set(_api().list_repo_files(EMB_REPO, repo_type="dataset"))
    out = {}
    for f in sorted(files):
        if f.startswith("sweep/") and f.endswith(".csv"):
            out[os.path.basename(f)[:-4]] = pd.read_csv(_dl(f))
            print(f"  {f}: {len(out[os.path.basename(f)[:-4]])} rows")
    return out


# ---------------------------------------------------------------------------
# 3. report
# ---------------------------------------------------------------------------

# Outside the project folder: the project directory is shared with two other
# overnight agents and this run must not leave stray files in it.
SCRATCH = os.environ.get(
    "HF_GPU_SCRATCH",
    "/private/tmp/claude-501/-Users-paing/d4fb984d-553b-4e43-9e85-1aafddec5769"
    "/scratchpad/ens")


def headline(arm, censored="tied"):
    """RESULTS.md section 2 estimator, computed here from the stored predictions.

    Calls the project's own run_experiment.ensemble_metrics rather than
    reimplementing it. That function writes ensemble_<arm>.csv into the current
    directory, and for the EXISTING arms that file belongs to another agent's
    run, so the call is made from a scratch directory holding symlinks to the
    prediction parquets. Nothing in the project folder is touched.
    """
    import run_experiment as R
    os.makedirs(SCRATCH, exist_ok=True)
    src = os.path.join(HERE, f"predictions_{arm}.parquet")
    dst = os.path.join(SCRATCH, f"predictions_{arm}.parquet")
    if not os.path.exists(dst):
        os.symlink(src, dst)
    df = R.load_df()                      # load from HERE before chdir
    cwd = os.getcwd()
    try:
        os.chdir(SCRATCH)
        ens = R.ensemble_metrics(arm, censored=censored, df=df)
    finally:
        os.chdir(cwd)
    e = ens[ens.status == "ok"]
    return (float(e.spearman.median()) if len(e) else np.nan,
            e.set_index("fold_name").spearman, e)


def report():
    import metrics
    from scipy import stats

    rows = []
    perfold = {}
    for arm, pair, what in NEW_ARMS:
        if not os.path.exists(os.path.join(HERE, f"predictions_{arm}.parquet")):
            print(f"[skip] {arm}: no predictions file")
            continue
        tied, pf, e = headline(arm, "tied")
        drop, _, _ = headline(arm, "drop")
        perfold[arm] = pf
        rows.append({"arm": arm, "what": what, "pairs_with": pair,
                     "tied": tied, "drop": drop, "n_folds": len(pf),
                     "n_seeds": int(e.n_seeds.max()) if len(e) else 0,
                     "top10": float(e.top10_precision.median()) if len(e) else np.nan})
    t = pd.DataFrame(rows)

    print("=" * 96)
    print("HEADLINE  median per-allele Spearman, 21 groove-held-out folds,")
    print("          5-seed ensemble mean prediction, censored kept as ties")
    print("=" * 96)
    if len(t):
        print(t[["arm", "what", "tied", "drop", "top10", "n_folds", "n_seeds"]]
              .to_string(index=False))

    print()
    print("against the published reference ladder:")
    for k, (v, lab) in sorted(REFERENCE.items(), key=lambda kv: -kv[1][0]):
        print(f"    {v:.3f}  {lab}")

    # the integrity gate
    print()
    bad = t[t.tied > LEAK_CEILING] if len(t) else t
    if len(bad):
        print("!" * 96)
        print(f"STOP: {len(bad)} arm(s) score above {LEAK_CEILING}. On this dataset,")
        print("80.9% of test rows share their peptide with a training row, so a")
        print("number this high is a leak, not a result. Do NOT report it.")
        print(bad[["arm", "tied"]].to_string(index=False))
        print("!" * 96)
    else:
        print(f"integrity gate: no arm above {LEAK_CEILING}  OK")

    # paired tests against the arm each one is the direct equivalent of
    print()
    print("=" * 96)
    print("PAIRED over the same 21 folds (Wilcoxon signed-rank on per-fold rho)")
    print("=" * 96)
    for arm, pair, what in NEW_ARMS:
        if arm not in perfold:
            continue
        for ref in (pair, "A_supervised_nn"):
            if not os.path.exists(os.path.join(HERE, f"predictions_{ref}.parquet")):
                continue
            if ref not in perfold:
                perfold[ref] = headline(ref, "tied")[1]
            a_, b_ = perfold[arm].align(perfold[ref], join="inner")
            d = (a_ - b_).dropna()
            if len(d) < 5:
                continue
            p = stats.wilcoxon(a_[d.index], b_[d.index]).pvalue
            print(f"  {arm:<26} vs {ref:<24} median {a_.median():+.3f} vs "
                  f"{b_.median():+.3f}  delta {d.median():+.4f}  "
                  f"wins {int((d > 0).sum())}/{len(d)}  p={p:.2g}")

    if len(t):
        t.to_csv(os.path.join(HERE, "results_N_gpu_summary.csv"), index=False)
        print(f"\nwrote results_N_gpu_summary.csv")
    return t


def control():
    """How far can the SAME arm move just by changing hardware and seed draw?

    N_A_gpu_control is arm_A_supervised_nn.py's code, features, folds, seeds and
    hyperparameters, fitted on an A10G instead of Apple MPS. torch.manual_seed()
    does not make a CUDA RNG stream equal an MPS one, so seed s draws a
    different init, dropout mask and shuffle on the two devices: the control is
    arm A with a different draw of 5 seeds, not a bit-reproduction of it.

    This routine puts a number on that, so the GPU arms below can be read
    against the right yardstick. The comparison for every GPU arm is the GPU
    control, not the published 0.307.
    """
    import itertools
    import run_experiment as R
    from scipy import stats

    df = R.load_df()
    out = {}
    for arm in ("A_supervised_nn", "N_A_gpu_control"):
        p = pd.read_parquet(os.path.join(HERE, f"predictions_{arm}.parquet"))
        seeds = sorted(p.seed.unique())
        tied, pf, _ = headline(arm, "tied")
        out[arm] = (tied, pf, p, seeds)
        print(f"{arm:<20} 5-seed ensemble headline {tied:+.4f}  "
              f"(seeds {seeds})")

    a, b = out["A_supervised_nn"], out["N_A_gpu_control"]
    d = (a[1] - b[1]).dropna()
    print(f"\npaired over {len(d)} folds: laptop - GPU median {d.median():+.4f}, "
          f"laptop wins {int((d > 0).sum())}/{len(d)}, "
          f"Wilcoxon p={stats.wilcoxon(d).pvalue:.3g}")

    # How much does the headline move if you simply ensemble a different
    # SUBSET of the same 5 seeds? That is the same kind of noise.
    print("\nheadline under every 3-of-5 and 4-of-5 sub-ensemble of the SAME run:")
    for arm, (_, _, p, seeds) in out.items():
        vals = []
        for k in (3, 4):
            for combo in itertools.combinations(seeds, k):
                sub = p[p.seed.isin(combo)]
                g = sub.groupby(["fold_name", "row_id"], sort=False).y_pred.mean()
                rows = []
                for fold, s in g.groupby(level=0):
                    te = df.loc[s.index.get_level_values(1).to_numpy()]
                    import metrics
                    rho = metrics.spearman_per_allele(te, s.to_numpy(), censored="tied")
                    if len(rho):
                        rows.append(float(np.median(rho)))
                vals.append(float(np.median(rows)))
        print(f"  {arm:<20} min {min(vals):+.4f}  median {float(np.median(vals)):+.4f}  "
              f"max {max(vals):+.4f}   ({len(vals)} sub-ensembles)")
    return out


def run_joint_C_local():
    """Arm C's head on the 650M joint encoding, fitted here on the laptop CPU.

    This arm is sklearn's MLPRegressor, which is CPU-bound: it measured 180 s a
    fit on the GPU container's vCPUs (5+ GPU-hours for 105 fits, at GPU prices,
    for a CPU workload). It runs in a fraction of that on this machine's own
    CPU, and it never touches MPS, which another overnight job is using.
    """
    import json as _json
    import embed
    import run_experiment as R

    j = np.load(os.path.join(HERE, "emb650_joint.npy"))
    ji = _json.load(open(os.path.join(HERE, "emb650_joint_index.json")))
    assert j.shape[0] == len(ji)
    embed.load = lambda name: {"joint": (j, ji)}[name]
    import arm_C_esm_joint as C
    C._MAT = C._IDX = None
    print(f"650M joint {j.shape}; arm C head, 21 folds x 5 seeds, CPU")
    return R.run_arm("N_650m_joint_C", C.featurize, C.make_model,
                     seeds=5, censored="tied")


def sweep_report():
    """Read the sweep CSVs and rank candidates PAIRED against arm A's BEST."""
    d = fetch_sweep()
    if not d:
        print("no sweep CSVs yet")
        return None
    full = pd.concat(d.values(), ignore_index=True)
    full = full[full.stage != "smoke"]
    key = ["outer", "inner", "seed"]
    base_lab = "ARM A BEST (baseline)"
    # One baseline for everything: arm A's own configuration, run on the same
    # (outer, inner, seed) cells. The GBM stage has no baseline row of its own,
    # so it is paired against stage 1's, which used identical cells.
    ref = full[(full.stage == "stage1") & (full.cand == base_lab)].set_index(key).rho
    out = []
    for stage, sub in full.groupby("stage"):
        b = (sub[sub.cand == base_lab].set_index(key).rho
             if base_lab in set(sub.cand) else ref)
        for c, g in sub.groupby("cand"):
            s = g.set_index(key).rho
            delta = (s - b).dropna()
            out.append({"stage": stage, "cand": c, "mean_rho": float(s.mean()),
                        "sd": float(s.std()), "n": int(len(s)),
                        "paired_delta": float(delta.mean()) if len(delta) else np.nan,
                        "wins": int((delta > 0).sum()), "of": int(len(delta)),
                        "cfg": g.cfg.iloc[0], "mode": g["mode"].iloc[0],
                        "mean_epochs": float(g.epochs.mean())})
    r = pd.DataFrame(out).sort_values(["stage", "paired_delta"], ascending=[True, False])
    print("=" * 110)
    print("SWEEP: inner-validation median per-allele Spearman, TRAIN-only splits.")
    print("These are NOT headline numbers -- the inner folds are easier than the")
    print("outer ones. Only the RANKING is used; the winner is then scored once,")
    print("through run_experiment.run_arm, on the real 21 outer folds.")
    print("=" * 110)
    print(r[["stage", "cand", "mode", "mean_rho", "sd", "n", "paired_delta",
             "wins", "of", "mean_epochs"]].to_string(index=False))
    r.to_csv(os.path.join(HERE, "results_N_sweep_ranked.csv"), index=False)
    print("\nwrote results_N_sweep_ranked.csv")
    return r


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    rc = {"verify": verify, "fetch": fetch, "report": report,
          "sweep": sweep_report, "control": control,
          "jointC": run_joint_C_local}[cmd]()
    raise SystemExit(rc if isinstance(rc, int) else 0)
