# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.6.0",
#   "transformers==4.49.0",
#   "numpy<2.3",
#   "pandas",
#   "scipy",
#   "scikit-learn",
#   "xgboost",
#   "pyarrow",
#   "huggingface_hub",
# ]
# ///
"""Score the 650M joint and un-pooled arms through the project's own harness.

WHAT RUNS HERE
--------------
Nothing in this file defines a metric, a fold, a split or a head. It imports
run_experiment.py, metrics.py, splits.py, data.py, supertypes.py,
arm_A_supervised_nn.py, arm_I_perresidue.py and arm_C_esm_joint.py UNCHANGED
(byte-for-byte copies mounted from a private dataset repo) and calls
run_experiment.run_arm, exactly as every existing arm does. The only
intervention is arm_G_650m.py's documented shim: swap the embedding tables
underneath the existing heads.

    embed.load = lambda name: (emb650_<name>.npy, emb650_<name>_index.json)

so if an existing head is wrong, these arms are wrong in the same way, which is
the point.

THE ARMS
    A_gpu_control          arm A verbatim, on CUDA instead of MPS. A CONTROL,
                           not a result: if this does not reproduce the
                           published 0.307, then no number produced on this
                           hardware is comparable and the rest is void.
    650m_joint_C           650M joint encoding (1280) -> arm C's head.
                           Matched pair with C_esm_joint (150M joint, 0.077).
    650m_joint_Amlp        650M joint (1280) -> arm A's tuned MLP head.
    650m_joint_pseudo_mlp  [650M joint | 650M pseudo] (2560) -> arm A's head.
    650m_perres_B_mlp      [650M per-residue peptide 9x1280 | 650M pseudo]
                           (12800) -> arm A's tuned MLP. Matched pair with
                           I_perres_B_pseudo_mlp (150M un-pooled, 0.195).
    650m_perres_B_ridge    same 12800 features -> arm I's RidgeHead, alpha
                           swept on a peptide-grouped inner split of TRAIN.
                           Matched pair with I_perres_B_pseudo (0.125).

Hyperparameters are frozen imports from the existing arms. Nothing is tuned
here, and no test fold is ever read outside run_experiment.

OUTPUT
    results_N_<arm>.csv and predictions_N_<arm>.parquet, uploaded to the output
    dataset repo as they finish, so a dead job loses one arm, not the run. The
    HEADLINE is not computed here: it is computed on the laptop by
    run_experiment.ensemble_metrics over the downloaded predictions.
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

HARNESS = ("data.py", "splits.py", "supertypes.py", "metrics.py",
           "run_experiment.py", "embed.py", "arm_A_supervised_nn.py",
           "arm_I_perresidue.py", "arm_C_esm_joint.py", "hf_gpu_sweep.py")
DATA = ("stability.txt", "alleles.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/inputs")
    ap.add_argument("--emb-repo", required=True)
    ap.add_argument("--out-repo", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--folds", type=int, default=0, help="first N folds only (smoke)")
    ap.add_argument("--only", default="", help="comma-separated arm keys")
    ap.add_argument("--winner", default="",
                    help='JSON [{"name":..,"kind":"mlp"|"gbm","mode":..,'
                         '"kw":{..},"seeds":N}, ..] -- the sweep winners, scored '
                         'through run_arm on the real outer folds')
    a = ap.parse_args()

    t_container = time.time()
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()

    work = "/tmp/work"
    os.makedirs(work, exist_ok=True)
    for f in HARNESS + DATA:
        src = os.path.join(a.inputs, "harness", f)
        if not os.path.exists(src):
            src = os.path.join(a.inputs, f)
        shutil.copy(src, os.path.join(work, f))
    os.chdir(work)
    sys.path.insert(0, work)

    import torch
    cuda = torch.cuda.is_available()
    dev = torch.device("cuda" if cuda else "cpu")
    print(f"device {dev} | {torch.cuda.get_device_name(0) if cuda else 'cpu'} | "
          f"torch {torch.__version__}", flush=True)

    # ---- pull the embeddings -------------------------------------------
    def get(fn):
        return hf_hub_download(a.emb_repo, fn, repo_type="dataset")

    files = set(api.list_repo_files(a.emb_repo, repo_type="dataset"))
    shards = sorted(f for f in files if f.startswith("joint/shard_"))
    print(f"emb repo: {len(files)} files, {len(shards)} joint shards", flush=True)

    perres = np.load(get("emb650_perres.npy"))
    perres_idx = json.load(open(get("perres_index.json")))
    pseudo = np.load(get("emb650_gpu_pseudo.npy"))
    pseudo_idx = json.load(open(get("pseudo_index.json")))
    joint_idx = json.load(open(get("joint_index.json")))
    joint = np.vstack([np.load(get(s)) for s in shards])
    print(f"perres {perres.shape}  pseudo {pseudo.shape}  joint {joint.shape}", flush=True)
    assert joint.shape[0] == len(joint_idx), (joint.shape, len(joint_idx))
    assert perres.shape[0] == len(perres_idx)

    # The mean over the 9 position blocks of the un-pooled matrix must equal a
    # mean-pooled embedding of the same peptides. Same check arm_I does at 150M.
    gpu_pooled = np.load(get("emb650_gpu_peptides.npy"))
    recon = perres.reshape(len(perres_idx), 9, -1).mean(1)
    err = float(np.abs(recon - gpu_pooled).max())
    print(f"max |mean(per-residue) - mean-pooled| = {err:.3e}", flush=True)
    assert err < 1e-3, "per-residue tokens do not average to the pooled embedding"

    # ---- the shim, exactly as arm_G_650m.py does it ---------------------
    import embed
    _TABLES = {"peptides": (gpu_pooled, perres_idx), "pseudo": (pseudo, pseudo_idx),
               "joint": (joint, joint_idx)}
    embed.load = lambda name: _TABLES[name]

    import arm_A_supervised_nn as A           # noqa: E402  (must follow the shim)
    import arm_C_esm_joint as C               # noqa: E402
    import arm_I_perresidue as I              # noqa: E402
    import run_experiment as R                # noqa: E402
    import splits                             # noqa: E402

    df = R.load_df()
    folds = splits.choose_held_out(df)
    assert len(folds) == 21, f"expected 21 folds, got {len(folds)}"
    use_folds = folds[:a.folds] if a.folds else None
    print(f"{len(df)} rows, {len(folds)} folds "
          f"({'first %d' % a.folds if a.folds else 'all'})", flush=True)

    # ---- featurizers -----------------------------------------------------
    # The joint arms use arm_C's own featurize(), so the 650M-vs-150M joint
    # pair differs in the embedding table and nothing else.
    f_joint = C.featurize
    n_missing = sum(1 for h, p in zip(df.HLA, df.Pep) if f"{h}|{p}" not in joint_idx)
    assert n_missing == 0, f"{n_missing} (allele, peptide) pairs missing from the joint index"

    def f_joint_pseudo(sub):
        j = joint[[joint_idx[f"{h}|{p}"] for h, p in zip(sub.HLA, sub.Pep)]]
        s = pseudo[[pseudo_idx[h] for h in sub.HLA]]
        return np.hstack([j, s]).astype(np.float64)

    def f_perres_pseudo(sub):
        p = perres[[perres_idx[x] for x in sub.Pep]]
        s = pseudo[[pseudo_idx[h] for h in sub.HLA]]
        return np.hstack([p, s]).astype(np.float64)

    DIM = int(pseudo.shape[1])
    NPC_JOINT = DIM                       # the whole joint vector is "peptide"
    NPC_PERRES = 9 * DIM

    af = A.featurizer(A.ENCODING)

    def mk_A(npc):
        def make(seed):
            return A.TorchMLP(seed=seed, n_pep_cols=npc, device=dev, **A.BEST)
        return make

    def mk_ridge(npc):
        return lambda seed: I.RidgeHead(seed=seed, n_pep_cols=npc, alphas=I.ALPHAS)

    # key -> (arm name, featurize, make_model, seeds)
    ARMS = {
        "control":  ("N_A_gpu_control",        af,              mk_A(af.n_pep_cols), a.seeds),
        "jointC":   ("N_650m_joint_C",         f_joint,         C.make_model,        a.seeds),
        "jointA":   ("N_650m_joint_Amlp",      f_joint,         mk_A(NPC_JOINT),     a.seeds),
        "jointPS":  ("N_650m_joint_pseudo_mlp", f_joint_pseudo, mk_A(NPC_JOINT),     a.seeds),
        "perresA":  ("N_650m_perres_B_mlp",    f_perres_pseudo, mk_A(NPC_PERRES),    a.seeds),
        "perresR":  ("N_650m_perres_B_ridge",  f_perres_pseudo, mk_ridge(NPC_PERRES), 1),
    }
    # --- the sweep winners, scored on the real outer folds ----------------
    # hf_gpu_sweep.SweepMLP with loss='mse' was verified on the laptop to be
    # bit-identical to arm A's TorchMLP, and its featurizer bit-identical to
    # arm A's, so running a winner through it is running arm A's arm with
    # different hyperparameters -- not a different implementation.
    if a.winner:
        import hf_gpu_sweep as S
        for spec in json.loads(a.winner):
            kw = {k: (tuple(v) if k == "hidden" else v)
                  for k, v in spec.get("kw", {}).items()}
            f = S.featurizer(spec.get("mode", "both"))
            if spec.get("kind", "mlp") == "gbm":
                def mk(seed, f=f, kw=kw):
                    return S.XGBHead(seed=seed, n_pep_cols=f.n_pep_cols, **kw)
            else:
                def mk(seed, f=f, kw=kw):
                    return S.SweepMLP(seed=seed, n_pep_cols=f.n_pep_cols,
                                      device=dev, **{**S.BASE, **kw})
            ARMS[spec["name"]] = (spec["name"], f, mk, spec.get("seeds", a.seeds))

    order = [k.strip() for k in a.only.split(",") if k.strip()] or list(ARMS)

    done = set(api.list_repo_files(a.out_repo, repo_type="dataset"))
    for key in order:
        name, feat, make, seeds = ARMS[key]
        if f"arms/results_{name}.csv" in done and not a.folds:
            print(f"[{name}] already in the output repo, skipping", flush=True)
            continue
        print(f"\n{'=' * 70}\n{key}: {name}  seeds={seeds}\n{'=' * 70}", flush=True)
        t0 = time.time()
        try:
            R.run_arm(name, feat, make, seeds=seeds, censored="tied", folds=use_folds)
        except Exception as e:
            import traceback
            print(f"[{name}] ARM FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        print(f"[{name}] wall {time.time() - t0:.1f}s", flush=True)
        for fn in (f"results_{name}.csv", f"predictions_{name}.parquet"):
            if os.path.exists(fn):
                try:
                    api.upload_file(path_or_fileobj=fn, path_in_repo=f"arms/{fn}",
                                    repo_id=a.out_repo, repo_type="dataset")
                    print(f"  uploaded arms/{fn}", flush=True)
                except Exception as e:
                    print(f"  upload failed {fn}: {type(e).__name__}: {e}", flush=True)

    print(f"\nCONTAINER WALL {time.time() - t_container:.1f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
