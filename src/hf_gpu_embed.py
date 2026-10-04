# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.6.0",
#   "transformers==4.49.0",
#   "numpy<2.3",
#   "pandas",
#   "huggingface_hub",
# ]
# ///
"""ESM-2 650M joint + per-residue embeddings on a Hugging Face Jobs GPU.

WHY THIS FILE EXISTS
--------------------
modal_embed.py was written to produce three 650M encodings (peptides, pseudo,
joint) but Modal's GPU tier is payment-gated, so only the two CPU-cheap ones
(emb650_peptides.npy, emb650_pseudo.npy) were ever built. The 650M scale arm has
therefore only ever been tested under MEAN POOLING -- which is the weakest
encoding at 150M scale. The two encodings that most helped the 150M model
(joint, and un-pooled per-residue) have never been run at 650M.

This file ports modal_embed.py's embedding core to `hf jobs`, unchanged. The
pooling functions below are copied verbatim from modal_embed.py (which was
itself verified bit-identical to embed.py, max abs diff 0.0). They are NOT
reimplemented -- that was an explicit instruction, because an off-by-one in the
tail mask would silently make the 650M-vs-150M comparison a pooling comparison.

OUTPUTS (uploaded to a private HF dataset repo, downloaded by the caller)
    emb650_joint.npy        (28166, 1280)   key "HLA|PEPTIDE"
    emb650_perres.npy       (5633, 9*1280)  key peptide, positional order
    emb650_gpu_peptides.npy (5633, 1280)    parity check vs the existing CPU file
    emb650_gpu_pseudo.npy   (75, 1280)      parity check vs the existing CPU file

RESUMABILITY
    The joint matrix is written as shards of SHARD pairs. Each shard is uploaded
    as soon as it is finished. A restarted job lists the shards already in the
    output repo and skips them, so a job that dies at 80% loses one shard, not
    the run. Shard order is the row order of stability.txt, so the final
    concatenation is exactly the order modal_embed.py would have produced.

USAGE
    hf jobs uv run --flavor a10g-small --secrets HF_TOKEN \
        -v hf://datasets/<user>/phla-inputs:/data \
        hf_gpu_embed.py --out-repo <user>/phla-emb650
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi
from transformers import AutoModel, AutoTokenizer

MODEL = "facebook/esm2_t33_650M_UR50D"
LINKER = "GGGGSGGGGS"          # identical to embed.py / modal_embed.py
PEPTIDE_LEN = 9
SHARD = 4000                   # joint pairs per resumable shard


# ---------------------------------------------------------------------------
# pooling -- copied verbatim from modal_embed.py, do not "improve"
# ---------------------------------------------------------------------------

def full_mask(attention_mask, torch):
    """All real residues, BOS and EOS masked out. embed.py's `tail is None`."""
    m = attention_mask.clone()
    m[:, 0] = 0
    idx = torch.arange(m.shape[0], device=m.device)
    m[idx, attention_mask.sum(1) - 1] = 0
    return m


def tail_mask(attention_mask, tail, torch):
    """The last `tail` residue positions before EOS. embed.py's `tail=k`."""
    m = torch.zeros_like(attention_mask)
    end = attention_mask.sum(1) - 1
    for j in range(m.shape[0]):
        m[j, end[j] - tail: end[j]] = 1
    return m


def embed(seqs, tok, model, device, batch=64, tail=None, log=""):
    """Mean-pooled embeddings, one row per sequence. Mirrors embed.py:embed()."""
    out, t0 = [], time.time()
    for i in range(0, len(seqs), batch):
        chunk = seqs[i: i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        m = full_mask(enc["attention_mask"], torch) if tail is None \
            else tail_mask(enc["attention_mask"], tail, torch)
        m = m.unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
        if log and i % (batch * 20) == 0:
            done, el = i + len(chunk), time.time() - t0
            print(f"  {log} {done}/{len(seqs)}  {el:.1f}s  ({done / max(el, 1e-9):.1f}/s)",
                  flush=True)
    return np.vstack(out), time.time() - t0


def verify_tail_mask(tok, pairs):
    """Decode the tokens the tail mask selects; assert they ARE the peptide.

    Verbatim from modal_embed.py. Deliberately a MIXED-LENGTH batch (the alleles
    are 181, 338 and 341 residues) so right padding is exercised. Runs BEFORE
    any GPU time is spent, as instructed.
    """
    seqs = [a + LINKER + p for a, p in pairs]
    enc = tok(seqs, return_tensors="pt", padding=True)
    ids, am = enc["input_ids"], enc["attention_mask"]

    assert tok.padding_side == "right", f"tail mask assumes right padding, got {tok.padding_side}"
    assert len({len(s) for s in seqs}) > 1, "verification batch must have mixed lengths"
    assert ids.shape[1] > min(am.sum(1)).item(), "verification batch is not actually padded"

    m = tail_mask(am, PEPTIDE_LEN, torch)
    for j, (allele_seq, pep) in enumerate(pairs):
        sel = m[j].nonzero().flatten().tolist()
        got = "".join(tok.convert_ids_to_tokens(ids[j, sel].tolist()))
        end = int(am[j].sum()) - 1
        assert len(sel) == PEPTIDE_LEN, f"row {j}: mask selects {len(sel)}, want {PEPTIDE_LEN}"
        assert sel == list(range(sel[0], sel[0] + PEPTIDE_LEN)), f"row {j}: not contiguous"
        assert sel[-1] == end - 1, f"row {j}: does not end immediately before EOS"
        assert got == pep, f"row {j}: mask decodes to {got!r}, want {pep!r}"
        print(f"    len(seq)={len(allele_seq):4d} pad={ids.shape[1] - int(am[j].sum()):3d} "
              f"pos={sel[0]}..{sel[-1]} -> {got}  OK", flush=True)
    print(f"  tail-mask check passed on {len(pairs)} mixed-length rows", flush=True)


# ---------------------------------------------------------------------------
# per-residue (un-pooled) peptide embedding -- mirrors arm_I_perresidue.py
# ---------------------------------------------------------------------------

def embed_perresidue(peps, tok, model, device, batch=256):
    """All 9 residue token vectors per peptide, positional order -> (n, 9*dim).

    ESM-2 tokenises as [BOS] r1..r9 [EOS] and every peptide is exactly 9 long,
    so there is no padding and no mask to get wrong -- asserted, not assumed.
    Identical slice to arm_I_perresidue.build_embeddings().
    """
    out, t0 = [], time.time()
    with torch.no_grad():
        for i in range(0, len(peps), batch):
            chunk = peps[i: i + batch]
            enc = tok(chunk, return_tensors="pt", padding=True).to(device)
            assert int(enc["attention_mask"].sum(1).min()) == PEPTIDE_LEN + 2
            assert int(enc["attention_mask"].sum(1).max()) == PEPTIDE_LEN + 2
            h = model(**enc).last_hidden_state[:, 1: PEPTIDE_LEN + 1, :]
            out.append(h.reshape(len(chunk), -1).float().cpu().numpy())
            if i % (batch * 5) == 0:
                print(f"  perres {i + len(chunk)}/{len(peps)}  {time.time() - t0:.1f}s",
                      flush=True)
    return np.vstack(out), time.time() - t0


# ---------------------------------------------------------------------------

def load_inputs(data_dir):
    import pandas as pd
    df = pd.read_csv(Path(data_dir) / "stability.txt", sep=r"\s+")
    alle = json.loads((Path(data_dir) / "alleles.json").read_text())
    peps = sorted(df.Pep.unique())
    pnames = [n for n in sorted(alle) if alle[n]["pseudo"]]
    pairs = list(zip(df.HLA, df.Pep))
    return df, alle, peps, pnames, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/data")
    ap.add_argument("--out-repo", required=True)
    ap.add_argument("--batch-joint", type=int, default=32)
    ap.add_argument("--batch-pep", type=int, default=256)
    ap.add_argument("--probe", type=int, default=0, help="only this many joint pairs, nothing saved")
    a = ap.parse_args()

    t_container = time.time()
    api = HfApi()
    work = Path("/tmp/out")
    work.mkdir(parents=True, exist_ok=True)

    cuda = torch.cuda.is_available()
    dev = "cuda" if cuda else "cpu"
    print(f"{torch.cuda.get_device_name(0) if cuda else 'cpu-only'} | "
          f"torch {torch.__version__} | device {dev}", flush=True)

    df, alle, peps, pnames, pairs = load_inputs(a.data_dir)
    print(f"data: {len(df)} rows, {len(peps)} peptides, {len(pnames)} pseudo, "
          f"{len(pairs)} pairs", flush=True)

    # --- what is already done? (resume) -------------------------------------
    try:
        existing = set(api.list_repo_files(a.out_repo, repo_type="dataset"))
    except Exception as e:
        print(f"output repo not readable yet ({type(e).__name__}), starting clean", flush=True)
        existing = set()
    print(f"output repo already holds {len(existing)} files", flush=True)

    def push(local: Path, remote: str):
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                        repo_id=a.out_repo, repo_type="dataset")
        print(f"  uploaded {remote} ({local.stat().st_size / 1e6:.1f} MB)", flush=True)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float32).eval().to(dev)
    dim = model.config.hidden_size
    print(f"loaded {MODEL}  dim={dim}  "
          f"params={sum(p.numel() for p in model.parameters()) / 1e6:.0f}M  "
          f"in {time.time() - t0:.1f}s", flush=True)

    # --- prove the pooling mask BEFORE spending GPU time on it --------------
    lens = sorted({len(v["seq"]) for v in alle.values()})
    probe_alleles = [next(n for n in pnames if len(alle[n]["seq"]) == L) for L in lens]
    verify_tail_mask(tok, [(alle[n]["seq"], p)
                           for n, p in zip(probe_alleles, peps[:len(probe_alleles)])])
    rng = np.random.default_rng(0)
    sample = [pairs[i] for i in rng.choice(len(pairs), 12, replace=False)]
    verify_tail_mask(tok, [(alle[h]["seq"], p) for h, p in sample])

    if a.probe:
        seqs = [alle[h]["seq"] + LINKER + p for h, p in pairs[:a.probe]]
        _, t = embed(seqs, tok, model, dev, batch=a.batch_joint, tail=PEPTIDE_LEN, log="probe")
        print(f"PROBE: {a.probe} pairs in {t:.1f}s ({a.probe / t:.1f}/s) -> "
              f"all {len(pairs)} = {t / a.probe * len(pairs) / 60:.1f} min", flush=True)
        return

    # --- parity encodings: reproduce the two files that already exist -------
    # Cheap (<1 min) and it is the only end-to-end proof that this port and the
    # existing 650M files are the same computation.
    if "emb650_gpu_peptides.npy" not in existing:
        mat, t = embed(peps, tok, model, dev, batch=a.batch_pep, log="pep")
        np.save(work / "emb650_gpu_peptides.npy", mat)
        push(work / "emb650_gpu_peptides.npy", "emb650_gpu_peptides.npy")
        print(f"peptides: {len(peps)} in {t:.1f}s", flush=True)
    if "emb650_gpu_pseudo.npy" not in existing:
        mat, t = embed([alle[n]["pseudo"] for n in pnames], tok, model, dev, batch=64)
        np.save(work / "emb650_gpu_pseudo.npy", mat)
        push(work / "emb650_gpu_pseudo.npy", "emb650_gpu_pseudo.npy")
        (work / "pseudo_index.json").write_text(json.dumps({k: i for i, k in enumerate(pnames)}))
        push(work / "pseudo_index.json", "pseudo_index.json")
        print(f"pseudo: {len(pnames)} in {t:.1f}s", flush=True)

    # --- per-residue peptides (un-pooled) -----------------------------------
    if "emb650_perres.npy" not in existing:
        mat, t = embed_perresidue(peps, tok, model, dev, batch=a.batch_pep)
        assert mat.shape == (len(peps), PEPTIDE_LEN * dim), mat.shape
        assert np.isfinite(mat).all()
        np.save(work / "emb650_perres.npy", mat.astype(np.float32))
        (work / "perres_index.json").write_text(json.dumps({p: i for i, p in enumerate(peps)}))
        push(work / "emb650_perres.npy", "emb650_perres.npy")
        push(work / "perres_index.json", "perres_index.json")
        print(f"per-residue: {mat.shape} in {t:.1f}s", flush=True)

    # --- joint, sharded and resumable ---------------------------------------
    (work / "joint_index.json").write_text(
        json.dumps({f"{h}|{p}": i for i, (h, p) in enumerate(pairs)}))
    if "joint_index.json" not in existing:
        push(work / "joint_index.json", "joint_index.json")

    n_shards = (len(pairs) + SHARD - 1) // SHARD
    t_joint, n_done = 0.0, 0
    for s in range(n_shards):
        name = f"joint/shard_{s:04d}.npy"
        lo, hi = s * SHARD, min((s + 1) * SHARD, len(pairs))
        if name in existing:
            print(f"shard {s + 1}/{n_shards} [{lo}:{hi}] already done, skipping", flush=True)
            continue
        seqs = [alle[h]["seq"] + LINKER + p for h, p in pairs[lo:hi]]
        mat, t = embed(seqs, tok, model, dev, batch=a.batch_joint,
                       tail=PEPTIDE_LEN, log=f"joint s{s}")
        assert mat.shape == (hi - lo, dim), mat.shape
        assert np.isfinite(mat).all()
        local = work / f"shard_{s:04d}.npy"
        np.save(local, mat.astype(np.float32))
        push(local, name)
        local.unlink()
        t_joint += t
        n_done += hi - lo
        print(f"shard {s + 1}/{n_shards} [{lo}:{hi}] {t:.1f}s "
              f"({(hi - lo) / t:.1f}/s)  elapsed {time.time() - t_container:.0f}s", flush=True)

    if n_done:
        print(f"joint: {n_done} new pairs in {t_joint:.1f}s ({n_done / t_joint:.1f}/s)", flush=True)
    print(f"\nCONTAINER WALL {time.time() - t_container:.1f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
