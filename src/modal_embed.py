"""ESM-2 650M embeddings on a Modal GPU, matching embed.py's conventions exactly.

Why this exists
---------------
Everything in this project so far uses facebook/esm2_t30_150M_UR50D (640-dim),
computed locally on Apple MPS. The open question is whether a 4x larger protein
language model changes any conclusion. This job produces the same three
encodings from facebook/esm2_t33_650M_UR50D (1280-dim) so the two can be
swapped underneath the identical downstream code.

A negative answer ("650M buys nothing here") is a real result and is cheap to
get: the whole run is ~12 min on one A10, i.e. well under a dollar. That is the
reason for an A10 and not an H100 -- 650M inference is nowhere near big enough
to need one, and the brief marks compute judgement.

Outputs (written into this directory, same convention as embed.py: an .npy
matrix plus a sibling <name>_index.json mapping key string -> row index):

    emb650_peptides.npy   (5633, 1280)   key: peptide
    emb650_pseudo.npy     (75, 1280)     key: allele name
    emb650_joint.npy      (28166, 1280)  key: "HLA|PEPTIDE"

Row order is identical to the 150M files, so emb650_*.npy is a drop-in
replacement for emb_*.npy.

Encodings (copied verbatim from embed.py, see parity test below):
  peptides  mean-pooled last hidden state over the 9 residues, BOS/EOS masked
  pseudo    mean-pooled over the 34-residue pseudo-sequence, BOS/EOS masked
  joint     allele_full_seq + "GGGGSGGGGS" + peptide in ONE forward pass,
            mean-pooled over the PEPTIDE positions ONLY (the last 9 residue
            positions before EOS)

Usage
-----
    # one-off, no Modal account needed: proves the pooling maths is identical
    # to embed.py's, bit for bit, on the 150M model running locally
    .venv/bin/python modal_embed.py --selftest

    # cheap throughput probe on the GPU (200 joint pairs, ~2 min, ~$0.03)
    ~/.local/bin/modal run modal_embed.py --probe 200

    # the real run
    ~/.local/bin/modal run modal_embed.py

    # no-GPU fallback: fine for peptides + pseudo, far too slow for all
    # 28,166 joint pairs (use --probe N to cap the joint work)
    ESM650_NO_GPU=1 ~/.local/bin/modal run modal_embed.py --probe 200

    # a different HF encoder (see ESM-C note at the bottom of this file)
    ~/.local/bin/modal run modal_embed.py --hf-model <repo> --prefix emb<tag>

Requires ~/.modal.toml, created by running `~/.local/bin/modal token new`.

GPU containers additionally require a payment method on the Modal workspace.
Promotional credits alone are NOT enough: without a card on file, every GPU
type returns "Please add a payment method to use <TYPE> GPU functions."
(verified for T4, L4 and A10, 2026-10-03).
"""

import json
import os
import sys
import time
from pathlib import Path

try:
    import modal
except ModuleNotFoundError:
    # The Modal CLI lives in its own uv tool venv; .venv (which has torch) does
    # not have modal. --selftest needs torch, not modal, so it still runs there.
    modal = None

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent

MODEL = "facebook/esm2_t33_650M_UR50D"   # 650M params, 1280-dim, 33 layers
BASELINE_MODEL = "facebook/esm2_t30_150M_UR50D"  # what the project already uses
PREFIX = "emb650"

# Flexible Gly/Ser linker. Identical to embed.py. Arbitrary but standard, and it
# keeps the peptide's token positions trivially identifiable (last 9 before EOS).
LINKER = "GGGGSGGGGS"
PEPTIDE_LEN = 9

# A10: 24 GB, ~$0.000306/sec ($1.10/hr). 650M in fp32 is 2.6 GB of weights and
# the workload is ~360-token sequences, so this fits with room to spare. An L4
# ($0.000222/s) would also fit but has half the memory bandwidth; an H100 would
# be ~3.6x the price for a model this small. A10 is the sensible point.
GPU = "A10"
GPU_USD_PER_SEC = 0.000306      # modal.com/pricing, checked 2026-10-03
CPU_USD_PER_CORE_SEC = 0.0000131
MEM_USD_PER_GIB_SEC = 0.00000222
N_CPU = 2.0
CPU_ONLY_CORES = 16.0
# Modal validates EVERY function in the app at startup, so simply *declaring* a
# GPU function is enough to fail a workspace with no payment method. The no-GPU
# fallback therefore has to be chosen at import time, before the app is built:
#   ESM650_NO_GPU=1 ~/.local/bin/modal run modal_embed.py --probe 200
NO_GPU = os.environ.get("ESM650_NO_GPU") == "1"
MEM_MB = 16384

HF_CACHE = "/opt/hf"
DATA_DIR = "/data"
OUT_DIR = "/out"

# Pinned to the versions in the local .venv so remote numbers are
# comparable with the local 150M numbers.
TORCH = "torch==2.14.1"
TRANSFORMERS = "transformers==5.18.0"


def _download_weights():
    """Build step: pull the weights into the image so no run re-downloads them."""
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL, max_workers=8)


if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(TORCH, TRANSFORMERS, "numpy", "pandas", "huggingface_hub")
        # HF_HUB_ENABLE_HF_TRANSFER is deprecated in current huggingface_hub;
        # HF_XET_HIGH_PERFORMANCE is the replacement.
        .env({"HF_HOME": HF_CACHE, "HF_XET_HIGH_PERFORMANCE": "1",
              "TOKENIZERS_PARALLELISM": "false"})
        # Weights are baked into an image layer here, not fetched at runtime.
        .run_function(_download_weights, timeout=30 * 60)
        # Mounted at container start (not baked), so editing the data does not
        # invalidate the 2.5 GB weights layer. Must come after all build steps.
        .add_local_file(HERE / "stability.txt", f"{DATA_DIR}/stability.txt")
        .add_local_file(HERE / "alleles.json", f"{DATA_DIR}/alleles.json")
    )
    app = modal.App("esm650-embed", image=image)
    out_volume = modal.Volume.from_name("esm650-embeddings", create_if_missing=True)
else:
    image = app = out_volume = None


# --------------------------------------------------------------------------
# pooling -- these two functions are the load-bearing part of the whole job
# --------------------------------------------------------------------------
#
# They are transcribed from embed.py's `embed()` and must stay numerically
# identical to it, otherwise the 150M vs 650M comparison measures the pooling
# change rather than the model change. `--selftest` proves the equality against
# the real embed.py on the real 150M model; `verify_tail_mask()` proves the
# tail mask selects exactly the 9 peptide residues by decoding the tokens.


def full_mask(attention_mask, torch):
    """All real residues, BOS and EOS masked out. embed.py's `tail is None` branch."""
    m = attention_mask.clone()
    m[:, 0] = 0                                                        # BOS
    idx = torch.arange(m.shape[0], device=m.device)
    m[idx, attention_mask.sum(1) - 1] = 0                              # EOS
    return m


def tail_mask(attention_mask, tail, torch):
    """The last `tail` residue positions before EOS. embed.py's `tail=k` branch.

    attention_mask.sum(1) - 1 is the EOS index for a RIGHT-padded row, so the
    slice [end-tail : end] is the peptide and never touches padding, EOS or the
    linker. Right padding is asserted in verify_tail_mask().
    """
    m = torch.zeros_like(attention_mask)
    end = attention_mask.sum(1) - 1                                    # EOS index
    for j in range(m.shape[0]):
        m[j, end[j] - tail: end[j]] = 1
    return m


def embed(seqs, tok, model, device, torch, np, batch=64, tail=None, log=""):
    """Mean-pooled embeddings, one row per sequence. Mirrors embed.py:embed()."""
    out, t0 = [], time.time()
    for i in range(0, len(seqs), batch):
        chunk = seqs[i: i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        if tail is None:
            m = full_mask(enc["attention_mask"], torch)
        else:
            m = tail_mask(enc["attention_mask"], tail, torch)
        m = m.unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
        if log and i % (batch * 20) == 0:
            done = i + len(chunk)
            el = time.time() - t0
            print(f"  {log} {done}/{len(seqs)}  {el:.1f}s  ({done / max(el, 1e-9):.1f}/s)", flush=True)
    return np.vstack(out), time.time() - t0


def verify_tail_mask(tok, pairs, torch):
    """Decode the tokens the tail mask selects; assert they ARE the peptide.

    Deliberately uses a MIXED-LENGTH batch (the dataset has 181, 338 and 341
    residue alleles) so that right padding is actually exercised. If the mask
    were off by one, or counted from the padded end instead of the real EOS,
    this would catch it.
    """
    seqs = [a + LINKER + p for a, p in pairs]
    enc = tok(seqs, return_tensors="pt", padding=True)
    ids, am = enc["input_ids"], enc["attention_mask"]

    assert tok.padding_side == "right", f"tail mask assumes right padding, got {tok.padding_side}"
    assert len({len(s) for s in seqs}) > 1, "verification batch must have mixed lengths"
    assert ids.shape[1] > min(am.sum(1)).item(), "verification batch is not actually padded"

    m = tail_mask(am, PEPTIDE_LEN, torch)
    report = []
    for j, (allele_seq, pep) in enumerate(pairs):
        sel = m[j].nonzero().flatten().tolist()
        got = "".join(tok.convert_ids_to_tokens(ids[j, sel].tolist()))
        end = int(am[j].sum()) - 1
        assert len(sel) == PEPTIDE_LEN, f"row {j}: mask selects {len(sel)} tokens, want {PEPTIDE_LEN}"
        assert sel == list(range(sel[0], sel[0] + PEPTIDE_LEN)), f"row {j}: mask is not contiguous"
        assert sel[-1] == end - 1, f"row {j}: mask does not end immediately before EOS"
        assert got == pep, f"row {j}: mask decodes to {got!r}, want {pep!r}"
        report.append(f"    len(seq)={len(allele_seq):4d} pad={ids.shape[1] - int(am[j].sum()):3d} "
                      f"pos={sel[0]}..{sel[-1]} -> {got}  OK")
    print(f"  tail-mask check passed on {len(pairs)} mixed-length rows "
          f"(padded width {ids.shape[1]}):", flush=True)
    print("\n".join(report), flush=True)
    return True


# --------------------------------------------------------------------------
# data loading (same reads as data.load() / alleles.load(), no project imports)
# --------------------------------------------------------------------------

def load_inputs(data_dir):
    import pandas as pd

    df = pd.read_csv(Path(data_dir) / "stability.txt", sep=r"\s+")      # == data.load()
    alle = json.loads((Path(data_dir) / "alleles.json").read_text())    # == alleles.load()

    peps = sorted(df.Pep.unique())
    pnames = [n for n in sorted(alle) if alle[n]["pseudo"]]
    pairs = list(zip(df.HLA, df.Pep))
    return df, alle, peps, pnames, pairs


# --------------------------------------------------------------------------
# the GPU job
# --------------------------------------------------------------------------

def _embed_job(hf_model: str = MODEL, prefix: str = PREFIX, probe: int = 0,
               batch_pep: int = 256, batch_joint: int = 64,
               n_cpu: float = 0.0, mem_mb: int = 0):
    """Embed peptides, pseudo-sequences and joint pairs. Returns a manifest."""
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    t_container = time.time()
    outdir = Path(OUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    cuda = torch.cuda.is_available()
    dev = "cuda" if cuda else "cpu"
    hw = torch.cuda.get_device_name(0) if cuda else "cpu-only"
    print(f"{hw} | torch {torch.__version__} | device {dev}", flush=True)
    df, alle, peps, pnames, pairs = load_inputs(DATA_DIR)
    print(f"data: {len(df)} rows, {len(peps)} peptides, {len(pnames)} pseudo, "
          f"{len(pairs)} pairs", flush=True)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(hf_model)
    model = AutoModel.from_pretrained(hf_model, dtype=torch.float32).eval().to(dev)
    dim = model.config.hidden_size
    print(f"loaded {hf_model}  dim={dim}  "
          f"params={sum(p.numel() for p in model.parameters()) / 1e6:.0f}M  "
          f"in {time.time() - t0:.1f}s", flush=True)

    # --- prove the pooling mask before spending any GPU time on it ----------
    lens = sorted({len(v["seq"]) for v in alle.values()})
    probe_alleles = [next(n for n in pnames if len(alle[n]["seq"]) == L) for L in lens]
    verify_tail_mask(
        tok,
        [(alle[n]["seq"], p) for n, p in zip(probe_alleles, peps[:len(probe_alleles)])],
        torch,
    )

    manifest, timings = {}, {}

    def save(name, keys, mat):
        assert mat.shape == (len(keys), dim), f"{name}: {mat.shape} vs {(len(keys), dim)}"
        assert np.isfinite(mat).all(), f"{name}: non-finite values"
        npy, idx = outdir / f"{prefix}_{name}.npy", outdir / f"{prefix}_{name}_index.json"
        np.save(npy, mat)
        idx.write_text(json.dumps({k: i for i, k in enumerate(keys)}))
        manifest[npy.name] = npy.stat().st_size
        manifest[idx.name] = idx.stat().st_size
        print(f"{npy.name} {mat.shape}", flush=True)

    # --- peptides -----------------------------------------------------------
    mat, t = embed(peps, tok, model, dev, torch, np, batch=batch_pep, log="pep")
    timings["peptides"] = (len(peps), t)
    print(f"peptides: {len(peps)} in {t:.1f}s ({len(peps) / t:.0f}/s)", flush=True)
    save("peptides", peps, mat)

    # --- pseudo-sequences ---------------------------------------------------
    mat, t = embed([alle[n]["pseudo"] for n in pnames], tok, model, dev, torch, np, batch=64)
    timings["pseudo"] = (len(pnames), t)
    print(f"pseudo: {len(pnames)} in {t:.1f}s", flush=True)
    save("pseudo", pnames, mat)

    # --- joint --------------------------------------------------------------
    use = pairs[:probe] if probe else pairs
    seqs = [alle[h]["seq"] + LINKER + p for h, p in use]
    mat, t = embed(seqs, tok, model, dev, torch, np, batch=batch_joint, tail=PEPTIDE_LEN, log="joint")
    timings["joint"] = (len(use), t)
    print(f"joint: {len(use)} pairs in {t:.1f}s ({t / len(use) * 1000:.1f} ms/pair, "
          f"{len(use) / t:.1f}/s) -> all {len(pairs)} = {t / len(use) * len(pairs) / 60:.1f} min",
          flush=True)
    if probe:
        print(f"PROBE ONLY: joint matrix for {probe} pairs NOT saved", flush=True)
    else:
        save("joint", [f"{h}|{p}" for h, p in use], mat)

    out_volume.commit()

    # Cost is computed from measured container wall time at Modal's published
    # rates (modal.com/pricing, 2026-10-03). It is an estimate of the compute
    # line only -- the billed figure is on modal.com/settings/usage.
    wall = time.time() - t_container
    cores = n_cpu or N_CPU
    gib = (mem_mb or MEM_MB) / 1024
    rate = (GPU_USD_PER_SEC if cuda else 0.0) + cores * CPU_USD_PER_CORE_SEC + gib * MEM_USD_PER_GIB_SEC
    cost = wall * rate
    print(f"\ncontainer wall {wall:.1f}s -> compute cost ${cost:.4f} "
          f"({'A10 $%.6f/s + ' % GPU_USD_PER_SEC if cuda else 'no GPU + '}"
          f"{cores} cpu + {gib:.0f} GiB)", flush=True)

    return {"manifest": manifest, "timings": timings, "wall_s": wall,
            "cost_usd": cost, "dim": dim, "model": hf_model,
            "hw": hw, "cuda": cuda, "probe": probe}


# --------------------------------------------------------------------------
# local entrypoint
# --------------------------------------------------------------------------

def main(hf_model: str = MODEL, prefix: str = PREFIX, probe: int = 0,
         batch_pep: int = 256, batch_joint: int = 64):
    t0 = time.time()
    cores = CPU_ONLY_CORES if NO_GPU else N_CPU
    if NO_GPU:
        print(f"ESM650_NO_GPU=1: no GPU, {cores} cores. Joint pairs are slow here; "
              f"use --probe N to cap them.", flush=True)
    r = run.remote(hf_model=hf_model, prefix=prefix, probe=probe,
                   batch_pep=batch_pep, batch_joint=batch_joint,
                   n_cpu=cores, mem_mb=MEM_MB)

    print("\n--- downloading ---", flush=True)
    for name, size in r["manifest"].items():
        dest = HERE / name
        with open(dest, "wb") as f:
            for chunk in out_volume.read_file(name):
                f.write(chunk)
        got = dest.stat().st_size
        assert got == size, f"{name}: downloaded {got} bytes, remote has {size}"
        print(f"{dest}  {got / 1e6:.1f} MB")

    print("\n--- summary ---")
    print(f"model      {r['model']}  dim {r['dim']}")
    print(f"hardware   {r['hw']}")
    for k, (n, t) in r["timings"].items():
        print(f"{k:<10} {n:>6} seqs in {t:7.1f}s  ({n / t:7.1f}/s)")
    print(f"remote     {r['wall_s']:.1f}s container wall")
    print(f"cost       ${r['cost_usd']:.4f} compute, from measured wall time at "
          f"published rates (billed figure: modal.com/settings/usage)")
    print(f"total wall {time.time() - t0:.1f}s (incl. image pull + download)")
    if r["probe"]:
        print(f"\nPROBE RUN ({r['probe']} joint pairs): joint matrix was not saved.")


# Decorate only when modal is importable, so --selftest works under .venv.
if modal is not None:
    _common = dict(memory=MEM_MB, timeout=2 * 60 * 60, volumes={OUT_DIR: out_volume})
    _gpu_kw = {} if NO_GPU else dict(gpu=GPU)
    run = app.function(cpu=CPU_ONLY_CORES if NO_GPU else N_CPU, **_gpu_kw, **_common)(_embed_job)
    main = app.local_entrypoint()(main)


# --------------------------------------------------------------------------
# local self-test: no Modal account, no GPU, no network beyond the HF cache
# --------------------------------------------------------------------------

def selftest():
    """Prove this file's pooling is bit-identical to embed.py's.

    Runs BOTH implementations on the SAME 150M model on this laptop and
    compares. If max |difference| is 0 for all three encodings, then swapping
    150M for 650M is the only thing that changes in the comparison.
    """
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    sys.path.insert(0, str(HERE))
    import embed as local_embed                       # read-only, never modified

    df, alle, peps, pnames, pairs = load_inputs(HERE)
    tok = AutoTokenizer.from_pretrained(BASELINE_MODEL)

    print(f"tokenizer: {BASELINE_MODEL}  padding_side={tok.padding_side}")
    lens = sorted({len(v["seq"]) for v in alle.values()})
    print(f"allele sequence lengths in the data: {lens} -> mixed batch is padded\n")

    # 1. the mask selects exactly the 9 peptide residues, under right padding
    probe_alleles = [next(n for n in pnames if len(alle[n]["seq"]) == L) for L in lens]
    verify_tail_mask(tok, [(alle[n]["seq"], p) for n, p in
                           zip(probe_alleles, peps[:len(probe_alleles)])], torch)

    # 1b. same check with a deliberately ragged, shuffled batch
    rng = np.random.default_rng(0)
    sample = [pairs[i] for i in rng.choice(len(pairs), 24, replace=False)]
    verify_tail_mask(tok, [(alle[h]["seq"], p) for h, p in sample], torch)

    # 2. numerical parity against embed.py itself
    dev = local_embed.device()
    model = AutoModel.from_pretrained(BASELINE_MODEL).eval().to(dev)
    print(f"\nparity vs embed.py on {BASELINE_MODEL} ({dev}):")

    cases = [
        ("peptides", peps[:64], dict(batch=32), dict(batch=32)),
        ("pseudo", [alle[n]["pseudo"] for n in pnames[:32]], dict(batch=16), dict(batch=16)),
        ("joint", [alle[h]["seq"] + LINKER + p for h, p in sample],
         dict(batch=7, tail=PEPTIDE_LEN), dict(batch=7, tail=PEPTIDE_LEN)),
    ]
    ok = True
    for name, seqs, mine, theirs in cases:
        a, _ = embed(seqs, tok, model, dev, torch, np, **mine)
        b, _ = local_embed.embed(seqs, tok, model, **theirs)
        d = float(np.abs(a - b).max())
        ok &= (d == 0.0)
        print(f"  {name:<9} n={len(seqs):<4} shape={a.shape}  max|diff| = {d:.3e}"
              f"  {'IDENTICAL' if d == 0.0 else 'MISMATCH'}")

    # 3. the joint encoding really is allele-conditioned: the same peptide with
    #    two different alleles must not give the same row.
    h1, h2 = pnames[0], pnames[-1]
    pep = peps[0]
    two, _ = embed([alle[h1]["seq"] + LINKER + pep, alle[h2]["seq"] + LINKER + pep],
                   tok, model, dev, torch, np, batch=2, tail=PEPTIDE_LEN)
    sep = float(np.abs(two[0] - two[1]).max())
    print(f"  joint is allele-conditioned: max|{h1} - {h2}| on peptide {pep} = {sep:.4f}"
          f"  {'OK' if sep > 1e-4 else 'SUSPICIOUS'}")

    print("\nSELFTEST", "PASS" if ok and sep > 1e-4 else "FAIL")
    return 0 if (ok and sep > 1e-4) else 1


# ESM-C note
# ----------
# `--hf-model <repo> --prefix <tag>` runs any HF encoder with a last_hidden_state
# through exactly this pipeline, which is the generic "second model" hook.
# ESM-C specifically (EvolutionaryScale/esmc-600m-2024-12) is NOT wired up and
# has NOT been run: it is a gated repo needing an accepted licence plus an HF
# token as a modal.Secret, and it loads through the `esm` package rather than
# transformers' AutoModel, so `.pip_install("esm")` and a different load path
# would be needed. Left out deliberately -- one working model beats two
# half-working ones.

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    print(__doc__)
    print("Run the GPU job with:  ~/.local/bin/modal run modal_embed.py")
