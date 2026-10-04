"""ESM-2 embedding cache. Fallback used until embed.py (other agent) lands.

model.py prefers `embed` and falls back to this. Same contract either way:

    get("pep")   -> (array [n_pep, D],  {peptide_string: row})
    get("hla")   -> (array [n_hla, D],  {allele_string: row})
    get("joint") -> (array [n_pair, D], {(peptide, allele): row})

THE COMPARISON THIS FILE EXISTS TO MAKE FAIR
Encoding (a) "concat": mean-pool the peptide alone, mean-pool the HLA alone,
concatenate. The peptide representation cannot possibly depend on the allele.
Encoding (b) "joint": push <cls> PEP <eos> HLA <eos> through ESM-2 in ONE pass,
then mean-pool the peptide positions and the HLA positions SEPARATELY and
concatenate those. Same dimension (2*640), same pooling, same model, same
layer. The only difference is that in (b) attention has let the HLA residues
modulate the peptide tokens. If (b) wins it is because of that context, not
because it got a bigger or differently-normalised feature vector.

Mean-pool, not <cls>: ESM-2 was not trained with a sequence-level objective, so
its <cls> is not a sentence embedding. Mean over residues is the standard choice.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

MODEL = "facebook/esm2_t30_150M_UR50D"
DIM = 640
CACHE = "emb_cache"


def _device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load_model():
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    mod = AutoModel.from_pretrained(MODEL).eval().to(_device())
    return tok, mod


@torch.no_grad()
def _embed_single(seqs, tok, mod, batch=16):
    """Mean-pooled representation of each sequence on its own."""
    out = np.zeros((len(seqs), DIM), dtype=np.float32)
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True).to(_device())
        h = mod(**enc).last_hidden_state
        # drop <cls>/<eos>/pad: special_tokens_mask marks them
        m = (~tok(chunk, return_tensors="pt", padding=True,
                  return_special_tokens_mask=True)["special_tokens_mask"].bool()
             ).to(_device()).unsqueeze(-1)
        out[i:i + batch] = ((h * m).sum(1) / m.sum(1)).float().cpu().numpy()
    return out


@torch.no_grad()
def _embed_pairs(pairs, pep_seq, hla_seq, tok, mod, batch=8, log_every=200):
    """One forward pass per (peptide, allele): <cls> PEP <eos> HLA <eos>.

    Returns [n, 2*DIM]: peptide-in-context mean || HLA-in-context mean.
    """
    cls_id, eos_id, pad_id = tok.cls_token_id, tok.eos_token_id, tok.pad_token_id
    out = np.zeros((len(pairs), 2 * DIM), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        ids, pm, am = [], [], []
        for pep, hla in chunk:
            p = tok(pep, add_special_tokens=False)["input_ids"]
            a = tok(hla_seq[hla], add_special_tokens=False)["input_ids"]
            seq = [cls_id] + p + [eos_id] + a + [eos_id]
            ids.append(seq)
            pmask = [0] + [1] * len(p) + [0] * (len(a) + 2)
            amask = [0] * (len(p) + 2) + [1] * len(a) + [0]
            pm.append(pmask)
            am.append(amask)
        L = max(len(s) for s in ids)
        dev = _device()
        att = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in ids], device=dev)
        inp = torch.tensor([s + [pad_id] * (L - len(s)) for s in ids], device=dev)
        pmt = torch.tensor([m + [0] * (L - len(m)) for m in pm], device=dev).unsqueeze(-1)
        amt = torch.tensor([m + [0] * (L - len(m)) for m in am], device=dev).unsqueeze(-1)
        h = mod(input_ids=inp, attention_mask=att).last_hidden_state
        pv = (h * pmt).sum(1) / pmt.sum(1)
        av = (h * amt).sum(1) / amt.sum(1)
        out[i:i + batch] = torch.cat([pv, av], -1).float().cpu().numpy()
        if i and i % (batch * log_every) == 0:
            done = i + len(chunk)
            rate = done / (time.time() - t0)
            print(f"  {done}/{len(pairs)} pairs  {rate:.1f}/s  "
                  f"eta {(len(pairs) - done) / rate / 60:.1f} min", flush=True)
    return out


# ---------------------------------------------------------------------------
# WHICH PART OF THE HLA CHAIN TO EMBED
# 'full'   the whole mature heavy chain, 338-341 aa. What we built first, and
#          almost certainly the wrong choice: the ~34 residues that line the
#          peptide groove are under 10% of a 341-residue mean, so the surviving
#          between-allele variation is dominated by alpha3 / framework
#          differences that track locus and lineage, not binding preference.
#          It also puts the 3 exon-2-3-only alleles in a different feature space
#          from the other 72, since a groove-only mean is not a whole-chain mean.
# 'groove' alpha1 + alpha2, mature residues 1-182. The domains that actually
#          form the binding cleft. Consistent across all 75 alleles.
# 'pseudo' the 34 NetMHCpan pseudo-sequence positions (polymorphic residues
#          within 4 A of the peptide). Maximum signal-to-dilution, and the same
#          representation NetMHCstabpan itself uses, so a win here is not
#          explainable by "we gave our model more information than the baseline".
# ---------------------------------------------------------------------------

GROOVE_END = 182


def region_seq(entry, region):
    """Slice one alleles.json entry down to `region`. Honours `offset`, which is
    how many mature residues are missing from the front of a partial record."""
    seq, off = entry["seq"], entry.get("offset", 0)
    if region == "full":
        return seq
    if region == "groove":
        return seq[max(0, 1 - 1 - off): GROOVE_END - off]
    if region == "pseudo":
        p = entry.get("pseudo")
        if p is None:
            raise ValueError("no pseudo-sequence for this allele")
        return p
    raise ValueError(region)


def build(df=None, alleles=None, region="full"):
    """Build every cache that is missing. Idempotent."""
    import data as _data
    import alleles as _alleles
    df = _data.load() if df is None else df
    entries = _alleles.load() if alleles is None else alleles
    hla_seq = {k: region_seq(v, region) for k, v in entries.items()}
    sfx = "" if region == "full" else "_" + region
    os.makedirs(CACHE, exist_ok=True)
    tok = mod = None

    peps = sorted(df.Pep.unique())
    hlas = sorted(df.HLA.unique())
    pairs = list(map(tuple, df[["Pep", "HLA"]].drop_duplicates().values))

    jobs = [("pep", peps), ("hla" + sfx, hlas), ("joint" + sfx, pairs)]
    for name, keys in jobs:
        npy, idx = f"{CACHE}/{name}.npy", f"{CACHE}/{name}.index.npy"
        if os.path.exists(npy):
            continue
        if tok is None:
            tok, mod = _load_model()
            print(f"loaded {MODEL} on {_device()}")
        t0 = time.time()
        if name == "pep":
            arr = _embed_single(peps, tok, mod)
        elif name.startswith("hla"):
            arr = _embed_single([hla_seq[h] for h in hlas], tok, mod, batch=4)
        else:
            # shorter HLA region => bigger batch fits and it runs much faster
            bs = 8 if region == "full" else (32 if region == "groove" else 64)
            arr = _embed_pairs(pairs, None, hla_seq, tok, mod, batch=bs)
        np.save(npy, arr)
        np.save(idx, np.array([str(k) if not name.startswith("joint")
                               else "\t".join(k) for k in keys]))
        print(f"{name}: {arr.shape} in {time.time() - t0:.1f}s")


def get(name):
    arr = np.load(f"{CACHE}/{name}.npy")
    keys = np.load(f"{CACHE}/{name}.index.npy")
    if name.startswith("joint"):
        index = {tuple(k.split("\t")): i for i, k in enumerate(keys)}
    else:
        index = {str(k): i for i, k in enumerate(keys)}
    return arr, index


if __name__ == "__main__":
    import sys
    for r in (sys.argv[1:] or ["full"]):
        build(region=r)
    for n in ("pep", "hla", "joint"):
        a, ix = get(n)
        print(n, a.shape, len(ix))
