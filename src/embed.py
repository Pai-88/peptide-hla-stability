"""Cache ESM-2 embeddings for peptides, alleles, and (optionally) pairs.

Model: facebook/esm2_t30_150M_UR50D (640-dim), already in the local HF cache.

Three encodings:
  peptides  mean-pooled last hidden state over the 9 residues        -> emb_peptides.npy
  alleles   mean-pooled over the whole mature chain                  -> emb_alleles.npy
  joint     allele + linker + peptide through the model, mean-pooled
            over the PEPTIDE positions only, so HLA context modulates
            the peptide representation                               -> emb_joint.npy

Each .npy is accompanied by a .json index mapping key string -> row.

Usage:
    python embed.py                    # peptides + alleles
    python embed.py --joint 200        # + time a 200-pair joint sample
    python embed.py --joint all        # + all 28,166 pairs (slow, see report)
"""

import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "facebook/esm2_t30_150M_UR50D"
# Flexible Gly/Ser linker. Arbitrary but standard, and it keeps the peptide's
# token positions trivially identifiable (last 9 before EOS).
LINKER = "GGGGSGGGGS"


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval().to(device())
    return tok, model


@torch.no_grad()
def embed(seqs, tok, model, batch=64, tail=None, log=""):
    """Mean-pooled embeddings, one row per sequence.

    tail=None pools over all real residues (special tokens masked out).
    tail=k    pools over the last k residue positions only, which is how the
              joint encoding isolates the peptide.
    """
    out, t0 = [], time.time()
    for i in range(0, len(seqs), batch):
        chunk = seqs[i: i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True).to(device())
        h = model(**enc).last_hidden_state
        if tail is None:
            m = enc["attention_mask"].clone()
            m[:, 0] = 0                                       # BOS
            m[torch.arange(len(chunk)), enc["attention_mask"].sum(1) - 1] = 0  # EOS
        else:
            m = torch.zeros_like(enc["attention_mask"])
            end = enc["attention_mask"].sum(1) - 1             # EOS index
            for j in range(len(chunk)):
                m[j, end[j] - tail: end[j]] = 1
        m = m.unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
        if log and i % (batch * 20) == 0:
            print(f"  {log} {i + len(chunk)}/{len(seqs)}  {time.time() - t0:.1f}s", flush=True)
    return np.vstack(out), time.time() - t0


def save(name, keys, mat):
    np.save(f"emb_{name}.npy", mat)
    json.dump({k: i for i, k in enumerate(keys)}, open(f"emb_{name}_index.json", "w"))
    print(f"emb_{name}.npy {mat.shape}  ({time.time():.0f})")


def load(name):
    return np.load(f"emb_{name}.npy"), json.load(open(f"emb_{name}_index.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", default=None, help="number of pairs to encode, or 'all'")
    args = ap.parse_args()

    import alleles as alleles_mod
    import data

    df = data.load()
    alle = alleles_mod.load()
    tok, model = load_model()
    print(f"device {device()}  model {MODEL}")

    peps = sorted(df.Pep.unique())
    mat, t = embed(peps, tok, model, batch=256, log="pep")
    print(f"peptides: {len(peps)} in {t:.1f}s ({len(peps) / t:.0f}/s)")
    save("peptides", peps, mat)

    names = sorted(alle)
    mat, t = embed([alle[n]["seq"] for n in names], tok, model, batch=8, log="hla")
    print(f"alleles: {len(names)} in {t:.1f}s")
    save("alleles", names, mat)

    # The 34-residue pseudo-sequence. Full-chain mean pooling barely separates
    # alleles (they differ at a handful of residues out of 338); the pseudo-
    # sequence is all polymorphic contact residues, so it separates much better.
    pnames = [n for n in names if alle[n]["pseudo"]]
    mat, t = embed([alle[n]["pseudo"] for n in pnames], tok, model, batch=64)
    print(f"pseudo: {len(pnames)} in {t:.1f}s")
    save("pseudo", pnames, mat)

    if args.joint:
        pairs = list(zip(df.HLA, df.Pep))
        if args.joint != "all":
            rng = np.random.default_rng(0)
            pairs = [pairs[i] for i in rng.choice(len(pairs), int(args.joint), replace=False)]
        seqs = [alle[h]["seq"] + LINKER + p for h, p in pairs]
        mat, t = embed(seqs, tok, model, batch=8, tail=9, log="joint")
        n = len(pairs)
        print(f"joint: {n} pairs in {t:.1f}s ({t / n * 1000:.0f} ms/pair) "
              f"-> all 28,166 would take {t / n * 28166 / 60:.1f} min")
        save("joint", [f"{h}|{p}" for h, p in pairs], mat)


if __name__ == "__main__":
    main()
