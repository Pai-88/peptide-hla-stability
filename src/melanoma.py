"""Melanoma neoantigen scan: which mutant 9-mers are presented stably enough to target?

THE PIPELINE, run forwards.

  1. a real melanoma driver mutation, looked up rather than derived
  2. every 9-mer window spanning the mutated residue (9 of them)
  3. the matched wild-type 9-mer for each
  4. predicted peptide-HLA half-life for both, across a panel of HLA alleles
  5. rank by MUTANT stability, and by the mutant-over-wild-type ratio
  6. refuse for any allele with no measured stability data

WHY HIGH STABILITY IS THE TARGET, not low.
A mutant peptide that stays bound is displayed long enough for a T cell to find it, and the
tumour cell dies. A mutant peptide that falls off is never seen: that is immune escape, not a
drug. So vaccine candidates are the ones where the mutation holds on, ideally better than the
wild-type sequence it replaced, because a peptide that only the tumour presents well is one
the immune system has not been tolerised to.

WHAT THIS DOES NOT DO. It does not fold anything, recover any sequence, reassemble any
protein, or infer any nucleotide. The driver mutations are catalogued (COSMIC, cBioPortal)
and the protein sequences are in UniProt. The only thing worth computing is which of the
resulting fragments a given patient's HLA will actually display.
"""

import json
import os
import urllib.request

import numpy as np
import pandas as pd

import data
import design

# Catalogued melanoma drivers. acc = UniProt, pos = 1-based residue in the MATURE chain,
# wt/mut = the substitution. Verified against the fetched sequence before use.
DRIVERS = {
    "BRAF V600E": dict(acc="P15056", gene="BRAF", pos=600, wt="V", mut="E",
                       note="~50% of cutaneous melanomas; c.1799T>A"),
    "BRAF V600K": dict(acc="P15056", gene="BRAF", pos=600, wt="V", mut="K",
                       note="second most common BRAF variant in melanoma"),
    "NRAS Q61R":  dict(acc="P01111", gene="NRAS", pos=61,  wt="Q", mut="R",
                       note="~15-20% of melanomas"),
    "NRAS Q61K":  dict(acc="P01111", gene="NRAS", pos=61,  wt="Q", mut="K",
                       note="NRAS hotspot"),
}

SEQ_CACHE = "driver_seqs.json"
K = 9


def fetch(acc):
    c = json.load(open(SEQ_CACHE)) if os.path.exists(SEQ_CACHE) else {}
    if acc not in c:
        url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
        with urllib.request.urlopen(url, timeout=30) as r:
            lines = r.read().decode().split("\n")
        c[acc] = "".join(lines[1:])
        json.dump(c, open(SEQ_CACHE, "w"))
    return c[acc]


def windows(driver):
    """Every 9-mer containing the mutated residue, wild-type and mutant, matched.

    A substitution at position p appears in 9 distinct 9-mers: the one starting at p-8
    through the one starting at p. Each is a separate candidate epitope because each
    places the mutation at a different position in the groove, and position matters
    enormously: P2 and P9 are anchors, the middle faces the T cell receptor.
    """
    d = DRIVERS[driver]
    seq = fetch(d["acc"])
    i = d["pos"] - 1
    if seq[i] != d["wt"]:
        raise ValueError(f"{driver}: residue {d['pos']} is {seq[i]}, expected {d['wt']}")
    mutseq = seq[:i] + d["mut"] + seq[i + 1:]

    out = []
    for start in range(max(0, i - K + 1), min(i + 1, len(seq) - K + 1)):
        out.append(dict(
            start=start + 1,
            mut_pos_in_peptide=i - start + 1,   # 1..9, which groove position the mutation sits at
            wt_pep=seq[start:start + K],
            mut_pep=mutseq[start:start + K],
        ))
    return out


def scan(driver, alleles=None, ens=None, min_fold=None):
    """Score every mutant window and its matched wild type across an HLA panel."""
    d = DRIVERS[driver]
    ens = ens or design.train()
    df_all = data.load()
    alleles = alleles or sorted(df_all.HLA.unique())

    rows = []
    for a in alleles:
        v, headline = design._ood(a)
        if v == "out":
            rows.append(dict(allele=a, verdict=v, note=headline))
            continue
        w = windows(driver)
        q = pd.DataFrame({"HLA": a, "Pep": [x["mut_pep"] for x in w] + [x["wt_pep"] for x in w]})
        mean, sd = ens.predict(q)
        n = len(w)
        for j, x in enumerate(w):
            rows.append(dict(
                allele=a, verdict=v,
                mut_pep=x["mut_pep"], wt_pep=x["wt_pep"],
                mut_at=x["mut_pos_in_peptide"],
                mut_thalf=10 ** mean[j], wt_thalf=10 ** mean[n + j],
                sd_log10=sd[j],
                fold_vs_wt=10 ** (mean[j] - mean[n + j]),
            ))
    out = pd.DataFrame(rows)
    if min_fold is not None and "fold_vs_wt" in out:
        out = out[out.fold_vs_wt.fillna(0) >= min_fold]
    return out


if __name__ == "__main__":
    import sys
    driver = sys.argv[1] if len(sys.argv) > 1 else "BRAF V600E"
    d = DRIVERS[driver]

    print(f"{driver}  ({d['gene']}, UniProt {d['acc']})")
    print(f"  {d['note']}")
    w = windows(driver)
    print(f"\n{len(w)} nine-mer windows span residue {d['pos']}:")
    for x in w:
        star = "".join("^" if k + 1 == x["mut_pos_in_peptide"] else " " for k in range(K))
        print(f"  {x['wt_pep']}  ->  {x['mut_pep']}   mutation at groove position {x['mut_pos_in_peptide']}")
        print(f"  {' ' * K}      {star}")

    print(f"\ntraining the {design.N_SEEDS}-seed ensemble")
    ens = design.train(verbose=False)

    res = scan(driver, ens=ens)
    scored = res.dropna(subset=["mut_thalf"]).copy()
    refused = res[res.verdict == "out"]

    print(f"\nscored {scored.allele.nunique()} alleles, refused {refused.allele.nunique()}")
    print("\nTOP CANDIDATES by predicted mutant half-life:")
    top = scored.sort_values("mut_thalf", ascending=False).head(12)
    print(top[["allele", "mut_pep", "mut_at", "mut_thalf", "wt_thalf", "fold_vs_wt", "sd_log10"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3g}"))

    print("\nMOST TUMOUR-SPECIFIC: mutant holds on, wild type does not")
    sel = scored[scored.mut_thalf > 1.0].sort_values("fold_vs_wt", ascending=False).head(8)
    print(sel[["allele", "wt_pep", "mut_pep", "mut_at", "wt_thalf", "mut_thalf", "fold_vs_wt"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print("\nA high fold_vs_wt with a high mut_thalf is the ideal vaccine candidate: presented")
    print("well on tumour cells, poorly on healthy ones, so the T cell response is tumour-specific.")
