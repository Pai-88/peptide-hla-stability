"""Does the model know where the anchors are, and what belongs in them?

We never told it. It saw (allele, 9-mer, half-life) triples and nothing else: no pocket
definitions, no motif tables, no structures. If it has recovered the anchor positions and
their residue preferences, that is evidence it learned the binding chemistry rather than
memorising peptides.

METHOD, per allele:
  take K peptides actually measured for that allele
  for each, score all 171 single-point mutants
  position importance = spread of predicted log half-life across the 19 substitutions at
    that position, averaged over the K peptides. A position the groove grips hard should
    swing the prediction; a solvent-exposed one should not.
  preferred residue  = the substitution with the highest mean predicted half-life there

Then compare against the published motifs, which the model has never seen.
"""

import numpy as np
import pandas as pd

import data
import design

AA = design.AA
K_PEPTIDES = 12

# Published anchor preferences, from the immunology literature, for scoring the model.
# P2 is pocket B, P9 is pocket F. These are the classical dominant residues only.
KNOWN = {
    "HLA-A*02:01": dict(P2="LM",   P9="VLI",  note="the textbook A2 motif"),
    "HLA-B*27:05": dict(P2="R",    P9="KRLF", note="P2 arginine is near-absolute"),
    "HLA-B*07:02": dict(P2="P",    P9="LF",   note="P2 proline defines the B7 supertype"),
    "HLA-A*03:01": dict(P2="LVMI", P9="KR",   note="basic C-terminus, A3 supertype"),
    "HLA-A*11:01": dict(P2="VTMI", P9="KR",   note="basic C-terminus, A3 supertype"),
    "HLA-B*57:01": dict(P2="ATS",  P9="WF",   note="aromatic C-terminus, B58 supertype"),
    "HLA-B*08:01": dict(P2="",     P9="LF",   note="unusually P3/P5 driven, weak P2"),
}


def profile(ens, allele, df, k=K_PEPTIDES, seed=0):
    """(importance per position, preferred residue per position, best-residue table)."""
    pool = df[df.HLA == allele].Pep.unique()
    if len(pool) == 0:
        return None
    rng = np.random.default_rng(seed)
    peps = rng.choice(pool, size=min(k, len(pool)), replace=False)

    rows, meta = [], []
    for p in peps:
        for pos in range(9):
            for aa in AA:
                rows.append(p[:pos] + aa + p[pos + 1:])
                meta.append((p, pos, aa))
    q = pd.DataFrame({"HLA": allele, "Pep": rows})
    mean, _ = ens.predict(q)

    m = pd.DataFrame(meta, columns=["base", "pos", "aa"])
    m["pred"] = mean

    # importance: spread across substitutions at a position, averaged over base peptides
    imp = m.groupby(["base", "pos"]).pred.std().groupby("pos").mean()
    # preference: mean prediction per (pos, aa), centred within position
    pref = m.groupby(["pos", "aa"]).pred.mean().unstack()
    pref = pref.sub(pref.mean(axis=1), axis=0)
    return imp, pref, len(peps)


def top_residues(pref, pos, n=3):
    return "".join(pref.loc[pos].sort_values(ascending=False).head(n).index)


if __name__ == "__main__":
    df = data.load()
    print(f"training the {design.N_SEEDS}-seed ensemble")
    ens = design.train(verbose=False)

    print(f"\n{'allele':14s} {'position importance, P1..P9 (higher = model grips harder)':58s}")
    print("-" * 78)
    results = {}
    for a in KNOWN:
        if a not in set(df.HLA):
            continue
        out = profile(ens, a, df)
        if out is None:
            continue
        imp, pref, n = out
        results[a] = (imp, pref)
        bars = "".join("#" if v >= imp.max() * 0.75 else ("+" if v >= imp.max() * 0.5 else ".")
                       for v in imp)
        nums = " ".join(f"{v:.2f}" for v in imp)
        top2 = list(imp.sort_values(ascending=False).head(2).index + 1)
        print(f"{a:14s} {bars}   {nums}   top two: P{top2[0]}, P{top2[1]}")

    print("\n\nDoes it recover the published motif? Model's top-3 residues vs the literature.")
    print("-" * 78)
    print(f"{'allele':14s} {'pos':4s} {'model top 3':12s} {'published':10s} {'hit':4s}  note")
    hits = tot = 0
    for a, (imp, pref) in results.items():
        k = KNOWN[a]
        for pos, want in ((1, k["P2"]), (8, k["P9"])):
            if not want:
                continue
            got = top_residues(pref, pos)
            ok = any(c in want for c in got)
            hits += ok
            tot += 1
            print(f"{a:14s} P{pos+1:<3d} {got:12s} {want:10s} {'YES' if ok else 'no':4s}  "
                  f"{k['note'] if pos == 1 else ''}")
    print("-" * 78)
    print(f"recovered the published anchor preference in {hits} of {tot} tested positions")
    print("\nNothing in the training data labelled these positions or residues. The model saw")
    print("only (allele, peptide, half-life).")
