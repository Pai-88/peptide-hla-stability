"""Which pseudo-sequence positions line which peptide anchor pocket? DERIVED, not quoted.

The diagnostics in this project say the signal concentrates at peptide positions P2 and
P9, and that the preferred residue there is ALLELE-SPECIFIC (A*02:01 wants L at P2,
B*27:05 wants R, B*07:02 wants P). That is an INTERACTION between a peptide anchor and
the groove residues that hold it. Arms A and I only concatenate the two blocks and hope
the network finds it.

To encode the interaction explicitly we need to know which of the 34 pseudo-sequence
positions pair with which peptide position. Published pocket definitions exist; this file
deliberately does not use them. It measures the pairing from the training rows, and the
published assignment is only ever used afterwards as an independent check (see __main__).

-------------------------------------------------------------------------------
THE STATISTIC
-------------------------------------------------------------------------------
For a peptide position p:

  1. r = y - mean(y | allele).  Removes the allele main effect, which is huge (alleles
     differ enormously in mean half-life) and has nothing to do with motif.
  2. Profile  P[a, aa] = weighted mean of r over rows with allele a and pep[p] == aa.
     This is "how much does allele a like residue aa at position p".
  3. Subtract the across-allele mean profile. What is left, Pc, is the ALLELE-SPECIFIC
     part of the position-p preference: exactly the interaction we want to explain.
  4. For pseudo position q, group the alleles by the residue they carry at q and compute
     eta2[p, q] = (weighted SS of the group-mean profiles) / (weighted SS of Pc).
     That is the fraction of allele-specific position-p preference explained by knowing
     one residue of the groove.

eta2 is biased upward by the number of distinct residues at q (more groups fit more), so
it is never used raw to assign a pocket. The assignment uses

  spec[p, q] = eta2[p, q] - mean over all nine p' of eta2[p', q]

which compares a position against ITSELF across peptide positions, so the group-count
bias cancels exactly. A pseudo position that merely tags allele-cluster identity raises
every eta2[., q] together and gets spec ~ 0 everywhere; a position that really lines one
pocket is high for one p and low for the rest.

A matched random-grouping null (same number of groups, allele labels shuffled) is
reported alongside, so "how big is eta2 for free" is measured rather than asserted.

-------------------------------------------------------------------------------
LEAKAGE
-------------------------------------------------------------------------------
fit_pockets() must only ever be handed TRAIN rows. The featurizers in arm_M_interact.py
call it once per fold on that fold's training side and assert that the held-out alleles
were absent. Deriving the pockets once on the full table would be a (small) use of test
labels for feature selection, so it is not done anywhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}
PEP_LEN = 9
PSEUDO_LEN = 34

MIN_CELL = 3        # rows needed before an (allele, residue) profile cell is trusted
MIN_GROUP = 2       # alleles needed before a pseudo-residue group is trusted

_PSEUDO = None


def pseudo_map(path="alleles.json"):
    """allele -> 34-residue pseudo-sequence. Same source arm A uses."""
    global _PSEUDO
    if _PSEUDO is None:
        d = json.load(open(path))
        _PSEUDO = {k: v["pseudo"] for k, v in d.items()}
        bad = [k for k, v in _PSEUDO.items() if not v or len(v) != PSEUDO_LEN]
        if bad:
            raise ValueError(f"alleles without a 34-mer pseudo-sequence: {bad}")
    return _PSEUDO


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------

def _profiles(df, p):
    """(Pc, W, alleles): allele-specific preference profile at peptide position p.

    Pc[i, j] is allele i's deviation-from-average liking of residue j at position p,
    W[i, j] the number of rows behind it. Cells with fewer than MIN_CELL rows are
    zero-weighted rather than dropped, so every allele keeps the same 20 columns.
    """
    alleles = np.array(sorted(df.HLA.unique()))
    a_idx = {a: i for i, a in enumerate(alleles)}
    ai = df.HLA.map(a_idx).to_numpy()
    res = np.array([AA_IDX.get(s[p], -1) for s in df.Pep.to_numpy()])
    y = df.y.to_numpy(dtype=np.float64)

    # 1. allele main effect out
    amean = np.bincount(ai, weights=y, minlength=len(alleles)) / np.bincount(ai, minlength=len(alleles))
    r = y - amean[ai]

    ok = res >= 0
    flat = ai[ok] * 20 + res[ok]
    n = np.bincount(flat, minlength=len(alleles) * 20).reshape(len(alleles), 20)
    s = np.bincount(flat, weights=r[ok], minlength=len(alleles) * 20).reshape(len(alleles), 20)
    W = np.where(n >= MIN_CELL, n, 0).astype(np.float64)
    P = np.divide(s, n, out=np.zeros_like(s), where=n > 0)

    # 2. across-allele mean profile out -> what is left is allele SPECIFIC
    col_w = W.sum(axis=0)
    gm = np.divide((P * W).sum(axis=0), col_w, out=np.zeros(20), where=col_w > 0)
    Pc = (P - gm) * (W > 0)
    return Pc, W, alleles


def _eta2(Pc, W, groups):
    """Fraction of the weighted SS of Pc explained by a grouping of the alleles."""
    tss = float((W * Pc ** 2).sum())
    if tss <= 0:
        return np.nan
    ess = 0.0
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < MIN_GROUP:
            continue
        w = W[m]
        cw = w.sum(axis=0)
        gmean = np.divide((Pc[m] * w).sum(axis=0), cw, out=np.zeros(20), where=cw > 0)
        ess += float((w * gmean ** 2).sum())
    return ess / tss


def anchor_variance(df):
    """var_spec[p]: how much the position-p residue preference DIFFERS BETWEEN ALLELES.

    The weighted mean square of Pc, in (log10 hours)^2. This is the quantity that says
    "position p is an allele-specific anchor" and it is what fit_pockets ranks anchors
    by. eta2 below is a FRACTION of this, so it is not comparable across p: a position
    with almost no allele-specific preference can still have most of the little it has
    explained by one groove residue. Ranking anchors by summed eta2 picks up exactly
    that artefact, which is why it is not used.
    """
    out = np.zeros(PEP_LEN)
    for p in range(PEP_LEN):
        Pc, W, _ = _profiles(df, p)
        out[p] = float((W * Pc ** 2).sum() / W.sum()) if W.sum() else 0.0
    return out


def interaction_table(df, pseudo=None, n_null=20, seed=0):
    """eta2[p, q] for all 9 peptide x 34 pseudo positions, plus a matched null.

    Returns (eta2, null, spec, n_groups) as (9, 34) / (9, 34) / (9, 34) / (34,) arrays.
    `null` is the mean eta2 from random allele groupings with the SAME group sizes, so
    the chance level of every cell is measured rather than assumed.
    """
    pseudo = pseudo or pseudo_map()
    rng = np.random.default_rng(seed)
    eta = np.full((PEP_LEN, PSEUDO_LEN), np.nan)
    null = np.full((PEP_LEN, PSEUDO_LEN), np.nan)
    ngroups = np.zeros(PSEUDO_LEN, dtype=int)

    cached = [_profiles(df, p) for p in range(PEP_LEN)]
    alleles = cached[0][2]
    pseq = np.array([list(pseudo[a]) for a in alleles])          # (n_alleles, 34)

    for q in range(PSEUDO_LEN):
        g = pd.factorize(pseq[:, q])[0]
        ngroups[q] = len(np.unique(g))
        perms = [rng.permutation(g) for _ in range(n_null)]
        for p in range(PEP_LEN):
            Pc, W, _ = cached[p]
            eta[p, q] = _eta2(Pc, W, g)
            null[p, q] = (float(np.mean([_eta2(Pc, W, gp) for gp in perms]))
                          if perms else 0.0)

    # A pseudo column with a single residue across the training alleles explains
    # nothing by construction; treat it as 0 rather than letting a NaN propagate.
    e0 = np.nan_to_num(eta, nan=0.0)
    spec = e0 - e0.mean(axis=0, keepdims=True)
    return eta, null, spec, ngroups


# ---------------------------------------------------------------------------
# the derived specification the featurizers consume
# ---------------------------------------------------------------------------

@dataclass
class PocketSpec:
    """Which pseudo positions to pair with which peptide position."""
    pockets: dict = field(default_factory=dict)       # pep position -> [pseudo positions]
    anchors: list = field(default_factory=list)       # pep positions used, in order
    eta2: np.ndarray = None
    spec: np.ndarray = None
    var_spec: np.ndarray = None
    n_train_alleles: int = 0
    train_alleles: frozenset = frozenset()

    def describe(self):
        return "; ".join(f"P{p+1}<-{'/'.join(str(q+1) for q in self.pockets[p])}"
                         for p in self.anchors)


def fit_pockets(df_train, anchors=None, k=6, n_anchors=2, pseudo=None, n_null=0, seed=0):
    """Derive the pocket memberships from TRAIN rows only.

    anchors    peptide positions (0-indexed) to build pockets for. None = the
               n_anchors positions with the largest allele-specific preference
               variance, measured on these same training rows. 'all' = all nine.
    k          pseudo positions per pocket, taken by `spec` (eta2 minus that pseudo
               position's own across-peptide-position mean, which cancels the
               group-count bias). k >= 34 means every pseudo position.
    """
    eta, null, spec, ngroups = interaction_table(df_train, pseudo, n_null=n_null, seed=seed)
    var_spec = anchor_variance(df_train)
    if anchors is None:
        anchors = sorted(np.argsort(-var_spec)[:n_anchors].tolist())
    elif anchors == "all":
        anchors = list(range(PEP_LEN))
    pockets = {}
    for p in anchors:
        pockets[p] = (list(range(PSEUDO_LEN)) if k >= PSEUDO_LEN
                      else sorted(int(q) for q in np.argsort(-spec[p])[:k]))
    return PocketSpec(pockets=pockets, anchors=[int(a) for a in anchors], eta2=eta,
                      spec=spec, var_spec=var_spec,
                      n_train_alleles=int(df_train.HLA.nunique()),
                      train_alleles=frozenset(df_train.HLA.unique()))


# ---------------------------------------------------------------------------
# an independent, MEASURED cross-check: contacts in a real pMHC crystal structure
# ---------------------------------------------------------------------------

import alleles as _alleles       # for PSEUDO_POS, the HLA residue numbers behind the 34


def _pdb_atoms(path):
    """{(chain, resnum): (n_atoms, 3) heavy-atom coordinates}."""
    out = {}
    for line in open(path):
        if line.startswith("ATOM") and line[76:78].strip() != "H":
            key = (line[21], int(line[22:26]))
            out.setdefault(key, []).append(
                [float(line[30 + 8 * i:38 + 8 * i]) for i in range(3)])
    return {k: np.array(v) for k, v in out.items()}


def structural_contacts(path="3gsn.pdb.txt", mhc_chain="H", pep_chain="P"):
    """D[p, q] = minimum heavy-atom distance (A) between peptide residue p+1 and the
    HLA residue behind pseudo-sequence column q.

    This uses no stability labels at all, so it cannot leak anything into the
    evaluation. It is a measurement off a deposited structure, not a remembered
    pocket definition: the 34 HLA residue numbers come from alleles.PSEUDO_POS, which
    that module validated by reproducing NetMHCpan's own pseudo-sequence file.
    """
    at = _pdb_atoms(path)
    npep = max(r for c, r in at if c == pep_chain)
    D = np.full((npep, PSEUDO_LEN), np.nan)
    for i in range(1, npep + 1):
        pa = at.get((pep_chain, i))
        if pa is None:
            continue
        for j, q in enumerate(_alleles.PSEUDO_POS):
            ha = at.get((mhc_chain, q))
            if ha is None:
                continue
            D[i - 1, j] = np.sqrt(((pa[:, None, :] - ha[None, :, :]) ** 2).sum(-1)).min()
    return D


def structural_pockets(anchors, k=6, path="3gsn.pdb.txt", **kw):
    """{pep position: [pseudo positions]} by nearest measured contact distance."""
    D = structural_contacts(path, **kw)
    out = {}
    for p in anchors:
        row = D[p] if p < len(D) else D[-1]
        out[p] = (list(range(PSEUDO_LEN)) if k >= PSEUDO_LEN
                  else sorted(int(q) for q in np.argsort(row)[:k]))
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _report():
    import data
    import splits

    df = data.load()
    print("=" * 78)
    print("INTERACTION MAP, measured on all rows (report only; the arms refit per fold)")
    print("=" * 78)
    eta, null, spec, ng = interaction_table(df, n_null=20)
    vs = anchor_variance(df)

    print("\nALLELE-SPECIFIC preference variance by PEPTIDE position, (log10 h)^2.\n"
          "How much the residue preference at that position differs between alleles:\n")
    for p in range(PEP_LEN):
        bar = "#" * int(round(40 * vs[p] / vs.max()))
        print(f"  P{p+1}  {vs[p]:7.4f}  {bar}")
    top2 = sorted(np.argsort(-vs)[:2] + 1)
    print(f"\n  -> the two strongest anchor positions, derived: P{top2[0]} and P{top2[1]}")
    print("     (the project's own mutant-scan diagnostic, anchors.py, independently "
          "names\n      P2 and P9; this measurement puts P9 first and P1 just above P2.)")

    print("\n" + "=" * 78)
    print("PER PSEUDO POSITION: which peptide position does it explain best?")
    print("=" * 78)
    print(f"{'q':>3} {'ngrp':>4} {'eta2(P2)':>9} {'null':>6} {'eta2(P9)':>9} {'null':>6} "
          f"{'argmax':>7} {'spec(P2)':>9} {'spec(P9)':>9}")
    for q in range(PSEUDO_LEN):
        am = int(np.nanargmax(eta[:, q])) + 1
        print(f"{q+1:>3} {ng[q]:>4} {eta[1, q]:>9.3f} {null[1, q]:>6.3f} "
              f"{eta[8, q]:>9.3f} {null[8, q]:>6.3f} {'P'+str(am):>7} "
              f"{spec[1, q]:>+9.3f} {spec[8, q]:>+9.3f}")

    sp = fit_pockets(df, k=6)
    print(f"\nDERIVED FROM LABELS (k=6): {sp.describe()}")

    print("\n" + "=" * 78)
    print("INDEPENDENT CHECK: measured contacts in 3gsn (a 9-mer pMHC crystal "
          "structure\nalready in this repo). No stability labels involved.")
    print("=" * 78)
    D = structural_contacts()
    print(f"{'pep':>4}  nearest 6 pseudo columns by min heavy-atom distance (A)")
    for p in range(9):
        o = np.argsort(D[p])[:6]
        print(f"  P{p+1}  " + "  ".join(f"q{q+1}({D[p, q]:.1f})" for q in o))
    print("\n  label-derived vs structure-derived pocket overlap, k=6:")
    sdp = structural_pockets(sp.anchors, k=6)
    for p in sp.anchors:
        a, b = set(sp.pockets[p]), set(sdp[p])
        print(f"    P{p+1}: label {[q+1 for q in sorted(a)]}  "
              f"structure {[q+1 for q in sorted(b)]}  overlap {len(a & b)}/6")
    print("\n  The two disagree. The pseudo-sequence columns are strongly correlated\n"
          "  across the 75 alleles (co-inherited groove haplotypes), so a column can\n"
          "  explain an allele's anchor preference without touching the peptide. Both\n"
          "  pocket definitions are therefore carried forward as candidates and the\n"
          "  choice is made on inner splits of TRAIN, never on a test fold.")

    print("\n" + "=" * 78)
    print("STABILITY of the derivation across the 21 training sides")
    print("=" * 78)
    folds = splits.choose_held_out(df)
    full = {p: set(sp.pockets[p]) for p in sp.anchors}
    agree = []
    for name, alleles in folds:
        tr = df[~df.HLA.isin(alleles)]
        s = fit_pockets(tr, k=6, n_null=4)
        ov = {p: len(set(s.pockets.get(p, [])) & full.get(p, set())) for p in s.anchors}
        agree.append((name, s.anchors, ov))
        print(f"  {name:<18} anchors {[p+1 for p in s.anchors]}  overlap with full-data "
              f"pockets {[f'P{p+1}:{v}/6' for p, v in ov.items()]}")
    same = sum(1 for _, a, _ in agree if a == sp.anchors)
    print(f"\n  same two anchor positions on {same}/{len(folds)} training sides")


if __name__ == "__main__":
    _report()
