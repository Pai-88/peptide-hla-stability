"""Is this allele unlike anything we have stability data for?

This is the claim the project rests on, so it gets its own module and its own
honest statement of what it measured.

THE QUESTION. Not "is this allele in the training set" — every allele in the
dropdown is. The useful question is the leave-one-allele-out one: if this allele
had never been measured, how close is the nearest allele that WAS? That is the
situation a real patient puts you in, because stability has been measured for
~75 alleles out of thousands that people actually carry.

So score(allele) = distance from that allele to its nearest neighbour among the
OTHER 74. High score = when this allele is held out, the model has nothing
similar to lean on, and the ranking it gives you is a guess.

TWO BASES, and the app says which one it used:

  "groove"  Hamming distance over the 34 peptide-binding pocket residues of the
            mature class I heavy chain. These are the residues that actually
            determine which peptides fit, so two alleles agreeing here will
            genuinely behave alike. Needs alleles.py to have fetched sequences.

  "name"    Fallback while sequences are still being fetched. Distance from the
            allele NAME: same first field (A*02:01 vs A*02:06) is near, same
            locus is middling, different locus is far. This is a provisional
            stand-in, it is labelled as such on screen, and it must not be
            reported as a result.

MEASURED vs UNMEASURED. stability.txt measured 75 alleles, all of them A or B.
HLA-C was never measured at all. So the dropdown offers two groups:

  measured    the 75 in the data. Scored leave-one-out, against the other 74.
  unmeasured  common alleles with no stability data, HLA-C especially. Scored
              against all 75. These are the real patient case and the reason
              the flag exists; without them the flag never fires on screen.

Peptide-binding residue positions are mature-chain 1-based, the standard class I
pocket set (Saper/Madden pocket definitions, as used for supertype assignment).
"""

import json
import os

import data

GROOVE = [7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81, 84,
          95, 97, 99, 114, 116, 118, 123, 143, 146, 147, 152, 156, 159, 163,
          167, 171]

# Cut points on the 0-1 score. Chosen so that, on the groove basis, alleles with
# a same-first-field sibling in the data land IN, and alleles that are the only
# representative of their locus-family land OUT.
UNMEASURED = "unmeasured.json"

# Cut points on the 0-1 score. Set as counts of differing pocket residues out
# of the 34, not tuned to make the screen look good:
#   6/34  specificity has drifted; the number is no longer reliable.
#   9/34  far enough to be crossing supertypes; the ranking is a guess.
# Run `python ood.py` to see the distribution these cut.
BORDERLINE = 6 / len(GROOVE)
OUT = 9 / len(GROOVE)

_STATE = None
_STAMP = None


def _stamp():
    """Fingerprint of the sequence caches, so a fetch finishing mid-demo is
    picked up on the next prediction instead of needing a restart."""
    out = []
    for p in ("alleles.json", UNMEASURED):
        try:
            st = os.stat(p)
            out.append((p, st.st_mtime_ns, st.st_size))
        except OSError:
            out.append((p, 0, 0))
    return tuple(out)


def _name_key(allele):
    """'HLA-B*14:02(C67S)' -> ('B', '14', '02')."""
    s = allele.replace("HLA-", "").split("(")[0]
    locus, _, rest = s.partition("*")
    f = (rest.split(":") + ["", ""])[:2]
    return locus, f[0], f[1]


def _name_distance(a, b):
    la, g1, _ = _name_key(a)
    lb, g2, _ = _name_key(b)
    if la != lb:
        return 1.0
    if g1 != g2:
        return 0.5
    return 0.1


def _groove(seq):
    if not seq or len(seq) < max(GROOVE):
        return None
    return "".join(seq[p - 1] for p in GROOVE)


def _groove_distance(ga, gb):
    return sum(x != y for x, y in zip(ga, gb)) / len(ga)


def _load_json(path):
    if not os.path.exists(path):
        return {}
    try:
        return json.load(open(path))
    except Exception:  # noqa: BLE001  partial write by a concurrent fetcher
        return {}


def _build():
    df = data.load()
    measured = sorted(df.HLA.unique())
    summary = data.allele_summary(df)

    seqs = _load_json("alleles.json")
    unseqs = _load_json(UNMEASURED)
    unmeasured = sorted(k for k in unseqs if k not in set(measured))

    grooves = {a: _groove(seqs.get(a, {}).get("seq")) for a in measured}
    grooves.update({a: _groove(unseqs[a].get("seq")) for a in unmeasured})

    # Only trust the groove basis once every MEASURED allele has a sequence: a
    # half-full table would compare each allele against a handful of neighbours
    # and understate every distance. Unmeasured alleles without a sequence yet
    # are simply dropped from the dropdown rather than scored on a worse basis.
    n_have = sum(1 for a in measured if grooves[a])
    basis = "groove" if n_have == len(measured) else "name"
    if basis == "groove":
        unmeasured = [a for a in unmeasured if grooves[a]]

    def dist(a, b):
        if basis == "groove":
            return _groove_distance(grooves[a], grooves[b])
        return _name_distance(a, b)

    nearest, nn = {}, {}
    for a in measured + unmeasured:
        # Measured alleles are scored leave-one-out; unmeasured ones have
        # nothing to leave out, so they see all 75.
        others = [b for b in measured if b != a]
        d = {b: dist(a, b) for b in others}
        b = min(d, key=d.get)
        nearest[a], nn[a] = d[b], b

    return {
        "basis": basis,
        "n_with_seq": n_have,
        "measured": measured,
        "unmeasured": unmeasured,
        "alleles": measured + unmeasured,
        "score": nearest,
        "neighbour": nn,
        "support": {a: int(summary.n[a]) for a in measured},
        "censored": {a: float(summary.censored_frac[a]) for a in measured},
        "fallback_seq": sorted(a for a in measured
                               if seqs.get(a, {}).get("reference_fallback")),
    }


def state():
    """Rebuilt automatically if a sequence cache changed on disk."""
    global _STATE, _STAMP
    now = _stamp()
    if _STATE is None or now != _STAMP:
        _STATE, _STAMP = _build(), now
    return _STATE


def alleles():
    return state()["alleles"]


def basis():
    return state()["basis"]


def is_measured(allele):
    return allele in set(state()["measured"])


def score(allele):
    """0 = well covered by measured alleles, 1 = unlike all of them."""
    return state()["score"].get(allele, 1.0)


def verdict(allele):
    """('in'|'borderline'|'out', headline, detail) for the flag on screen."""
    st = state()
    s = score(allele)
    nn = st["neighbour"].get(allele, "\u2014")
    prov = "" if st["basis"] == "groove" else " Provisional allele-name basis \u2014 sequences still downloading."
    where = "binding groove" if st["basis"] == "groove" else "allele name"

    if not is_measured(allele):
        return ("out",
                "NO STABILITY DATA EXISTS FOR THIS ALLELE",
                f"Nobody has ever measured it. Nearest allele that was measured "
                f"is {nn}, {s:.0%} different across the {where}. Treat this "
                f"ranking as a guess.{prov}")

    n = st["support"].get(allele, 0)
    if s >= OUT:
        return ("out",
                "THIS ALLELE IS UNLIKE ANYTHING ELSE MEASURED",
                f"Held out, its nearest measured relative {nn} is {s:.0%} "
                f"different across the {where}. Treat this ranking as a "
                f"guess.{prov}")
    if s >= BORDERLINE:
        return ("borderline",
                "WEAK SUPPORT FOR THIS ALLELE",
                f"Nearest measured relative {nn} differs by {s:.0%} across the "
                f"{where}. Direction is probably right, the number is not.{prov}")
    return ("in",
            "WELL SUPPORTED",
            f"{nn} is a close measured relative ({s:.0%} different); "
            f"{n:,} measurements on this allele itself.{prov}")


def calibrate():
    """Print the score distribution the thresholds are cut from."""
    import numpy as np
    st = state()
    m = np.array([score(a) for a in st["measured"]])
    u = np.array([score(a) for a in st["unmeasured"]]) if st["unmeasured"] else np.array([])
    print(f"basis {st['basis']}  measured {len(m)}  unmeasured {len(u)}")
    q = [0, 10, 25, 50, 75, 90, 100]
    print("measured   percentiles", dict(zip(q, np.round(np.percentile(m, q), 3))))
    if len(u):
        print("unmeasured percentiles", dict(zip(q, np.round(np.percentile(u, q), 3))))
    print(f"thresholds: borderline >= {BORDERLINE}, out >= {OUT}")


if __name__ == "__main__":
    import collections
    st = state()
    print(f"basis: {st['basis']}  sequences: {st['n_with_seq']}/{len(st['measured'])}")
    if st["fallback_seq"]:
        print(f"reference-fallback seqs (not the real allele): {st['fallback_seq']}")
    print()
    calibrate()
    print()
    for group in ("measured", "unmeasured"):
        rows = sorted(((a, score(a)) for a in st[group]), key=lambda kv: -kv[1])
        print(f"--- {group} ({len(rows)}) ---")
        for a, s in rows[:12]:
            print(f"{a:24s} {s:6.3f} nearest {st['neighbour'][a]:24s} {verdict(a)[0]}")
        print()
    print(collections.Counter(verdict(a)[0] for a in st["alleles"]))
