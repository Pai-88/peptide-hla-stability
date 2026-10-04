"""Fetch the real protein sequence for every HLA allele in the dataset.

Source: IPD-IMGT/HLA REST API (https://www.ebi.ac.uk/cgi-bin/ipd/api/).
The allele record carries sequence.protein (full precursor, signal + mature)
and feature.protein, which gives the signal-peptide length. HLA residue
numbering is of the MATURE chain, so mature = protein[signal_len:].

Three alleles in the dataset are lab constructs written "HLA-B*14:02(C67S)".
We strip the parenthetical, fetch the base allele, and apply the substitution
at mature position 67 ourselves (asserting the wild-type residue is C first).

Output: alleles.json, one entry per dataset allele string:
    {"seq": <mature chain>, "source": "IPD", "ipd_name": ..., "ipd_accession": ...,
     "mutation": "C67S" or null, "reference_fallback": false}
`reference_fallback` is true only if we had to fall back to the UniProt locus
reference sequence instead of the true allele. That is a scientifically
different object and must never be silent.
"""

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

API = "https://www.ebi.ac.uk/cgi-bin/ipd/api"
CACHE = "alleles.json"
MIN_MATURE = 330  # full mature class I heavy chain
MIN_PARTIAL = 170  # exon 2-3 only: alpha1+alpha2, the binding groove  # class I mature heavy chain; shorter = partial cDNA record

# Locus reference proteins, used ONLY as a last-resort fallback.
UNIPROT_REF = {"A": "P04439", "B": "P01889", "C": "P10321"}


def _get(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            print(f"  retry {attempt + 1} ({e})", file=sys.stderr)
            time.sleep(2 * (attempt + 1))


def parse_allele(s):
    """'HLA-B*14:02(C67S)' -> ('B*14:02', 'C67S'). 'HLA-A*02:01' -> ('A*02:01', None)."""
    m = re.fullmatch(r"HLA-([A-C]\*\d+:\d+)(?:\(([A-Z]\d+[A-Z])\))?", s)
    if not m:
        raise ValueError(f"cannot parse allele {s!r}")
    return m.group(1), m.group(2)


def search(base):
    """All IPD alleles named `base` or `base`:..., following pagination.

    Older alleles (A*43:01, B*08:03) have no sub-fields at all, so we query the
    bare prefix and filter, rather than requiring a trailing colon.
    """
    q = urllib.parse.quote(f'startsWith(name,"{base}")', safe="")
    url = f"{API}/allele?query={q}"
    out = []
    while url:
        d = _get(url)
        out += d["data"]
        nxt = d.get("meta", {}).get("next")
        url = f"{API}/allele{nxt}" if nxt else None
    return [h for h in out if h["name"] == base or h["name"].startswith(base + ":")]


def rank(base, hits):
    """Candidates in preference order: lowest-numbered sub-allele first.

    Field count is NOT a good tiebreak on its own: a 3-field record like
    A*23:01:02 can be a partial cDNA while A*23:01:01:01 is complete. So we
    order by the numeric sub-fields and let the caller reject short sequences.
    """
    def key(h):
        f = h["name"].split(":")[2:]
        return ([int(re.sub(r"\D", "", x) or 0) for x in f] + [0, 0])[:3] + [len(f)]

    if not hits:
        raise LookupError(f"no IPD allele for {base}")
    return sorted(hits, key=key)


def mature_from_record(rec):
    """Return (mature_chain, offset).

    offset = how many mature residues are missing from the FRONT. It is 0 for
    full-length records. Exon-2-only cDNA records start at CDS nt 74, but the
    mature chain starts at nt 73, so their first residue is actually mature
    residue 2 and indexing them from 1 would shift every position by one.
    """
    prot = rec["sequence"]["protein"]
    feats = rec["feature"]["protein"]
    sig = next((f["length"] for f in feats if f["type"] == "signal"), 24)
    mat = next((f for f in feats if f["type"] == "mature"), None)
    seq = prot[sig:]
    if mat:
        seq = prot[mat["start"] - 1: mat["start"] - 1 + mat["length"]]
    seq = seq.replace("*", "").replace("X", "")

    offset = 0
    st = rec.get("sequence_status") or {}
    if not st.get("full", True) and st.get("type") == "cDNA":
        mature_cds_start = 3 * sig + 1
        if st.get("start", 1) > mature_cds_start:
            offset = (st["start"] - mature_cds_start + 2) // 3
    return seq, offset


def apply_mutation(seq, mut, offset=0):
    """C67S: mature residue 67 (1-based) must be C, becomes S."""
    wt, pos, new = mut[0], int(mut[1:-1]), mut[-1]
    pos -= offset
    got = seq[pos - 1]
    if got != wt:
        raise AssertionError(f"{mut}: mature position {pos} is {got}, not {wt}")
    return seq[: pos - 1] + new + seq[pos:]


def uniprot_ref(locus):
    url = f"https://rest.uniprot.org/uniprotkb/{UNIPROT_REF[locus]}.fasta"
    with urllib.request.urlopen(url, timeout=30) as r:
        txt = r.read().decode()
    prot = "".join(l.strip() for l in txt.splitlines() if not l.startswith(">"))
    return prot[24:]  # 24-residue signal peptide


def fetch_all(allele_strings, cache=CACHE):
    out = json.load(open(cache)) if os.path.exists(cache) else {}
    for s in allele_strings:
        if s in out:
            continue
        base, mut = parse_allele(s)
        try:
            best = None
            # Walk candidates until one has a full-length mature chain. Class I
            # heavy chains are ~338-365 aa; anything much shorter is a partial
            # cDNA record and would silently corrupt the embedding.
            for hit in rank(base, search(base))[:8]:
                rec = _get(f"{API}/allele/{hit['accession']}")
                seq, off = mature_from_record(rec)
                if best is None or len(seq) > len(best[1]):
                    best = (hit, seq, off)
                if len(seq) >= MIN_MATURE:
                    break
            hit, seq, off = best
            if len(seq) < MIN_PARTIAL:
                raise LookupError(f"sequence too short ({len(seq)} aa) for {base}")
            partial = len(seq) < MIN_MATURE
            # A few old alleles are deposited as exon 2-3 cDNA only (~181 aa).
            # That is the alpha1+alpha2 pair, i.e. the whole peptide-binding
            # groove, and it is the allele's REAL sequence. Keeping it, flagged,
            # beats substituting a different allele's reference sequence.
            entry = {"seq": seq, "source": "IPD-partial" if partial else "IPD",
                     "ipd_name": hit["name"], "ipd_accession": hit["accession"],
                     "mutation": mut, "reference_fallback": False, "partial": partial,
                     "offset": off}
        except Exception as e:  # noqa: BLE001
            print(f"  !! IPD failed for {s}: {e} -> UniProt reference", file=sys.stderr)
            entry = {"seq": uniprot_ref(base[0]), "source": f"UniProt/{UNIPROT_REF[base[0]]}",
                     "ipd_name": None, "ipd_accession": None, "mutation": mut,
                     "reference_fallback": True, "partial": False, "offset": 0}
        if mut:
            entry["seq"] = apply_mutation(entry["seq"], mut, entry["offset"])
        out[s] = entry
        print(f"{s:22s} {entry['ipd_name'] or entry['source']:20s} {len(entry['seq'])} aa")
        json.dump(out, open(cache, "w"), indent=1)
    return out


# The 34 NetMHCpan pseudo-sequence positions: polymorphic residues within 4.0 A
# of the peptide. 1-based over the MATURE chain. Source: netMHCpan's own bundled
# data/all_varcontacts.nlist (identical byte-for-byte in netMHCstabpan-1.0),
# described in Nielsen et al., PLoS ONE 2007;2(8):e796.
# Verified, not trusted: indexing our 69 full-length non-mutant IPD sequences at
# these positions reproduces NetMHCpan's MHC_pseudo.dat string exactly, 69 of 69.
PSEUDO_POS = [7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81,
              84, 95, 97, 99, 114, 116, 118, 143, 147, 150, 152, 156, 158, 159,
              163, 167, 171]


def pseudo(seq, offset=0):
    """The 34-residue pseudo-sequence, or None if the chain does not reach 171."""
    if len(seq) + offset < max(PSEUDO_POS) or offset >= min(PSEUDO_POS):
        return None
    return "".join(seq[p - 1 - offset] for p in PSEUDO_POS)


def add_pseudo(cache=CACHE):
    d = json.load(open(cache))
    for v in d.values():
        v["pseudo"] = pseudo(v["seq"], v.get("offset", 0))
    json.dump(d, open(cache, "w"), indent=1)
    return d


def load(cache=CACHE):
    return json.load(open(cache))


if __name__ == "__main__":
    import data

    alleles = sorted(data.load().HLA.unique())
    out = fetch_all(alleles)
    out = add_pseudo()
    fb = [k for k, v in out.items() if v["reference_fallback"]]
    pa = [k for k, v in out.items() if v.get("partial")]
    print(f"\n{len(out)} alleles; real per-allele sequences: {len(out) - len(fb)}; "
          f"reference fallbacks: {len(fb)} {fb}")
    print(f"partial (exon 2-3, binding groove only): {len(pa)} {pa}")
    ps = {v["pseudo"] for v in out.values()}
    print(f"pseudo-sequences: {sum(v['pseudo'] is not None for v in out.values())}/{len(out)}, {len(ps)} distinct")
    print(f"distinct sequences: {len({v['seq'] for v in out.values()})}")
