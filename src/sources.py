"""Map the dataset's 9-mer peptides back to the proteins they came from.

Exact substring matching only. These peptides are cut intact out of real
proteins, so a 9-mer either occurs in a proteome or it does not. No fuzzy
matching, no "closest hit": a wrong protein label on a slide is worse than
an honest blank.

Build (needs network, once):      python sources.py build
Offline afterwards:               from sources import label; label("VTTEVAFGL")

Caches:
  fasta_cache/*.fasta     reviewed (Swiss-Prot) sequences, one file per source
  peptide_sources.json    peptide -> list of matches, for the demo app
"""

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

FASTA_DIR = "fasta_cache"
CACHE = "peptide_sources.json"
STREAM = "https://rest.uniprot.org/uniprotkb/stream"

# Swiss-Prot (reviewed) only. Small downloads, clean protein names.
# Human is the reference proteome; the rest are the pathogens these
# published epitope panels are overwhelmingly drawn from.
SOURCES = {
    "human":       ("Human",                 "(proteome:UP000005640) AND (reviewed:true)"),
    "hiv1":        ("HIV-1",                 "(taxonomy_id:11676) AND (reviewed:true)"),
    "influenza_a": ("Influenza A",           "(taxonomy_id:11320) AND (reviewed:true)"),
    "ebv":         ("Epstein-Barr virus",    "(taxonomy_id:10376) AND (reviewed:true)"),
    "cmv":         ("Cytomegalovirus",       "(taxonomy_id:10358) AND (reviewed:true)"),
    "hbv":         ("Hepatitis B virus",     "(taxonomy_id:10407) AND (reviewed:true)"),
    # HCV and HPV have no reviewed entries at the species/family node itself,
    # so match them by organism name instead of taxon id.
    "hcv":         ("Hepatitis C virus",     '(organism_name:"Hepatitis C virus") AND (reviewed:true)'),
    "hpv":         ("Human papillomavirus",  '(organism_name:"Human papillomavirus") AND (reviewed:true)'),
    "sars_cov_2":  ("SARS-CoV-2",            "(taxonomy_id:2697049) AND (reviewed:true)"),
    "htlv1":       ("HTLV-1",                "(taxonomy_id:11908) AND (reviewed:true)"),
    "pfalciparum": ("P. falciparum",         "(taxonomy_id:5833) AND (reviewed:true)"),
    "dengue":      ("Dengue virus",          "(taxonomy_id:12637) AND (reviewed:true)"),
    "vaccinia":    ("Vaccinia virus",        "(taxonomy_id:10245) AND (reviewed:true)"),
    "mtb":         ("M. tuberculosis",       "(taxonomy_id:1773) AND (reviewed:true)"),
}

K = 9


# ---------------------------------------------------------------- download

def fetch(key, force=False):
    """Download one source's reviewed FASTA to fasta_cache/. Idempotent."""
    path = os.path.join(FASTA_DIR, key + ".fasta")
    if os.path.exists(path) and not force:
        return path
    os.makedirs(FASTA_DIR, exist_ok=True)
    url = f"{STREAM}?query={urllib.parse.quote(SOURCES[key][1])}&format=fasta"
    t = time.time()
    with urllib.request.urlopen(url, timeout=600) as r, open(path, "wb") as f:
        f.write(r.read())
    print(f"  {key:12s} {os.path.getsize(path)/1e6:6.1f} MB  {time.time()-t:5.1f}s")
    return path


# ---------------------------------------------------------------- parsing

HEADER = re.compile(r"^>(?:sp|tr)\|([^|]+)\|\S+\s+(.*?)\s+OS=(.*?)\s+(?:OX|GN|PE|SV)=")


def parse_fasta(path):
    """Yield (accession, protein_name, organism, sequence)."""
    acc = name = org = None
    seq = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if acc:
                    yield acc, name, org, "".join(seq)
                m = HEADER.match(line)
                if m:
                    acc, name, org = m.group(1), m.group(2), m.group(3)
                else:  # unexpected header shape; keep the accession, drop the rest
                    acc, name, org = line[1:].split()[0], line[1:].strip(), "?"
                seq = []
            else:
                seq.append(line.strip())
    if acc:
        yield acc, name, org, "".join(seq)


def build_index(keys=None):
    """k-mer -> list of (acc, protein, organism, source_key, 1-based start)."""
    index = {}
    for key in keys or SOURCES:
        path = os.path.join(FASTA_DIR, key + ".fasta")
        n_prot = n_res = 0
        for acc, name, org, seq in parse_fasta(path):
            n_prot += 1
            n_res += len(seq)
            for i in range(len(seq) - K + 1):
                index.setdefault(seq[i:i + K], []).append(
                    (acc, name, org, key, i + 1))
        print(f"  {key:12s} {n_prot:6d} proteins  {n_res/1e6:6.2f}M residues")
    return index


# ---------------------------------------------------------------- lookup

_sources = None


def _load():
    global _sources
    if _sources is None:
        with open(CACHE) as f:
            _sources = json.load(f)
    return _sources


_proteome = None


def _scan(peptide):
    """Exact scan of the cached FASTA for a peptide not in the JSON cache.

    The JSON cache only covers the 5,633 dataset peptides; the demo can be
    typed any 9-mer. First call loads ~16 MB of sequence (~2 s), then it is
    fast. Offline: reads only fasta_cache/.
    """
    global _proteome
    if _proteome is None:
        _proteome = []
        for key in SOURCES:
            path = os.path.join(FASTA_DIR, key + ".fasta")
            if os.path.exists(path):
                _proteome += [(a, n, o, key, s) for a, n, o, s in parse_fasta(path)]
    out = []
    for acc, name, org, key, seq in _proteome:
        i = seq.find(peptide)
        while i != -1:
            out.append({"acc": acc, "protein": name, "organism": org,
                        "source": key, "start": i + 1, "end": i + len(peptide)})
            i = seq.find(peptide, i + 1)
    return out


def matches(peptide, scan=True):
    """All exact source-protein matches for a peptide. [] if unmatched.

    Each match: {acc, protein, organism, source, start, end}.
    Dataset peptides come from the JSON cache; anything else falls back to an
    exact scan of the cached FASTA (set scan=False to skip that).
    """
    p = peptide.strip().upper()
    hits = _load().get(p)
    if hits is None and scan:
        hits = _scan(p)
    return hits or []


def label(peptide, default="", scan=True):
    """Short display string for the demo, e.g.

        "SLYNTVATL, Gag polyprotein (HIV-1), residues 77-85"

    Collapses the many strain-level duplicates onto the commonest protein
    name, and says so when the peptide is shared across organisms.
    Returns `default` when nothing matched: it never guesses.
    """
    ms = matches(peptide, scan=scan)
    if not ms:
        return default
    from collections import Counter
    by_name = Counter((m["source"], m["protein"]) for m in ms)
    (src, name), _ = by_name.most_common(1)[0]
    first = next(m for m in ms if m["source"] == src and m["protein"] == name)
    short = SOURCES.get(src, (first["organism"],))[0]
    n_org = len({m["source"] for m in ms})
    extra = f" (also in {n_org - 1} other organism{'s' * (n_org > 2)})" if n_org > 1 else ""
    return (f"{peptide.strip().upper()}, {name} ({short}), "
            f"residues {first['start']}-{first['end']}{extra}")


# ---------------------------------------------------------------- build CLI

def main():
    import data

    print("downloading reviewed (Swiss-Prot) FASTA ...")
    for key in SOURCES:
        fetch(key)
    total = sum(os.path.getsize(os.path.join(FASTA_DIR, k + ".fasta"))
                for k in SOURCES)
    print(f"  cache total {total/1e6:.1f} MB\n")

    print("indexing 9-mers ...")
    t = time.time()
    index = build_index()
    print(f"  {len(index):,} distinct 9-mers in {time.time()-t:.1f}s\n")

    peps = sorted(data.load().Pep.unique())
    out, per_source, unmatched = {}, {}, []
    for p in peps:
        hits = index.get(p, [])
        if not hits:
            unmatched.append(p)
            continue
        out[p] = [{"acc": a, "protein": n, "organism": o, "source": s,
                   "start": i, "end": i + K - 1} for a, n, o, s, i in hits]
        for s in {h[3] for h in hits}:
            per_source[s] = per_source.get(s, 0) + 1

    with open(CACHE, "w") as f:
        json.dump(out, f)

    n = len(peps)
    print(f"peptides           {n}")
    print(f"matched            {len(out)} ({100*len(out)/n:.1f}%)")
    print(f"UNMATCHED          {len(unmatched)} ({100*len(unmatched)/n:.1f}%)")
    print("\nper source (a peptide can hit several):")
    for k, (nice, _) in SOURCES.items():
        c = per_source.get(k, 0)
        print(f"  {nice:22s} {c:5d}  {100*c/n:5.1f}%")
    multi = sum(1 for v in out.values() if len({m['acc'] for m in v}) > 1)
    print(f"\nmatched >1 protein {multi}")
    print(f"cache {CACHE} {os.path.getsize(CACHE)/1e6:.1f} MB")
    print("\nexamples:")
    for p in peps[:5] + unmatched[:3]:
        print("  " + (label(p) or p + "  -> no exact match"))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        main()
    else:
        print(__doc__)
