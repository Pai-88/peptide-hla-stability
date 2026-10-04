"""Fetch sequences for alleles the dataset NEVER measured.

Separate cache from alleles.json so this cannot race the dataset fetcher.
HLA-C is absent from stability.txt entirely -- all 75 measured alleles are A or
B -- so every C allele a patient carries is unmeasured. Plus a few A/B first
fields with no representative in the data at all.
"""
import json, alleles

WANTED = [
    # HLA-C: the whole locus is unmeasured. C*07:01/07:02 are among the most
    # common class I alleles worldwide.
    "HLA-C*07:01", "HLA-C*07:02", "HLA-C*04:01", "HLA-C*06:02", "HLA-C*05:01",
    "HLA-C*03:04", "HLA-C*02:02", "HLA-C*12:03",
    # B first fields with no representative in stability.txt.
    "HLA-B*52:01", "HLA-B*53:01", "HLA-B*38:01", "HLA-B*49:01", "HLA-B*50:01",
    "HLA-B*37:01", "HLA-B*47:01", "HLA-B*73:01",
    # Divergent A alleles with no first-field representative.
    "HLA-A*36:01", "HLA-A*34:02",
]
if __name__ == "__main__":
    out = alleles.fetch_all(WANTED, cache="unmeasured.json")
    print(f"\n{len(out)} unmeasured alleles cached")
    print("fallbacks:", [k for k, v in out.items() if v["reference_fallback"]])
