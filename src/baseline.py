"""Fetch and cache published per-peptide NetMHCstabpan predictions.

Source: Fasoulis, Rigo, Antunes, Paliouras, Kavraki, "Transfer learning improves
pMHC kinetic stability and immunogenicity predictions", ImmunoInformatics 13
(2024) 100030, doi:10.1016/j.immuno.2023.100030.
Data: https://github.com/KavrakiLab/TL-MHC (TLStab/misc/datasets/*_pan_results_v2.csv)

These are NetMHCstabpan 1.0 predictions on two EXTERNAL stability datasets
(Ebola virus, Pox virus) that are NOT the NetMHCstabpan training set. They are
therefore the only leak-free NetMHCstabpan numbers we can obtain without
downloading the licensed standalone binary from DTU.

No registration, no licence acceptance, no form submission is involved.

Run:  python baseline.py
Writes: baseline_cache/{Ebola,Pox}_pan_results_v2.csv and baseline_metrics.json
"""

from __future__ import annotations

import json
import pathlib
import urllib.request

import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

BASE = (
    "https://raw.githubusercontent.com/KavrakiLab/TL-MHC/master/"
    "TLStab/misc/datasets/"
)
SETS = {"Ebola": "Ebola_pan_results_v2.csv", "Pox": "Pox_pan_results_v2.csv"}

ROOT = pathlib.Path(__file__).resolve().parent
CACHE = ROOT / "baseline_cache"
TRAIN = ROOT / "stability.txt"

# Columns in the published CSVs that are comparable scores (higher = more stable).
TRUTH = "Stability"  # rescaled experimental half-life, s = 2 ** (-t0 / t_half)
METHODS = ["NetMHCstabpan", "NetMHCpan4.1BA", "NetMHCpan4.1EL", "mhcflurry_presentation_score"]


def fetch() -> dict[str, pd.DataFrame]:
    CACHE.mkdir(exist_ok=True)
    out = {}
    for name, fn in SETS.items():
        dst = CACHE / fn
        if not dst.exists():
            urllib.request.urlretrieve(BASE + fn, dst)
            print(f"downloaded {dst}")
        out[name] = pd.read_csv(dst)
    return out


def leakage_report(df: pd.DataFrame) -> dict:
    """How much of this evaluation set appears in NetMHCstabpan's training data?"""
    if not TRAIN.exists():
        return {"checked": False}
    tr = pd.read_csv(TRAIN, sep=r"\s+")
    tr_pep = set(tr["Pep"])
    tr_allele = {a.replace("*", "").replace("HLA-", "") for a in tr["HLA"]}
    tr_pair = {
        (p, a.replace("*", "").replace("HLA-", ""))
        for p, a in zip(tr["Pep"], tr["HLA"])
    }
    ev_allele = {a.replace("*", "").replace("HLA-", "") for a in df["allele"]}
    pairs = [
        (p, a.replace("*", "").replace("HLA-", ""))
        for p, a in zip(df["peptide"], df["allele"])
    ]
    return {
        "checked": True,
        "n_rows": len(df),
        "peptides_in_train": int(sum(p in tr_pep for p in df["peptide"])),
        "pairs_in_train": int(sum(pr in tr_pair for pr in pairs)),
        "alleles": sorted(ev_allele),
        "alleles_in_train": sorted(ev_allele & tr_allele),
        "alleles_not_in_train": sorted(ev_allele - tr_allele),
    }


def metrics(df: pd.DataFrame) -> dict:
    res = {}
    y = df[TRUTH]
    for m in METHODS:
        if m not in df.columns:
            continue
        x = df[m]
        # mhcflurry_affinity-style columns would need sign flip; presentation score does not.
        res[m] = {
            "pearson_r": round(float(pearsonr(x, y)[0]), 4),
            "spearman_rho": round(float(spearmanr(x, y)[0]), 4),
            "kendall_tau_b": round(float(kendalltau(x, y, variant="b")[0]), 4),
            "n": int(len(df)),
        }
    return res


def main() -> None:
    data = fetch()
    report = {}
    for name, df in data.items():
        report[name] = {
            "metrics_all_rows": metrics(df),
            "per_allele_netmhcstabpan_pearson": {
                a: round(float(pearsonr(g["NetMHCstabpan"], g[TRUTH])[0]), 4)
                for a, g in df.groupby("allele")
                if len(g) >= 10 and g["NetMHCstabpan"].nunique() > 1
            },
            "leakage": leakage_report(df),
        }
    (ROOT / "baseline_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
