# Peptide-HLA class I stability: do protein foundation models help?

**Short answer: no.** On held-out binding grooves, a conventional supervised network beats
every protein-language-model configuration we tested, across three encoder families and
eleven-plus configurations.

Built for the Serova "drug and protein design" track at the
**London AI x Science Hackathon**, 3 to 4 October 2026.
Team **Fridaymerchants**: Paing Hein Htet, Radin Moradi, Sina Ahani, Andy Kaça.

Serova's brief asked whether protein foundation models are *useful* for this problem, not to
prove that they are. We took that literally.

**Live site: https://peptidehla.com**  
Ten models, a per-allele peptide lookup against the measured value, interactive 3D
structures, the anchor-optimisation scan, and a tiered abacavir / HLA-B\*57:01 case study.

---

## The result

Median per-allele Spearman across **21 leave-one-groove-cluster-out folds**.
Higher is better; 0 is no skill, 1 is a perfect ordering.

| | Spearman |
|---|---|
| **Conventional supervised net** — BLOSUM62 + one-hot, tuned MLP, no foundation model | **0.265** |
| ESM-2 150M un-pooled, tuned head — best foundation-model arm | 0.164 |
| ESM-2 150M fine-tuned end to end | 0.127 |
| ESM-2 650M, mean-pooled — 4x the parameters | 0.122 |
| ESM-C 600M un-pooled — EvolutionaryScale | 0.119 |
| SaProt 650M — structure-aware vocabulary, run without structure tokens | 0.115 |
| ProtT5 1.2B un-pooled — Rostlab | 0.098 |
| ESM-2 150M off the shelf, mean-pooled — how most people use it | 0.092 |
| CONTROL: peptide only, never told which allele | 0.056 |
| NULL: one constant per allele | 0.000 |

Expressed as pairwise concordance, the conventional net orders a random pair of peptides
correctly **61.5%** of the time. Off-the-shelf ESM-2 manages **54.1%**. A coin is 50%.

### Which estimator

Spearman can be aggregated several ways and the numbers differ, so we fix one and say so.
Every figure above is **per-fold mean, then median across folds**.
`results/RECONCILED_TABLE.csv` carries the same arms under the alternative convention
(median over all cells) so the two can be compared directly. Mixing them was the single
easiest way to produce a misleading table, and it caught us more than once.

---

## Why the splits are the hard part

Leave-one-allele-out is impossible on this dataset: every allele has a near-twin, so holding
one out still leaves the model a nearly identical groove to learn from. Instead the folds are
**clusters at 80% identity over the 34-residue peptide-contact pseudo-sequence**, which yields
21 usable folds. `supertypes.py` independently recovers the published A24, B58 and B7
supertypes from the measured data, which is the check that the clustering is doing real work.

**Censoring.** 5,679 of 28,166 measurements (20.2%) sit at exactly 0 h, meaning the complex
dissociated faster than the assay could resolve. These are left-censored, not zeros. Headline
numbers use `censored='tied'`; every table is also reported under `censored='drop'`, and the
ranking does not change.

---

## Three independent lines pointing the same way

1. **Features.** Swapping the conventional features for ESM embeddings under a matched tuned
   head loses most of the signal (p = 1.00 for the foundation-model advantage).
2. **Predictions.** Ensembling the conventional net with the foundation-model arms made it
   *worse*, 0.307 to 0.217. They are not contributing independent signal.
3. **Controls.** Off-the-shelf ESM-2 is statistically indistinguishable from a control that is
   never told which allele it is predicting for.

An earlier version of this work claimed no ESM arm beat a three-line lookup table. That became
false once un-pooling and fine-tuning landed, and it was corrected. What survives is narrower
and still true: **off-the-shelf mean-pooled ESM-2 (0.092) loses to a lookup table (0.130).**

---

## What the model learned, and where it refuses

`anchors.py` scores all 171 single-point mutants of measured peptides and asks which positions
move the prediction. It recovers **P2 and P9**, the two anchors, and their residue preferences,
from (allele, peptide, half-life) triples alone — no pocket definitions, no structures, no motif
tables. It matched the published motif at 11 of 13 strictly-scored positions, and reproduced
B\*08:01 as the known flat exception.

`design.py` enumerates single mutants rather than searching sequence space, because a forward
model at this accuracy would be driven straight into its own blind spots. `ood.py` refuses to
rank alleles with no measured stability at all, rather than returning a confident guess.

---

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/data.py              # dataset summary and censoring counts
python src/supertypes.py        # groove clusters, and the supertype recovery check
python src/run_experiment.py    # the arms; writes results/
python src/anchors.py           # anchor recovery against the published motifs
```

Embedding caches (`*.npy`, several GB) are not committed; `embed.py` and `hf_gpu_embed.py`
regenerate them. GPU arms were run on Hugging Face Jobs.

---

## Data

`data/stability.txt` — the NetMHCstabpan stability dataset, **Rasmussen et al. 2016**.
28,166 measurements, 75 HLA alleles, 5,633 distinct 9-mers, half-life in hours.
Serova's brief makes this dataset mandatory for the track.

**NetMHCstabpan itself is not used as a comparator.** It trained on the entirety of this
dataset, so the brief calls it unfair as a direct baseline. Every number here is scored
against the *measured* half-life instead.

---

## Interactive site

`site/index.html` — all ten models, a per-allele peptide lookup against the measured value,
3D structures, the anchor-optimisation scan, and a tiered case study of abacavir and
HLA-B\*57:01. Serve the `site/` directory over HTTP (it fetches its data by relative path):

```bash
python -m http.server 8000 --directory site
```

---

## Citations

Every DOI in [CITATIONS.md](CITATIONS.md) was resolved against Crossref and the returned title,
first author and year compared against what we were calling it. That check caught one wrong
first author in our own draft. References that failed to resolve are listed separately and are
cited nowhere.

## Licence

Code is MIT, see [LICENSE](LICENSE). The stability dataset is redistributed under the terms of
its original publication and belongs to Rasmussen et al.; the PDB coordinates belong to their
depositors. Neither is ours to relicense.
