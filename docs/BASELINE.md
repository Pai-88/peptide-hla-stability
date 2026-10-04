# NetMHCstabpan baseline: what is honest, what is obtainable

Status as of 2026-10-03. Every number below is sourced. Nothing here is estimated or inferred.

---

## 1. The trap, stated plainly

`stability.txt` (28,166 rows / 75 alleles / 5,633 distinct 9-mers) **is the NetMHCstabpan 1.0
training set**, not a benchmark. The paper says so directly:

> "the complete data set of 28,166 9-mer measurements covering 75 HLA alleles"
> — Rasmussen et al. 2016, *The NetMHCstabpan server* section (PMC4976001)

So any NetMHCstabpan score we generate on these rows is a score on its own training data.
It is not a baseline. If we put such a number on a slide we are claiming a comparison we did
not make.

Worse for our specific pitch: our story is **held-out alleles**. NetMHCstabpan saw all 75.
There is no fold, no split, and no published artefact that gives us a NetMHCstabpan prediction
for an allele it never trained on, because no such allele exists inside this dataset.

### How the published model was actually cross-validated

Directly from Materials and Methods (*Artificial neural network training*):

> "the pool of unique peptides was split into five sets in a typical fivefold cross-validation
> scheme with all peptide-HLA-I stability data for a given peptide placed in the same group
> ... in this way, no peptide can belong to more than one group"

**The partitioning is by peptide, not by allele.** Every fold's training half contains every
allele. There is no leave-one-allele-out experiment anywhere in the paper. This is the single
most important fact for us: the published 0.676 is a *held-out-peptide* number, and our
held-out-allele number is measuring a strictly harder task.

Architecture, for the record: feed-forward ANN, BLOSUM50 (normalisation 5) or sparse encoding,
40 / 50 / 60 hidden neurons, peptide + 34-residue MHC pseudosequence input, half-lives rescaled
as `s = 2^(-t0/t_half)`, 1,000 random natural 9-mers per allele added as `t_half = 0` negatives
(selected to have NetMHCpan-2.8 predicted affinity weaker than 20,000 nM).

---

## 2. Numbers we can quote, with their exact context

### From Rasmussen et al. 2016 (open access, PMC4976001, doi:10.4049/jimmunol.1600582)

| Number | What it actually is | Safe phrasing |
|---|---|---|
| **PCC = 0.676** | Cross-validated test-set Pearson correlation of the final pan-specific network, using the **global** rescaling threshold `t0 = 1 h`. This is the published model. | "NetMHCstabpan reports a cross-validated Pearson of 0.676 on held-out *peptides*, with all 75 alleles seen in training." |
| **PCC = 0.693** | Same CV, but using **allele-specific** rescaling. The paper rejects this variant as unusable for a pan-specific method. Not the shipped model. | Only quote as "an upper variant the authors discarded (0.693)". |
| p = 0.901 | Binomial test, ties excluded: 0.676 vs 0.693 are statistically comparable. | — |
| 28,166 measurements / 75 HLA alleles | Training set size (filtered from 28,939 measurements / 80 allotypes). | — |
| 0.9 : 0.1 (ligands), 0.8 : 0.2 (epitopes) | Optimal affinity : stability weighting, 5-fold CV, on 1,058 MHC ligands (31 allotypes) and 598 T-cell epitopes (23 allotypes) from IEDB + SYFPEITHI. | — |
| p = 0.0013 (ligands), p = 0.0005 (epitopes) | Binomial tests, combined vs each method alone. | — |
| 0.7 : 0.3 (ligands), 0.6 : 0.4 (epitopes) | Same, but with size-balanced affinity/stability training sets (17,998 points, 58 alleles each). | — |

**Do not quote any AUC figure from this paper.** The epitope/ligand AUCs live only in Figures 1–3;
no numeric AUC appears in the text, and we have not read the figure values off the axes. The
metric reported in text for the stability task itself is **PCC only**.

### From Fasoulis et al. 2024 — computed by us from their released predictions

Fasoulis R, Rigo MM, Antunes DA, Paliouras G, Kavraki LE. "Transfer learning improves pMHC
kinetic stability and immunogenicity predictions." *ImmunoInformatics* 13 (2024) 100030.
Data: <https://github.com/KavrakiLab/TL-MHC> (`TLStab/misc/datasets/{Ebola,Pox}_pan_results_v2.csv`)

These files contain **real per-peptide NetMHCstabpan 1.0 predictions** on two external
stability datasets. Metrics below were computed by `baseline.py` in this repo from their
`NetMHCstabpan` column against their `Stability` ground-truth column — they are our
computation over their published predictions, not quoted from their paper (their own
comparisons are in Fig. 3B/3C, which we have not read numerically).

| Set | n | NetMHCstabpan Pearson r | Spearman | Kendall tau-b |
|---|---|---|---|---|
| Ebola | 1,023 | **0.410** | 0.281 | 0.195 |
| Pox | 541 | **0.444** | 0.358 | 0.248 |

For context, on the same rows: NetMHCpan4.1 BA gets r = 0.331 (Ebola) / 0.397 (Pox);
MHCflurry2.0 presentation gets 0.301 / 0.337.

Per-allele NetMHCstabpan Pearson on Ebola ranges from **0.016 (HLA-A\*24:02)** to
**0.697 (HLA-A\*03:01)**; on Pox from **-0.031 (HLA-A\*24:02)** to **0.674 (HLA-A\*11:01)**.
That spread is itself a usable slide point.

---

## 3. Can we get real per-peptide predictions ourselves?

| Route | Available? | Blocker |
|---|---|---|
| **Published external predictions (Fasoulis 2024)** | **YES — already fetched and cached** | None. Public GitHub, no registration. See `baseline.py`. |
| `data.tar.gz` (networks + %rank thresholds) | **YES — openly downloadable**, HTTP 200, 6.8 MB, no auth | None. Contains `data/syn/synaps` = all **30 trained networks** (5 CV folds x {40,50,60} hidden x {BLOSUM, sparse}), `data/MHC_pseudo.dat` (3,724 allele pseudosequences), and 2,126 per-allele %rank threshold files. |
| **Standalone executable** (Linux 1.0b/1.0cstatic, Darwin 1.0a) | Requires the DTU software-request form at `services.healthtech.dtu.dk/cgi-bin/sw_request` — an **academic licence agreement**. | **The user must do this personally.** I did not and will not submit it. |
| **Web server** at DTU | Exists (paste peptides or FASTA). | Manual, rate-limited, and submitting 28k rows through it is both impractical and pointless (training data). |
| **Per-fold CV predictions in the supplement** | **NO.** The supplement is one file (`SD1`); the article text never offers per-fold predictions. Only the aggregate PCC values are published. | Not obtainable. |

**Note on the weights.** Because `synaps` is open, a from-scratch forward-pass reimplementation
is technically possible (input layer is 903 = 43 positions x 21, i.e. 9 peptide + 34
pseudosequence residues). **I did not do this and recommend against it for the hackathon**:
DTU ships no reference input/output pair, so there is nothing to validate a reimplementation
against. An unvalidated reimplementation producing a "NetMHCstabpan number" is exactly the kind
of unfalsifiable figure that would sink the project. If we ever want it, the validation gate is:
obtain the standalone, run `data/B0702.fsa`, and match our forward pass to its output to 3 d.p.
first.

---

## 4. What the honest comparison is

**The primary comparison (recommended).** Evaluate our held-out-allele model and NetMHCstabpan
on the **same external rows** — the Fasoulis Ebola (1,023) and Pox (541) sets. Neither set is
NetMHCstabpan's training data, and overlap with `stability.txt` is tiny and measured:

- Ebola: 66 of 1,023 peptides and **24 exact (peptide, allele) pairs** appear in `stability.txt` (2.3%).
- Pox: 13 of 541 peptides and **6 exact pairs** (1.1%).

`baseline.py` reports these, and we should drop those rows and re-report as a sensitivity check.

**The caveat that must survive onto the slide:** all 8 alleles in both external sets
(A\*01:01, A\*02:01, A\*03:01, A\*11:01, A\*24:02, B\*07:02, B\*08:01, B\*15:01) **are inside
NetMHCstabpan's 75 training alleles**. So this is held-out *peptides and assay*, not held-out
alleles, for NetMHCstabpan. If we train on `stability.txt` with those 8 alleles withheld and
then evaluate here, the comparison is **asymmetric in NetMHCstabpan's favour**. That asymmetry
is a feature, not a bug: it makes any win of ours conservative and any loss uninformative.

**The secondary comparison (clearly labelled).** Our leave-allele-out cross-validated PCC on
`stability.txt` against the published 0.676. This is *not* a head-to-head — different split,
different task difficulty — and must be labelled "reference point, not a controlled comparison".

**What we must never do:** run NetMHCstabpan on `stability.txt` rows and present the result as
a baseline.

---

## 5. Fallback if the above collapses

In priority order:

1. **Ablation baselines we fully control.** NetMHCpan4.1 BA and MHCflurry presentation columns
   are already in the cached Fasoulis CSVs (r = 0.33 / 0.30 on Ebola). "Our stability model vs
   affinity-only prediction, on identical held-out rows" is a complete, defensible story even
   with zero NetMHCstabpan involvement.
2. **Allele-mean and nearest-pseudosequence-neighbour baselines** computed from `stability.txt`
   itself. These are the honest floor for a held-out-allele task and cost nothing.
3. **Published-number-only framing.** State 0.676 as the literature reference and show our
   leave-allele-out number beside it with the asymmetry spelled out. Weaker, but not dishonest.

---

## 6. What we say on the slide (draft)

> We benchmark against NetMHCstabpan 1.0 (Rasmussen et al., *J Immunol* 2016), the standard
> pan-specific pMHC-I stability predictor. One thing has to be said up front: the 28,166-measurement
> DTU stability set is NetMHCstabpan's *training* data, and its published Pearson of 0.676 comes
> from five-fold cross-validation partitioned **by peptide, not by allele** — every fold saw all
> 75 alleles. There is no published leave-one-allele-out NetMHCstabpan result, so we did not
> manufacture one. Instead we compare on two external datasets (Ebola, n = 1,023; Pox, n = 541)
> where NetMHCstabpan's per-peptide predictions were independently published by Fasoulis et al.
> (2024); on those rows NetMHCstabpan achieves Pearson r = 0.41 and 0.44 respectively. Our model
> is evaluated on the identical rows with those alleles **withheld from training**, while
> NetMHCstabpan trained on all of them — so the comparison is deliberately tilted against us.
> Overlap between those external sets and the training data is 24 and 6 exact peptide-allele
> pairs, which we remove in a sensitivity check.

---

## 7. Reproducing

```
python baseline.py       # fetches, caches, writes baseline_metrics.json
```

Outputs `baseline_cache/{Ebola,Pox}_pan_results_v2.csv` and `baseline_metrics.json`
(metrics, per-allele breakdown, and the leakage audit).

## Sources

- Rasmussen M, et al. *J Immunol* 2016;197(4):1517-24. PMID 27402703, PMCID PMC4976001,
  doi:10.4049/jimmunol.1600582. Full text via NCBI efetch (`db=pmc&id=PMC4976001`).
- Fasoulis R, et al. *ImmunoInformatics* 2024;13:100030. Data: <https://github.com/KavrakiLab/TL-MHC>
- DTU service page: <https://services.healthtech.dtu.dk/services/NetMHCstabpan-1.0/>
- Open data bundle: <https://services.healthtech.dtu.dk/services/NetMHCstabpan-1.0/data.tar.gz>
