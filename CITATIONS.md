# Citations, every DOI verified against Crossref

Checked Sat 3 Oct 2026, 16:20. Each line resolved live against `api.crossref.org` and the
returned title, first author and year compared against what we were calling it.

**One error found and fixed**: we had been calling the geographic-bias paper "Barton et al.".
The first author is **Atkins**. It was wrong in `brief.html` and is now corrected.

| Used for | Verified reference |
|---|---|
| **The dataset. Serova's brief says this one is mandatory.** | Rasmussen M, et al. Pan-Specific Prediction of Peptide-MHC Class I Complex Stability, a Correlate of T Cell Immunogenicity. *J Immunol* 2016;197(4):1517-24. doi:10.4049/jimmunol.1600582 |
| Why stability and not affinity | Harndahl M, et al. Peptide-MHC class I stability is a better predictor than peptide affinity of CTL immunogenicity. *Eur J Immunol* 2012;42(6):1405-16. doi:10.1002/eji.201141774 |
| Supertypes, the fold-design check | Sidney J, et al. HLA class I supertypes: a revised and updated classification. *BMC Immunol* 2008;9:1. doi:10.1186/1471-2172-9-1 |
| The 34 peptide-contact positions | Reynisson B, et al. NetMHCpan-4.1 and NetMHCIIpan-4.0. *Nucleic Acids Res* 2020;48(W1):W449-54. doi:10.1093/nar/gkaa379 |
| HLA sequences | Barker DJ, et al. The IPD-IMGT/HLA Database. *Nucleic Acids Res* 2023;51(D1):D1053-60. doi:10.1093/nar/gkac1011 |
| ESM-2 | Lin Z, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* 2023;379(6637):1123-30. doi:10.1126/science.ade2574 |
| The counter-evidence: binding prediction survives the allele gap | **Atkins** C, et al. Geographically Biased Composition of NetMHCpan Training Datasets and Evaluation of MHC-Peptide Binding Prediction Accuracy on Novel Alleles. bioRxiv 2023. doi:10.1101/2023.09.03.556092 |

## Not yet verified, do not cite until checked

- The bioRxiv preprint on peptide:MHC stability with protein language models. We found it in a
  search this morning and never resolved its DOI or read it. **Do not reference it on a slide.**
- MINT, the interaction-aware model pretrained on affinity and fine-tuned on half-life. Same:
  surfaced in a search, never verified.
- Fasoulis et al. 2024, *ImmunoInformatics*, the source of the external NetMHCstabpan
  predictions in BASELINE.md. Verify its DOI before quoting those numbers.

## Also credit in the repo

GDM Science Skills (Apache-2.0). UniProt reviewed proteomes, for the epitope source mapping.
Modal, Anthropic, Hugging Face and Amass credits where their tooling was used.

## Rule for tomorrow

No reference goes on a slide or in the write-up unless it appears in the table above. If
someone wants to add one, resolve the DOI against Crossref first. Takes ten seconds and it
already caught one wrong author name today.
