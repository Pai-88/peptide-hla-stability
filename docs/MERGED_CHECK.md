# Supertype-merge robustness check

**ADDITIONAL analysis. It replaces nothing.** The 21-fold structure in `RESULTS.md`,
`PITCH.md` and `SUBMISSION.md` stands as the headline. This file asks one question:
*if two groups of our folds are really one supertype each, does the conclusion survive
merging them?*

**It does. The conclusion holds on all five arms re-run, and on the foundation-model
side it gets sharper.** The conventional net still beats the best frozen ESM-2 arm
(+0.066, 14 of 17 folds, p = 1.3e-03) and the pooled frozen arm (+0.173, 16 of 17,
p = 3.1e-05). The pooled frozen arm, already indistinguishable from a control that
never sees the allele, now sits *slightly below* that control (-0.005, 8/17, p = 0.68).
Two things move and must be said out loud if the merged numbers are quoted: the
conventional net's own absolute score falls **0.307 to 0.246**, and its margin over the
best foundation-model arm narrows from +0.080 on 19 of 21 folds to **+0.066 on 14 of
17**. The ladder does not reorder.

Run 3 Oct 2026. Nothing outside `merged_check.py`, `results_merged.csv`,
`predictions_merged.parquet` and this file was written or modified. No fine-tuning was
re-run. No browser tool was used.

---

## 1. The merged fold definitions

Folds are groove clusters from 34-residue pseudo-sequence identity at 0.80
(`supertypes.clusters(0.80)`, unmodified). Two groups of those clusters share a single
P2 anchor preference. **Radin Moradi, the team's biochemist, was asked directly whether
those clusters are single supertypes split too finely by the 0.80 cut, and he confirmed
these are single supertypes for both the B44 set and the B07 set.** The merged fold
definitions below follow that expert supertype assignment. They are not an automatic
rule derived from the P2 column, and they are not an inference made by this analysis.
The merge collapses each group into one fold.

`supertypes.py` and `splits.py` are untouched. The merge is defined in `merged_folds()`
in `merged_check.py`, which first asserts that all six clusters exist exactly as written
in `supertypes.clusters(0.80)` and raises if they do not.

**MERGED B07-like**: 8 alleles, 3,196 rows. Was three folds (1,740 + 1,103 + 353 rows).

```
{B*07:02, B*42:01, B*42:02, B*81:01}  +  {B*54:01, B*55:01, B*56:01}  +  {B*51:01}
```

**MERGED B44-like**: 6 alleles, 1,768 rows. Was three folds (1,057 + 359 + 352 rows).

```
{B*40:01, B*40:02, B*41:01, B*45:01}  +  {B*18:01}  +  {B*44:05}
```

**21 folds becomes 17.** The other 15 folds are unchanged in membership *and* in their
training side, because `split_by_allele` depends only on the held-out allele list.

### Corroboration in our own measured data

This is a consistency check on the biochemist's assignment, not the basis for it.
Top-decile peptides per allele, P2 residue frequency over the dataset-wide P2 background:

| group | P2 preference | enrichment |
|---|---|---|
| B07-like (8 alleles) | **P** is the top-enriched P2 residue in all 8 | 5.9x to 6.4x |
| B44-like (6 alleles) | **E** is enriched in all 6, and is the top residue in 4 (B\*18:01, B\*40:01, B\*41:01, B\*44:05) | E 12.7x to 17.0x in all six |

Two B44-like members have a different top residue with E immediately behind it:
B\*45:01 is D 74.5x then E 14.6x (acidic either way), and B\*40:02 is K 17.0x then
E 14.3x. **B\*40:02 has only 19 measured rows**, below `metrics.MIN_N = 20`, so it is
never scored in either fold design and cannot affect any number here. Its P2 reading
rests on 19 peptides and should not be leaned on.

### What the merge actually removes

For each merged-away allele, the closest groove the 21-fold design still left in its
training set:

| allele | nearest allele left in training under 21 folds | identity |
|---|---|---|
| B\*42:01 / B\*55:01 / B\*56:01 | B\*54:01 / B\*42:01 / B\*42:01 | 28/34 (0.824) |
| B\*07:02 / B\*42:02 / B\*81:01 | B\*55:01 / B\*54:01 / B\*56:01 | 27/34 (0.794) |
| B\*51:01 | B\*55:01 | 23/34 (0.676) |
| B\*45:01 / B\*44:05 | B\*44:05 / B\*45:01 | 27/34 (0.794) |
| B\*40:02 / B\*41:01 / B\*18:01 | B\*18:01 / B\*18:01 / B\*41:01 | 26/34 (0.765) |
| B\*40:01 | B\*18:01 | 25/34 (0.735) |

**Honest caveat on the merge itself:** every one of these pairs sits *below* the 0.80
identity cut. The merge is therefore **stricter than our own clustering rule requires**.
It rests on the biochemist's supertype assignment, corroborated by the shared P2 anchor,
not on groove identity. That makes it a conservative stress test, which is what it is
for, but it is not on its own evidence that the 0.80 cut was wrong.

---

## 2. The comparison table

Estimator, unchanged from `PITCH.md` line 4 and `RESULTS.md` section 2: seed-ensemble
**mean** prediction across 5 seeds, `censored='tied'`, per-fold = median per-allele
Spearman over that fold's held-out alleles at `metrics.MIN_N = 20`, headline = median
over folds. Same `run_experiment.run_arm` code path, same featurizers, same model
constructors, same seeds 0 to 4. The implementation in `merged_check._estimator`
reproduces every published 21-fold number to four decimal places before any merged
number is computed, so the deltas are like-for-like.

| arm | merged 17 folds | original 21 folds | delta |
|---|---:|---:|---:|
| **A, conventional supervised net** (BLOSUM62 + one-hot, no FM) | **0.2463** | **0.3069** | **-0.0606** |
| **BEST frozen ESM-2 150M**, peptide \| pseudo-seq un-pooled, tuned MLP | **0.1948** | **0.1948** | **-0.0000** |
| frozen ESM-2 150M, peptide \| pseudo-seq pooled, MLP head | 0.1125 | 0.1059 | +0.0066 |
| *CONTROL* ESM-2 150M peptide only (allele never shown) | 0.0643 | 0.0643 | 0.0000 |
| *NULL* allele mean + tie-break | -0.0063 | -0.0082 | +0.0018 |

The best frozen arm's -0.0000 is a coincidence, not an identity: it is 0.194781 merged
against 0.194806 original, because two adjacent folds both read 0.1948 and one of them
is the median in each design. Do not present it as "exactly unchanged"; present it as
"unchanged to four decimal places".

Spread, merged vs original. Arm A: IQR 0.142 to 0.422 (was 0.219 to 0.422), worst fold
B\*46:01 +0.061 either way, 0/17 folds at or below zero. Best frozen arm: IQR 0.076 to
0.277, worst A\*01:01 -0.018, 1/17 at or below zero. Pooled frozen arm: IQR 0.023 to
0.157, 3/17 at or below zero.

### Paired over folds (Wilcoxon signed rank, two-sided)

| comparison | merged: median delta | folds won | p | 21-fold: median delta | folds won | p |
|---|---:|---:|---:|---:|---:|---:|
| **conventional net - BEST frozen ESM-2** | **+0.0663** | **14/17** | **1.3e-03** | +0.0800 | 19/21 | 1.3e-05 |
| conventional net - pooled ESM-2 MLP | +0.1731 | 16/17 | 3.1e-05 | +0.1691 | 21/21 | 9.5e-07 |
| conventional net - peptide-only control | +0.1790 | 16/17 | 3.8e-04 | +0.2026 | 20/21 | 1.8e-05 |
| conventional net - null | +0.2338 | 17/17 | 1.5e-05 | +0.3015 | 21/21 | 9.5e-07 |
| *UNMATCHED* BEST frozen ESM-2 - peptide-only control | +0.0992 | 14/17 | 6.7e-03 | +0.0992 | 18/21 | 6.1e-04 |
| **head-matched: pooled ESM-2 MLP - peptide-only control** | **-0.0052** | **8/17** | **0.68** | +0.0258 | 13/21 | 0.23 |
| pooled ESM-2 MLP - null | +0.1158 | 12/17 | 4.6e-03 | +0.0884 | 18/21 | 2.0e-04 |

The *UNMATCHED* label carries over verbatim from `RESULTS.md`: the un-pooled arm and the
peptide-only control differ in head, encoding and seed count at once, so that row is not
a clean control test under the merged folds either. The clean, head-matched control test
is the pooled row, and it is the one that moves to -0.005.

### The two merged folds, which are the whole point

| arm | MERGED B07-like | was (3 folds) | MERGED B44-like | was (3 folds) |
|---|---:|---|---:|---|
| conventional net | **+0.4256** | +0.389 / +0.521 / +0.526 | **+0.1143** | +0.335 / +0.280 / +0.307 |
| BEST frozen ESM-2 | +0.2594 | +0.220 / +0.251 / +0.484 | **+0.1682** | +0.270 / +0.019 / +0.145 |
| pooled ESM-2 MLP | +0.2431 | +0.246 / +0.224 / +0.374 | +0.1209 | +0.006 / +0.028 / +0.138 |
| peptide-only control | +0.1011 | -0.017 / +0.203 / +0.185 | +0.0026 | +0.138 / -0.153 / +0.055 |
| null | -0.0093 | -0.008 / +0.040 / -0.071 | -0.0063 | -0.039 / -0.011 / +0.030 |

The two merged folds behave in opposite directions, and that is the most informative
thing in this file:

- **B44-like is where the leakage was, and it is the one place the result locally
  inverts.** The conventional net falls from about 0.31 across its three component folds
  to **+0.114** once B\*18:01 and B\*44:05 can no longer be predicted from a
  B\*40/41/45 groove left in training. On that single fold **both frozen ESM-2 arms are
  now ahead of it** (+0.168 and +0.121 against +0.114). One fold out of seventeen is not
  a result and the paired tests over all 17 folds are unambiguous, but if a judge asks
  "is there any fold where the foundation model wins", the honest answer is yes, this
  one, and it is the hardest one we built.
- **B07-like is not leakage, it is difficulty averaging.** The merged fold scores
  **+0.426** for the conventional net, above two of its three components. Merging eight
  alleles into one median does not make this groove family harder. The B7 motif is
  strong and the model gets it from the pseudo-sequence.

### Where the -0.061 on arm A comes from

The 15 unchanged folds are **bit-identical** between the merged and the original run for
the conventional net, the best frozen arm, the control and the null (median, min and max
delta all exactly 0.0000), as they must be: same held-out alleles, same training rows,
same seeds. So the conventional net's entire drop is structural, not retraining and not
noise. Three folds averaging about 0.31 collapse into one fold at +0.114, and two folds
above +0.52 collapse into one at +0.426. Together that pulls the median of the remaining
list from 0.307 down to 0.246.

**One reproducibility caveat.** The pooled frozen ESM-2 MLP arm is *not* bit-reproducible:
on the 15 unchanged folds it moved by -0.057 to +0.027 (median -0.006) purely from being
re-run under a different BLAS thread count. Its +0.0066 merged-versus-original shift is
inside that noise and should be read as "unchanged", not as an improvement.

---

## 3. Does the conclusion change?

**No. The conclusion holds.**

The conclusion under test was: *the conventional net beats every foundation-model arm,
and the ESM arms sit near a control that cannot see the allele.*

1. **The conventional net still beats every foundation-model arm that was re-run.**
   0.2463 against 0.1948 and 0.1125. Paired, +0.066 on 14 of 17 folds (p = 1.3e-03)
   against the best frozen arm and +0.173 on 16 of 17 (p = 3.1e-05) against the pooled
   one.
2. **The head-matched "indistinguishable from an allele-blind control" claim gets
   stronger.** On 21 folds the pooled frozen arm beat its peptide-only control by +0.026
   (p = 0.23). On 17 merged folds it is **-0.005, 8 folds of 17, p = 0.68**. Under a
   harder fold design the allele-aware foundation-model arm does not beat a model that
   is never shown the allele at all.
3. **The ladder does not reorder.** 0.246 > 0.195 > 0.113 > 0.064 > -0.006, the same
   order as 0.307 > 0.195 > 0.106 > 0.064 > -0.008.

**What changes, and must be said if a merged number is ever quoted:**

- The conventional net's absolute score falls from **0.307 to 0.246**, a -0.061 drop,
  all of it from the merged B44-like fold.
- Its margin over the **best** foundation-model arm narrows from +0.080 on 19 of 21
  folds (p = 1.3e-05) to **+0.066 on 14 of 17 folds (p = 1.3e-03)**. Still significant,
  but two orders of magnitude weaker in p and three fewer folds won. Do not keep saying
  "nineteen of twenty-one" if you are quoting merged folds.
- On the single hardest fold, MERGED B44-like, both frozen ESM-2 arms beat the
  conventional net.

If a judge asks "is 0.31 robust to your fold design?", the honest answer is that the
*ordering* and the *claim* are robust, the *absolute number* is not, and under a stricter
supertype-merged design it reads 0.25.

### What was NOT re-run

**The fine-tuned arm (`FG_finetune_gpu`, 0.1697 on 21 folds) was not re-scored on the
merged folds.** It was out of scope by instruction and re-running 85 GPU fits was not
possible here. So the sentence "the conventional net beats *every* foundation-model arm"
is re-verified on the merged folds for the frozen arms only. The 650M arm, the hybrid
arm and every ridge-head arm were also not re-run. Nothing in this file touches the
three-best-shots band or any fine-tuning claim.

Arm A's merged 0.2463 must not be compared against any arm's 21-fold number. Different
fold sets. Do not subtract across the two columns.

---

## 4. Paragraph for `RESULTS.md` (paste as a robustness section)

> **Robustness to supertype-merged folds.** The 21 folds are groove clusters at 0.80
> pseudo-sequence identity. Two groups of those clusters share a single P2 anchor
> preference, and the team's biochemist, Radin Moradi, confirmed that each group is a
> single supertype split too finely by the 0.80 cut. The B07-like group is B\*07:02,
> B\*42:01, B\*42:02, B\*81:01, B\*54:01, B\*55:01, B\*56:01, B\*51:01 (P at P2 enriched
> 5.9x to 6.4x in every member's top decile); the B44-like group is B\*40:01, B\*40:02,
> B\*41:01, B\*45:01, B\*18:01, B\*44:05 (E at P2 enriched 12.7x to 17.0x in all six).
> The 21-fold design can therefore leave a groove that takes the same key in training
> while a member of the same supertype is held out. Collapsing each group into one fold
> gives 17 strictly harder folds, and the five arms re-run on them under the identical
> estimator give: conventional net 0.2463 (21-fold 0.3069), best frozen ESM-2 un-pooled
> tuned MLP 0.1948 (0.1948), pooled frozen ESM-2 MLP 0.1125 (0.1059), peptide-only
> control 0.0643 (0.0643), allele-mean null -0.0063 (-0.0082). The conclusion is
> unchanged and the head-matched foundation-model finding is sharpened: the conventional
> net still leads the best frozen arm by +0.066 on 14 of 17 folds (p = 1.3e-03) and the
> pooled arm by +0.173 on 16 of 17 (p = 3.1e-05), while the pooled arm, which beat its
> allele-blind control by a non-significant +0.026 on 21 folds, now sits at -0.005
> against it (8/17, p = 0.68). Two costs are real and are stated rather than buried: the
> conventional net's absolute score falls by 0.061, entirely attributable to the merged
> B44-like fold (+0.114, against about 0.31 across its three former components, with the
> 15 unchanged folds bit-identical between the two runs), and its margin over the best
> foundation-model arm weakens from +0.080 on 19 of 21 folds to +0.066 on 14 of 17. On
> MERGED B44-like alone, the hardest fold in the design, both frozen ESM-2 arms score
> above the conventional net. The fine-tuned arm (0.1697) was not re-scored on the
> merged folds and is not covered by this check.

---

## 5. The sentence to say on stage

> **"Yes. Our biochemist flagged two sets of our folds as single supertypes split too
> finely, so we merged them, which takes twenty-one folds to seventeen harder ones, and
> the conventional network still wins, by seven hundredths over the best frozen ESM-2
> arm on fourteen of seventeen folds, while the pooled frozen arm drops to dead level
> with the allele-blind control; our own number falls from zero point three one to zero
> point two five, and all of that is in the write-up."**

If pressed for one more sentence: *"We did not re-run the fine-tuned arm on the merged
folds, and on the single hardest merged fold the frozen arms do beat us."*

---

## 6. Reproducing this

```
python merged_check.py folds                     # the 17 merged folds
python merged_check.py worker A|I|B|D|E          # run one arm (run it in a scratch cwd)
python merged_check.py combine <dir> [<dir> ...] # -> results_merged.csv,
                                                 #    predictions_merged.parquet, table
```

Workers were run with the project directory symlinked into five scratch directories so
that `run_arm`'s `results_*` and `predictions_*` writes could not touch the project.
`results_merged.csv` holds 425 rows (5 arms x 17 folds x 5 seeds, all status `ok`, none
skipped, none failed). `predictions_merged.parquet` holds 703,975 per-row predictions
with an added `arm` column.
