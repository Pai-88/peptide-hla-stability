"""Per-allele evaluation metrics for peptide-HLA stability prediction.

Everything here returns a DISTRIBUTION over held-out alleles, never a pooled
number. A pooled Spearman over all test rows is dominated by the between-allele
differences in mean stability and can look good while every single allele is
ranked at chance. The claim is "generalises to unseen alleles", so the unit of
evaluation is the allele.

Interface: every function takes a long dataframe with columns HLA, Thalf, y,
censored (from data.load()) restricted to the test rows, plus a prediction
array aligned to it. That makes it agnostic to how the split was produced, so
it works with split_random, split_by_peptide, or split_by_allele once written.

-------------------------------------------------------------------------------
CENSORING (the 20.2% at Thalf == 0.0)
-------------------------------------------------------------------------------
Those rows mean "dissociated faster than the assay resolves", not "half-life is
zero". Every metric here takes an explicit `censored` argument. No silent
default behaviour; you have to say which you used on the slide.

  'tied'  - keep censored rows, all sharing the lowest rank (a single tie block).
            This is the honest reading of the data: we know they are all at the
            bottom, we do not know their order. Spearman's tie correction
            handles this correctly. Deflates rho slightly versus 'drop' because
            ~20% of the pairs carry no orderable information.
  'drop'  - exclude censored rows entirely. Answers "among peptides the assay
            could actually measure, do we rank them right?". Higher numbers, a
            narrower claim, and it discards exactly the easy negatives a real
            screen wants excluded, so do not quote it alone.
  'binary'- replace the measured value by in-the-top-decile / not, i.e. treat
            the task as detecting stable binders. Only offered for top-k.

Recommendation: report 'tied' as the headline and 'drop' as a robustness row.
"""

import numpy as np
import pandas as pd
from scipy import stats

MIN_N = 20          # fewer test rows than this and a per-allele rho is noise
MIN_NONCENSORED = 10


def _prep(df, pred, censored):
    """Return (df, pred) after applying the censoring policy."""
    pred = np.asarray(pred, dtype=float)
    assert len(pred) == len(df), "predictions must align with the test rows"
    if censored == "drop":
        m = ~df.censored.values
        return df[m], pred[m]
    if censored == "tied":
        return df, pred
    raise ValueError("censored must be 'tied' or 'drop'")


def spearman_per_allele(df, pred, censored="tied", min_n=MIN_N):
    """Spearman rho between predicted and measured half-life, one per allele.

    Ranking, not absolute hours: the use case is "pick the best few peptides for
    this allele", and a model with a constant offset is still useful for that.
    Measured value used is Thalf itself (rank-identical to data.y, since the log
    and the 0.05 floor are both monotone, so the choice of floor cannot move rho).

    Returns a Series indexed by allele. Alleles with too few usable test rows,
    or with no variance left to rank (e.g. every row censored), are dropped and
    reported via .attrs['skipped'].
    """
    df, pred = _prep(df, pred, censored)
    out, skipped = {}, {}
    for allele, idx in df.groupby("HLA").indices.items():
        if len(idx) < min_n:
            skipped[allele] = f"n={len(idx)} < {min_n}"
            continue
        yt, yp = df.Thalf.values[idx], pred[idx]
        if np.ptp(yt) == 0 or np.ptp(yp) == 0:
            skipped[allele] = "no variance"
            continue
        out[allele] = stats.spearmanr(yp, yt).statistic
    s = pd.Series(out, dtype=float).sort_index()
    s.attrs["skipped"] = skipped
    s.attrs["censored_policy"] = censored
    return s


def top10_precision_per_allele(df, pred, k=10, decile=0.9, censored="tied",
                               min_n=MIN_N):
    """Of the k peptides we rank highest for an allele, what fraction are truly
    in that allele's top decile?

    The application metric: a screen tests a handful of peptides per patient
    allele, and only the hits matter. Chance level is 1 - decile (0.10 by
    default), which makes the bar on the slide self-interpreting.

    The true top decile is defined PER ALLELE on the test rows of that allele,
    by Thalf. Ties at the threshold are resolved inclusively (>= the quantile),
    so with heavy ties the true set can exceed 10% of rows; that only ever makes
    the metric more generous, and censored rows can never enter it since they
    are the minimum.

    'binary' is not a separate policy here: this metric is already a binarised
    target. 'tied' keeps censored rows in the candidate pool (they can be ranked
    highly by a bad model and correctly count against it, which is what we want);
    'drop' removes them from the pool and makes the task easier.
    """
    df, pred = _prep(df, pred, censored)
    out, skipped = {}, {}
    for allele, idx in df.groupby("HLA").indices.items():
        if len(idx) < max(min_n, k):
            skipped[allele] = f"n={len(idx)} too small for k={k}"
            continue
        yt, yp = df.Thalf.values[idx], pred[idx]
        thresh = np.quantile(yt, decile)
        true_top = yt >= thresh
        if true_top.all():
            skipped[allele] = "degenerate: every row at/above the decile cut"
            continue
        picked = np.argsort(-yp, kind="stable")[:k]
        out[allele] = true_top[picked].mean()
    s = pd.Series(out, dtype=float).sort_index()
    s.attrs["skipped"] = skipped
    s.attrs["chance"] = 1 - decile
    s.attrs["censored_policy"] = censored
    return s


def calibration_per_allele(df, pred, sigma, censored="tied", min_n=MIN_N):
    """Does the claimed uncertainty track the actual error?

    DEFINITION CHOSEN: Spearman rank correlation between the per-prediction
    uncertainty (ensemble std across seeds) and the absolute error
    |pred - y| in log10 half-life. Call it the error-uncertainty correlation
    (EUC); it is the rank-based form of the standard "sparsification" check.

    Why this one, over the two obvious alternatives:
      - Expected calibration error / coverage of a 68% interval requires the
        ensemble std to be an absolute, correctly-scaled sigma. An ensemble of
        a handful of seeds is systematically over-confident, so its raw ECE
        measures mostly that scaling artefact, not whether the model knows when
        it is lost.
      - NLL under a Gaussian mixes calibration with accuracy in one number, so
        a more accurate model wins on NLL even if its uncertainty is useless.
      EUC is scale-free and monotone-invariant, which is exactly the property an
      over-confident but correctly-ORDERED ensemble needs. It answers the
      decision-relevant question: "if we only trust the predictions the model is
      confident about, do we do better?" Positive is good, 0 is useless, and it
      is bounded in [-1, 1] so it plots next to the Spearman panel.

    Because it is scale-free, it CANNOT detect over-confidence in absolute terms.
    So we also return per-allele empirical coverage of the nominal 68% interval
    (fraction with |err| <= sigma) as a second column. If EUC is healthy but
    coverage is far below 0.68, the ranking of uncertainty is fine and the
    ensemble simply needs a temperature scaling; say that rather than claiming
    calibrated uncertainty.

    Error is measured on y (log10 hours), the space the model is trained in.
    Under 'tied' the censored rows contribute an error against the 0.05 floor,
    which is a lower bound on the true error, so the coverage column is
    optimistic there; use 'drop' if coverage is the number being quoted.

    Returns a DataFrame indexed by allele with columns euc, coverage68, n.
    """
    sigma = np.asarray(sigma, dtype=float)
    assert len(sigma) == len(df), "sigma must align with the test rows"
    # filter sigma exactly as _prep filters df/pred
    if censored == "drop":
        sigma = sigma[~df.censored.values]
    df, pred = _prep(df, pred, censored)
    rows, skipped = {}, {}
    for allele, idx in df.groupby("HLA").indices.items():
        if len(idx) < min_n:
            skipped[allele] = f"n={len(idx)} < {min_n}"
            continue
        err = np.abs(pred[idx] - df.y.values[idx])
        sg = sigma[idx]
        euc = (np.nan if np.ptp(sg) == 0 or np.ptp(err) == 0
               else stats.spearmanr(sg, err).statistic)
        rows[allele] = {"euc": euc,
                        "coverage68": float((err <= sg).mean()),
                        "n": len(idx)}
    out = pd.DataFrame(rows).T.sort_index()
    out.attrs["skipped"] = skipped
    out.attrs["censored_policy"] = censored
    return out


def summarise(s, name):
    """One line for the slide: median and spread across held-out alleles."""
    v = np.asarray(s, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return f"{name}: no alleles scored"
    return (f"{name}: median {np.median(v):.3f}  "
            f"IQR [{np.percentile(v, 25):.3f}, {np.percentile(v, 75):.3f}]  "
            f"mean {v.mean():.3f} +/- {v.std(ddof=1) if len(v) > 1 else 0:.3f}  "
            f"worst {v.min():.3f}  n_alleles {len(v)}")


# ---------------------------------------------------------------------------
# Verification. Synthetic predictions with a KNOWN correlation to the real
# labels, so each metric can be checked against a value we can derive.
# ---------------------------------------------------------------------------

def _synth(y, rho_target, rng):
    """Gaussian-copula style: mix the true signal with noise so that the
    Pearson correlation on ranks is approximately rho_target."""
    z = stats.zscore(stats.rankdata(y))
    n = rng.standard_normal(len(y))
    n = stats.zscore(n)
    return rho_target * z + np.sqrt(max(0.0, 1 - rho_target ** 2)) * n


def _verify():
    import data
    rng = np.random.default_rng(0)
    df = data.load()
    elig = data.loao_eligible(df)
    test = df[df.HLA.isin(elig[:12])].copy()
    print(f"verification set: {len(test)} rows, {test.HLA.nunique()} alleles, "
          f"{100 * test.censored.mean():.1f}% censored\n")

    print("=" * 72)
    print("1. SPEARMAN recovers the correlation it was built with")
    print("=" * 72)
    for target in [0.0, 0.3, 0.6, 0.9]:
        # build predictions per allele so the per-allele rho is the known one
        pred = np.empty(len(test))
        for _, idx in test.groupby("HLA").indices.items():
            pred[idx] = _synth(test.y.values[idx], target, rng)
        for pol in ["tied", "drop"]:
            s = spearman_per_allele(test, pred, censored=pol)
            print(f"  target rho={target:.1f}  policy={pol:5s}  "
                  f"median={np.median(s):+.3f}  min={s.min():+.3f}  max={s.max():+.3f}")
    print("\n  Both policies recover the target to ~0.03, and 0.0 gives ~0. Note")
    print("  'tied' comes out slightly ABOVE 'drop', not below: the synthetic")
    print("  predictor is built from the full ranking, so it separates the")
    print("  censored block from the rest easily, and those pairs are free marks.")
    print("  A real model gets the same free marks (telling a non-binder from a")
    print("  binder is the easy half of the problem). So 'tied' is the generous")
    print("  policy here, not the strict one - the opposite of the naive guess,")
    print("  which is exactly why the policy has to be stated rather than assumed.\n")

    print("=" * 72)
    print("2. SPEARMAN sign and extremes")
    print("=" * 72)
    perfect = test.y.values.copy()
    s = spearman_per_allele(test, perfect, censored="drop")
    print(f"  perfect predictor (drop):  median={np.median(s):+.4f}  (expect +1.000)")
    s = spearman_per_allele(test, perfect, censored="tied")
    print(f"  perfect predictor (tied):  median={np.median(s):+.4f}  "
          f"(expect +1.000: pred IS y, so its ties line up with y's ties)")
    oracle_noisy = perfect + rng.standard_normal(len(test)) * 1e-6
    s = spearman_per_allele(test, oracle_noisy, censored="tied")
    print(f"  oracle, untied (tied):     median={np.median(s):+.4f}  "
          f"(expect <1: a real model breaks the tie block in a random order,")
    print(f"                                       and is penalised for guessing)")
    s = spearman_per_allele(test, -perfect, censored="drop")
    print(f"  inverted predictor (drop): median={np.median(s):+.4f}  (expect -1.000)")
    s = spearman_per_allele(test, rng.standard_normal(len(test)), censored="tied")
    print(f"  pure noise:                median={np.median(s):+.4f}  (expect ~0)\n")

    print("=" * 72)
    print("3. TOP-10 PRECISION against its analytic chance level")
    print("=" * 72)
    print("  chance = 1 - decile = 0.100")
    for target in [0.0, 0.3, 0.6, 0.9]:
        pred = np.empty(len(test))
        for _, idx in test.groupby("HLA").indices.items():
            pred[idx] = _synth(test.y.values[idx], target, rng)
        p = top10_precision_per_allele(test, pred)
        print(f"  target rho={target:.1f}  median precision={np.median(p):.3f}  "
              f"mean={p.mean():.3f}")
    p = top10_precision_per_allele(test, test.y.values)
    print(f"  oracle               median precision={np.median(p):.3f}  (expect 1.000)")
    p = top10_precision_per_allele(test, -test.y.values)
    print(f"  anti-oracle          median precision={np.median(p):.3f}  (expect 0.000)")
    # Monte-Carlo the null so the chance level is checked, not asserted
    null = []
    for _ in range(200):
        null.append(np.median(top10_precision_per_allele(
            test, rng.standard_normal(len(test)))))
    print(f"  random, 200 draws    mean of medians={np.mean(null):.3f} "
          f"+/- {np.std(null):.3f}  (expect ~0.100)\n")

    print("=" * 72)
    print("4. CALIBRATION: EUC is +ve only when sigma really tracks the error")
    print("=" * 72)
    truth = test.y.values
    base = _synth(truth, 0.6, rng)
    # construct errors of a known, varying magnitude
    scale = rng.uniform(0.05, 1.5, len(test))
    pred = truth + rng.standard_normal(len(test)) * scale
    for label, sigma in [
        ("honest   sigma = the true error scale", scale),
        ("noisy    sigma = scale + noise       ", np.abs(scale + rng.standard_normal(len(test)) * 0.4)),
        ("useless  sigma = random              ", rng.uniform(0.05, 1.5, len(test))),
        ("inverted sigma = 1.55 - scale        ", 1.55 - scale),
    ]:
        c = calibration_per_allele(test, pred, sigma)
        print(f"  {label}  median EUC={np.nanmedian(c.euc):+.3f}  "
              f"median coverage68={np.nanmedian(c.coverage68):.3f}")
    print("\n  Expected: honest sigma gives a clearly positive EUC and coverage")
    print("  near 0.68 (|N(0,s)| <= s happens 68.3% of the time); random sigma")
    print("  gives EUC ~0; inverted gives a strongly negative EUC. Coverage is")
    print("  near 0.68 for the inverted case too even though it is a terrible")
    print("  uncertainty, which is exactly why EUC is the headline and coverage")
    print("  is only the scale check.\n")

    print("=" * 72)
    print("5. OVER-CONFIDENCE is invisible to EUC, visible in coverage")
    print("=" * 72)
    for f in [0.25, 0.5, 1.0, 2.0]:
        c = calibration_per_allele(test, pred, scale * f)
        print(f"  sigma scaled x{f:<4}  EUC={np.nanmedian(c.euc):+.3f}  "
              f"coverage68={np.nanmedian(c.coverage68):.3f}")
    print("  EUC is identical across all four (scale-free). Coverage moves.")
    print("  Quote both or the slide overstates the uncertainty claim.\n")

    print("=" * 72)
    print("6. POOLING would have hidden a failure (why per-allele matters)")
    print("=" * 72)
    pred = np.empty(len(test))
    alleles = sorted(test.HLA.unique())
    for i, a in enumerate(alleles):
        idx = test.groupby("HLA").indices[a]
        # half the alleles predicted well, half at chance, plus a big per-allele
        # offset so the pooled correlation is carried by between-allele spread
        r = 0.8 if i % 2 == 0 else 0.0
        pred[idx] = _synth(test.y.values[idx], r, rng) + 3 * test.y.values[idx].mean()
    pooled = stats.spearmanr(pred, test.Thalf.values).statistic
    s = spearman_per_allele(test, pred)
    good = [a for i, a in enumerate(alleles) if i % 2 == 0 and a in s.index]
    bad = [a for i, a in enumerate(alleles) if i % 2 == 1 and a in s.index]
    print(f"  POOLED rho over all test rows : {pooled:+.3f}   <- looks fine")
    print(f"  per-allele median             : {np.median(s):+.3f}")
    print(f"  good half, median             : {np.median(s[good]):+.3f}")
    print(f"  FAILING half, median          : {np.median(s[bad]):+.3f}   <- at chance")
    print("  Half the alleles are at chance and pooling conceals it entirely.\n")

    print("=" * 72)
    print("7. GUARDS")
    print("=" * 72)
    s = spearman_per_allele(df, rng.standard_normal(len(df)))
    print(f"  full data, all 75 alleles -> {len(s)} scored, "
          f"{len(s.attrs['skipped'])} skipped (small n / no variance)")
    for a, why in list(s.attrs["skipped"].items())[:4]:
        print(f"    {a}: {why}")
    const = np.zeros(len(test))
    s = spearman_per_allele(test, const)
    print(f"  constant predictor -> {len(s)} scored, "
          f"{len(s.attrs['skipped'])} skipped as 'no variance' (not scored as 0)")
    print()
    print(summarise(spearman_per_allele(test, _synth(test.y.values, 0.6, rng)),
                    "example slide line"))


if __name__ == "__main__":
    _verify()
