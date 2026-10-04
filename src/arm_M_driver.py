"""Overnight driver for arm M. Staged inner-split tuning, then the full 21-fold runs.

Stage 1  features           (MLP + squared error, so only the features move)
Stage 2  loss and capacity  (around the stage-1 winner)
Stage 3  alternative heads  (gradient boosting, ridge) on the stage-1 features
Stage 4  the full protocol  21 groove folds x 5 seeds for a shortlist, plus a
                            replication of arm A through this same harness as a control
Stage 5  scoring            ensemble, censored='drop' robustness, paired Wilcoxon
                            against arm A's stored per-fold numbers

Every stage writes its own csv as it goes, so an interrupted night still leaves the
finished parts behind. Nothing in stages 1 to 3 ever loads an outer test fold.

    python arm_M_driver.py            # everything
    python arm_M_driver.py tune       # stages 1-3 only
    python arm_M_driver.py full       # stage 4-5 from the shortlist already on disk
    python arm_M_driver.py score      # stage 5 only
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy import stats

import arm_M_interact as M
import metrics
import run_experiment as R
import splits

DEVICE = os.environ.get("ARM_M_DEVICE") or None
N_OUTER = int(os.environ.get("ARM_M_N_OUTER", 5))
TUNE_SEEDS = (0, 1, 2)
FULL_SEEDS = int(os.environ.get("ARM_M_SEEDS", 5))
WIDTH_CAP = 9000

ARM_A = "A_supervised_nn"
# Same features as arm A, byte for byte (asserted in arm_M_interact), run through this
# file's model wrapper. It differs from arm A only in WHICH alleles land in the inner
# early-stopping split, which is pure nuisance variance -- so it is the fair paired
# reference for everything arm M adds, and the gap between it and arm A's stored 0.307
# is itself a measurement of how much the harness alone moves the headline.
ARM_CTRL = "M_ctrl_armA_repl"
SHORTLIST_JSON = "results_M_shortlist.json"


def _banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


# ---------------------------------------------------------------------------
# stage 1: which interaction features, if any
# ---------------------------------------------------------------------------

def stage1_candidates():
    base = dict(model="mlp", loss="mse", base="both")
    c = []
    c.append(("00 no interaction (arm A features)", M.Cfg(inter="none", **base)))
    # where the pockets come from
    for src in ("structure", "label", "all34"):
        k = 34 if src == "all34" else 6
        c.append((f"10 pca6 k6 {src:<9} P2,P9",
                  M.Cfg(inter="pca6", pocket_src=src, k=k, anchors=(1, 8), **base)))
    # which anchor positions
    for lab, anch in [("P2,P9", (1, 8)), ("derive2", "derive2"), ("derive3", "derive3"),
                      ("P1,P2,P9", (0, 1, 8)), ("all nine", "all")]:
        c.append((f"20 pca6 k6 structure {lab}",
                  M.Cfg(inter="pca6", pocket_src="structure", k=6, anchors=anch, **base)))
    # pocket size
    for k in (3, 10, 34):
        c.append((f"30 pca6 k{k:<2} structure P2,P9",
                  M.Cfg(inter="pca6", pocket_src="structure", k=k, anchors=(1, 8), **base)))
    # interaction richness
    for it in ("pca3", "pca10", "pca20", "sim", "pca6+sim"):
        c.append((f"40 {it:<9} k6 structure P2,P9",
                  M.Cfg(inter=it, pocket_src="structure", k=6, anchors=(1, 8), **base)))
    # full grid of cheap pair similarities
    c.append(("41 sim k34 structure all nine",
              M.Cfg(inter="sim", pocket_src="all34", k=34, anchors="all", **base)))
    # is the base block still needed?
    c.append(("50 interaction only, no base block",
              M.Cfg(inter="pca10", pocket_src="structure", k=10, anchors="all",
                    model="mlp", loss="mse", base="none")))
    return c


# ---------------------------------------------------------------------------
# stage 2: loss and capacity around the stage-1 winner
# ---------------------------------------------------------------------------

def stage2_candidates(best: M.Cfg, best_nointer: M.Cfg):
    c = [("60 winner, squared error", best),
         ("61 winner, TOBIT", best.__class__(**{**best.__dict__, "loss": "tobit"})),
         ("62 arm-A features, TOBIT",
          best_nointer.__class__(**{**best_nointer.__dict__, "loss": "tobit"}))]
    for lab, kw in [
        ("70 hidden 512-256", dict(hidden=(512, 256))),
        ("71 hidden 512-256-128", dict(hidden=(512, 256, 128))),
        ("72 hidden 256", dict(hidden=(256,))),
        ("73 dropout 0.2", dict(dropout=0.2)),
        ("74 dropout 0.3", dict(dropout=0.3)),
        ("75 lr 1e-3", dict(lr=1e-3)),
        ("76 lr 1e-4", dict(lr=1e-4)),
        ("77 weight_decay 1e-4", dict(weight_decay=1e-4)),
        ("78 weight_decay 1e-3", dict(weight_decay=1e-3)),
        ("79 batch 256", dict(batch=256)),
    ]:
        c.append((lab, best.__class__(**{**best.__dict__, **kw})))
    return c


# ---------------------------------------------------------------------------
# stage 3: alternative heads
# ---------------------------------------------------------------------------

def stage3_candidates(best: M.Cfg, best_nointer: M.Cfg):
    def hgb(**kw):
        return dict(model="hgb", hgb=kw)
    c = [("80 HGB default, arm-A residues only",
          best_nointer.__class__(**{**best_nointer.__dict__, **hgb()})),
         ("81 HGB default + interaction",
          best.__class__(**{**best.__dict__, **hgb()}))]
    for lab, kw in [
        ("82 HGB lr 0.03 iter 800", dict(learning_rate=0.03, max_iter=800)),
        ("83 HGB leaves 63", dict(max_leaf_nodes=63)),
        ("84 HGB leaves 15, lr 0.1", dict(max_leaf_nodes=15, learning_rate=0.1)),
        ("85 HGB min_leaf 100", dict(min_samples_leaf=100)),
        ("86 HGB l2 10", dict(l2_regularization=10.0)),
    ]:
        c.append((lab, best.__class__(**{**best.__dict__, **hgb(**kw)})))
    c.append(("90 ridge alpha 1 on interaction features",
              best.__class__(**{**best.__dict__, "model": "ridge", "alpha": 1.0})))
    c.append(("91 ridge alpha 30 on interaction features",
              best.__class__(**{**best.__dict__, "model": "ridge", "alpha": 30.0})))
    return c


# ---------------------------------------------------------------------------
# sweep runner
# ---------------------------------------------------------------------------

def run_stage(name, cands, out):
    _banner(f"STAGE {name}: {len(cands)} candidates, inner splits of TRAIN only "
            f"({N_OUTER} outer folds x {len(TUNE_SEEDS)} seeds)")
    rows = []
    for label, cfg in cands:
        w = M.width(cfg)
        if w > WIDTH_CAP:
            print(f"\n[{label}] SKIPPED, width {w} over the {WIDTH_CAP} cap", flush=True)
            rows.append({"label": label, "tag": cfg.tag(), "inner_rho": np.nan,
                         "width": w, "seconds": 0.0, "error": "width cap",
                         "detail": "[]", "cfg": json.dumps(_cfg_json(cfg))})
            pd.DataFrame(rows).to_csv(out, index=False)
            continue
        print(f"\n[{label}]  {cfg.tag()}  width {w}", flush=True)
        t0 = time.time()
        try:
            mean, det = M.inner_eval(cfg, n_outer=N_OUTER, seeds=TUNE_SEEDS, device=DEVICE)
            err = ""
        except Exception as e:
            traceback.print_exc()
            mean, det, err = np.nan, [], f"{type(e).__name__}: {e}"
        print(f"  => inner rho {mean:+.4f}   {time.time()-t0:.0f}s"
              + (f"   FAILED {err}" if err else ""), flush=True)
        rows.append({"label": label, "tag": cfg.tag(), "inner_rho": mean, "width": w,
                     "seconds": round(time.time() - t0, 1), "error": err,
                     "detail": json.dumps(det), "cfg": json.dumps(_cfg_json(cfg))})
        pd.DataFrame(rows).to_csv(out, index=False)
    t = pd.DataFrame(rows).sort_values("inner_rho", ascending=False)
    print(f"\n--- stage {name} ranking (inner validation, TRAIN only) ---")
    print(t[["label", "inner_rho", "width", "seconds"]].to_string(index=False), flush=True)
    return t


def extra_candidates():
    """Stage 4: the within-allele target transform, the learned bilinear term, and the
    best combinations stages 1-3 found.

    Built from whatever the earlier stages wrote to disk, so this can be run on its own
    after the fact. Still inner-split only.
    """
    s1 = pd.read_csv("results_M_tuning_stage1.csv")
    best = pick(s1)
    nointer = _cfg_from_json(json.loads(
        s1[s1.label.str.startswith("00")].iloc[0]["cfg"]))
    # The two references are re-run here on purpose. They make this stage readable on
    # its own, and because inner_eval is deterministic they must reproduce their
    # stage-1 numbers exactly -- a free check that nothing drifted between processes.
    c = [("R0 reference: arm-A features", nointer),
         ("R1 reference: stage-1 winner", best)]
    # the within-allele target transform, with and without the interaction block, so
    # it is clear whether the two are additive
    c.append(("B0 target=centred, arm-A features",
              M.Cfg(**{**nointer.__dict__, "target": "centred"})))
    c.append(("B1 target=centred + interaction",
              M.Cfg(**{**best.__dict__, "target": "centred"})))
    c.append(("B2 target=standardised + interaction",
              M.Cfg(**{**best.__dict__, "target": "standardised"})))
    c.append(("B3 target=centred + interaction + TOBIT",
              M.Cfg(**{**best.__dict__, "target": "centred", "loss": "tobit"})))
    # the learned rank-d bilinear term
    c.append(("A0 towers d64, arm-A features",
              M.Cfg(**{**nointer.__dict__, "net": "towers", "d_tower": 64})))
    c.append(("A1 towers d64 + interaction block",
              M.Cfg(**{**best.__dict__, "net": "towers", "d_tower": 64})))
    c.append(("A2 towers d64 + interaction + TOBIT",
              M.Cfg(**{**best.__dict__, "net": "towers", "d_tower": 64,
                       "loss": "tobit"})))
    try:
        t2 = pd.read_csv("results_M_tuning_stage2.csv")
        b2 = pick(t2)
        c.append(("A3 stage-2 winner + TOBIT",
                  M.Cfg(**{**b2.__dict__, "loss": "tobit"})))
        c.append(("A4 stage-2 winner + towers d64",
                  M.Cfg(**{**b2.__dict__, "net": "towers", "d_tower": 64})))
        c.append(("A5 stage-2 winner + target=centred",
                  M.Cfg(**{**b2.__dict__, "target": "centred"})))
    except Exception:
        traceback.print_exc()
    # a structurally different model: predict the unseen groove's PWM from its 34
    # residues. Weaker on its own by construction, but wrong in a different way.
    for al in (3.0, 10.0, 30.0, 100.0):
        c.append((f"P0 PWM-from-pseudo-sequence, ridge alpha {al:g}",
                  M.Cfg(model="pwm", inter="none", base="none", alpha=al)))
    try:
        t3 = pd.read_csv("results_M_tuning_stage3.csv")
        b3 = pick(t3)
        c.append(("A6 stage-3 head + target=centred",
                  M.Cfg(**{**b3.__dict__, "target": "centred"})))
    except Exception:
        traceback.print_exc()
    return c


def overlap_split(arms, censored="tied"):
    """How much of each arm's score lives on the peptide-overlap surface?

    80.9% of test rows share their peptide with a training row paired against a
    different allele. Split every arm's stored predictions on that, and score the two
    halves separately with the same metric. This does not change any headline; it says
    where the headline comes from, and whether the interaction features earn anything
    on peptides the model has genuinely never seen.
    """
    _banner("DIAGNOSTIC: score split by whether the test peptide is also in train")
    df = R.load_df()
    folds = splits.choose_held_out(df)
    seen_by_fold = {}
    for name, alleles in folds:
        tr, _ = splits.split_by_allele(df, alleles)
        seen_by_fold[name] = set(df.loc[tr].Pep.unique())
    rows = []
    for arm in arms:
        path = f"predictions_{arm}.parquet"
        if not os.path.exists(path):
            continue
        p = pd.read_parquet(path)
        p = p.groupby(["fold_name", "row_id"], as_index=False).y_pred.mean()
        per = {"seen": [], "novel": []}
        frac = []
        for name, sub in p.groupby("fold_name"):
            te = df.loc[sub.row_id.to_numpy()]
            seen = te.Pep.isin(seen_by_fold[name]).to_numpy()
            frac.append(seen.mean())
            for key, m in (("seen", seen), ("novel", ~seen)):
                if m.sum() < 40:
                    continue
                s = metrics.spearman_per_allele(te[m], sub.y_pred.to_numpy()[m],
                                                censored=censored)
                if len(s):
                    per[key].append(float(np.median(s)))
        rows.append({"arm": arm,
                     "frac_test_rows_peptide_seen": float(np.mean(frac)),
                     "median_rho_seen_peptides": float(np.median(per["seen"])) if per["seen"] else np.nan,
                     "n_folds_seen": len(per["seen"]),
                     "median_rho_novel_peptides": float(np.median(per["novel"])) if per["novel"] else np.nan,
                     "n_folds_novel": len(per["novel"])})
    t = pd.DataFrame(rows)
    t.to_csv("results_M_overlap_split.csv", index=False)
    print(t.round(4).to_string(index=False), flush=True)
    return t


def _cfg_json(cfg):
    d = dict(cfg.__dict__)
    d["hidden"] = list(d["hidden"])
    d["anchors"] = d["anchors"] if isinstance(d["anchors"], str) else list(d["anchors"])
    return d


def _cfg_from_json(d):
    d = dict(d)
    d["hidden"] = tuple(d["hidden"])
    if not isinstance(d["anchors"], str):
        d["anchors"] = tuple(d["anchors"])
    return M.Cfg(**d)


# ---------------------------------------------------------------------------
# stage 4 + 5
# ---------------------------------------------------------------------------

def full_runs(shortlist):
    _banner(f"STAGE 4: full protocol, 21 groove folds x {FULL_SEEDS} seeds, "
            f"{len(shortlist)} arms")
    done = []
    for name, cfg in shortlist:
        if os.path.exists(f"results_M_{name}.csv"):
            try:
                r = pd.read_csv(f"results_M_{name}.csv")
                if (r.status == "ok").sum() >= 21 * FULL_SEEDS:
                    print(f"[M_{name}] already complete, skipping", flush=True)
                    done.append((name, cfg))
                    continue
            except Exception:
                pass
        print(f"\n--- M_{name}  {cfg.tag()} ---", flush=True)
        try:
            M.run(name, cfg, seeds=FULL_SEEDS, device=DEVICE)
            done.append((name, cfg))
        except Exception as e:
            traceback.print_exc()
            print(f"[M_{name}] FAILED {type(e).__name__}: {e}", flush=True)
    return done


_FOLD_CACHE = {}


def _fold_series(arm, censored="tied"):
    """Per-fold ensemble median per-allele Spearman, indexed by fold name."""
    key = (arm, censored)
    if key not in _FOLD_CACHE:
        ens = R.ensemble_metrics(arm, censored=censored)
        ens = ens[ens.status == "ok"]
        _FOLD_CACHE[key] = ens.set_index("fold_name").spearman
    return _FOLD_CACHE[key]


def _paired(s, ref, prefix, row):
    common = ref.index.intersection(s.index)
    d = (s[common] - ref[common]).values
    row[f"{prefix}_delta"] = float(np.median(d))
    row[f"{prefix}_wins"] = int((d > 0).sum())
    row[f"{prefix}_n"] = int(len(common))
    if len(common) >= 6 and np.any(d != 0):
        w = stats.wilcoxon(s[common].values, ref[common].values)
        row[f"{prefix}_p"] = float(w.pvalue)
    else:
        row[f"{prefix}_p"] = np.nan


def score_all(names, censored_list=("tied", "drop")):
    _banner("STAGE 5: scoring on the 21 folds; paired Wilcoxon vs arm A and vs the "
            "same-features control")
    ref_a = {c: _fold_series(ARM_A, c) for c in censored_list}
    ref_c = {}
    for c in censored_list:
        if os.path.exists(f"predictions_{ARM_CTRL}.parquet"):
            try:
                ref_c[c] = _fold_series(ARM_CTRL, c)
            except Exception:
                pass
    rows = []
    for arm in names:
        if not os.path.exists(f"predictions_{arm}.parquet"):
            print(f"  {arm}: no predictions file, skipped", flush=True)
            continue
        row = {"arm": arm}
        for c in censored_list:
            try:
                s = _fold_series(arm, c)
            except Exception as e:
                row[f"median_{c}"] = np.nan
                row[f"err_{c}"] = f"{type(e).__name__}: {e}"
                continue
            row[f"median_{c}"] = float(np.median(s))
            row[f"nfolds_{c}"] = int(len(s))
            _paired(s, ref_a[c], f"vsA_{c}", row)
            if c in ref_c and arm != ARM_CTRL:
                _paired(s, ref_c[c], f"vsCTRL_{c}", row)
        rows.append(row)
    t = pd.DataFrame(rows)
    a = pd.DataFrame([{"arm": ARM_A,
                       **{f"median_{c}": float(np.median(ref_a[c])) for c in censored_list},
                       **{f"nfolds_{c}": int(len(ref_a[c])) for c in censored_list}}])
    t = pd.concat([a, t], ignore_index=True)
    t.to_csv("results_M_headline.csv", index=False)
    show = ["arm"] + [c for c in t.columns
                      if c.startswith(("median_", "vsA_tied", "vsCTRL_tied"))]
    print(t[show].to_string(index=False), flush=True)
    print("\n(full table incl. censored='drop' in results_M_headline.csv)", flush=True)
    return t


def per_fold_table(names, censored="tied"):
    """One column per arm, one row per fold. Written for the write-up."""
    out = {ARM_A: _fold_series(ARM_A, censored)}
    for arm in names:
        if os.path.exists(f"predictions_{arm}.parquet"):
            try:
                out[arm] = _fold_series(arm, censored)
            except Exception:
                pass
    t = pd.DataFrame(out)
    t.to_csv(f"results_M_per_fold_{censored}.csv")
    print(f"\n--- per-fold median per-allele Spearman, censored='{censored}' ---")
    print(t.round(3).to_string(), flush=True)
    print("\nmedian over folds:")
    print(t.median().round(4).to_string(), flush=True)
    return t


# ---------------------------------------------------------------------------
# blends, computed from predictions already on disk (no new fitting)
# ---------------------------------------------------------------------------

def _cfg_key(cfg):
    """A canonical string for a Cfg, stable across the stage files even though those
    were written before `net`, `target` and `split_seed` existed (both sides are run
    through the current Cfg, so the defaults fill in identically)."""
    return json.dumps(_cfg_json(cfg), sort_keys=True)


def inner_rho_lookup():
    """{canonical cfg key: inner-validation rho} over every tuning stage on disk."""
    out = {}
    for s in (1, 2, 3, 4):
        p = f"results_M_tuning_stage{s}.csv"
        if not os.path.exists(p):
            continue
        for _, r in pd.read_csv(p).iterrows():
            if pd.isna(r.inner_rho):
                continue
            try:
                out[_cfg_key(_cfg_from_json(json.loads(r["cfg"])))] = float(r.inner_rho)
            except Exception:
                pass
    return out


def blend_members(shortlist, margin=0.02):
    """Which arms go into the blend, decided on INNER validation only.

    Every arm whose inner-validation rho is within `margin` of the best inner rho among
    the shortlisted arms. The control arms and anything never scored on an inner split
    are excluded. No test fold is consulted, and the margin is fixed in advance rather
    than tuned, so the blend inherits the protocol.
    """
    look = inner_rho_lookup()
    scored = [(n, look[_cfg_key(c)]) for n, c in shortlist
              if _cfg_key(c) in look and not n.startswith("ctrl")]
    if not scored:
        return []
    best = max(v for _, v in scored)
    keep = [(n, v) for n, v in scored if v >= best - margin]
    print(f"  blend members (inner rho within {margin} of {best:+.4f}):", flush=True)
    for n, v in sorted(keep, key=lambda t: -t[1]):
        print(f"    M_{n:<18} inner {v:+.4f}", flush=True)
    dropped = [(n, v) for n, v in scored if v < best - margin]
    for n, v in sorted(dropped, key=lambda t: -t[1]):
        print(f"    (dropped M_{n:<14} inner {v:+.4f})", flush=True)
    return [f"M_{n}" for n, _ in keep]


def blend(name, arms, weights=None):
    """Average the stored per-row predictions of several arms into a new arm.

    Every component was fitted without ever seeing a test fold, and the weights here
    are fixed (equal unless given), chosen without looking at any test score, so the
    blend inherits the protocol. Rank-averaging is used rather than value-averaging
    because the metric is a rank correlation and the arms are not on a common scale.
    """
    frames = []
    for a in arms:
        p = pd.read_parquet(f"predictions_{a}.parquet")
        p = p.groupby(["fold_name", "row_id"], as_index=False).agg(
            HLA=("HLA", "first"), Pep=("Pep", "first"),
            y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
        p["y_pred"] = p.groupby("fold_name").y_pred.rank(pct=True)
        frames.append(p.set_index(["fold_name", "row_id"]))
    w = weights or [1.0] * len(frames)
    base = frames[0].copy()
    base["y_pred"] = sum(wi * f.y_pred for wi, f in zip(w, frames)) / sum(w)
    base = base.reset_index()
    base["seed"] = np.int32(0)
    base = base[["row_id", "fold_name", "seed", "HLA", "Pep", "y_true", "y_pred"]]
    base.to_parquet(f"predictions_{name}.parquet", index=False)
    print(f"[{name}] blended {arms} -> {len(base)} rows", flush=True)
    return name


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def pick(t, exclude_label_prefix=None):
    t = t[t.inner_rho.notna()]
    if exclude_label_prefix:
        t = t[~t.label.str.startswith(exclude_label_prefix)]
    if not len(t):
        return None
    return _cfg_from_json(json.loads(t.sort_values("inner_rho", ascending=False)
                                     .iloc[0]["cfg"]))


def extend_shortlist(new):
    """Append (name, cfg) pairs to the shortlist on disk, skipping duplicate tags."""
    with open(SHORTLIST_JSON) as fh:
        cur = [(n, _cfg_from_json(d)) for n, d in json.load(fh)]
    seen = {c.tag() for _, c in cur}
    for n, c in new:
        if c.tag() in seen:
            print(f"  (extend: {n} duplicates an existing config, dropped)", flush=True)
            continue
        seen.add(c.tag())
        cur.append((n, c))
    with open(SHORTLIST_JSON, "w") as fh:
        json.dump([(n, _cfg_json(c)) for n, c in cur], fh, indent=1)
    return cur


def report():
    """Print everything BEST_MODEL.md needs, from the files on disk."""
    pd.set_option("display.width", 200)
    with open(SHORTLIST_JSON) as fh:
        shortlist = [(n, _cfg_from_json(d)) for n, d in json.load(fh)]
    _banner("SHORTLIST (what each arm M_* actually is)")
    look = inner_rho_lookup()
    for n, c in shortlist:
        ir = look.get(_cfg_key(c))
        print(f"  M_{n:<18} inner {('%+.4f' % ir) if ir is not None else '    n/a'}  "
              f"{c.tag()}")
    for s in (1, 2, 3, 4):
        p = f"results_M_tuning_stage{s}.csv"
        if not os.path.exists(p):
            continue
        t = pd.read_csv(p).sort_values("inner_rho", ascending=False)
        _banner(f"TUNING STAGE {s} (inner splits of TRAIN, NOT comparable to 0.307)")
        print(t[["label", "inner_rho", "width", "seconds", "error"]]
              .to_string(index=False))
    arms = [f"M_{n}" for n, _ in shortlist] + ["M_blend_inner", "M_blend_net_tree", "M_blend_diverse"]
    arms = [a for a in arms if os.path.exists(f"predictions_{a}.parquet")]
    score_all(arms)
    per_fold_table(arms, "tied")
    per_fold_table(arms, "drop")
    for p, title in [("results_M_leakcheck.csv", "LEAK CHECK"),
                     ("results_M_permutation_control.csv", "PERMUTATION CONTROL"),
                     ("results_M_overlap_split.csv", "PEPTIDE-OVERLAP SPLIT")]:
        if os.path.exists(p):
            _banner(title)
            print(pd.read_csv(p).to_string(index=False))


def main(what="all"):
    t_start = time.time()
    print(f"arm M driver  device={DEVICE or 'auto'}  n_outer={N_OUTER}  "
          f"tune seeds={TUNE_SEEDS}  full seeds={FULL_SEEDS}", flush=True)
    M._check_log_ndtr(DEVICE)

    shortlist = []
    if what in ("all", "tune"):
        t1 = run_stage("1 FEATURES", stage1_candidates(), "results_M_tuning_stage1.csv")
        best = pick(t1)
        nointer = _cfg_from_json(json.loads(
            t1[t1.label.str.startswith("00")].iloc[0]["cfg"]))
        print(f"\nstage 1 winner: {best.tag()}", flush=True)

        t2 = run_stage("2 LOSS + CAPACITY", stage2_candidates(best, nointer),
                       "results_M_tuning_stage2.csv")
        best2 = pick(t2)
        print(f"\nstage 2 winner: {best2.tag()}", flush=True)

        t3 = run_stage("3 ALTERNATIVE HEADS", stage3_candidates(best, nointer),
                       "results_M_tuning_stage3.csv")
        best3 = pick(t3)
        print(f"\nstage 3 winner: {best3.tag()}", flush=True)

        # the shortlist that goes to the full protocol. Names are fixed so a rerun
        # picks the finished files back up.
        shortlist = [
            ("ctrl_armA_repl", nointer),                      # control: must land ~0.307
            ("feat_best", best),                              # stage 1 winner
            ("tune_best", best2),                             # stage 2 winner
            ("tobit", M.Cfg(**{**best.__dict__, "loss": "tobit"})),
            ("tobit_base", M.Cfg(**{**nointer.__dict__, "loss": "tobit"})),
            ("head_best", best3),                             # stage 3 winner
        ]
        # drop duplicates by tag, keep first
        seen, uniq = set(), []
        for n, c in shortlist:
            if c.tag() in seen:
                print(f"  (shortlist: {n} duplicates an earlier config, dropped)")
                continue
            seen.add(c.tag())
            uniq.append((n, c))
        shortlist = uniq
        with open(SHORTLIST_JSON, "w") as fh:
            json.dump([(n, _cfg_json(c)) for n, c in shortlist], fh, indent=1)
        print(f"\nshortlist -> {SHORTLIST_JSON}", flush=True)
        for n, c in shortlist:
            print(f"  M_{n:<16} {c.tag()}  width {M.width(c)}", flush=True)

    if what == "extra":
        t4 = run_stage("4 LEARNED BILINEAR + COMBINATIONS", extra_candidates(),
                       "results_M_tuning_stage4.csv")
        best4 = pick(t4)
        print(f"\nstage 4 winner: {best4.tag()}", flush=True)
        add = [("stage4_best", best4)]
        t4ok = t4[t4.inner_rho.notna()]
        for pre, nm in (("A0", "towers_best"), ("A1", "towers_best"), ("P0", "pwm_best")):
            sel = t4ok[t4ok.label.str.startswith(pre)]
            if len(sel) and nm not in [n for n, _ in add]:
                add.append((nm, _cfg_from_json(json.loads(
                    sel.sort_values("inner_rho", ascending=False).iloc[0]["cfg"]))))
        # How much does the headline move if ONLY the inner early-stopping split
        # changes? Same features, same hyperparameters, same 21 folds, same 5 seeds;
        # just a different relabelling of the alleles before the split picks some.
        # Nothing is selected on these -- they are the error bar for everything else.
        s1 = pd.read_csv("results_M_tuning_stage1.csv")
        nointer = _cfg_from_json(json.loads(
            s1[s1.label.str.startswith("00")].iloc[0]["cfg"]))
        for sp in (1, 2):
            add.append((f"ctrl_split{sp}",
                        M.Cfg(**{**nointer.__dict__, "split_seed": sp})))
        shortlist = extend_shortlist(add)
        what = "full"

    if what in ("all", "full", "score"):
        with open(SHORTLIST_JSON) as fh:
            shortlist = [(n, _cfg_from_json(d)) for n, d in json.load(fh)]

    if what in ("all", "full"):
        full_runs(shortlist)

    if what in ("all", "full", "score"):
        arms = [f"M_{n}" for n, _ in shortlist]
        arms = [a for a in arms if os.path.exists(f"predictions_{a}.parquet")]
        # Blends, assembled from stored predictions. Membership is decided on inner
        # validation only (blend_members); no test fold is consulted.
        try:
            mem = [a for a in blend_members(shortlist) if a in arms]
            if len(mem) >= 2:
                blend("M_blend_inner", mem)
                arms.append("M_blend_inner")
        except Exception:
            traceback.print_exc()
        # Two more blends, both specified before any test number was looked at.
        #   net_tree : the best net and the best tree head, the two that disagree most
        #   diverse  : one arm from each model family, each the best of its family on
        #              inner validation, equal weights
        for nm, want in [("M_blend_net_tree", ["M_feat_best", "M_head_best"]),
                         ("M_blend_diverse", ["M_feat_best", "M_head_best", "M_pwm_best"])]:
            have = [a for a in want if a in arms]
            if len(have) >= 2 and nm not in arms:
                try:
                    blend(nm, have)
                    arms.append(nm)
                except Exception:
                    traceback.print_exc()
        score_all(arms)
        per_fold_table(arms, "tied")
        per_fold_table(arms, "drop")
        try:
            overlap_split([ARM_A] + arms)
        except Exception:
            traceback.print_exc()

    print(f"\ndriver finished in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    report() if cmd == "report" else main(cmd)
