"""The predict interface the app is built against, plus a placeholder model.

THE CONTRACT. Anything that satisfies this can be dropped in by editing the one
line marked SWAP HERE at the bottom of this file.

    predict_batch(peptides: list[str], allele: str) -> list[Prediction]

    Prediction.thalf  float  predicted half-life, HOURS (not log10)
    Prediction.lo     float  low edge of the band, hours
    Prediction.hi     float  high edge of the band, hours
    Prediction.seeds  list[float]  per-seed point predictions, hours

Band convention: lo/hi are the min/max across the seed ensemble, in hours. The
app draws them literally and labels them "seed ensemble", so a model whose seeds
all agree will honestly show a hairline band.

The real model is expected to widen its own band out of distribution. The
placeholder below fakes that widening so the demo reads correctly today; the
fake is confined to _ood_inflation() and goes away with the swap.

PLACEHOLDER: ridge regression on one-hot peptide + one-hot allele, fit on the
real stability.txt at import (~0.4 s, cached to predict_cache.npz). It is not
the project's model and makes no use of foundation-model embeddings. It is here
so the app runs end to end, and because a crude model that learned real anchor
preferences makes the mutant scan show structure instead of noise.
"""

import os
from dataclasses import dataclass

import numpy as np

import data

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}
PEPLEN = 9
N_SEEDS = 5
CACHE = "predict_cache.npz"


@dataclass
class Prediction:
    thalf: float
    lo: float
    hi: float
    seeds: list


def valid_peptide(pep):
    """(ok, message). The app shows `message` inline and refuses to predict."""
    pep = (pep or "").strip().upper()
    if not pep:
        return False, "Enter a 9-residue peptide."
    if len(pep) != PEPLEN:
        return False, f"Need exactly 9 residues — you gave {len(pep)}."
    bad = sorted({c for c in pep if c not in AA_IDX})
    if bad:
        return False, f"Not standard amino acids: {', '.join(bad)}"
    return True, ""


# --------------------------------------------------------------------------
# placeholder model
# --------------------------------------------------------------------------

def _onehot(peps):
    X = np.zeros((len(peps), PEPLEN * 20), dtype=np.float32)
    for r, p in enumerate(peps):
        for i, c in enumerate(p.upper()):
            X[r, i * 20 + AA_IDX[c]] = 1.0
    return X


class _Ridge:
    """Seed ensemble of ridge fits on bootstrap resamples. Deterministic."""

    def __init__(self):
        df = data.load()
        self.alleles = sorted(df.HLA.unique())
        self.a_idx = {a: i for i, a in enumerate(self.alleles)}
        # Per-allele intercept, learned separately so an unseen allele can fall
        # back to the global mean rather than failing.
        self.global_mean = float(df.y.mean())
        self.a_mean = df.groupby("HLA").y.mean().to_dict()

        if os.path.exists(CACHE):
            z = np.load(CACHE)
            if z["n"] == len(df) and z["W"].shape == (N_SEEDS, PEPLEN * 20):
                self.W = z["W"]
                return

        from sklearn.linear_model import Ridge
        X = _onehot(df.Pep.tolist())
        resid = df.y.values - df.HLA.map(self.a_mean).values  # allele-free part
        W = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed)
            take = rng.integers(0, len(X), len(X))
            m = Ridge(alpha=10.0, fit_intercept=False).fit(X[take], resid[take])
            W.append(m.coef_.astype(np.float32))
        self.W = np.stack(W)
        np.savez(CACHE, W=self.W, n=len(df))

    def __call__(self, peps, allele):
        X = _onehot(peps)                       # (n, 180)
        base = self.a_mean.get(allele, self.global_mean)
        return X @ self.W.T + base              # (n, N_SEEDS) in log10 hours


_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = _Ridge()
    return _MODEL


def _ood_inflation(allele):
    """PLACEHOLDER ONLY. Widen the band out of distribution.

    The real model's seeds should disagree more on an allele it has no support
    for, because that is the finding. The placeholder's seeds differ only by
    bootstrap noise on the peptide term and are blind to the allele, so we
    inflate by hand to keep the demo honest about what it is claiming. Delete
    this with the swap.
    """
    import ood
    return 1.0 + 2.5 * ood.score(allele)


def _stub_predict_batch(peptides, allele):
    Y = _model()(peptides, allele)              # (n, N_SEEDS) log10 hours
    k = _ood_inflation(allele)
    out = []
    for row in Y:
        mid = float(row.mean())
        spread = (row - mid) * k
        seeds = np.clip(10.0 ** (mid + spread), 0.0, 1e4)
        out.append(Prediction(
            thalf=float(10.0 ** mid),
            lo=float(seeds.min()),
            hi=float(seeds.max()),
            seeds=[float(v) for v in seeds],
        ))
    return out


# ======================= SWAP HERE =======================
# Replace the right-hand side with the real ensemble, e.g.
#     from model import predict_batch
# and delete _ood_inflation. Nothing else in the app changes.
predict_batch = _stub_predict_batch
IS_STUB = predict_batch is _stub_predict_batch
# =========================================================


def predict_one(peptide, allele):
    return predict_batch([peptide.strip().upper()], allele)[0]


def warm():
    """Touch everything slow so the first click is fast. Called at app import."""
    _model()
    predict_one("A" * PEPLEN, _model().alleles[0])


if __name__ == "__main__":
    warm()
    for a in ["HLA-A*02:01", "HLA-B*07:02"]:
        p = predict_one("SLYNTVATL", a)
        print(f"{a}  {p.thalf:6.2f} h  [{p.lo:.2f}, {p.hi:.2f}]")


# ======================= REAL MODEL ADAPTER ==============================
# Wraps design.py's 5-seed arm-A ensemble (the conventional supervised net,
# the arm that won) in the predict_batch contract above. Trains once on first
# call, ~2 min, then cached. Unmeasured alleles are supported by deriving their
# 34-residue pseudo-sequence from unmeasured.json via ood.GROOVE, the same
# positions arm A was trained on.
_ENS = None


def _real_ensemble():
    global _ENS
    if _ENS is None:
        import json
        import arm_A_supervised_nn as armA
        import design
        import ood
        pm = armA.pseudo_map()                      # 75 measured alleles
        for name, rec in json.load(open(ood.UNMEASURED)).items():
            if name not in pm:
                g = ood._groove(rec.get("seq", ""))
                if g and len(g) == armA.PSEUDO_LEN:
                    pm[name] = g                    # make it featurizable
        _ENS = design.train(verbose=False)
    return _ENS


def _real_predict_batch(peptides, allele):
    import numpy as np
    import pandas as pd
    ens = _real_ensemble()
    df = pd.DataFrame({"HLA": allele, "Pep": list(peptides)})
    P = np.stack([m.predict(ens.featurize(df)) for m in ens.models])   # log10 h
    H = np.power(10.0, P)                                             # hours
    return [Prediction(thalf=float(H[:, i].mean()),
                       lo=float(H[:, i].min()),
                       hi=float(H[:, i].max()),
                       seeds=[float(v) for v in H[:, i]])
            for i in range(H.shape[1])]


predict_batch = _real_predict_batch
IS_STUB = predict_batch is _stub_predict_batch
# =========================================================================
