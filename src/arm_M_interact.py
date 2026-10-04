"""Arm M: explicit anchor-pocket interaction features, and a censored-aware loss.

Two things arm A does not do, both pointed at by this project's own diagnostics:

  1. ANCHOR x POCKET INTERACTION. The anchor preference is allele-specific (A*02:01
     wants L at P2, B*27:05 wants R). Arm A concatenates the peptide block and the
     pseudo-sequence block and leaves the network to discover the pairing. Here the
     pairing is a feature: an outer product between the residue at an anchor position
     and the residue at each pseudo-sequence position that lines its pocket. Pocket
     membership is DERIVED (arm_M_pockets.py), either from the training labels or from
     measured contact distances in a crystal structure; never quoted.

  2. CENSORED-AWARE LOSS. 20.2% of rows sit at Thalf == 0.0, a left-censored detection
     floor. Arm A trains them against a fabricated log10(0.05). Here they can instead
     enter a Tobit likelihood as "below the detection limit", which is what they are.

Everything else is held identical to arm A on purpose: same folds, same early-stopping
protocol (allele-grouped inner split of TRAIN, stopping on median per-allele Spearman),
same metrics, same runner. So a difference in the headline is a difference from these
two changes and not from the harness.

    python arm_M_interact.py tune      # inner-split sweep, TRAIN only
    python arm_M_interact.py run NAME  # 21 groove folds x 5 seeds for one config
    python arm_M_interact.py smoke
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

import arm_M_pockets as PK
import data
import metrics
import run_experiment as R
import splits

AA = PK.AA
AA_IDX = PK.AA_IDX
PEP_LEN = 9
PSEUDO_LEN = 34

# A side of a fold with at least this many distinct alleles is a TRAINING side. The
# biggest held-out groove cluster has 9 alleles; the smallest training side has 66.
# Asserted at use, so the gap cannot close silently.
TRAIN_ALLELE_MIN = 20

# Bookkeeping columns appended to every X by the featurizers here, and stripped by
# every model here before anything is fitted. They are NOT features.
#   -2  allele id   : only used to build the allele-grouped early-stopping split,
#                     exactly as arm A does (arm A recovers it by random-projecting
#                     the pseudo-sequence block; doing it explicitly is the same
#                     information, just not reconstructed).
#   -1  censored    : 1.0 if that row sat at the Thalf == 0.0 detection floor. Only
#                     the Tobit likelihood reads it, and only for TRAIN rows.
N_BOOK = 2


# ---------------------------------------------------------------------------
# residue encodings
# ---------------------------------------------------------------------------

_B62_ORDER = "ARNDCQEGHILKMFPSTWYV"
_B62_ROWS = """
 4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
-1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
-2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
-2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
 0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
-1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
 0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
-2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
-1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
-1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
-1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
-2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
 1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
 0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
-2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
 0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""
BLOSUM_SCALE = 5.0


def _blosum62():
    m = np.array([[float(x) for x in r.split()] for r in _B62_ROWS.strip().splitlines()])
    assert m.shape == (20, 20) and (m == m.T).all(), "BLOSUM62 is not symmetric"
    p = [_B62_ORDER.index(a) for a in AA]
    m = m[np.ix_(p, p)]
    for a, v in [("W", 11), ("C", 9), ("H", 8), ("P", 7), ("Y", 7), ("G", 6)]:
        assert m[AA_IDX[a], AA_IDX[a]] == v, f"BLOSUM62 diagonal wrong at {a}"
    return m / BLOSUM_SCALE


B62 = _blosum62()
EYE = np.eye(20)
TABLES = {"blosum": [B62], "onehot": [EYE], "both": [B62, EYE]}


def residue_embedding(d):
    """20 x d residue vectors whose inner products approximate BLOSUM62.

    Eigendecomposition of the (symmetric) BLOSUM62 matrix, top-d components by |lambda|,
    each column standardised to unit variance so no component dominates the first layer.
    Fixed constant: it is a function of BLOSUM62 only, with no data fitted into it, so
    train and test are encoded identically and nothing can leak through it.

    d = 20 recovers a full-rank rotation of the 20-dim one-hot basis, which makes the
    outer-product block below an exact linear reparametrisation of the 400-dim one-hot
    pair indicator. So d is a smoothness knob, not a different kind of feature.
    """
    w, v = np.linalg.eigh(B62)
    order = np.argsort(-np.abs(w))[:d]
    E = v[:, order] * np.sqrt(np.abs(w[order]))
    E = E / (E.std(axis=0, keepdims=True) + 1e-12)
    return E


# ---------------------------------------------------------------------------
# feature configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Cfg:
    """One featurizer + model configuration."""
    # base block, identical to arm A
    base: str = "both"             # "both" | "blosum" | "onehot" | "none"
    # interaction block
    inter: str = "pca6"            # "none" | "pca{d}" | "sim" | "pca{d}+sim"
    pocket_src: str = "structure"  # "label" | "structure" | "all34"
    anchors: tuple = (1, 8)        # 0-indexed peptide positions; "derive2"/"derive3"/"all"
    k: int = 6                     # pseudo positions per pocket
    # model
    model: str = "mlp"             # "mlp" | "ridge" | "hgb" | "pwm"
    net: str = "mlp"               # "mlp" | "towers"  (model == "mlp" only)
    d_tower: int = 64
    loss: str = "mse"              # "mse" | "tobit"
    target: str = "raw"            # "raw" | "centred" | "standardised"
    hidden: tuple = (256, 128)
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 1e-5
    batch: int = 512
    max_epochs: int = 300
    patience: int = 25
    alpha: float = 1.0             # ridge
    hgb: dict = field(default_factory=dict)
    # Relabels the alleles before the early-stopping split picks some of them, so two
    # configs differing ONLY in split_seed draw genuinely different inner validation
    # sets. 0 is the default behaviour. Used to measure how much of the headline is
    # nuisance from that choice; never used to select anything.
    split_seed: int = 0

    def tag(self):
        a = self.anchors if isinstance(self.anchors, str) else \
            "".join(str(p + 1) for p in self.anchors)
        t = (f"{self.model}_{self.loss}_{self.target}_{self.base}_{self.inter}_"
             f"{self.pocket_src}{self.k}_P{a}")
        if self.model == "mlp":
            t += (f"_{self.net}{self.d_tower if self.net == 'towers' else ''}"
                  f"_h{'-'.join(map(str, self.hidden))}_do{self.dropout}"
                  f"_lr{self.lr:g}_wd{self.weight_decay:g}_b{self.batch}")
        elif self.model == "hgb" and self.hgb:
            t += "_" + "-".join(f"{k}{v:g}" for k, v in sorted(self.hgb.items()))
        elif self.model in ("ridge", "pwm"):
            t += f"_a{self.alpha:g}"
        if self.split_seed:
            t += f"_sp{self.split_seed}"
        return t


def _parse_inter(s):
    """'pca6+sim' -> (6, True); 'sim' -> (0, True); 'none' -> (0, False)."""
    d, sim = 0, False
    for part in s.split("+"):
        part = part.strip()
        if part.startswith("pca"):
            d = int(part[3:])
        elif part == "sim":
            sim = True
        elif part not in ("", "none"):
            raise ValueError(f"bad inter spec {s!r}")
    return d, sim


# ---------------------------------------------------------------------------
# the featurizer
# ---------------------------------------------------------------------------

class Featurizer:
    """featurize(df_subset) -> X, with the pocket spec derived from TRAIN only.

    run_experiment.run_arm calls this once with the fold's training rows and once with
    its test rows, train first. A side with >= TRAIN_ALLELE_MIN distinct alleles is the
    training side and (re)derives the pocket spec; a smaller side reuses the cached one
    and ASSERTS that none of its alleles were in the set the spec was derived from. So
    a held-out groove can never influence its own features. Both branches are asserted
    rather than assumed, and the assertion fires loudly inside run_arm's per-fold
    try/except rather than silently producing a leaked number.
    """

    def __init__(self, cfg: Cfg, all_alleles=None):
        self.cfg = cfg
        self.d_pca, self.use_sim = _parse_inter(cfg.inter)
        self.E = residue_embedding(self.d_pca) if self.d_pca else None
        self.tables = [] if cfg.base == "none" else TABLES[cfg.base]
        self.pm = PK.pseudo_map()
        self.allele_id = {a: i for i, a in enumerate(sorted(all_alleles or self.pm))}
        self._spec = None
        self._spec_key = None
        self._pep_cache, self._hla_cache = {}, {}
        self.n_derivations = 0

    # -- pocket spec ------------------------------------------------------
    def _spec_for(self, d):
        al = frozenset(d.HLA.unique())
        if len(al) >= TRAIN_ALLELE_MIN:
            if al != self._spec_key:
                self._spec_key = al
                self._spec = self._derive(d)
                self.n_derivations += 1
            return self._spec
        assert self._spec is not None, (
            "featurize() saw a small (test-looking) subset before any training side; "
            "the pocket spec has not been derived")
        overlap = al & self._spec.train_alleles
        assert not overlap, (
            f"LEAK GUARD: held-out alleles {sorted(overlap)} were present when the "
            f"pocket spec was derived")
        return self._spec

    def _derive(self, d):
        c = self.cfg
        anchors = c.anchors
        if anchors == "all":
            anchors = list(range(PEP_LEN))
        elif isinstance(anchors, str) and anchors.startswith("derive"):
            n = int(anchors[6:])
            anchors = sorted(np.argsort(-PK.anchor_variance(d))[:n].tolist())
        anchors = [int(p) for p in anchors]

        if c.pocket_src == "label":
            return PK.fit_pockets(d, anchors=anchors, k=c.k)
        pockets = (PK.structural_pockets(anchors, k=c.k) if c.pocket_src == "structure"
                   else {p: list(range(PSEUDO_LEN)) for p in anchors})
        return PK.PocketSpec(pockets=pockets, anchors=anchors,
                             n_train_alleles=int(d.HLA.nunique()),
                             train_alleles=frozenset(d.HLA.unique()))

    # -- encoding ---------------------------------------------------------
    def _pep_idx(self, peps):
        c = self._pep_cache
        for p in peps:
            if p not in c:
                c[p] = np.array([AA_IDX[ch] for ch in p], dtype=np.int64)
        return np.stack([c[p] for p in peps])

    def _hla_idx(self, hlas):
        c = self._hla_cache
        for a in hlas:
            if a not in c:
                c[a] = np.array([AA_IDX[ch] for ch in self.pm[a]], dtype=np.int64)
        return np.stack([c[a] for a in hlas])

    def _base_block(self, pi, hi):
        """[peptide block | pseudo-sequence block], each the concatenation over the
        encoding tables. Column order is arm A's exactly, so that with inter='none'
        this featurizer is byte-identical to arm A's and the control arm really is a
        replication rather than a lookalike."""
        if not self.tables:
            return np.zeros((len(pi), 0))
        pep = np.hstack([t[pi].reshape(len(pi), -1) for t in self.tables])
        hla = np.hstack([t[hi].reshape(len(hi), -1) for t in self.tables])
        return np.hstack([pep, hla])

    def _inter_block(self, pi, hi, spec):
        if not (self.d_pca or self.use_sim):
            return np.zeros((len(pi), 0))
        out = []
        for p in spec.anchors:
            a_idx = pi[:, p]
            for q in spec.pockets[p]:
                b_idx = hi[:, q]
                if self.d_pca:
                    u, v = self.E[a_idx], self.E[b_idx]
                    out.append((u[:, :, None] * v[:, None, :]).reshape(len(pi), -1))
                if self.use_sim:
                    out.append(B62[a_idx, b_idx][:, None])
        return np.hstack(out)

    def __call__(self, df_subset):
        spec = self._spec_for(df_subset)
        peps = df_subset.Pep.to_numpy()
        hlas = df_subset.HLA.to_numpy()
        pi, hi = self._pep_idx(peps), self._hla_idx(hlas)
        book = np.stack([
            np.array([self.allele_id[a] for a in hlas], dtype=np.float64),
            df_subset.censored.to_numpy(dtype=np.float64),
        ], axis=1)
        return np.hstack([self._base_block(pi, hi), self._inter_block(pi, hi, spec), book])

    # -- for the tree arm -------------------------------------------------
    def categorical(self, df_subset):
        """Integer-coded view: 9 peptide + 34 pseudo residues, then the interaction
        block, then the two bookkeeping columns. Column indices 0..42 are categorical."""
        spec = self._spec_for(df_subset)
        peps, hlas = df_subset.Pep.to_numpy(), df_subset.HLA.to_numpy()
        pi, hi = self._pep_idx(peps), self._hla_idx(hlas)
        book = np.stack([
            np.array([self.allele_id[a] for a in hlas], dtype=np.float64),
            df_subset.censored.to_numpy(dtype=np.float64),
        ], axis=1)
        return np.hstack([pi.astype(np.float64), hi.astype(np.float64),
                          self._inter_block(pi, hi, spec), book])


def width(cfg):
    """Feature width this config produces, for the log. Excludes bookkeeping."""
    d, sim = _parse_inter(cfg.inter)
    anchors = (9 if cfg.anchors == "all" else
               int(cfg.anchors[6:]) if isinstance(cfg.anchors, str) else len(cfg.anchors))
    k = min(cfg.k, PSEUDO_LEN)
    inter = anchors * k * (d * d + (1 if sim else 0))
    if cfg.model in ("hgb", "pwm"):             # integer-coded categorical view
        return PEP_LEN + PSEUDO_LEN + inter
    base = 0 if cfg.base == "none" else (PEP_LEN + PSEUDO_LEN) * 20 * len(TABLES[cfg.base])
    return base + inter


# ---------------------------------------------------------------------------
# models.  Every one strips the N_BOOK bookkeeping columns first.
# ---------------------------------------------------------------------------

def _strip(X):
    """(features, allele_id, censored_mask). Asserts the bookkeeping columns are sane."""
    X = np.asarray(X)
    book = X[:, -N_BOOK:]
    cens = book[:, 1]
    assert np.all((cens == 0) | (cens == 1)), "censored bookkeeping column is not binary"
    return X[:, :-N_BOOK], book[:, 0].astype(np.int64), cens > 0.5


def _device(name=None):
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MLP(nn.Module):
    def __init__(self, d_in, hidden, dropout):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# Censoring threshold in y = log10(Thalf) space. A censored row was recorded as
# Thalf == 0.0; the smallest value the assay ever reports is 0.1 h, so all we know is
# Thalf < 0.1, i.e. y < log10(0.1) = -1. data.py imputes log10(0.05) = -1.301 for the
# squared-error path; the Tobit likelihood never uses that imputed number, only the
# indicator and this threshold.
TOBIT_C = float(np.log10(0.1))

_HALF_LOG_2PI = 0.9189385332046727
_SQRT2 = 1.4142135623730951
_TAIL = -8.0


def log_ndtr(x):
    """log Phi(x), stable, using only ops MPS implements.

    torch.special.log_ndtr has no MPS kernel (PyTorch 2.14), and routing it to the CPU
    per minibatch would force a device sync inside the training loop. Two branches:
    log(0.5 erfc(-x/sqrt2)) where erfc is comfortably representable, and the standard
    asymptotic expansion below x = -8. Both branches are clamped before evaluation so
    neither produces a NaN that torch.where would then propagate into the gradient.
    Checked against scipy.special.log_ndtr in _check_log_ndtr().
    """
    xm = x.clamp(min=_TAIL)
    main = torch.log(0.5 * torch.erfc(-xm / _SQRT2))
    xt = x.clamp(max=_TAIL)
    iv = 1.0 / (xt * xt)
    series = torch.log1p(-iv * (1.0 - 3.0 * iv * (1.0 - 5.0 * iv)))
    tail = -0.5 * xt * xt - _HALF_LOG_2PI - torch.log(-xt) + series
    return torch.where(x >= _TAIL, main, tail)


def _check_log_ndtr(device=None):
    from scipy.special import log_ndtr as ref
    dev = _device(device)
    x = np.concatenate([np.linspace(-40, 10, 2001), [-8.0, -7.999, -8.001]])
    t = torch.tensor(x, dtype=torch.float32, device=dev).requires_grad_(True)
    got = log_ndtr(t)
    got.sum().backward()
    r = ref(x)
    v = got.detach().cpu().numpy()
    err = np.abs(v - r)
    big = np.abs(r) > 1e-3          # relative error is meaningless where log Phi ~ 0
    rel = (err[big] / np.abs(r[big])).max()
    g = t.grad.cpu().numpy()
    print(f"log_ndtr on {dev}: max abs err {err.max():.2e}  "
          f"max rel err (|log Phi|>1e-3) {rel:.2e}  "
          f"grad finite {bool(np.isfinite(g).all())}  value finite "
          f"{bool(np.isfinite(v).all())}")
    assert err.max() < 1e-3 and rel < 1e-4 and np.isfinite(g).all() and np.isfinite(v).all()
    return True


class TorchModel:
    """arm A's TorchMLP protocol, with a selectable loss.

    Identical early stopping to arm A so the comparison is about the loss and the
    features: hold out ~15% of TRAIN by whole alleles, stop on the median per-allele
    Spearman over the held-out alleles, restore the best epoch. The test fold is never
    touched. The validation criterion is a RANK metric either way, so the two losses
    are early-stopped on exactly the same quantity.

    loss='tobit': the network emits the latent mean mu and a single learned log sigma.
      uncensored  -> Gaussian NLL of y
      censored    -> -log Phi((c - mu) / sigma), i.e. "the true value is below c"
    Predictions are mu in both cases, which is the ranking-relevant quantity.
    """

    def __init__(self, seed=0, cfg: Cfg = None, device=None, blocks=None):
        self.seed = int(seed)
        self.cfg = cfg or Cfg()
        self.device = _device(device)
        self.blocks = blocks          # (n_pep, n_hla) for the two-tower net

    def _make_net(self, d_in):
        c = self.cfg
        if c.net == "mlp":
            return MLP(d_in, tuple(c.hidden), c.dropout)
        n_pep, n_hla = self.blocks
        assert n_pep + n_hla <= d_in, f"block widths {n_pep}+{n_hla} exceed {d_in}"
        return TowerNet(n_pep, n_hla, d_in - n_pep - n_hla, d=c.d_tower,
                        hidden=tuple(c.hidden), dropout=c.dropout)

    def _split(self, gid, rng):
        groups = np.unique(gid)
        if self.cfg.split_seed:
            groups = np.random.default_rng(90000 + self.cfg.split_seed).permutation(groups)
        order = rng.permutation(groups)
        k0 = min(max(1, int(round(0.15 * len(groups)))), len(groups) - 1)
        m = np.isin(gid, order[:k0])
        for k in range(k0 + 1, len(groups)):
            if m.sum() >= 300:
                break
            m = np.isin(gid, order[:k])
        return ~m, m

    def _loss(self, mu, y, cens, thr):
        if self.cfg.loss == "mse":
            return nn.functional.mse_loss(mu, y)
        s = torch.exp(self.log_sigma).clamp(1e-3, 10.0)
        z = (y - mu) / s
        nll_obs = 0.5 * z ** 2 + self.log_sigma
        nll_cen = -log_ndtr((thr - mu) / s)
        return torch.where(cens, nll_cen, nll_obs).mean()

    def fit(self, X, y):
        Xf, gid, cens = _strip(X)
        Xf = np.asarray(Xf, dtype=np.float32)
        # Within-allele centring is strictly monotone inside each allele, so the
        # early-stopping criterion (median per-allele Spearman) is unchanged by it.
        y, thr = retarget(y, gid, cens, self.cfg.target)
        y = y.astype(np.float32)
        rng = np.random.default_rng(1000 + self.seed)
        tr, va = self._split(gid, rng)
        dev, c = self.device, self.cfg

        Xtr = torch.from_numpy(Xf[tr]).to(dev)
        ytr = torch.from_numpy(y[tr]).to(dev)
        ctr = torch.from_numpy(cens[tr]).to(dev)
        ttr = torch.from_numpy(thr[tr].astype(np.float32)).to(dev)
        Xva = torch.from_numpy(Xf[va]).to(dev)
        yva_np = y[va].astype(np.float64)
        gva = gid[va]
        vgroups = [np.where(gva == u)[0] for u in np.unique(gva)
                   if (gva == u).sum() >= metrics.MIN_N]

        torch.manual_seed(self.seed)
        self.model = self._make_net(Xf.shape[1]).to(dev)
        params = list(self.model.parameters())
        if c.loss == "tobit":
            self.log_sigma = nn.Parameter(torch.tensor(np.log(0.5), dtype=torch.float32,
                                                       device=dev))
            params = params + [self.log_sigma]
        opt = torch.optim.Adam(params, lr=c.lr, weight_decay=c.weight_decay)
        gen = torch.Generator(device="cpu").manual_seed(self.seed)

        best, best_state, bad, n = np.inf, None, 0, len(Xtr)
        ep = 0
        for ep in range(c.max_epochs):
            self.model.train()
            perm = torch.randperm(n, generator=gen).to(dev)
            for i in range(0, n, c.batch):
                b = perm[i:i + c.batch]
                opt.zero_grad(set_to_none=True)
                self._loss(self.model(Xtr[b]), ytr[b], ctr[b], ttr[b]).backward()
                opt.step()
            self.model.eval()
            with torch.no_grad():
                pv = self.model(Xva).cpu().numpy().astype(np.float64)
            r = [stats.spearmanr(pv[i], yva_np[i]).statistic for i in vgroups]
            r = [x for x in r if np.isfinite(x)]
            v = -float(np.median(r)) if r else np.inf
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = {k: t.detach().clone()
                              for k, t in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= c.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.val_score_, self.epochs_run_ = best, ep + 1
        return self

    def predict(self, X):
        Xf, _, _ = _strip(X)
        self.model.eval()
        t = torch.from_numpy(np.asarray(Xf, dtype=np.float32)).to(self.device)
        out = []
        with torch.no_grad():
            for i in range(0, len(t), 4096):
                out.append(self.model(t[i:i + 4096]).cpu().numpy())
        return np.concatenate(out).astype(np.float64)


class TowerNet(nn.Module):
    """Arm A's MLP with a LEARNED rank-d bilinear peptide x groove term bolted on.

        z_p = A x_pep,  z_a = B x_pseudo,  both in R^d
        the head sees [ the whole original feature vector , LayerNorm(z_p * z_a) ]

    sum_k (a_k . x_pep)(b_k . x_pseudo) is a rank-d bilinear form in the two blocks, so
    this is the learned counterpart of the fixed outer-product features: nothing tells
    it which groove residue pairs with which anchor, it has to find d directions that
    do. The towers are linear on purpose, so the term is a genuine bilinear form rather
    than another opaque MLP branch, and the head keeps the raw features so the model is
    a strict superset of arm A's rather than a different architecture competing with it.

    z_a is a function of the pseudo-sequence, never an allele lookup table, so it is
    defined for a groove cluster the model has never seen. An allele-embedding table
    would not be, which is why idea 4 is done this way round.
    """

    def __init__(self, n_pep, n_hla, n_extra, d=64, hidden=(256, 128), dropout=0.1):
        super().__init__()
        self.n_pep, self.n_hla = n_pep, n_hla
        self.tp = nn.Linear(n_pep, d, bias=False)
        self.ta = nn.Linear(n_hla, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.head = MLP(n_pep + n_hla + n_extra + d, tuple(hidden), dropout)

    def forward(self, x):
        p = x[:, :self.n_pep]
        a = x[:, self.n_pep:self.n_pep + self.n_hla]
        return self.head(torch.cat([x, self.norm(self.tp(p) * self.ta(a))], dim=1))


class RidgeModel:
    """Linear reference on the same features. Deterministic: every seed is identical,
    so its across-seed sigma is 0 and run_experiment blanks the calibration columns."""

    def __init__(self, seed=0, cfg: Cfg = None):
        from sklearn.linear_model import Ridge
        self.cfg = cfg or Cfg()
        self.m = Ridge(alpha=self.cfg.alpha)

    def fit(self, X, y):
        Xf, gid, cens = _strip(X)
        self.m.fit(Xf, retarget(y, gid, cens, self.cfg.target)[0])
        return self

    def predict(self, X):
        return self.m.predict(_strip(X)[0])


class HGBModel:
    """HistGradientBoostingRegressor on the integer-coded view.

    Columns 0..42 are the 9 peptide and 34 pseudo-sequence residues as categories;
    anything after them is the float interaction block. Trees cannot take the Tobit
    likelihood, so censored rows keep data.py's imputed floor; that difference is
    stated rather than hidden, and is one reason the tree arm is reported separately.
    """

    N_CAT = PEP_LEN + PSEUDO_LEN

    def __init__(self, seed=0, cfg: Cfg = None):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.cfg = cfg or Cfg()
        kw = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                  min_samples_leaf=40, l2_regularization=1.0,
                  early_stopping=True, validation_fraction=0.15, n_iter_no_change=25)
        kw.update(self.cfg.hgb)
        self.m = HistGradientBoostingRegressor(random_state=seed, **kw)

    def fit(self, X, y):
        Xf, gid, cens = _strip(X)
        cat = np.zeros(Xf.shape[1], dtype=bool)
        cat[:self.N_CAT] = True
        self.m.set_params(categorical_features=cat)
        self.m.fit(Xf, retarget(y, gid, cens, self.cfg.target)[0])
        return self

    def predict(self, X):
        return self.m.predict(_strip(X)[0])


class PWMModel:
    """Predict the unseen groove's 9x20 scoring matrix from its 34 residues.

    A different model class entirely, and a much smaller one. For every TRAINING allele
    estimate the empirical allele-centred preference profile Pc[a] (9 x 20, shrunk
    towards zero by the cell count), then ridge-regress those 180 numbers on the
    allele's one-hot 34-residue pseudo-sequence. Only 66 training examples, but 680
    inputs and a very structured output, so the fit is essentially "blend the profiles
    of the grooves this one resembles". A test peptide is scored by summing its
    predicted PWM entries.

    It cannot represent anything about the peptide beyond an additive position model,
    which is exactly why it is worth having next to the MLP: when two models that wrong
    in different ways are rank-averaged, the average is usually better than either.
    Consumes the integer-coded (categorical) feature view.
    """

    N_CAT = PEP_LEN + PSEUDO_LEN

    def __init__(self, seed=0, cfg: Cfg = None):
        self.cfg = cfg or Cfg()
        self.alpha = self.cfg.alpha
        self.shrink = 5.0          # cells with this many rows get half weight

    def fit(self, X, y):
        Xf, gid, cens = _strip(X)
        y, _ = retarget(y, gid, cens, self.cfg.target)
        pep = Xf[:, :PEP_LEN].astype(np.int64)
        pse = Xf[:, PEP_LEN:self.N_CAT].astype(np.int64)
        alleles, inv = np.unique(gid, return_inverse=True)
        na = len(alleles)

        # allele main effect out, then the per-(allele, position, residue) mean
        n = np.bincount(inv, minlength=na).astype(float)
        mu = np.bincount(inv, weights=y, minlength=na) / np.maximum(n, 1)
        r = y - mu[inv]
        P = np.zeros((na, PEP_LEN, 20))
        for p in range(PEP_LEN):
            flat = inv * 20 + pep[:, p]
            cnt = np.bincount(flat, minlength=na * 20).reshape(na, 20)
            s = np.bincount(flat, weights=r, minlength=na * 20).reshape(na, 20)
            # shrink towards 0 by count: a cell seen twice should not swing the fit
            P[:, p, :] = s / (cnt + self.shrink)

        # one pseudo-sequence row per allele (constant within an allele by construction)
        A = np.zeros((na, PSEUDO_LEN * 20))
        first = np.zeros(na, dtype=np.int64)
        for i in range(na):
            first[i] = np.argmax(inv == i)
        for j in range(PSEUDO_LEN):
            A[np.arange(na), j * 20 + pse[first, j]] = 1.0

        from sklearn.linear_model import Ridge
        self.ridge = Ridge(alpha=self.alpha).fit(A, P.reshape(na, -1))
        return self

    def predict(self, X):
        Xf, _, _ = _strip(X)
        pep = Xf[:, :PEP_LEN].astype(np.int64)
        pse = Xf[:, PEP_LEN:self.N_CAT].astype(np.int64)
        n = len(Xf)
        A = np.zeros((n, PSEUDO_LEN * 20))
        rows = np.arange(n)
        for j in range(PSEUDO_LEN):
            A[rows, j * 20 + pse[:, j]] = 1.0
        pwm = self.ridge.predict(A).reshape(n, PEP_LEN, 20)
        return pwm[rows[:, None], np.arange(PEP_LEN)[None, :], pep].sum(axis=1)


MIN_SD = 0.15


def retarget(y, gid, cens, mode):
    """(y', c') -- the training target and the per-row Tobit threshold, after an
    optional within-allele transform.

    The headline metric is a per-allele rank correlation, so a constant offset or
    scale INSIDE one allele cannot change it. Plain squared error nevertheless spends
    most of its capacity on the between-allele spread of mean half-life, which is both
    the largest part of the variance and the part the metric throws away. Centring the
    target within each training allele removes that part and leaves the model fitting
    only the within-allele ordering that is actually scored. Standardising removes the
    per-allele scale too.

    Both use TRAIN labels only, through the allele id that _strip() recovers from the
    bookkeeping column. Nothing allele-specific is needed at prediction time: the model
    predicts on the transformed scale and the metric only ranks within an allele, so no
    unseen allele's mean or spread ever has to be known.
    """
    y = np.asarray(y, dtype=np.float64)
    c = np.full(len(y), TOBIT_C)
    if mode == "raw":
        return y, c
    n = np.bincount(gid, minlength=gid.max() + 1).astype(float)
    s = np.bincount(gid, weights=y, minlength=len(n))
    mu = np.divide(s, n, out=np.zeros_like(s), where=n > 0)
    out = y - mu[gid]
    c = c - mu[gid]
    if mode == "standardised":
        s2 = np.bincount(gid, weights=out ** 2, minlength=len(n))
        sd = np.sqrt(np.divide(s2, np.maximum(n - 1, 1), out=np.zeros_like(s2),
                               where=n > 1))
        sd = np.clip(np.where(n > 1, sd, 1.0), MIN_SD, None)
        out = out / sd[gid]
        c = c / sd[gid]
    elif mode != "centred":
        raise ValueError(f"bad target mode {mode!r}")
    return out, c


MODELS = {"mlp": TorchModel, "ridge": RidgeModel, "hgb": HGBModel,
          "pwm": PWMModel}


def build(cfg: Cfg, df=None, device=None):
    """(featurize, make_model) for run_experiment.run_arm."""
    df = df if df is not None else R.load_df()
    f = Featurizer(cfg, all_alleles=sorted(df.HLA.unique()))
    featurize = f.categorical if cfg.model in ("hgb", "pwm") else f.__call__
    if cfg.model == "mlp":
        nt = len(f.tables)
        blocks = (PEP_LEN * 20 * nt, PSEUDO_LEN * 20 * nt)
        if cfg.net == "towers" and cfg.base == "none":
            raise ValueError("the two-tower net needs the base block to split on")

        def make_model(seed):
            return TorchModel(seed=seed, cfg=cfg, device=device, blocks=blocks)
    else:
        cls = MODELS[cfg.model]

        def make_model(seed):
            return cls(seed=seed, cfg=cfg)
    return featurize, make_model, f


# ---------------------------------------------------------------------------
# inner-split tuning.  TRAIN only, never a test fold.
# ---------------------------------------------------------------------------

def inner_eval(cfg: Cfg, n_outer=3, seeds=(0, 1), device=None, verbose=True, df=None):
    """Mean inner-validation median per-allele Spearman for one config.

    Mirrors arm A's tune(): for each of the n_outer largest outer folds, take its TRAIN
    side only, hold out one further groove cluster from inside it, fit, and score on
    that inner cluster. The outer test rows are never loaded, so nothing here can
    inform the headline through the back door.
    """
    df = df if df is not None else R.load_df()
    outer = splits.choose_held_out(df)
    scores, details = [], []
    for oi in range(n_outer):
        o_name, o_alleles = outer[oi]
        tr_idx, _ = splits.split_by_allele(df, o_alleles)
        train = df.loc[tr_idx]
        elig = [(n, a) for n, a in outer if n != o_name and train.HLA.isin(a).sum() >= 300]
        i_name, i_alleles = elig[oi % len(elig)]
        i_tr, i_te = splits.split_by_allele(train, i_alleles)
        fit_df, val_df = train.loc[i_tr], train.loc[i_te]

        featurize, make_model, f = build(cfg, df=df, device=device)
        Xf, Xv = featurize(fit_df), featurize(val_df)
        yf = fit_df.y.to_numpy()
        rs = []
        for s in seeds:
            R.set_seed(s)
            m = make_model(s).fit(Xf, yf)
            p = m.predict(Xv)
            rs.append(float(np.median(metrics.spearman_per_allele(val_df, p))))
        scores.append(float(np.mean(rs)))
        details.append({"outer": o_name, "inner": i_name, "rho": scores[-1],
                        "seeds": rs, "dim": Xf.shape[1] - N_BOOK})
        if verbose:
            print(f"    outer {o_name:<18} inner {i_name:<18} "
                  f"rho={scores[-1]:+.4f} ({' '.join(f'{r:+.3f}' for r in rs)})",
                  flush=True)
    return float(np.mean(scores)), details


def sweep(cands, n_outer=3, seeds=(0, 1), device=None, out="results_M_tuning.csv"):
    rows = []
    for label, cfg in cands:
        t0 = time.time()
        print(f"\n[{label}]  {cfg.tag()}  (nominal width {width(cfg)})", flush=True)
        try:
            mean, det = inner_eval(cfg, n_outer=n_outer, seeds=seeds, device=device)
            ok, err = True, ""
        except Exception as e:
            import traceback
            traceback.print_exc()
            mean, det, ok, err = np.nan, [], False, f"{type(e).__name__}: {e}"
        print(f"  => MEAN inner rho {mean:+.4f}   ({time.time()-t0:.0f}s)"
              + ("" if ok else f"  FAILED {err}"), flush=True)
        rows.append({"label": label, "tag": cfg.tag(), "inner_rho": mean,
                     "width": width(cfg), "seconds": round(time.time() - t0, 1),
                     "n_outer": n_outer, "n_seeds": len(seeds), "error": err,
                     "detail": json.dumps(det)})
        pd.DataFrame(rows).to_csv(out, index=False)
    t = pd.DataFrame(rows).sort_values("inner_rho", ascending=False)
    print("\n=== inner-validation ranking (TRAIN only) ===")
    print(t[["label", "inner_rho", "width", "seconds"]].to_string(index=False))
    return t


# ---------------------------------------------------------------------------
# the real run
# ---------------------------------------------------------------------------

def run(name, cfg: Cfg, seeds=5, device=None, censored="tied"):
    featurize, make_model, f = build(cfg, device=device)
    t0 = time.time()
    res = R.run_arm(f"M_{name}", featurize, make_model, seeds=seeds, censored=censored)
    print(f"[M_{name}] {cfg.tag()}   pocket derivations: {f.n_derivations}  "
          f"wall {time.time()-t0:.0f}s", flush=True)
    if f._spec is not None:
        print(f"[M_{name}] last pocket spec: {f._spec.describe()}", flush=True)
    return res


def _smoke(device=None):
    df = R.load_df()
    folds = splits.choose_held_out(df)[:2]
    for label, cfg in [
        ("arm-A replication (base only, MSE)", Cfg(inter="none", model="mlp")),
        ("base + structural pockets, pca6",    Cfg(inter="pca6", pocket_src="structure")),
        ("base + pca6 + tobit",                Cfg(inter="pca6", loss="tobit")),
        ("ridge on base + pca6",               Cfg(inter="pca6", model="ridge")),
        ("hgb categorical + pca6",             Cfg(inter="pca6", model="hgb")),
    ]:
        featurize, make_model, f = build(cfg, device=device)
        print(f"\n### {label}  [{cfg.tag()}]  width~{width(cfg)}")
        R.run_arm(f"M_smoke", featurize, make_model, seeds=1, folds=folds, censored="tied")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if cmd == "smoke":
        _smoke()
    else:
        raise SystemExit(f"unknown command {cmd!r}")
