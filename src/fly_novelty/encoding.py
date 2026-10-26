"""From a SCADA state vector to projection neuron (PN) input: per-signal Gaussianisation and a population code.

The fly's antennal lobe hands the mushroom body a normalised pattern: the mean PN firing rate is nearly the same
for every odour and concentration (Olsen et al. 2010). A plain standardised SCADA vector is a poor stand-in,
because the winner-take-all step that follows only sees the direction of its input, not its size: a gearbox
bearing 2 sigma or 8 sigma too hot would give the same tag. Each signal is therefore coded by a small bank of
overlapping tuning curves (a population code, the way sensory neurons cover a continuous quantity):

1. every signal is mapped to a standard normal score through its empirical quantiles on the training rows, so
   skewed or bimodal signals (power, pitch) use the code evenly; values outside the training range land on the
   outermost bin;
2. the score ``z`` activates ``n_bins`` channels with Gaussian tuning curves at evenly spaced centres, normalised
   to sum to 1 (a soft one-hot code with open-ended outer bins). A missing value leaves its channels silent.

Every row therefore carries the same total input (one unit per available signal), and how extreme a value is
shows up as which channel is active, so a value far in the tail activates a rarely used channel.

Usage:
    enc = PopulationEncoder(n_bins=4).fit(X_train)   # X: (rows, signals), NaN allowed
    code = enc.transform(X)                          # (rows, signals * n_bins), float32
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri

PROBS = np.linspace(0.001, 0.999, 201)
CENTRE_MAX = 2.25  # outermost tuning-curve centre in normal scores; with 4 bins: -2.25, -0.75, 0.75, 2.25


class PopulationEncoder:
    """Quantile Gaussianisation per signal followed by a normalised Gaussian population code."""

    def __init__(self, n_bins: int = 4, centre_max: float = CENTRE_MAX):
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        self.n_bins = n_bins
        self.centres = np.linspace(-centre_max, centre_max, n_bins)
        self.width = (self.centres[1] - self.centres[0]) / 2

    def fit(self, X: np.ndarray) -> "PopulationEncoder":
        X = np.asarray(X, dtype=float)
        self.quantiles_ = []
        for j in range(X.shape[1]):
            col = X[:, j]
            col = col[np.isfinite(col)]
            if len(col) == 0:
                raise ValueError(f"signal {j} has no finite training values")
            q = np.quantile(col, PROBS)
            # tied quantiles (discrete or clipped signals): one knot per value at the mean probability
            values, inverse = np.unique(q, return_inverse=True)
            probs = np.bincount(inverse, weights=PROBS) / np.bincount(inverse)
            self.quantiles_.append((values, probs))
        return self

    @property
    def n_signals(self) -> int:
        return len(self.quantiles_)

    @property
    def n_channels(self) -> int:
        return self.n_signals * self.n_bins

    def gaussianise(self, X: np.ndarray) -> np.ndarray:
        """Normal scores in [ndtri(0.001), ndtri(0.999)]; NaN stays NaN."""
        X = np.asarray(X, dtype=float)
        z = np.full(X.shape, np.nan)
        for j, (values, probs) in enumerate(self.quantiles_):
            col = X[:, j]
            ok = np.isfinite(col)
            if len(values) == 1:  # constant in training: anything else is off-scale
                p = np.where(col[ok] > values[0], PROBS[-1], np.where(col[ok] < values[0], PROBS[0], 0.5))
            else:
                p = np.interp(col[ok], values, probs)
            z[ok, j] = ndtri(p)
        return z

    def code(self, z: np.ndarray) -> np.ndarray:
        """Soft one-hot population code of normal scores, (rows, signals * n_bins)."""
        finite = np.isfinite(z)
        d = (np.where(finite, z, 0.0)[:, :, None] - self.centres[None, None, :]) / self.width
        logits = -0.5 * d * d
        logits -= logits.max(axis=2, keepdims=True)
        act = np.exp(logits)
        act /= act.sum(axis=2, keepdims=True)
        act[~finite] = 0.0
        return act.reshape(len(z), -1).astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.code(self.gaussianise(X))

    def nbytes(self) -> int:
        return sum(v.nbytes + p.nbytes for v, p in self.quantiles_)
