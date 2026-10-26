"""Standard unsupervised anomaly scores on the same SCADA state vector, for comparison with the fly detector.

All take a (rows, signals) array with NaN allowed, are fitted on normal operation only and return one score per
row (higher = more anomalous). Signals are standardised with the training mean and standard deviation and missing
values set to 0 (the training mean), except for the robust z-score.

- ``zscore``: largest absolute robust z (median, MAD) over signals; univariate, no interactions;
- ``pca``: squared reconstruction error outside the principal components that explain 95 % of the variance (SPE);
- ``iforest``: isolation forest (100 trees of 256 samples), negated ``score_samples``;
- ``autoencoder``: a 3-hidden-layer MLP trained to reproduce its input through a bottleneck, mean squared error;
- ``knn``: mean distance to the 5 nearest of up to 10,000 stored training rows; the exact version of what the fly
  Bloom filter approximates (Eq. 1 of Dasgupta et al. 2018: distance to the closest stored item).

Usage:
    model = make("iforest", seed=0).fit(X_train)
    score = model.score(X)
"""
from __future__ import annotations

import pickle
import warnings

import numpy as np

BASELINES = ("zscore", "pca", "iforest", "autoencoder", "knn")
MAX_FIT_ROWS = 20_000
KNN_MEMORY = 10_000
KNN_K = 5
CHUNK = 8192


class _Standardised:
    def _fit_scale(self, X: np.ndarray) -> np.ndarray:
        self.mean_ = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        self.sd_ = np.where(sd > 0, sd, 1.0)
        return self._scale(X)

    def _scale(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, dtype=float) - self.mean_) / self.sd_
        return np.where(np.isfinite(Z), Z, 0.0)

    def nbytes(self) -> int:
        return len(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))


def _subsample(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return X if len(X) <= n else X[np.sort(rng.choice(len(X), n, replace=False))]


class ZScore(_Standardised):
    def fit(self, X):
        self.med_ = np.nanmedian(X, axis=0)
        mad = 1.4826 * np.nanmedian(np.abs(X - self.med_), axis=0)
        sd = np.nanstd(X, axis=0)
        self.spread_ = np.where(mad > 0, mad, np.where(sd > 0, sd, 1.0))
        return self

    def score(self, X):
        z = np.abs((np.asarray(X, dtype=float) - self.med_) / self.spread_)
        z = np.where(np.isfinite(z), z, -np.inf)
        out = z.max(axis=1)
        return np.where(np.isfinite(out), out, np.nan)


class PCAModel(_Standardised):
    def __init__(self, variance: float = 0.95):
        self.variance = variance

    def fit(self, X):
        Z = self._fit_scale(X)
        _, s, vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
        explained = np.cumsum(s ** 2) / np.sum(s ** 2)
        n = int(np.searchsorted(explained, self.variance) + 1)
        self.centre_ = Z.mean(axis=0)
        self.components_ = vt[:n]
        return self

    def residuals(self, X):
        """Per-signal part of the standardised state outside the principal subspace, (rows, signals)."""
        Z = self._scale(X) - self.centre_
        return Z - (Z @ self.components_.T) @ self.components_

    def score(self, X):
        return np.mean(self.residuals(X) ** 2, axis=1)


class IForest(_Standardised):
    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X):
        from sklearn.ensemble import IsolationForest

        Z = self._fit_scale(X)
        rng = np.random.default_rng(self.seed)
        self.model_ = IsolationForest(n_estimators=100, max_samples=256, random_state=self.seed, n_jobs=1)
        self.model_.fit(_subsample(Z, MAX_FIT_ROWS * 5, rng))
        return self

    def score(self, X):
        return -self.model_.score_samples(self._scale(X))


class AutoEncoder(_Standardised):
    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X):
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.neural_network import MLPRegressor

        Z = self._fit_scale(X)
        Z = _subsample(Z, MAX_FIT_ROWS, np.random.default_rng(self.seed))
        f = Z.shape[1]
        hidden = max(8, f // 2)
        bottleneck = max(3, f // 8)
        self.model_ = MLPRegressor(hidden_layer_sizes=(hidden, bottleneck, hidden), activation="tanh",
                                   alpha=1e-4, batch_size=256, learning_rate_init=1e-3, max_iter=60,
                                   early_stopping=True, n_iter_no_change=5, random_state=self.seed)
        with warnings.catch_warnings():  # a fixed budget of 60 epochs, convergence is not required
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model_.fit(Z, Z)
        return self

    def score(self, X):
        Z = self._scale(X)
        out = np.empty(len(Z))
        for start in range(0, len(Z), CHUNK):
            part = Z[start:start + CHUNK]
            out[start:start + CHUNK] = np.mean((self.model_.predict(part) - part) ** 2, axis=1)
        return out


class KNN(_Standardised):
    def __init__(self, seed: int = 0, memory: int = KNN_MEMORY, k: int = KNN_K):
        self.seed, self.memory, self.k = seed, memory, k

    def fit(self, X):
        from sklearn.neighbors import NearestNeighbors

        Z = self._fit_scale(X)
        self.model_ = NearestNeighbors(n_neighbors=self.k, algorithm="brute")
        self.model_.fit(_subsample(Z, self.memory, np.random.default_rng(self.seed)))
        return self

    def score(self, X):
        Z = self._scale(X)
        out = np.empty(len(Z))
        for start in range(0, len(Z), CHUNK):
            dist, _ = self.model_.kneighbors(Z[start:start + CHUNK])
            out[start:start + CHUNK] = dist.mean(axis=1)
        return out


def make(name: str, seed: int = 0):
    if name == "zscore":
        return ZScore()
    if name == "pca":
        return PCAModel()
    if name == "iforest":
        return IForest(seed)
    if name == "autoencoder":
        return AutoEncoder(seed)
    if name == "knn":
        return KNN(seed)
    raise ValueError(f"unknown baseline {name!r}, choose from {BASELINES}")
