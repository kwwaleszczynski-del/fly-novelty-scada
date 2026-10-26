"""Winner-take-all tags and the fly Bloom filter (Dasgupta, Sheehan, Stevens, Navlakha 2018, PNAS 115:13093).

Tag: the ``k`` Kenyon cells with the largest summed input (feedback inhibition through the APL neuron leaves about
5 % of KCs active). Filter: one weight per KC, the KC -> MBON-alpha'3 synapse, initialised to 1. Storing an item
follows Eq. 2 of the paper:

    w_i <- delta * w_i          for the k KCs in the item's tag
    w_i <- min(1, w_i + eps)    for every other KC

and the novelty of an item is the mean weight over its tag, in [0, 1]: 1 when nothing like it was stored, near 0
when it is familiar. With ``eps = 0`` the order of storage does not matter and ``w_i = delta ** n_i``, ``n_i`` being
how often KC ``i`` was part of a stored tag (``delta = 0`` is the binary Bloom filter). That is how the training
year is stored: on 10-minute data ``n_i`` runs into the thousands for common states, so their weights are
vanishingly small and the novelty of a row is driven by the rarely used KCs in its tag. ``eps`` is the paper's
recovery, which makes the filter time sensitive (forgetting); it is switched on for streaming operation after the
training year. Switched on already while storing the training year it makes the filter forget the start of that
year, which costs more false alarms than it gains (development test in :mod:`inject`). Recovery is applied lazily,
each KC remembering when it was last touched, so storing one item costs O(k) instead of O(m).

Usage:
    det = FlyNovelty(wiring, expansion="flywire", seed=0).fit(X_train)
    novelty = det.score(X)                               # frozen filter, one value per row
    value = det.score_one(x, store=True)                 # one SCADA sample: score, then remember it (with recovery)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .connectome import Wiring
from .encoding import PopulationEncoder
from .expansion import Expansion, build

K_FRACTION = 0.05  # share of active KCs (Dasgupta et al.: about 5 %)
DELTA = 0.5        # decay of an active KC's output synapse per stored item
EPS = 1e-5         # streaming recovery per stored row: a habituated KC is new again after 100,000 rows (2-3 years)
WINDOW = 36        # trailing 6 h of 10-minute rows
MIN_PERIODS = 12
CHUNK = 8192


def wta(drive: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` largest entries of every row, ordered from the largest down (rows, k) int32."""
    if not 0 < k <= drive.shape[1]:
        raise ValueError("k must be between 1 and the number of KCs")
    top = np.argpartition(drive, drive.shape[1] - k, axis=1)[:, -k:]
    order = np.argsort(-np.take_along_axis(drive, top, axis=1), axis=1, kind="stable")
    return np.take_along_axis(top, order, axis=1).astype(np.int32)


def tags(expansion: Expansion, code: np.ndarray, k: int, chunk: int = CHUNK) -> tuple[np.ndarray, np.ndarray]:
    """WTA tags of every row of a channel code and the rows that carry any input at all."""
    out = np.empty((len(code), k), dtype=np.int32)
    for start in range(0, len(code), chunk):
        out[start:start + chunk] = wta(expansion.drive(code[start:start + chunk]), k)
    return out, code.sum(axis=1) > 0


class FlyBloomFilter:
    """KC -> MBON weights with multiplicative decay on storage and lazy additive recovery."""

    def __init__(self, n_kc: int, delta: float = DELTA, eps: float = 0.0):
        if not 0.0 <= delta < 1.0 or eps < 0.0:
            raise ValueError("need 0 <= delta < 1 and eps >= 0")
        self.delta, self.eps = delta, eps
        self.w = np.ones(n_kc)
        self.last = np.zeros(n_kc, dtype=np.int64)
        self.t = 0  # number of stored items

    def current(self, idx=slice(None)) -> np.ndarray:
        return np.minimum(1.0, self.w[idx] + self.eps * (self.t - self.last[idx]))

    def novelty(self, tag: np.ndarray) -> float:
        return float(self.current(tag).mean())

    def novelty_batch(self, tag_rows: np.ndarray) -> np.ndarray:
        """Novelty of many tags against the current weights (nothing is stored)."""
        return self.current()[tag_rows].mean(axis=1)

    def store(self, tag: np.ndarray) -> None:
        self.t += 1
        before = np.minimum(1.0, self.w[tag] + self.eps * (self.t - 1 - self.last[tag]))
        self.w[tag] = self.delta * before
        self.last[tag] = self.t

    def store_counts(self, counts: np.ndarray) -> None:
        """Store many items at once from per-KC tag counts (order-free, so only exact for ``eps = 0``)."""
        if self.eps > 0:
            raise ValueError("batch storage ignores recovery; set eps = 0 while storing in batch")
        self.w = self.current() * self.delta ** counts
        self.last[:] = self.t

    def nbytes(self) -> int:
        return self.w.nbytes + self.last.nbytes


def fitted_filter(n_kc: int, fit_tags: np.ndarray, delta: float = DELTA, eps: float = 0.0) -> FlyBloomFilter:
    """A filter that has stored the training tags in time order (all at once when there is no recovery)."""
    filt = FlyBloomFilter(n_kc, delta, eps)
    if eps == 0:
        filt.store_counts(np.bincount(fit_tags.ravel(), minlength=n_kc))
    else:
        for tag in fit_tags:
            filt.store(tag)
    return filt


def trailing_median(score: np.ndarray, usable: np.ndarray, window: int = WINDOW,
                    min_periods: int = MIN_PERIODS) -> np.ndarray:
    """Median of the usable scores in the trailing window (rows), NaN when fewer than ``min_periods``."""
    s = pd.Series(np.where(usable, score, np.nan))
    return s.rolling(window, min_periods=min_periods).median().to_numpy()


def stream(filt: FlyBloomFilter, tag_rows: np.ndarray, valid: np.ndarray, usable: np.ndarray, learn: np.ndarray,
           threshold: float | np.ndarray | None = None, gate: bool = True, force: np.ndarray | None = None,
           window: int = WINDOW, min_periods: int = MIN_PERIODS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay rows one at a time: score against the filter, smooth online, decide the alarm, then store the row.

    A row is stored when ``learn`` (normal operation) and it has input, unless ``gate`` is on and the smoothed
    score is above ``threshold`` (a number or one value per row; no learning while in alarm); ``force`` rows are
    stored regardless (an operator confirming a new normal after service). Returns the raw score, the smoothed
    score and the stored-row mask; the smoothed score equals :func:`trailing_median` of the raw score."""
    n = len(tag_rows)
    raw = np.full(n, np.nan)
    smooth = np.full(n, np.nan)
    stored = np.zeros(n, dtype=bool)
    buf = np.full(window, np.nan)
    thr = None if threshold is None else np.broadcast_to(np.asarray(threshold, dtype=float), (n,))
    for i in range(n):
        if valid[i]:
            raw[i] = filt.novelty(tag_rows[i])
        buf[i % window] = raw[i] if usable[i] else np.nan
        finite = buf[np.isfinite(buf)]
        if len(finite) >= min_periods:
            smooth[i] = np.median(finite)
        alarm = thr is not None and smooth[i] > thr[i]
        if valid[i] and ((learn[i] and not (gate and alarm)) or (force is not None and force[i])):
            filt.store(tag_rows[i])
            stored[i] = True
    return raw, smooth, stored


class FlyNovelty:
    """Population code -> PN -> KC expansion -> WTA tag -> fly Bloom filter, fitted on normal operation."""

    def __init__(self, wiring: Wiring, expansion: str = "flywire", seed: int = 0, n_bins: int = 4,
                 k_fraction: float = K_FRACTION, delta: float = DELTA, eps: float = EPS):
        self.wiring, self.kind, self.seed = wiring, expansion, seed
        self.encoder = PopulationEncoder(n_bins)
        self.k = max(1, int(round(k_fraction * wiring.n_kc)))
        self.filter = FlyBloomFilter(wiring.n_kc, delta, eps)
        self.expansion: Expansion | None = None

    def tags(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return tags(self.expansion, self.encoder.transform(X), self.k)

    def fit(self, X: np.ndarray) -> "FlyNovelty":
        self.encoder.fit(X)
        self.expansion = build(self.kind, self.wiring, self.encoder.n_channels, self.seed)
        code = self.encoder.transform(X)
        self.expansion.fit_centre(code)
        t, valid = tags(self.expansion, code, self.k)
        eps = self.filter.eps
        self.filter = fitted_filter(self.wiring.n_kc, t[valid], self.filter.delta, 0.0)
        self.filter.eps = eps  # recovery only from here on, for streaming use
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        t, valid = self.tags(X)
        out = self.filter.novelty_batch(t)
        out[~valid] = np.nan
        return out

    def score_one(self, x: np.ndarray, store: bool = False) -> float:
        """Novelty of one SCADA sample (the streaming path); with ``store`` the sample is remembered afterwards."""
        code = self.encoder.transform(np.asarray(x, dtype=float)[None, :])
        if not code.any():
            return float("nan")
        tag = wta(self.expansion.drive(code), self.k)[0]
        value = self.filter.novelty(tag)
        if store:
            self.filter.store(tag)
        return value

    def nbytes(self) -> int:
        return self.encoder.nbytes() + self.expansion.nbytes() + self.filter.nbytes()
