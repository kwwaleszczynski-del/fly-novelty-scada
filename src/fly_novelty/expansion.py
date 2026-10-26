"""Expansion from projection neurons (PNs) to Kenyon cells (KCs): the real FlyWire wiring and random controls.

Every variant sees the same input. The population-code channels (signal x tuning curve) are first assigned to the
149 PNs by a seeded, balanced random map: with more channels than PNs each PN sums a few channels (a broadly tuned
glomerulus), with fewer each channel drives several sister PNs. The same seed gives the same map for every
variant, so the variants differ only in the PN -> KC matrix:

- ``flywire``: the real synapse counts of the right mushroom body (149 PNs -> 2,364 KCs, 10,505 connections);
- ``shuffled``: the real matrix after degree-preserving edge swaps. Every KC keeps its number of PN partners and its
  synapse counts, every PN its number of KC partners; which PN reaches which KC is random. Comparing it with
  ``flywire`` isolates the specific pairings from the degree structure;
- ``flyhash``: FlyHash (Dasgupta et al. 2017): every KC sums ``claws`` PNs drawn uniformly, binary weights, with
  ``claws`` the rounded mean in-degree of the real KCs (4);
- ``dense``: a dense Gaussian random projection, the classic locality-sensitive hashing control;
- ``flyhash_full``: FlyHash without the PN bottleneck: every KC samples 6 channels directly.

Two steps are applied to every variant alike, because the raw synapse counts make the same few KCs win every time
(a KC with seven strong claws out-drives a KC with two, whatever the input), the filter then saturates and nothing
looks new:

- PN adaptation: every PN input is centred on its mean over the training rows (FlyHash centres its input in the
  same spirit), so a KC is driven by inputs that are above their usual level;
- KC compensation: every KC's weight row is scaled to unit length. KCs need several claws active to fire
  (Gruntman & Turner 2013) and KCs with more or stronger inputs are less excitable (compensatory variability,
  Abdelrahman et al. 2021), which keeps every KC's activity in the same range; the relative synapse counts within
  a KC are kept.

Usage:
    exp = build("flywire", wiring, n_channels=200, seed=0)
    exp.fit_centre(code_train)
    drive = exp.drive(code)   # (rows, n_kc)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .connectome import Wiring

EXPANSIONS = ("flywire", "shuffled", "flyhash", "dense", "flyhash_full")
FULL_CLAWS = 6  # FlyHash default: each KC samples about 6 of the 50 PN types
SWAPS_PER_EDGE = 10


@dataclass
class Expansion:
    """Input map (PN x channel, ``None`` when channels feed the KCs directly) and KC x input weights."""

    name: str
    input_map: sparse.csr_matrix | None
    weights: sparse.csr_matrix | np.ndarray
    centre: np.ndarray | None = None  # mean input over the training rows

    @property
    def n_kc(self) -> int:
        return self.weights.shape[0]

    def __post_init__(self):
        # dense float32 copies for the products (BLAS, rows x KC layout); the sparse matrices stay the model
        dense = lambda m: (m.toarray() if sparse.issparse(m) else np.asarray(m)).astype(np.float32)  # noqa: E731
        self._map_t = None if self.input_map is None else np.ascontiguousarray(dense(self.input_map).T)
        self._weights_t = np.ascontiguousarray(dense(self.weights).T)

    def inputs(self, code: np.ndarray) -> np.ndarray:
        """PN (or channel) activity, (rows, inputs)."""
        code = np.asarray(code, dtype=np.float32)
        return code if self._map_t is None else code @ self._map_t

    def fit_centre(self, code: np.ndarray) -> "Expansion":
        self.centre = self.inputs(code).mean(axis=0).astype(np.float32)
        return self

    def drive(self, code: np.ndarray) -> np.ndarray:
        """Input of every KC for every row of the channel code, (rows, n_kc) float32."""
        x = self.inputs(code)
        if self.centre is not None:
            x = x - self.centre
        return x @ self._weights_t

    def nbytes(self) -> int:
        def size(m):
            if m is None:
                return 0
            if sparse.issparse(m):
                return m.data.nbytes + m.indices.nbytes + m.indptr.nbytes
            return m.nbytes
        return size(self.input_map) + size(self.weights) + size(self.centre)


def input_map(n_channels: int, n_pn: int, rng: np.random.Generator) -> sparse.csr_matrix:
    """Balanced random assignment of channels to PNs (every channel and every PN used, loads differ by <= 1)."""
    n = max(n_channels, n_pn)
    pn = np.arange(n) % n_pn
    channel = rng.permutation(n) % n_channels
    m = sparse.csr_matrix((np.ones(n, np.float32), (pn, channel)), shape=(n_pn, n_channels))
    m.sum_duplicates()
    m.data[:] = 1.0
    return m


def degree_preserving_shuffle(matrix: sparse.spmatrix, rng: np.random.Generator,
                              swaps_per_edge: int = SWAPS_PER_EDGE) -> sparse.csr_matrix:
    """Random edge swaps (kc1, pn1), (kc2, pn2) -> (kc1, pn2), (kc2, pn1) that never create a duplicate pair.
    Row degrees, column degrees and every row's multiset of weights are kept."""
    coo = sparse.coo_matrix(matrix)
    kc, pn, w = coo.row.tolist(), coo.col.tolist(), coo.data.copy()
    n_pn = matrix.shape[1]
    present = {k * n_pn + p for k, p in zip(kc, pn)}
    e = len(kc)
    picks = rng.integers(0, e, size=(swaps_per_edge * e, 2)).tolist()
    for a, b in picks:
        k1, p1, k2, p2 = kc[a], pn[a], kc[b], pn[b]
        if k1 == k2 or p1 == p2:
            continue
        new1, new2 = k1 * n_pn + p2, k2 * n_pn + p1
        if new1 in present or new2 in present:
            continue
        present.difference_update((k1 * n_pn + p1, k2 * n_pn + p2))
        present.update((new1, new2))
        pn[a], pn[b] = p2, p1
    return sparse.csr_matrix((w, (kc, pn)), shape=matrix.shape)


def flyhash_matrix(n_kc: int, n_in: int, claws: int, rng: np.random.Generator) -> sparse.csr_matrix:
    """Every KC sums ``claws`` distinct inputs drawn uniformly, weight 1."""
    cols = np.argsort(rng.random((n_kc, n_in)), axis=1)[:, :claws]
    rows = np.repeat(np.arange(n_kc), claws)
    return sparse.csr_matrix((np.ones(n_kc * claws, np.float32), (rows, cols.ravel())), shape=(n_kc, n_in))


def unit_rows(weights: sparse.csr_matrix | np.ndarray) -> sparse.csr_matrix | np.ndarray:
    """Every KC's weight row scaled to unit Euclidean length (KC compensation)."""
    if sparse.issparse(weights):
        norm = np.sqrt(np.asarray(weights.multiply(weights).sum(axis=1)).ravel())
        return sparse.diags(1.0 / np.where(norm > 0, norm, 1.0)).astype(np.float32) @ weights
    norm = np.linalg.norm(weights, axis=1, keepdims=True)
    return (weights / np.where(norm > 0, norm, 1.0)).astype(np.float32)


def build(kind: str, wiring: Wiring, n_channels: int, seed: int) -> Expansion:
    """One expansion variant; ``seed`` fixes the channel -> PN map (shared by all variants) and the matrix."""
    if kind not in EXPANSIONS:
        raise ValueError(f"unknown expansion {kind!r}, choose from {EXPANSIONS}")
    n_kc, n_pn = wiring.matrix.shape
    imap = input_map(n_channels, n_pn, np.random.default_rng(seed))
    rng = np.random.default_rng([seed, EXPANSIONS.index(kind)])
    if kind == "flywire":
        weights = wiring.matrix.astype(np.float32).tocsr()
    elif kind == "shuffled":
        weights = degree_preserving_shuffle(wiring.matrix, rng).astype(np.float32)
    elif kind == "flyhash":
        claws = max(1, int(round(wiring.matrix.getnnz(axis=1).mean())))
        weights = flyhash_matrix(n_kc, n_pn, claws, rng)
    elif kind == "dense":
        weights = rng.standard_normal((n_kc, n_pn)).astype(np.float32)
    else:
        weights = flyhash_matrix(n_kc, n_channels, min(FULL_CLAWS, n_channels), rng)
        return Expansion(kind, None, unit_rows(weights).tocsr())
    weights = unit_rows(weights)
    return Expansion(kind, imap, weights.tocsr() if sparse.issparse(weights) else weights)
