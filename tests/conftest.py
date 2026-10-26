import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from fly_novelty.connectome import Wiring


def synthetic_wiring(n_kc: int = 400, n_pn: int = 30, seed: int = 0) -> Wiring:
    """A small PN -> KC matrix with a skewed PN popularity and 2-7 PN partners per KC, like the real one."""
    rng = np.random.default_rng(seed)
    popularity = rng.pareto(1.5, n_pn) + 0.2
    popularity /= popularity.sum()
    rows, cols, vals = [], [], []
    for kc in range(n_kc):
        partners = rng.choice(n_pn, size=rng.integers(2, 8), replace=False, p=popularity)
        rows += [kc] * len(partners)
        cols += partners.tolist()
        vals += rng.integers(5, 30, len(partners)).tolist()
    matrix = sparse.csr_matrix((np.array(vals, float), (rows, cols)), shape=(n_kc, n_pn))
    pn = pd.DataFrame({"node": np.arange(n_pn), "glomerulus": [f"G{i}" for i in range(n_pn)]})
    kc = pd.DataFrame({"node": np.arange(n_kc), "kc_class": "gamma"})
    return Wiring(matrix=matrix, pn=pn, kc=kc)


def synthetic_scada(n: int = 6000, seed: int = 0) -> np.ndarray:
    """Correlated operating signals: power follows wind, temperatures follow power and ambient."""
    rng = np.random.default_rng(seed)
    wind = np.clip(8 + np.cumsum(rng.normal(0, 0.3, n)) * 0.1 + rng.normal(0, 1.5, n), 0, 25)
    power = 2000 / (1 + np.exp(-(wind - 9)))
    ambient = 10 + 8 * np.sin(np.arange(n) * 2 * np.pi / 144) + rng.normal(0, 0.5, n)
    gear = ambient + 30 * power / 2000 + rng.normal(0, 0.8, n)
    gen = ambient + 45 * power / 2000 + rng.normal(0, 1.0, n)
    return np.column_stack([wind, power, ambient, gear, gen])


@pytest.fixture
def wiring():
    return synthetic_wiring()


@pytest.fixture
def scada():
    return synthetic_scada()
