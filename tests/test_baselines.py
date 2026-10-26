import numpy as np
import pytest

from fly_novelty import baselines


@pytest.mark.parametrize("name,offset", [(n, 80.0) for n in baselines.BASELINES]
                         + [(n, 25.0) for n in ("pca", "autoencoder", "knn")])
def test_baselines_rank_an_injected_fault_above_normal_rows(name, offset, scada):
    """+80 K is outside anything seen in training (every method must react); +25 K stays inside the range of the
    generator temperature and only breaks its relation to power and ambient (needs a multivariate method)."""
    model = baselines.make(name, seed=0).fit(scada[:4000])
    normal = model.score(scada[4000:5000])
    faulty = scada[5000:6000].copy()
    faulty[:, 4] += offset
    bad = model.score(faulty)
    assert np.isfinite(normal).all() and np.isfinite(bad).all()
    # the isolation forest only orders: one wild signal out of five shortens few of its random paths
    level = 0.5 if name == "iforest" else 0.95
    assert np.median(bad) > np.quantile(normal, level), name


def test_pca_score_is_the_mean_squared_residual_and_missing_values_are_neutral(scada):
    pca = baselines.PCAModel().fit(scada[:4000])
    r = pca.residuals(scada[4000:4100])
    assert r.shape == (100, scada.shape[1])
    np.testing.assert_allclose(pca.score(scada[4000:4100]), np.mean(r ** 2, axis=1))
    x = scada[4000:4001].copy()
    x[0, :] = np.nan
    assert pca.score(x)[0] == pytest.approx(0.0, abs=1e-12)  # all missing = the training mean


def test_unknown_baseline_raises():
    with pytest.raises(ValueError):
        baselines.make("svm")
