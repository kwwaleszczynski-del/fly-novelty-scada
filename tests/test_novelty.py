import numpy as np
import pytest

from fly_novelty.encoding import PopulationEncoder
from fly_novelty.expansion import EXPANSIONS, build, degree_preserving_shuffle, flyhash_matrix, input_map
from fly_novelty.novelty import FlyBloomFilter, FlyNovelty, stream, tags, trailing_median, wta


def test_wta_keeps_exactly_k_active_per_row():
    rng = np.random.default_rng(1)
    drive = rng.normal(size=(500, 2000)).astype(np.float32)
    k = 100  # 5 %
    t = wta(drive, k)
    assert t.shape == (500, 100)
    assert all(len(set(row)) == k for row in t)
    # the chosen KCs are the k largest, ordered from the largest down
    best = np.sort(drive, axis=1)[:, ::-1][:, :k]
    np.testing.assert_allclose(np.take_along_axis(drive, t, axis=1), best)


def test_wta_with_ties_still_keeps_k():
    drive = np.zeros((3, 50), dtype=np.float32)
    drive[:, :5] = 1.0  # five clear winners, the rest tied at zero
    t = wta(drive, 10)
    assert t.shape == (3, 10) and all(len(set(r)) == 10 for r in t)
    assert all(set(range(5)) <= set(r) for r in t)


def test_tags_activity_share_matches_k_fraction(wiring, scada):
    det = FlyNovelty(wiring, "flywire", k_fraction=0.05).fit(scada[:3000])
    t, valid = det.tags(scada[3000:])
    assert det.k == round(0.05 * wiring.n_kc)
    active = np.zeros((len(t), wiring.n_kc), bool)
    np.put_along_axis(active, t, True, axis=1)
    np.testing.assert_allclose(active.mean(axis=1), det.k / wiring.n_kc)
    assert valid.all()


def test_filter_new_pattern_is_novel_and_repeats_become_familiar():
    f = FlyBloomFilter(100, delta=0.5)
    a, b = np.arange(0, 10), np.arange(50, 60)
    assert f.novelty(a) == 1.0
    seen = []
    for _ in range(6):
        seen.append(f.novelty(a))
        f.store(a)
    assert seen == [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
    assert f.novelty(b) == 1.0  # an unrelated pattern is still new
    half = np.r_[np.arange(0, 5), np.arange(50, 55)]
    assert f.novelty(half) == pytest.approx((5 * 0.5 ** 6 + 5) / 10)  # graded: distance sensitive


def test_binary_filter_and_batch_counts_match_sequential_storage():
    rng = np.random.default_rng(3)
    stored = [rng.choice(200, 10, replace=False) for _ in range(40)]
    for delta in (0.0, 0.5):
        seq, batch = FlyBloomFilter(200, delta), FlyBloomFilter(200, delta)
        for t in stored:
            seq.store(t)
        batch.store_counts(np.bincount(np.concatenate(stored), minlength=200))
        np.testing.assert_allclose(seq.current(), batch.current())
    assert set(np.unique(FlyBloomFilter(200, 0.0).current())) == {1.0}


def test_recovery_restores_novelty_and_lazy_update_equals_eager_rule():
    rng = np.random.default_rng(4)
    m, delta, eps = 60, 0.3, 0.02
    lazy = FlyBloomFilter(m, delta, eps)
    eager = np.ones(m)  # Eq. 2 applied to every KC at every step
    for _ in range(300):
        tag = rng.choice(m, 5, replace=False)
        assert lazy.novelty(tag) == pytest.approx(eager[tag].mean())
        active = np.zeros(m, bool)
        active[tag] = True
        eager = np.where(active, eager * delta, np.minimum(1.0, eager + eps))
        lazy.store(tag)
    np.testing.assert_allclose(lazy.current(), eager)
    # a pattern stored once, then 1/eps unrelated items later, is new again
    f = FlyBloomFilter(100, delta=0.0, eps=0.01)
    f.store(np.arange(10))
    assert f.novelty(np.arange(10)) == 0.0
    for _ in range(100):
        f.store(np.arange(90, 100))
    assert f.novelty(np.arange(10)) == pytest.approx(1.0)


def test_fitted_filter_stores_in_time_order_and_does_not_saturate():
    from fly_novelty.novelty import fitted_filter

    rng = np.random.default_rng(5)
    common = rng.choice(500, 20, replace=False)
    rare = np.setdiff1d(np.arange(500), common)[:20]
    fit = [common] * 5000 + [rare] * 5  # one state seen all year, another a handful of times
    no_recovery = fitted_filter(500, np.array(fit), delta=0.5, eps=0.0)
    with_recovery = fitted_filter(500, np.array(fit), delta=0.5, eps=1e-3)
    assert no_recovery.novelty(common) == 0.0 and no_recovery.novelty(rare) == pytest.approx(0.5 ** 5)
    # with recovery the weights balance decay against recovery: frequent -> small, rare -> larger, never -> 1
    assert 0 < with_recovery.novelty(common) < with_recovery.novelty(rare) < 1.0
    assert with_recovery.novelty(np.setdiff1d(np.arange(500), np.r_[common, rare])[:20]) == 1.0
    seq = FlyBloomFilter(500, 0.5, 1e-3)
    for t in fit:
        seq.store(t)
    np.testing.assert_allclose(seq.current(), with_recovery.current())


def test_batch_storage_refuses_recovery():
    with pytest.raises(ValueError):
        FlyBloomFilter(10, 0.5, eps=0.1).store_counts(np.zeros(10))


def test_detector_flags_injected_fault_more_than_normal_rows(scada):
    from conftest import synthetic_wiring

    wiring = synthetic_wiring(n_kc=1000, n_pn=40)
    for kind in EXPANSIONS:
        det = FlyNovelty(wiring, kind, seed=0).fit(scada[:4000])
        normal = det.score(scada[4000:5000])
        faulty = scada[5000:6000].copy()
        faulty[:, 3] += 15.0  # gearbox 15 K hotter than ever seen at that power and ambient
        bad = det.score(faulty)
        assert np.mean(bad > np.quantile(normal, 0.99)) > 0.8, kind


def test_same_seed_same_result_other_seed_differs(wiring, scada):
    a = FlyNovelty(wiring, "shuffled", seed=7).fit(scada[:3000]).score(scada[3000:])
    b = FlyNovelty(wiring, "shuffled", seed=7).fit(scada[:3000]).score(scada[3000:])
    c = FlyNovelty(wiring, "shuffled", seed=8).fit(scada[:3000]).score(scada[3000:])
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_score_one_matches_batch_and_learns(wiring, scada):
    det = FlyNovelty(wiring, "flyhash", seed=1).fit(scada[:3000])
    x = scada[3500].copy()
    x[4] += 40.0
    assert det.score_one(x) == pytest.approx(det.score(x[None, :])[0])
    first = det.score_one(x, store=True)
    assert det.score_one(x) < first


def test_missing_signals_are_silent_and_all_missing_rows_have_no_score(wiring, scada):
    det = FlyNovelty(wiring, "flywire").fit(scada[:3000])
    x = scada[3000:3003].copy()
    x[1, 2] = np.nan
    x[2, :] = np.nan
    s = det.score(x)
    assert np.isfinite(s[:2]).all() and np.isnan(s[2])


def test_population_code_sums_to_one_and_tails_use_outer_bins(scada):
    enc = PopulationEncoder(n_bins=4).fit(scada)
    code = enc.transform(scada).reshape(len(scada), -1, 4)
    np.testing.assert_allclose(code.sum(axis=2), 1.0, atol=1e-5)
    extreme = scada[:2].copy()
    extreme[0, 3], extreme[1, 3] = 1e6, -1e6
    c = enc.transform(extreme).reshape(2, -1, 4)[:, 3]
    assert c[0].argmax() == 3 and c[1].argmax() == 0
    assert enc.n_channels == scada.shape[1] * 4


def test_input_map_is_balanced():
    rng = np.random.default_rng(0)
    for channels, pns in ((200, 149), (60, 149), (149, 149)):
        m = input_map(channels, pns, rng)
        per_pn, per_channel = np.asarray(m.sum(axis=1)).ravel(), np.asarray(m.sum(axis=0)).ravel()
        assert per_pn.min() >= 1 and per_channel.min() >= 1
        assert per_pn.max() - per_pn.min() <= 1 and per_channel.max() - per_channel.min() <= 1


def test_shuffle_keeps_degrees_and_weights_without_duplicates(wiring):
    rng = np.random.default_rng(0)
    s = degree_preserving_shuffle(wiring.matrix, rng)
    np.testing.assert_array_equal(s.getnnz(axis=1), wiring.matrix.getnnz(axis=1))
    np.testing.assert_array_equal(s.getnnz(axis=0), wiring.matrix.getnnz(axis=0))
    for i in range(0, wiring.n_kc, 37):
        np.testing.assert_array_equal(np.sort(s[i].data), np.sort(wiring.matrix[i].data))
    assert s.max() == wiring.matrix.max()  # no merged duplicate pairs
    overlap = (s.multiply(wiring.matrix) > 0).sum() / wiring.matrix.nnz
    assert overlap < 0.5


def test_flyhash_has_fixed_claws_and_binary_weights():
    m = flyhash_matrix(300, 40, 6, np.random.default_rng(0))
    assert (m.getnnz(axis=1) == 6).all() and set(np.unique(m.data)) == {1.0}


def test_build_variants_share_input_map(wiring):
    a, b = build("flywire", wiring, 50, seed=3), build("dense", wiring, 50, seed=3)
    assert (a.input_map != b.input_map).nnz == 0
    full = build("flyhash_full", wiring, 50, seed=3)
    assert full.input_map is None and full.weights.shape == (wiring.n_kc, 50)


def test_streaming_smooth_equals_offline_median_and_gate_blocks_learning():
    rng = np.random.default_rng(0)
    n, m = 400, 80
    tag_rows = np.array([rng.choice(m, 4, replace=False) for _ in range(n)])
    valid = np.ones(n, bool)
    usable = rng.random(n) > 0.2
    raw, smooth, stored = stream(FlyBloomFilter(m, 0.5), tag_rows, valid, usable, np.ones(n, bool))
    np.testing.assert_allclose(smooth, trailing_median(raw, usable), equal_nan=True)
    assert stored.all()
    # with a threshold below every smoothed value nothing is stored once the window has filled
    _, smooth2, stored2 = stream(FlyBloomFilter(m, 0.5), tag_rows, valid, usable, np.ones(n, bool), threshold=-1.0)
    assert not stored2[np.isfinite(smooth2)].any()
    force = np.zeros(n, bool)
    force[200:] = True
    _, _, stored3 = stream(FlyBloomFilter(m, 0.5), tag_rows, valid, usable, np.ones(n, bool), threshold=-1.0,
                           force=force)
    assert stored3[200:].all()


def test_tags_mark_rows_without_input():
    from fly_novelty.expansion import Expansion
    from scipy import sparse

    exp = Expansion("x", None, sparse.csr_matrix(np.eye(5, dtype=np.float32)))
    code = np.zeros((2, 5), np.float32)
    code[0, 2] = 1.0
    t, valid = tags(exp, code, 1)
    assert valid.tolist() == [True, False] and t[0, 0] == 2
