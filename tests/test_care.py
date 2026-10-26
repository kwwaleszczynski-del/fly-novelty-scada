import numpy as np
import pandas as pd
import pytest

from fly_novelty import care
from fly_novelty.inject import FAULTS, apply_fault, component_temperatures


def frame(n: int = 2000, unit: str = "ns") -> pd.DataFrame:
    """A dataset as :func:`care.load_dataset` returns it: 1500 training rows, 500 prediction rows."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-01", periods=n, freq="10min", name="timestamp").as_unit(unit)
    power = np.clip(rng.normal(1000, 600, n), 0, 2000)
    df = pd.DataFrame({"power_62": power, "sensor_1": 20 + power / 100 + rng.normal(0, 1, n),
                       "train_test": ["train"] * 1500 + ["prediction"] * (n - 1500),
                       "status_type_id": 0.0, "present": 1.0}, index=idx)
    df.iloc[100:110, df.columns.get_loc("status_type_id")] = 4.0
    return df


@pytest.mark.parametrize("unit", ["ns", "us", "s"])
def test_masks_split_training_and_mark_the_event_window(unit):
    df = frame(unit=unit)
    info = pd.Series({"event_start": pd.Timestamp("2022-01-12 00:00"), "event_end": pd.Timestamp("2022-01-13 00:00")})
    m = care.masks(df, "B", info)
    rows = np.arange(len(df))
    assert not (m["fit"] & m["cal"]).any()
    assert rows[m["fit"]].max() < rows[m["cal"]].min()
    assert rows[m["cal"]].max() < 1500 and m["pred"].sum() == 500
    assert not m["fit"][100:110].any()  # abnormal status never fitted
    assert m["usable"][m["fit"]].all()
    assert m["in_window"].sum() == 145  # 24 h of 10-minute rows, both ends included
    assert m["status_normal"][100:110].sum() == 0


def test_wind_farm_a_prediction_rows_ignore_status():
    df = frame()
    df.iloc[1600:1700, df.columns.get_loc("status_type_id")] = 4.0
    df = df.rename(columns={"power_62": "power_30"})
    info = pd.Series({"event_start": df.index[1800], "event_end": df.index[1900]})
    m = care.masks(df, "A", info)
    producing = df["power_30"].to_numpy() > 0.05 * np.nanpercentile(df["power_30"].to_numpy()[:1500], 99)
    assert (m["usable"][1600:1700] == producing[1600:1700]).all()
    assert (m["learn"][1600:1700] == producing[1600:1700]).all()  # the logged fault status must not gate learning
    assert m["status_normal"].all()


def test_select_signals_drops_sparse_constant_and_duplicate_signals():
    rng = np.random.default_rng(1)
    n = 3000
    a = rng.normal(size=n)
    sparse_col = np.where(rng.random(n) < 0.7, np.nan, a)
    X = np.column_stack([a, a * 2 + 1, np.full(n, 3.0), sparse_col, rng.normal(size=n), rng.integers(0, 2, n)])
    assert care.select_signals(X).tolist() == [0, 4]


def test_evaluate_column_detects_flags_inside_the_window():
    n = 1000
    s = pd.DataFrame({"cal": np.r_[np.ones(400, bool), np.zeros(600, bool)],
                      "pred": np.r_[np.zeros(400, bool), np.ones(600, bool)],
                      "usable": True, "status_normal": True,
                      "in_window": np.r_[np.zeros(700, bool), np.ones(200, bool), np.zeros(100, bool)]})
    score = np.random.default_rng(0).normal(0, 1, n)
    score[700:750] = 0.0
    score[750:900] += 10.0
    s["m"] = score
    ev = care.evaluate_column(s, "m", "anomaly")
    assert ev["detected"] and ev["first_flag_lead_days"] == pytest.approx((899 - 750) / 144)
    assert ev["false_flag_days"] < 0.1
    normal = care.evaluate_column(s.assign(in_window=False, m=np.random.default_rng(1).normal(0, 1, n)), "m",
                                  "normal")
    assert not normal["detected"] and normal["accuracy"] > 0.95


def test_faults_change_only_the_target_rows():
    X = np.tile(np.arange(10.0)[:, None], (1, 3))
    rows = np.arange(4, 8)
    for fault in FAULTS:
        Y = apply_fault(X, rows, 1, fault)
        assert np.array_equal(np.delete(Y, rows, axis=0), np.delete(X, rows, axis=0))
        assert np.array_equal(Y[:, [0, 2]], X[:, [0, 2]])
    assert apply_fault(X, rows, 1, "step_10K")[rows, 1].tolist() == [14, 15, 16, 17]
    assert apply_fault(X, rows, 1, "drift_10K")[rows, 1].tolist() == pytest.approx([4, 5 + 10 / 3, 6 + 20 / 3, 17])
    assert apply_fault(X, rows, 1, "stuck")[rows, 1].tolist() == [4, 4, 4, 4]


def test_component_temperatures_skip_air_temperatures():
    cand = pd.DataFrame({"sensor": ["s1", "s2", "s3", "s4"],
                         "description": ["Gearbox bearing temperature 1", "Outside temperature", "Active power",
                                         "Nacelle temperature"],
                         "unit": ["°C", "°C", "kW", "°C"]})
    assert component_temperatures(["s1", "s2", "s3", "s4"], cand) == [0]
