import numpy as np
import pandas as pd

from fly_novelty.report import _variant, bootstrap_difference, method_table, pick_events


def events_frame() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for i in range(12):
        label = "anomaly" if i < 6 else "normal"
        for m in ("fly_flywire_s0", "fly_flywire_s1", "fly_flywire_s2", "fly_shuffled_s0", "fly_shuffled_s1",
                  "fly_shuffled_s2", "pca", "autoencoder"):
            good = m.startswith("fly_flywire") or m == "pca"
            detected = bool(label == "anomaly" and (good or i < 2)) or (label == "normal" and i == 11 and good)
            rows.append({"farm": "B", "event_id": i, "label": label, "method": m, "event_description": "x",
                         "detected": detected, "accuracy": 0.99 if label == "normal" else 0.5,
                         "coverage": 0.4 if detected else 0.0, "earliness": 0.3 if detected else 0.0,
                         "lead_steps": float(rng.integers(100, 1000)) if detected else np.nan})
    return pd.DataFrame(rows)


def test_variant_names():
    assert _variant("fly_flywire_s2") == "fly_flywire" and _variant("pca") == "pca"
    assert _variant("stream_flywire_gated") == "stream_flywire_gated"


def test_bootstrap_difference_sign_and_interval():
    d = bootstrap_difference(events_frame(), ["fly_flywire_s0", "fly_flywire_s1"], ["fly_shuffled_s0"], n=200)
    assert d["difference"] > 0 and d["ci_low"] <= d["difference"] <= d["ci_high"]
    same = bootstrap_difference(events_frame(), ["pca"], ["pca"], n=50)
    assert same["difference"] == 0 and same["ci_low"] == same["ci_high"] == 0


def test_pick_events_follows_the_rule():
    picks = pick_events(events_frame())
    kinds = [k for _, _, k in picks]
    assert kinds.count("caught") == 4 and kinds.count("false alarm") == 1 and kinds.count("quiet") == 1
    assert ("B", 11, "false alarm") in picks


def test_method_table_averages_seeds():
    summary = pd.DataFrame([{"quantile": 0.995, "farm": "all", "method": m, "care": c, "events_detected": 3,
                             "normal_events_flagged": 1, "anomaly_events": 45, "normal_events": 50,
                             "median_lead_days": 2.0,
                             "false_flag_days_per_dataset": 0.1, "coverage": 0.1, "accuracy": 0.9,
                             "reliability": 0.3, "earliness": 0.1}
                            for m, c in (("fly_flywire_s0", 0.5), ("fly_flywire_s1", 0.6), ("pca", 0.55),
                                         ("grid_flywire_k02_d0_e0e+00", 0.4))])
    t = method_table(summary).set_index("key")
    assert t.loc["fly_flywire", "care"] == 0.55 and t.loc["fly_flywire", "seeds"] == [0.5, 0.6]
    assert "grid_flywire_k02_d0_e0e+00" not in t.index
    assert t.loc["pca", "skill"] == 3 / 45 - 1 / 50
