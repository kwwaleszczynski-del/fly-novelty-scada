import numpy as np
import pandas as pd
import pytest

from fly_novelty.care_score import care_score, criticality, earliness_weights, evaluate_event, fbeta


def test_fbeta_weights_precision_over_recall():
    assert fbeta(10, 0, 0) == 1.0
    assert fbeta(0, 5, 5) == 0.0
    # precision 1, recall 0.5 scores higher than precision 0.5, recall 1 with beta = 0.5
    assert fbeta(5, 0, 5) > fbeta(10, 10, 0)
    assert fbeta(5, 0, 5) == pytest.approx(1.25 * 1 * 0.5 / (0.25 * 1 + 0.5))


def test_criticality_counts_only_on_normal_status():
    pred = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1], bool)
    normal = np.array([1, 1, 0, 1, 0, 1, 1, 1, 1], bool)
    np.testing.assert_array_equal(criticality(pred, normal), [1, 2, 2, 1, 1, 0, 0, 1, 2])


def test_earliness_weights_shape():
    w = earliness_weights(101)
    assert w[:25].min() == 1.0 and w[-1] == 0.0
    assert w[50] == pytest.approx(0.5 / 0.75)
    assert np.all(np.diff(w) <= 0)


def test_anomaly_event_detected_early_scores_high_and_counts_lead():
    n = 500
    window = np.zeros(n, bool)
    window[200:400] = True
    pred = np.zeros(n, bool)
    pred[210:400] = True  # flags from early in the window on
    ev = evaluate_event(pred, np.ones(n, bool), window, "anomaly", event_id=1)
    assert ev["detected"] and ev["max_criticality"] == 190
    assert ev["lead_steps"] == 399 - (210 + 71)  # threshold 72 reached at index 281
    assert ev["earliness"] > 0.9
    assert ev["coverage"] == pytest.approx(fbeta(190, 0, 10))


def test_predictions_after_the_event_end_do_not_count_for_detection():
    n = 300
    window = np.zeros(n, bool)
    window[100:150] = True
    pred = np.zeros(n, bool)
    pred[160:300] = True  # only after the fault
    ev = evaluate_event(pred, np.ones(n, bool), window, "anomaly")
    assert not ev["detected"] and ev["earliness"] == 0.0


def test_abnormal_status_points_are_ignored():
    n = 200
    pred = np.zeros(n, bool)
    pred[50:150] = True
    normal = np.ones(n, bool)
    normal[50:150] = False  # the operator already knew: turbine in service mode
    ev = evaluate_event(pred, normal, np.ones(n, bool), "normal")
    assert ev["accuracy"] == 1.0 and not ev["detected"]


def _events(detect_anomaly: bool, flag_normal: bool) -> pd.DataFrame:
    n = 400
    window = np.zeros(n, bool)
    window[100:300] = True
    rows = []
    for i in range(3):
        pred = window.copy() if detect_anomaly else np.zeros(n, bool)
        rows.append(evaluate_event(pred, np.ones(n, bool), window, "anomaly", i))
    for i in range(3, 6):
        pred = np.zeros(n, bool)
        if flag_normal:
            pred[:100] = True
        rows.append(evaluate_event(pred, np.ones(n, bool), window, "normal", i))
    return pd.DataFrame(rows)


def test_care_score_special_cases_and_perfect_model():
    assert care_score(_events(False, False))["care"] == 0.0  # nothing detected
    perfect = care_score(_events(True, False))
    assert perfect["accuracy"] == 1.0 and perfect["reliability"] == 1.0 and perfect["coverage"] == 1.0
    # earliness of a window flagged from its first point is 1
    assert perfect["care"] == pytest.approx(1.0)
    noisy = care_score(_events(True, True))
    assert noisy["accuracy"] == 0.75 and noisy["normal_events_flagged"] == 3
    assert noisy["care"] == pytest.approx((1 + 1 + fbeta(3, 3, 0) + 2 * 0.75) / 5)


def test_care_score_falls_back_to_accuracy_when_worse_than_random():
    df = _events(True, False)
    df.loc[df.label == "normal", "accuracy"] = 0.4
    assert care_score(df)["care"] == pytest.approx(0.4)
