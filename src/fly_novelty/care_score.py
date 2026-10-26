"""CARE score for early fault detection (Gueck, Roelofs, Faulstich 2024, doi:10.3390/data9120138).

An own implementation of the published definition, kept in line with the authors' reference implementation
(EnergyFaultDetector, MIT) in the choices the paper leaves open; taken unchanged from the companion project
wind-thermal-nbm (same author, MIT), where it matches the reference code to 4 decimals on all 95 datasets, so that
both projects score CARE identically. Every dataset ("event") of the benchmark has a
prediction window; anomaly events contain a labelled window [event_start, event_end] that leads up to a fault.
Point predictions (anomaly yes/no per 10-minute timestamp) are scored on four aspects:

- Coverage: point-wise F-score (beta = 0.5) on anomaly events, ground truth = inside the event window;
- Accuracy: share of points predicted normal on normal events;
- Reliability: event-wise F-score (beta = 0.5); an event counts as detected when the criticality counter
  (+1 for a predicted anomaly, -1 otherwise, never below 0, unchanged on points with abnormal status) reaches
  72 (12 h of net anomalies) before the event end;
- Earliness: weighted share of detected points inside the event window, weight 1 in the first quarter of the
  window and falling linearly to 0 at its end.

Points whose operating status is not normal are left out of Coverage and Accuracy and do not move the
criticality (for wind farm A the status comes from a failure log and is ignored in the evaluation).

    CARE = 0                                   if no event is detected at all
         = mean Accuracy                       if mean Accuracy <= 0.5
         = (Cov + Earl + Rel + 2 Acc) / 5      otherwise

Usage:
    rows = [evaluate_event(pred, normal, in_window, "anomaly", event_id=3), ...]
    summary = care_score(pd.DataFrame(rows))
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BETA = 0.5
CRITICALITY_THRESHOLD = 72
DESCEND_START = 0.25
WEIGHTS = {"coverage": 1.0, "earliness": 1.0, "reliability": 1.0, "accuracy": 2.0}


def fbeta(tp: float, fp: float, fn: float, beta: float = BETA) -> float:
    """F-beta score from counts; 0 when there are no true positives."""
    b2 = beta * beta
    denom = (1 + b2) * tp + b2 * fn + fp
    return float((1 + b2) * tp / denom) if denom > 0 and tp > 0 else 0.0


def criticality(anomaly: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Counter that rises with predicted anomalies and falls with normal predictions, on normal-status points."""
    step = np.where(normal & anomaly, 1, np.where(normal & ~anomaly, -1, 0))
    out = np.empty(len(step), dtype=np.int64)
    c = 0
    for i, s in enumerate(step):
        c = max(0, c + int(s))
        out[i] = c
    return out


def earliness_weights(n: int, descend_start: float = DESCEND_START) -> np.ndarray:
    """Weight 1 up to ``descend_start`` of the window, then linearly down to 0 at its end."""
    if n == 1:
        return np.ones(1)
    pos = np.arange(n) / (n - 1)
    return np.where(pos < descend_start, 1.0, (1.0 - pos) / (1.0 - descend_start))


def evaluate_event(pred: np.ndarray, normal: np.ndarray, in_window: np.ndarray, label: str,
                   event_id: int | None = None) -> dict:
    """Score the point predictions of one dataset's prediction section.

    ``pred``: predicted anomaly per timestamp; ``normal``: normal operating status (all True to ignore the status);
    ``in_window``: timestamp inside [event_start, event_end]. The criticality is counted up to event_end (for an
    anomaly event the fault starts there); the window is the ground truth only for anomaly events."""
    pred = np.asarray(pred, dtype=bool)
    normal = np.asarray(normal, dtype=bool)
    in_window = np.asarray(in_window, dtype=bool)
    if label not in ("anomaly", "normal"):
        raise ValueError(f"unknown label {label!r}")
    window_idx = np.flatnonzero(in_window)
    stop = window_idx[-1] + 1 if len(window_idx) else len(pred)
    crit = criticality(pred[:stop], normal[:stop])
    truth = in_window if label == "anomaly" else np.zeros(len(pred), dtype=bool)
    p, t = pred[normal], truth[normal]
    tp, fp = int(np.sum(p & t)), int(np.sum(p & ~t))
    fn, tn = int(np.sum(~p & t)), int(np.sum(~p & ~t))
    out = {"event_id": event_id, "label": label, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
           "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
           "coverage": fbeta(tp, fp, fn) if label == "anomaly" else np.nan,
           "max_criticality": int(crit.max()) if len(crit) else 0,
           "detected": bool(len(crit) and crit.max() >= CRITICALITY_THRESHOLD)}
    if label == "anomaly" and len(window_idx):
        w = earliness_weights(len(window_idx))
        out["earliness"] = float(np.sum(w * pred[window_idx]) / np.sum(w))
        first = np.flatnonzero(crit >= CRITICALITY_THRESHOLD)
        out["lead_steps"] = int(stop - 1 - first[0]) if len(first) else np.nan
    else:
        out["earliness"] = np.nan
        out["lead_steps"] = np.nan
    return out


def care_score(events: pd.DataFrame) -> dict:
    """Combine evaluated events (rows of :func:`evaluate_event`) into the CARE score and its parts."""
    anomaly = events["label"] == "anomaly"
    if anomaly.all() or (~anomaly).all():
        raise ValueError("the CARE score needs at least one anomaly and one normal event")
    tp = int((events["detected"] & anomaly).sum())
    fp = int((events["detected"] & ~anomaly).sum())
    fn = int((~events["detected"] & anomaly).sum())
    parts = {
        "coverage": float(events.loc[anomaly, "coverage"].mean()),
        "accuracy": float(events.loc[~anomaly, "accuracy"].mean()),
        "reliability": fbeta(tp, fp, fn),
        "earliness": float(events.loc[anomaly, "earliness"].mean()),
        "events_detected": tp, "normal_events_flagged": fp, "anomaly_events": int(anomaly.sum()),
        "normal_events": int((~anomaly).sum()),
    }
    if tp + fp == 0:
        score = 0.0
    elif parts["accuracy"] <= 0.5:
        score = parts["accuracy"]
    else:
        score = sum(WEIGHTS[k] * parts[k] for k in WEIGHTS) / sum(WEIGHTS.values())
    return {"care": float(score), **parts}
