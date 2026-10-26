"""The fly detector as an operational tool: cost per sample, behaviour after a change of normal, live replay.

- :func:`cost_table`: fitting time, model state in bytes, batch time per row and single-sample latency of the
  fly detector (scoring and learning one sample at a time) and of the baselines, on one dataset;
- :func:`service_dataset`: what happens after a permanent change of normal operation (a +5 K step in one
  component temperature, as after a sensor swap or a new cooling setpoint) and during a slowly developing fault (a
  drift to +10 K), for four policies: frozen filter; streaming with learning blocked while in alarm (gated);
  streaming that always learns (ungated); gated with an operator acknowledgement that forces learning for 24 h
  after the service; plus a control without any change, which shows the alarms that come from the passing time
  alone. Label-free: training sections of the CARE datasets, as in :mod:`inject`;
- :func:`replay`: a dataset's prediction section streamed sample by sample with the gated detector, logging every
  alarm change and writing a compact trace for the dashboard.

Usage:
    table = cost_table(X_fit, X_test, wiring)
    rows = service_dataset(zip_path, conn_dir, "B", 34, info_row)
    trace = replay(zip_path, conn_dir, "C", 55, Path("docs/replay.json"))
"""
from __future__ import annotations

import copy
import json
import logging
import pickle
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import baselines, care, connectome
from .care_score import criticality
from .encoding import PopulationEncoder
from .expansion import build
from .inject import DETECTED_STEPS, apply_fault, component_temperatures
from .novelty import DELTA, EPS, FlyBloomFilter, FlyNovelty, fitted_filter, stream, tags

POLICIES = ("frozen", "gated", "ungated", "gated_ack")
SCENARIOS = {"none": None, "new_normal": "step_5K", "fault": "drift_10K"}  # none: control, nothing changes
ACK_ROWS = 144  # 24 h of forced learning after the operator confirms the service
CLEAR_ROWS = 144  # an alarm counts as cleared after 24 h without a flag
log = logging.getLogger(__name__)


def _latency(fn, X: np.ndarray, n: int) -> float:
    rows = X[np.linspace(0, len(X) - 1, n).astype(int)]
    t0 = time.perf_counter()
    for x in rows:
        fn(x)
    return (time.perf_counter() - t0) / n * 1e6


def cost_table(X_fit: np.ndarray, X_test: np.ndarray, wiring: connectome.Wiring, n_single: int = 300) -> pd.DataFrame:
    """Fit time [s], state [bytes], batch [us/row] and one-sample latency [us] per method."""
    rows = []
    t0 = time.perf_counter()
    det = FlyNovelty(wiring, "flywire", seed=0).fit(X_fit)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    det.score(X_test)
    batch = (time.perf_counter() - t0) / len(X_test) * 1e6
    single = _latency(lambda x: det.score_one(x, store=True), X_test, n_single)
    rows.append({"method": "fly (flywire)", "fit_s": fit_s, "state_bytes": det.nbytes(),
                 "learned_bytes": det.filter.nbytes(), "batch_us_per_row": batch, "single_us": single,
                 "learns_online": True})
    for name in baselines.BASELINES:
        t0 = time.perf_counter()
        model = baselines.make(name, seed=0).fit(X_fit)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        model.score(X_test)
        batch = (time.perf_counter() - t0) / len(X_test) * 1e6
        single = _latency(lambda x: model.score(x[None, :]), X_test, n_single)
        size = len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
        rows.append({"method": name, "fit_s": fit_s, "state_bytes": size, "learned_bytes": size,
                     "batch_us_per_row": batch, "single_us": single, "learns_online": False})
    return pd.DataFrame(rows)


def _cleared_after(flags: np.ndarray, start: int) -> float:
    """Days from ``start`` until the last flag that is followed by at least ``CLEAR_ROWS`` quiet rows."""
    idx = np.flatnonzero(flags[start:])
    if not len(idx):
        return 0.0
    gaps = np.diff(np.r_[idx, len(flags) - start + CLEAR_ROWS])
    quiet = np.flatnonzero(gaps >= CLEAR_ROWS)
    return float(idx[quiet[0]] / 144) if len(quiet) and idx[quiet[0]] + CLEAR_ROWS <= len(flags) - start \
        else np.nan


def service_dataset(zip_path: Path, conn_dir: Path, farm: str, event_id: int, info: pd.Series,
                    seed: int = 0) -> list[dict]:
    with zipfile.ZipFile(zip_path) as z:
        candidates = care.state_signals(z, farm)
        df = care.load_dataset(z, farm, event_id, candidates["sensor"].tolist())
    m = care.masks(df, farm, info)
    sensors = [s for s in candidates["sensor"] if s in df]
    X_all = df[sensors].to_numpy(float)
    chosen = care.select_signals(X_all[m["fit"]])
    names = [sensors[i] for i in chosen]
    X = X_all[:, chosen]
    temps = component_temperatures(names, candidates)
    if not temps:
        return []
    target = int(np.random.default_rng([seed, ord(farm), event_id, 1]).choice(temps))
    wiring = connectome.load(conn_dir)
    enc = PopulationEncoder().fit(X[m["fit"]])
    exp = build("flywire", wiring, enc.n_channels, 0)
    exp.fit_centre(enc.transform(X[m["fit"]]))
    k = max(1, int(round(care.MAIN_K * wiring.n_kc)))
    fit_tags, fit_valid = tags(exp, enc.transform(X[m["fit"]]), k)
    base = fitted_filter(wiring.n_kc, fit_tags[fit_valid], DELTA)
    base.eps = EPS  # recovery from the end of the training rows on (frozen policy stores nothing, so never recovers)
    cal = np.flatnonzero(m["cal"])
    span = np.arange(cal[0], cal[-1] + 1)
    change = int(np.searchsorted(span, cal[len(cal) // 2]))  # first row of the test half, relative to span
    usable, learn = m["usable"][span], m["learn"][span]
    ref = m["cal"][span][:change]
    no_learning = np.zeros(len(span), bool)

    def fresh() -> FlyBloomFilter:
        return copy.deepcopy(base)

    rows = []
    for scenario, fault in SCENARIOS.items():
        Y = X if fault is None else apply_fault(X, span[change:], target, fault)
        t_rows, valid = tags(exp, enc.transform(Y[span]), k)
        # thresholds from the reference half (before the change), frozen and learning as the policy would run
        thresholds = {}
        for live, learn_p in (("frozen", no_learning), ("streaming", learn)):
            _, ref_smooth, _ = stream(fresh(), t_rows[:change], valid[:change], usable[:change], learn_p[:change])
            thresholds[live] = float(np.nanquantile(ref_smooth[ref], care.SCORE_QUANTILE))
        for policy in POLICIES:
            frozen = policy == "frozen"
            thr = thresholds["frozen" if frozen else "streaming"]
            force = np.zeros(len(span), bool)
            if policy == "gated_ack" and scenario == "new_normal":
                force[change:change + ACK_ROWS] = learn[change:change + ACK_ROWS]
            thr_rows = np.where(np.arange(len(span)) >= change, thr, np.inf)
            _, smooth, _ = stream(fresh(), t_rows, valid, usable, no_learning if frozen else learn,
                                  threshold=thr_rows, gate=policy != "ungated", force=force)
            flags = smooth > thr
            after = flags[change:] & usable[change:]
            crit = criticality(flags[change:], usable[change:])
            first = np.flatnonzero(crit >= DETECTED_STEPS)
            rows.append({"farm": farm, "event_id": int(event_id), "target": names[target], "scenario": scenario,
                         "policy": policy, "flagged_share": float(after.sum() / max(usable[change:].sum(), 1)),
                         "detected": bool(len(first)), "delay_days": float(first[0] / 144) if len(first) else np.nan,
                         "cleared_after_days": _cleared_after(flags & usable, change),
                         "test_days": float((len(span) - change) / 144),
                         "false_flags_before": float(flags[:change][usable[:change]].mean())})
    return rows


def _run_service(args) -> list[dict]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from threadpoolctl import threadpool_limits

    zip_path, conn_dir, farm, event_id, info = args
    with threadpool_limits(2):
        return service_dataset(zip_path, conn_dir, farm, event_id, info)


def run_service(zip_path: Path, conn_dir: Path, out_csv: Path, jobs: int = 1) -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor

    with zipfile.ZipFile(zip_path) as z:
        info = care.event_info(z)
    todo = [(zip_path, conn_dir, r["farm"], int(r["event_id"]), r) for _, r in info.iterrows()]
    todo.sort(key=lambda a: a[2] != "C")
    rows = []
    with ProcessPoolExecutor(max_workers=max(jobs, 1)) as ex:
        for res in ex.map(_run_service, todo):
            rows.extend(res)
    out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def replay(zip_path: Path, conn_dir: Path, farm: str, event_id: int, out_json: Path | None = None,
           novel_weight: float = 0.05) -> dict:
    """Stream the prediction section of one dataset through the gated fly detector (flywire, seed 0) as it would
    run live: fitted on the fitting rows, threshold from the calibration rows, then one sample at a time. Alarm
    changes are logged; the trace (hourly) and the raster of the tag KCs whose weight is still above
    ``novel_weight`` (one tag per hour) go to JSON."""
    with zipfile.ZipFile(zip_path) as z:
        info = care.event_info(z)
        info = info[(info["farm"] == farm) & (info["event_id"] == event_id)].iloc[0]
        candidates = care.state_signals(z, farm)
        df = care.load_dataset(z, farm, event_id, candidates["sensor"].tolist())
    m = care.masks(df, farm, info)
    sensors = [s for s in candidates["sensor"] if s in df]
    X = df[sensors].to_numpy(float)[:, care.select_signals(df[sensors].to_numpy(float)[m["fit"]])]
    wiring = connectome.load(conn_dir)
    det = FlyNovelty(wiring, "flywire", seed=0).fit(X[m["fit"]])
    span = np.arange(np.flatnonzero(m["cal"])[0], len(df))
    t_rows, valid = det.tags(X[span])
    usable, learn = m["usable"][span], m["learn"][span]
    pred_start = int(np.flatnonzero(m["pred"][span])[0])
    state = (det.filter.w.copy(), det.filter.last.copy(), det.filter.t)
    _, cal_smooth, _ = stream(det.filter, t_rows[:pred_start], valid[:pred_start], usable[:pred_start],
                              learn[:pred_start])
    threshold = float(np.nanquantile(cal_smooth[m["cal"][span][:pred_start]], care.SCORE_QUANTILE))
    det.filter.w, det.filter.last, det.filter.t = state
    # live part: the same loop as :func:`stream`, unrolled to log alarms and keep the KC weights for the raster
    buf = np.full(36, np.nan)
    times = df.index[span]
    smooth = np.full(len(span), np.nan)
    raw = np.full(len(span), np.nan)
    alarm_on = False
    alarms = []
    raster_from = pred_start
    raster = []
    for i in range(len(span)):
        if valid[i]:
            raw[i] = det.filter.novelty(t_rows[i])
        buf[i % 36] = raw[i] if usable[i] else np.nan
        finite = buf[np.isfinite(buf)]
        if len(finite) >= 12:
            smooth[i] = np.median(finite)
        live = i >= pred_start
        alarm = live and smooth[i] > threshold
        if live and alarm != alarm_on:
            log.info("%s %s alarm %s (novelty %.4f, threshold %.4f)", times[i], f"{farm}-{event_id}",
                     "ON " if alarm else "off", smooth[i], threshold)
            alarms.append([str(times[i]), bool(alarm)])
            alarm_on = alarm
        if i >= raster_from and i % 6 == 0 and valid[i] and usable[i]:
            w = det.filter.current(t_rows[i])
            novel = w > novel_weight
            raster.append([int((i - raster_from) // 6), t_rows[i][novel].tolist(), np.round(w[novel], 2).tolist()])
        if valid[i] and learn[i] and not alarm:
            det.filter.store(t_rows[i])
    hourly = pd.DataFrame({"raw": raw, "smooth": smooth, "usable": usable,
                           "in_window": m["in_window"][span], "pred": m["pred"][span]}, index=times)
    hourly = hourly.iloc[pred_start:].resample("1h").agg({"raw": "mean", "smooth": "max", "usable": "max",
                                                          "in_window": "max", "pred": "max"})
    trace = {"farm": farm, "event_id": int(event_id), "label": info["event_label"],
             "description": str(info.get("event_description", "") or ""), "threshold": threshold,
             "t": [str(t) for t in hourly.index], "smooth": _round(hourly["smooth"]),
             "in_window": hourly["in_window"].astype(bool).tolist(), "raster_start": str(times[raster_from]),
             "raster_hours": int((len(span) - raster_from) // 6), "raster": raster, "n_kc": wiring.n_kc,
             "alarms": alarms}
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(trace), encoding="utf-8")
    return trace


def _round(s: pd.Series, digits: int = 4) -> list:
    return [None if not np.isfinite(v) else round(float(v), digits) for v in s.to_numpy(float)]
