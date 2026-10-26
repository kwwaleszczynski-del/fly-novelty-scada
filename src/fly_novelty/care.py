"""The fly novelty detector and standard anomaly scores on the CARE to Compare benchmark (Gueck et al. 2024).

CARE to Compare (doi:10.5281/zenodo.15846963, CC BY-SA 4.0) holds 95 datasets from three wind farms; each is one
turbine's 10-minute SCADA with about a year of training data and a prediction section that contains either a
labelled anomaly leading up to a fault or normal behaviour. The archive is read directly from the zip. The data
handling and the evaluation protocol follow the companion project wind-thermal-nbm, so the scores of its
temperature models (RC, gradient boosting) on the same datasets can be put next to these:

1. state vector: every 10-minute average of the farm's feature list except angles and counters, kept when at
   least half of the fitting rows have a value, its training distribution has at least 5 distinct quantiles and it
   is not a near copy (|r| > 0.99) of an earlier signal;
2. fitting rows: the first 75 % of the training section, normal status (``status_type_id == 0``), producing more
   than 5 % of rated power; calibration rows: the remaining 25 %, same conditions, never used for fitting;
3. every method gives a raw score per row; the alarm score is its trailing 6-hour median over producing rows,
   and a row is flagged when that exceeds the 99.5 % quantile of the alarm score on the calibration rows (fixed in
   the companion project before any prediction section was looked at);
4. the flags of the prediction section are scored with :mod:`care_score`.

Methods: the fly detector with every expansion variant and three seeds (fitted on the fitting rows, then frozen;
the main setting is 5 % active KCs, decay 0.5, no recovery while fitting), a sensitivity grid for the first seed of
the real, shuffled and FlyHash wiring (3 x 3 settings of active share x decay, and two recovery rates switched on
while fitting), the streaming fly detector (keeps learning from the calibration rows on, with recovery 1e-5 per
stored row; learning gated by alarms, and ungated), the fly detector fed with the PCA residuals of the state instead
of the state itself (real, shuffled and FlyHash wiring, three seeds), and the baselines of :mod:`baselines`.

Usage:
    meta = run_all(Path("data/care/raw/CARE_To_Compare.zip"), Path("connectome"), Path("data/care/scores"), jobs=4)
    events, summary = summarise(Path("data/care/scores"))
"""
from __future__ import annotations

import io
import logging
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import baselines, connectome
from .care_score import care_score, evaluate_event
from .encoding import PopulationEncoder
from .expansion import EXPANSIONS, build
from .novelty import EPS, fitted_filter, stream, tags, trailing_median

ROOT = "CARE_To_Compare"
FARMS = ("A", "B", "C")
STEP = pd.Timedelta(minutes=10)
POWER = {"A": "power_30", "B": "power_62", "C": "power_6"}  # grid active power, from feature_description.csv
FIT_FRACTION = 0.75
OPERATING_FRACTION = 0.05  # of the 99th percentile of power in the training section
MIN_FINITE = 0.5
MIN_KNOTS = 5
MAX_CORR = 0.99
SCORE_QUANTILE = 0.995
QUANTILES = (0.99, SCORE_QUANTILE, 0.999)  # sensitivity check only
SEEDS = (0, 1, 2)
K_GRID = (0.02, 0.05, 0.10)
DELTA_GRID = (0.0, 0.5, 0.9)
EPS_GRID = (1e-5, 3e-5)  # recovery already while fitting (sensitivity only)
MAIN_K, MAIN_DELTA, MAIN_EPS = 0.05, 0.5, 0.0
GRID_KINDS = ("flywire", "shuffled", "flyhash")
STREAM_KINDS = ("flywire", "flyhash")
RES_KINDS = ("flywire", "shuffled", "flyhash")  # the fly on PCA residuals instead of the state
log = logging.getLogger(__name__)


def grid_settings() -> list[tuple[float, float, float]]:
    """(active share, decay, recovery) of the sensitivity grid; the main setting is part of it."""
    return ([(k, d, MAIN_EPS) for k in K_GRID for d in DELTA_GRID]
            + [(MAIN_K, MAIN_DELTA, e) for e in EPS_GRID])


def grid_name(kind: str, k: float, delta: float, eps: float) -> str:
    return f"grid_{kind}_k{int(round(k * 100)):02d}_d{int(round(delta * 10))}_e{eps:.0e}"


def _read_csv(z: zipfile.ZipFile, name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(z.read(name)), sep=";", **kw)


def event_info(z: zipfile.ZipFile) -> pd.DataFrame:
    out = []
    for farm in FARMS:
        ev = _read_csv(z, f"{ROOT}/Wind Farm {farm}/event_info.csv").rename(columns={"asset": "asset_id"})
        ev["farm"] = farm
        out.append(ev)
    ev = pd.concat(out, ignore_index=True)
    for c in ("event_start", "event_end"):
        ev[c] = pd.to_datetime(ev[c])
    return ev


def state_signals(z: zipfile.ZipFile, farm: str) -> pd.DataFrame:
    """Candidate signals of a farm: every sensor with a 10-minute average that is neither an angle nor a counter."""
    fd = _read_csv(z, f"{ROOT}/Wind Farm {farm}/feature_description.csv", encoding="latin-1")
    flag = lambda c: fd[c].astype(str).str.lower().eq("true")  # noqa: E731
    keep = fd["statistics_type"].astype(str).str.contains("average") & ~flag("is_angle") & ~flag("is_counter")
    return pd.DataFrame({"sensor": fd.loc[keep, "sensor_name"].to_numpy(),
                         "description": fd.loc[keep, "description"].astype(str).str.strip().to_numpy(),
                         "unit": fd.loc[keep, "unit"].astype(str).str.strip().to_numpy()})


def load_dataset(z: zipfile.ZipFile, farm: str, event_id: int, sensors: list[str]) -> pd.DataFrame:
    """One dataset on a regular 10-minute grid with ``<sensor>`` columns, status and section labels."""
    name = f"{ROOT}/Wind Farm {farm}/datasets/{event_id}.csv"
    cols = {f"{s}_avg": s for s in sensors}
    base = ["time_stamp", "train_test", "status_type_id"]
    df = _read_csv(z, name, usecols=lambda c: c in cols or c in base)
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    df = df.copy()  # read_csv with hundreds of columns returns a fragmented frame
    df["present"] = 1.0  # rows that exist in the file (the grid below adds empty ones)
    df = df.drop_duplicates("time_stamp").set_index("time_stamp").sort_index().rename(columns=cols)
    df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq=STEP))
    df.index.name = "timestamp"
    for c in df.columns:
        if c != "train_test":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df["train_test"] = df["train_test"].ffill()
    return df


def masks(df: pd.DataFrame, farm: str, info: pd.Series) -> dict[str, np.ndarray]:
    """Row masks of one dataset (see the module docstring)."""
    n = len(df)
    train = (df["train_test"] == "train").to_numpy()
    normal = (df["status_type_id"] == 0).to_numpy()
    power = df[POWER[farm]].to_numpy()
    p99 = np.nanpercentile(power[train], 99)
    producing = np.nan_to_num(power, nan=-np.inf) > OPERATING_FRACTION * p99
    idx_train = np.flatnonzero(train)
    split = idx_train[int(FIT_FRACTION * len(idx_train))]
    pred = (df["train_test"].to_numpy() == "prediction") & (df["present"].to_numpy() == 1.0)
    # wind farm A's status comes from the failure log (two weeks before a failure are marked as downtime): it may
    # only filter the training data, so in the prediction section every producing row is used and scored
    status_ok = normal | pred if farm == "A" else normal
    rows = np.arange(n)
    return {
        "fit": train & normal & producing & (rows < split),
        "cal": train & normal & producing & (rows >= split),
        "pred": pred,
        "usable": status_ok & producing,
        # streaming learns from normal producing rows; on wind farm A the prediction status would leak the fault
        "learn": status_ok & producing,
        "status_normal": normal if farm in ("B", "C") else np.ones(n, dtype=bool),
        "in_window": ((df.index >= info["event_start"]) & (df.index <= info["event_end"])),
    }


def select_signals(X: np.ndarray) -> np.ndarray:
    """Indices of the usable, non-duplicate signals given the fitting rows ``X``."""
    from .encoding import PROBS

    keep = []
    for j in range(X.shape[1]):
        col = X[:, j]
        finite = col[np.isfinite(col)]
        if len(finite) < MIN_FINITE * len(col) or len(np.unique(np.quantile(finite, PROBS))) < MIN_KNOTS:
            continue
        keep.append(j)
    keep = np.array(keep, dtype=int)
    rng = np.random.default_rng(0)
    sample = X[rng.choice(len(X), min(len(X), 5000), replace=False)][:, keep]
    corr = pd.DataFrame(sample).corr(min_periods=500).abs().to_numpy()
    out = []
    for a in range(len(keep)):
        if not any(corr[a, b] > MAX_CORR for b in out):
            out.append(a)
    return keep[out]


def _threshold(smooth: np.ndarray, cal: np.ndarray, quantile: float = SCORE_QUANTILE) -> float:
    values = smooth[cal]
    values = values[np.isfinite(values)]
    return float(np.quantile(values, quantile)) if len(values) else np.inf


def score_dataset(zip_path: Path, conn_dir: Path, farm: str, event_id: int, info: pd.Series, out_dir: Path,
                  seeds=SEEDS) -> dict:
    """Run every method on one dataset; save the alarm scores of the calibration and prediction span to
    ``<farm>_<event>.parquet`` and return the metadata row (sizes, timings)."""
    with zipfile.ZipFile(zip_path) as z:
        candidates = state_signals(z, farm)
        df = load_dataset(z, farm, event_id, candidates["sensor"].tolist())
    m = masks(df, farm, info)
    sensors = [s for s in candidates["sensor"] if s in df]
    X_all = df[sensors].to_numpy(float)
    chosen = select_signals(X_all[m["fit"]])
    X = X_all[:, chosen]
    span = np.arange(np.flatnonzero(m["cal"])[0], len(df))  # calibration start to the end of the file
    usable = m["usable"][span]
    cal_s = m["cal"][span]
    pred_start = int(np.flatnonzero(m["pred"][span])[0])
    out = pd.DataFrame({k: m[k][span] for k in ("cal", "pred", "usable", "status_normal", "in_window")},
                       index=df.index[span])
    timing = {}

    t0 = time.perf_counter()
    enc = PopulationEncoder().fit(X[m["fit"]])
    code = enc.transform(X)
    t_enc = time.perf_counter() - t0
    wiring = connectome.load(conn_dir)
    k_of = {f: max(1, int(round(f * wiring.n_kc))) for f in K_GRID}
    k_max = max(k_of.values())
    for kind in EXPANSIONS:
        for seed in seeds:
            t0 = time.perf_counter()
            exp = build(kind, wiring, enc.n_channels, seed)
            exp.fit_centre(code[m["fit"]])
            tag_rows, valid = tags(exp, code, k_max)
            fit_tags = tag_rows[m["fit"] & valid]
            t_tags = time.perf_counter() - t0
            main = (MAIN_K, MAIN_DELTA, MAIN_EPS)
            grid = grid_settings() if (kind in GRID_KINDS and seed == seeds[0]) else [main]
            for kf, delta, eps in grid:
                k = k_of[kf]
                t0 = time.perf_counter()
                filt = fitted_filter(wiring.n_kc, fit_tags[:, :k], delta, eps)
                raw = filt.novelty_batch(tag_rows[span, :k])
                raw[~valid[span]] = np.nan
                name = f"fly_{kind}_s{seed}" if (kf, delta, eps) == main else grid_name(kind, kf, delta, eps)
                out[name] = trailing_median(raw, usable).astype(np.float32)
                if (kf, delta, eps) == main:
                    timing[f"fly_{kind}_s{seed}"] = t_enc + t_tags + time.perf_counter() - t0
            if kind in STREAM_KINDS and seed == seeds[0]:
                k = k_of[MAIN_K]
                for gate in (True, False):
                    filt = fitted_filter(wiring.n_kc, fit_tags[:, :k], MAIN_DELTA)
                    filt.eps = EPS
                    state = (filt.w.copy(), filt.last.copy(), filt.t)
                    args = (tag_rows[span, :k], valid[span], usable, m["learn"][span])
                    # pass 1: calibration rows only (all learned), to set the threshold the way it would be set live
                    _, smooth_cal, _ = stream(filt, *[a[:pred_start] for a in args])
                    thr = _threshold(smooth_cal, cal_s[:pred_start])
                    filt.w, filt.last, filt.t = state
                    threshold = np.where(np.arange(len(span)) >= pred_start, thr, np.inf)
                    t0 = time.perf_counter()
                    _, smooth, stored = stream(filt, *args, threshold=threshold, gate=gate)
                    name = f"stream_{kind}_{'gated' if gate else 'ungated'}"
                    out[name] = smooth.astype(np.float32)
                    timing[name] = time.perf_counter() - t0
                    timing[f"{name}_stored_pred"] = float(stored[pred_start:].mean())
            del tag_rows, fit_tags

    # the fly on the PCA residuals: which signals leave the usual correlation structure, and in which direction
    t0 = time.perf_counter()
    pca = baselines.PCAModel().fit(X[m["fit"]])
    R = pca.residuals(X)
    enc_r = PopulationEncoder().fit(R[m["fit"]])
    code_r = enc_r.transform(R)
    t_res = time.perf_counter() - t0
    k = k_of[MAIN_K]
    for kind in RES_KINDS:
        for seed in seeds:
            t0 = time.perf_counter()
            exp = build(kind, wiring, enc_r.n_channels, seed)
            exp.fit_centre(code_r[m["fit"]])
            tag_rows, valid = tags(exp, code_r, k)
            filt = fitted_filter(wiring.n_kc, tag_rows[m["fit"] & valid], MAIN_DELTA)
            raw = filt.novelty_batch(tag_rows[span])
            raw[~valid[span]] = np.nan
            out[f"fly_res_{kind}_s{seed}"] = trailing_median(raw, usable).astype(np.float32)
            timing[f"fly_res_{kind}_s{seed}"] = t_res + time.perf_counter() - t0

    for name in baselines.BASELINES:
        t0 = time.perf_counter()
        model = baselines.make(name, seed=0).fit(X[m["fit"]])
        t_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        raw = model.score(X[span])
        timing[name] = t_fit + time.perf_counter() - t0
        out[name] = trailing_median(raw, usable).astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / f"{farm}_{event_id}.parquet")
    row = {"farm": farm, "event_id": int(event_id), "asset_id": int(info["asset_id"]), "label": info["event_label"],
           "event_description": info.get("event_description", ""), "candidate_signals": len(sensors),
           "signals": len(chosen), "channels": enc.n_channels, "fit_rows": int(m["fit"].sum()),
           "cal_rows": int(m["cal"].sum()), "prediction_days": float(m["pred"].sum() / 144),
           "event_days": float((m["in_window"] & m["pred"]).sum() / 144), "span_rows": len(span)}
    row.update({f"time_{k}": v for k, v in timing.items()})
    log.info("%s %3d %-8s %3d signals, %d fit rows", farm, event_id, info["event_label"], len(chosen),
             row["fit_rows"])
    return row


def _run_one(args) -> dict:
    zip_path, conn_dir, farm, event_id, info, out_dir = args
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from threadpoolctl import threadpool_limits

    with threadpool_limits(2):
        return score_dataset(zip_path, conn_dir, farm, event_id, info, out_dir)


def run_all(zip_path: Path, conn_dir: Path, out_dir: Path, jobs: int = 1, farms=FARMS,
            events: list[int] | None = None) -> pd.DataFrame:
    """Score every dataset into ``out_dir``; ``meta.csv`` there makes an interrupted run resume where it stopped."""
    from concurrent.futures import ProcessPoolExecutor

    out_csv = out_dir / "meta.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        info = event_info(z)
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(columns=["farm", "event_id"])
    seen = set(zip(done["farm"], done["event_id"].astype(int)))
    todo = [(zip_path, conn_dir, r["farm"], int(r["event_id"]), r, out_dir) for _, r in info.iterrows()
            if r["farm"] in farms and (events is None or int(r["event_id"]) in events)
            and (r["farm"], int(r["event_id"])) not in seen]
    # the large wind farm C files first, so the pool is not left waiting on one of them at the end
    todo.sort(key=lambda a: a[2] != "C")
    log.info("CARE: %d datasets to run, %d cached", len(todo), len(seen))
    rows = [done] if len(done) else []
    with ProcessPoolExecutor(max_workers=max(jobs, 1)) as ex:
        for res in ex.map(_run_one, todo):
            frame = pd.DataFrame([res])
            if out_csv.exists():
                frame = pd.concat([pd.read_csv(out_csv), frame], ignore_index=True)
            frame.to_csv(out_csv, index=False)
            rows.append(frame.tail(1))
    return pd.read_csv(out_csv)


def evaluate_column(scores: pd.DataFrame, column: str, label: str, quantile: float = SCORE_QUANTILE) -> dict:
    """Threshold one alarm score at the given quantile of its calibration rows and score the event."""
    s = scores[column].to_numpy(float)
    threshold = _threshold(s, scores["cal"].to_numpy(), quantile)
    p = scores["pred"].to_numpy()
    flag = s[p] > threshold
    ev = evaluate_event(flag, scores["status_normal"].to_numpy()[p], scores["in_window"].to_numpy()[p], label)
    del ev["event_id"]
    normal = scores["status_normal"].to_numpy()[p]
    in_window = scores["in_window"].to_numpy()[p]
    outside = normal & ~in_window if label == "anomaly" else normal
    first = np.flatnonzero(flag & in_window)
    lead = (np.flatnonzero(in_window)[-1] - first[0]) / 144 if len(first) else np.nan
    return {**ev, "quantile": quantile, "threshold": threshold,
            "flagged_share": float(flag[normal].mean()) if normal.any() else np.nan,
            "false_flag_days": float((flag & outside).sum() / 144), "first_flag_lead_days": float(lead)}


def summarise(scores_dir: Path, quantiles=(SCORE_QUANTILE,)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluated events and CARE scores (all farms and per farm) for every method column and quantile."""
    meta = pd.read_csv(scores_dir / "meta.csv")
    events = []
    for r in meta.itertuples(index=False):
        s = pd.read_parquet(scores_dir / f"{r.farm}_{r.event_id}.parquet")
        methods = [c for c in s.columns if c not in ("cal", "pred", "usable", "status_normal", "in_window")]
        for q in quantiles:
            for method in methods:
                events.append({"farm": r.farm, "event_id": r.event_id, "asset_id": r.asset_id, "label": r.label,
                               "event_description": r.event_description, "method": method,
                               **evaluate_column(s, method, r.label, q)})
    events = pd.DataFrame(events)
    summary = []
    for (q, method), g in events.groupby(["quantile", "method"]):
        summary.append({"quantile": q, "method": method, "farm": "all", **_summary_row(g)})
        for farm, gf in g.groupby("farm"):
            summary.append({"quantile": q, "method": method, "farm": farm, **_summary_row(gf)})
    return events, pd.DataFrame(summary)


def _summary_row(g: pd.DataFrame) -> dict:
    anomaly = g["label"] == "anomaly"
    detected = g[anomaly & g["detected"]]
    return {**care_score(g), "median_lead_days": float(detected["lead_steps"].median() / 144) if len(detected)
            else np.nan, "false_flag_days_per_dataset": float(g["false_flag_days"].mean())}
