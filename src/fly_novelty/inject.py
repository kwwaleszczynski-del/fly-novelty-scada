"""Label-free validation: synthetic sensor faults injected into the training sections of the CARE datasets.

The 45 labelled faults of CARE are few and mixed (hydraulics, pitch, converter, ...); a question like "does the
real fly wiring beat a random one?" needs more trials. Here only the training section of every dataset is used,
the event labels are not touched:

1. fit every detector on the fitting rows (first 75 % of the training section, as in :mod:`care`);
2. split the calibration rows in time into a reference half (sets the 99.5 % threshold of the alarm score) and a
   test half;
3. inject one fault at a time into one component temperature in the test half and count how much of it is flagged,
   next to the flags on the clean test half (false alarms). Faults: a step of +5 K or +10 K (a bearing or winding
   running hot), a drift growing linearly to +10 K over the test half (a slowly developing fault) and a frozen
   sensor (the value held from the first test row on).

Two temperatures per dataset are chosen at random (seeded) among the components (unit degC, not ambient, nacelle or
hub air). The design choices of the fly detector (the number of tuning curves, recovery during fitting) were
checked on this test for a development subset of datasets.

Usage:
    rows = inject_dataset(zip_path, conn_dir, "B", 34, info_row, kinds=("flywire", "flyhash"), seeds=(0,))
"""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import baselines, care, connectome
from .care_score import criticality
from .encoding import PopulationEncoder
from .expansion import EXPANSIONS, build
from .novelty import DELTA, fitted_filter, tags, trailing_median

FAULTS = {"step_5K": ("step", 5.0), "step_10K": ("step", 10.0), "drift_10K": ("drift", 10.0),
          "stuck": ("stuck", 0.0)}
TARGETS_PER_DATASET = 2
NOT_COMPONENTS = ("ambient", "outside", "nacelle temperature", "hub temperature", "nose cone", "platform", "board")
DETECTED_STEPS = 72  # CARE's criticality threshold: 12 h of net flags
log = logging.getLogger(__name__)


def component_temperatures(names: list[str], candidates: pd.DataFrame) -> list[int]:
    """Positions in ``names`` of the component temperature signals."""
    info = candidates.set_index("sensor")
    out = []
    for i, s in enumerate(names):
        unit, desc = str(info.loc[s, "unit"]), str(info.loc[s, "description"]).lower()
        if (unit.endswith("C") or unit.lower() == "celsius") and not any(w in desc for w in NOT_COMPONENTS):
            out.append(i)
    return out


def apply_fault(X: np.ndarray, rows: np.ndarray, col: int, fault: str) -> np.ndarray:
    kind, size = FAULTS[fault]
    Y = X.copy()
    if kind == "step":
        Y[rows, col] += size
    elif kind == "drift":
        Y[rows, col] += size * np.linspace(0.0, 1.0, len(rows))
    else:
        start = rows[np.isfinite(X[rows, col])]
        Y[rows, col] = X[start[0], col] if len(start) else np.nan
    return Y


def _outcome(smooth: np.ndarray, threshold: float, test: np.ndarray, usable: np.ndarray) -> dict:
    flags = smooth[test] > threshold
    ok = usable[test]
    crit = criticality(flags, ok)
    first = np.flatnonzero(crit >= DETECTED_STEPS)
    return {"flagged": float(flags[ok].mean()) if ok.any() else np.nan, "detected": bool(len(first)),
            "delay_days": float(first[0] / 144) if len(first) else np.nan}


def inject_dataset(zip_path: Path, conn_dir: Path, farm: str, event_id: int, info: pd.Series,
                   kinds=EXPANSIONS, seeds=care.SEEDS, n_bins_options=(4,), eps_options=(0.0,),
                   residual_kinds=care.RES_KINDS, with_baselines: bool = True, seed: int = 0) -> list[dict]:
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
    rng = np.random.default_rng([seed, ord(farm), event_id])
    targets = rng.choice(temps, size=min(TARGETS_PER_DATASET, len(temps)), replace=False) if temps else []
    cal = np.flatnonzero(m["cal"])
    ref_end = cal[len(cal) // 2]
    span = np.arange(cal[0], cal[-1] + 1)
    usable = m["usable"][span]
    ref = (m["cal"][span]) & (span < ref_end)
    test_rows = np.flatnonzero(m["cal"] & (np.arange(len(df)) >= ref_end))
    test = np.isin(span, test_rows)
    variants = {}  # name -> function(X) returning a raw score per row of X[span]
    wiring = connectome.load(conn_dir)
    k = max(1, int(round(0.05 * wiring.n_kc)))
    for n_bins in n_bins_options:
        enc = PopulationEncoder(n_bins).fit(X[m["fit"]])
        code_fit = enc.transform(X[m["fit"]])
        for kind in kinds:
            for s in seeds:
                exp = build(kind, wiring, enc.n_channels, s)
                exp.fit_centre(code_fit)
                fit_tags, valid = tags(exp, code_fit, k)
                for eps in eps_options:
                    filt = fitted_filter(wiring.n_kc, fit_tags[valid], DELTA, eps)

                    def score(A, enc=enc, exp=exp, filt=filt):
                        t, v = tags(exp, enc.transform(A), k)
                        out = filt.novelty_batch(t)
                        out[~v] = np.nan
                        return out
                    suffix = "" if len(n_bins_options) == 1 else f"_b{n_bins}"
                    suffix += "" if len(eps_options) == 1 else f"_e{eps:.0e}"
                    variants[f"fly_{kind}_s{s}{suffix}"] = score
    if residual_kinds:
        pca = baselines.PCAModel().fit(X[m["fit"]])
        enc_r = PopulationEncoder().fit(pca.residuals(X[m["fit"]]))
        code_fit = enc_r.transform(pca.residuals(X[m["fit"]]))
        for kind in residual_kinds:
            for s in seeds:
                exp = build(kind, wiring, enc_r.n_channels, s)
                exp.fit_centre(code_fit)
                fit_tags, valid = tags(exp, code_fit, k)
                filt = fitted_filter(wiring.n_kc, fit_tags[valid], DELTA)

                def score_res(A, exp=exp, filt=filt):
                    t, v = tags(exp, enc_r.transform(pca.residuals(A)), k)
                    out = filt.novelty_batch(t)
                    out[~v] = np.nan
                    return out
                variants[f"fly_res_{kind}_s{s}"] = score_res
    if with_baselines:
        for name in baselines.BASELINES:
            variants[name] = baselines.make(name, seed=0).fit(X[m["fit"]]).score
    rows = []
    for method, fn in variants.items():
        clean_raw = fn(X[span])
        clean = trailing_median(clean_raw, usable)
        threshold = float(np.nanquantile(clean[ref], care.SCORE_QUANTILE))
        base = {"farm": farm, "event_id": int(event_id), "method": method}
        rows.append({**base, "fault": "none", "target": "", **_outcome(clean, threshold, test, usable)})
        for col in targets:
            for fault in FAULTS:
                dirty = apply_fault(X, test_rows, int(col), fault)
                raw = clean_raw.copy()
                raw[test] = fn(dirty[test_rows])  # only the test rows change
                smooth = trailing_median(raw, usable)
                rows.append({**base, "fault": fault, "target": names[col], **_outcome(smooth, threshold, test,
                                                                                         usable)})
    log.info("%s %3d injected into %s", farm, event_id, [names[c] for c in targets])
    return rows


def _run_one(args) -> list[dict]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from threadpoolctl import threadpool_limits

    zip_path, conn_dir, farm, event_id, info, kw = args
    with threadpool_limits(2):
        return inject_dataset(zip_path, conn_dir, farm, event_id, info, **kw)


def run_all(zip_path: Path, conn_dir: Path, out_csv: Path, jobs: int = 1, datasets=None, **kw) -> pd.DataFrame:
    """Injection study over the given (farm, event_id) pairs (all datasets by default)."""
    from concurrent.futures import ProcessPoolExecutor

    with zipfile.ZipFile(zip_path) as z:
        info = care.event_info(z)
    todo = [(zip_path, conn_dir, r["farm"], int(r["event_id"]), r, kw) for _, r in info.iterrows()
            if datasets is None or (r["farm"], int(r["event_id"])) in datasets]
    todo.sort(key=lambda a: a[2] != "C")
    rows = []
    with ProcessPoolExecutor(max_workers=max(jobs, 1)) as ex:
        for res in ex.map(_run_one, todo):
            rows.extend(res)
    out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def dev_subset(zip_path: Path, per_farm: int = 6, seed: int = 0) -> set[tuple[str, int]]:
    """A fixed random subset of datasets for design choices."""
    with zipfile.ZipFile(zip_path) as z:
        info = care.event_info(z)
    rng = np.random.default_rng(seed)
    out = set()
    for farm, g in info.groupby("farm"):
        ids = np.sort(g["event_id"].to_numpy())
        out |= {(farm, int(e)) for e in rng.choice(ids, size=min(per_farm, len(ids)), replace=False)}
    return out
