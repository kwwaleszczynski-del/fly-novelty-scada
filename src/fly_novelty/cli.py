"""Command line interface.

Usage:
    python -m fly_novelty fetch                  # FlyWire cell annotations (27 MB) and connections (852 MB)
    python -m fly_novelty fetch-care             # CARE to Compare benchmark (5.5 GB, CC BY-SA 4.0)
    python -m fly_novelty connectome             # PN -> KC wiring tables in connectome/ (committed)
    python -m fly_novelty care --jobs 5          # every method on the 95 CARE datasets, results/care/
    python -m fly_novelty inject --jobs 5        # synthetic faults in the training sections (label-free)
    python -m fly_novelty inject --dev --jobs 6  # the development study on the number of tuning curves
    python -m fly_novelty service --jobs 5       # change of normal vs slow fault for four learning policies
    python -m fly_novelty cost                   # memory and time per sample on one dataset
    python -m fly_novelty replay --farm B --event 7    # one prediction section streamed live, alarms logged
    python -m fly_novelty report                 # README figures and the dashboard in docs/
    python -m fly_novelty all --jobs 5           # care, inject, service, cost, replay, report

The CARE archive is read from $CARE_ZIP if set, else data/care/raw/CARE_To_Compare.zip; the FlyWire files from
$FLYWIRE_DATA_DIR if set, else data/flywire.
"""
from __future__ import annotations

import argparse
import logging
import os
import zipfile
from pathlib import Path

import numpy as np

log = logging.getLogger("fly_novelty")
COMMANDS = ("fetch", "fetch-care", "connectome", "care", "inject", "service", "cost", "replay", "report", "all")
REPLAY_DEFAULT = ("B", 7)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="fly-novelty",
                                description="Fruit fly mushroom body novelty detection on wind turbine SCADA.")
    p.add_argument("command", choices=COMMANDS)
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="downloads, caches, score files")
    p.add_argument("--conn-dir", type=Path, default=Path("connectome"), help="derived PN -> KC wiring tables")
    p.add_argument("--results-dir", type=Path, default=Path("results"), help="tables committed with the repo")
    p.add_argument("--docs-dir", type=Path, default=Path("docs"), help="dashboard and README figures")
    p.add_argument("--jobs", type=int, default=1, help="parallel processes (one CARE dataset each)")
    p.add_argument("--dev", action="store_true", help="inject: development subset, 4/8/12 tuning curves")
    p.add_argument("--farm", default=REPLAY_DEFAULT[0], help="replay: wind farm A, B or C")
    p.add_argument("--event", type=int, default=REPLAY_DEFAULT[1], help="replay: event id")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def care_zip(args) -> Path:
    env = os.environ.get("CARE_ZIP")
    return Path(env) if env else args.data_dir / "care" / "raw" / "CARE_To_Compare.zip"


def flywire_dir(args) -> Path:
    env = os.environ.get("FLYWIRE_DATA_DIR")
    return Path(env) if env else args.data_dir / "flywire"


def cmd_fetch(args) -> None:
    from .connectome import fetch
    fetch(flywire_dir(args))


def cmd_fetch_care(args) -> None:
    import hashlib

    import requests

    raw = care_zip(args)
    raw.parent.mkdir(parents=True, exist_ok=True)
    info = next(f for f in requests.get("https://zenodo.org/api/records/15846963", timeout=60).json()["files"]
                if f["key"].endswith(".zip"))
    done = raw.stat().st_size if raw.exists() else 0
    if done < info["size"]:
        headers = {"Range": f"bytes={done}-"} if done else {}
        with requests.get(info["links"]["self"], headers=headers, stream=True, timeout=120) as r, \
                raw.open("ab") as fh:
            r.raise_for_status()
            for chunk in r.iter_content(1 << 22):
                fh.write(chunk)
    h = hashlib.md5()
    with raw.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    algo, _, digest = info["checksum"].partition(":")
    if algo == "md5" and h.hexdigest() != digest:
        raise IOError(f"{raw.name}: MD5 mismatch, delete the file and fetch again")


def cmd_connectome(args) -> None:
    from .connectome import ANNOTATIONS_FILE, CONNECTIONS_FILE, build
    d = flywire_dir(args)
    build(d / CONNECTIONS_FILE, d / ANNOTATIONS_FILE, args.conn_dir)


def cmd_care(args) -> None:
    from . import care

    scores = args.data_dir / "care" / "scores"
    out = args.results_dir / "care"
    out.mkdir(parents=True, exist_ok=True)
    meta = care.run_all(care_zip(args), args.conn_dir, scores, jobs=args.jobs)
    meta.sort_values(["farm", "event_id"]).to_csv(out / "datasets.csv", index=False)
    events, summary = care.summarise(scores, quantiles=care.QUANTILES)
    summary.to_csv(out / "care_scores.csv", index=False)
    events[events["quantile"] == care.SCORE_QUANTILE].to_csv(out / "care_events.csv", index=False)
    main = summary[(summary["farm"] == "all") & (summary["quantile"] == care.SCORE_QUANTILE)]
    log.info("CARE scores:\n%s", main.round(3).to_string())


def cmd_inject(args) -> None:
    from . import inject

    if args.dev:
        dev = inject.dev_subset(care_zip(args))
        inject.run_all(care_zip(args), args.conn_dir, args.results_dir / "care" / "dev_tuning_curves.csv",
                       jobs=args.jobs, datasets=dev, kinds=("flywire", "flyhash"), seeds=(0,),
                       n_bins_options=(4, 8, 12), residual_kinds=())
        inject.run_all(care_zip(args), args.conn_dir, args.results_dir / "care" / "dev_recovery.csv",
                       jobs=args.jobs, datasets=dev, kinds=("flywire", "flyhash"), seeds=(0,),
                       n_bins_options=(4, 8), eps_options=(0.0, 3e-5), residual_kinds=(), with_baselines=False)
        out = inject.run_all(care_zip(args), args.conn_dir, args.results_dir / "care" / "dev_residual.csv",
                             jobs=args.jobs, datasets=dev, kinds=(), seeds=(0,),
                             residual_kinds=("flywire", "flyhash"), with_baselines=False)
    else:
        out = inject.run_all(care_zip(args), args.conn_dir, args.results_dir / "care" / "injection.csv",
                             jobs=args.jobs)
    log.info("injection study:\n%s",
             out.groupby(["method", "fault"])["detected"].mean().unstack().round(3).to_string())


def cmd_service(args) -> None:
    from .operational import run_service

    out = run_service(care_zip(args), args.conn_dir, args.results_dir / "care" / "service.csv", jobs=args.jobs)
    log.info("service study:\n%s", out.groupby(["scenario", "policy"])[["detected", "flagged_share"]].mean()
             .round(3).to_string())


def cmd_cost(args) -> None:
    from . import care, connectome
    from .operational import cost_table

    farm, event_id = "B", 34
    with zipfile.ZipFile(care_zip(args)) as z:
        info = care.event_info(z)
        info = info[(info["farm"] == farm) & (info["event_id"] == event_id)].iloc[0]
        cand = care.state_signals(z, farm)
        df = care.load_dataset(z, farm, event_id, cand["sensor"].tolist())
    m = care.masks(df, farm, info)
    sensors = [s for s in cand["sensor"] if s in df]
    X = df[sensors].to_numpy(float)
    X = X[:, care.select_signals(X[m["fit"]])]
    table = cost_table(X[m["fit"]], X[m["cal"]], connectome.load(args.conn_dir))
    table.insert(0, "dataset", f"{farm}-{event_id}")
    table.insert(1, "signals", X.shape[1])
    out = args.results_dir / "care" / "cost.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    log.info("cost per method:\n%s", table.round(2).to_string())


def cmd_replay(args) -> None:
    from .operational import replay
    replay(care_zip(args), args.conn_dir, args.farm, args.event,
           args.data_dir / "replay" / f"{args.farm}_{args.event}.json")


def cmd_report(args) -> None:
    from .report import report
    report(args.data_dir, args.conn_dir, args.results_dir, args.docs_dir, replay_key=f"{args.farm}_{args.event}",
           zip_path=care_zip(args))


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    np.seterr(invalid="ignore")
    if args.command == "all":
        for stage in ("care", "inject", "service", "cost", "replay", "report"):
            log.info("== %s", stage)
            globals()[f"cmd_{stage}"](args)
        return
    globals()[f"cmd_{args.command.replace('-', '_')}"](args)


if __name__ == "__main__":
    main()
