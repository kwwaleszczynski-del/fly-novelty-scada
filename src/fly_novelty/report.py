"""Figures for the README and the interactive dashboard (``docs/index.html``).

Figures derived from CARE to Compare go to ``results/care`` (CC BY-SA 4.0), the wiring figure (FlyWire, CC BY 4.0)
to ``docs/img``. The dashboard embeds its data as JSON in a static page (Plotly from a CDN).

Usage:
    report(data_dir=Path("data"), conn_dir=Path("connectome"), results_dir=Path("results"), docs_dir=Path("docs"))
"""
from __future__ import annotations

import json
import logging
import textwrap
from importlib import resources
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from . import care, connectome  # noqa: E402
from .expansion import EXPANSIONS, degree_preserving_shuffle, flyhash_matrix  # noqa: E402

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
LABELS = {"fly_flywire": "Fly, FlyWire wiring", "fly_shuffled": "Fly, shuffled wiring",
          "fly_flyhash": "Fly, FlyHash random", "fly_dense": "Fly, dense random",
          "fly_flyhash_full": "Fly, FlyHash no PN layer", "fly_res_flywire": "Fly on PCA residuals, FlyWire",
          "fly_res_shuffled": "Fly on PCA residuals, shuffled", "fly_res_flyhash": "Fly on PCA residuals, FlyHash",
          "zscore": "Robust z-score", "pca": "PCA residual",
          "iforest": "Isolation forest", "autoencoder": "Autoencoder", "knn": "k nearest neighbours",
          "stream_flywire_gated": "Fly streaming, gated", "stream_flywire_ungated": "Fly streaming, always learns",
          "stream_flyhash_gated": "Fly streaming FlyHash, gated",
          "stream_flyhash_ungated": "Fly streaming FlyHash, always learns",
          "nbm_rc": "RC thermal model (temps only)", "nbm_gbm": "Gradient boosting NBM (temps only)"}
COLORS = {"fly_flywire": "#2a78d6", "fly_shuffled": "#8a63d2", "fly_flyhash": "#1baf7a", "fly_dense": "#eda100",
          "fly_flyhash_full": "#e87ba4", "fly_res_flywire": "#0d4ea3", "fly_res_shuffled": "#5b3aa6",
          "fly_res_flyhash": "#0f7a55", "stream_flywire_gated": "#2a78d6", "stream_flywire_ungated": "#7fb0ea",
          "stream_flyhash_gated": "#1baf7a", "stream_flyhash_ungated": "#7fd3b4", "pca": "#eb6834",
          "autoencoder": "#8c6d31", "knn": "#d03b3b", "iforest": "#7d7b75", "zscore": "#b5b3ab", "nbm_rc": "#a8a69f",
          "nbm_gbm": "#a8a69f"}
CSS_COLORS = {"flywire": "--c1", "shuffled": "--c6", "flyhash": "--c3", "dense": "--c4", "flyhash_full": "--c5",
              "fly_flywire": "--c1", "fly_shuffled": "--c6", "fly_flyhash": "--c3", "fly_dense": "--c4",
              "fly_flyhash_full": "--c5", "fly_res_flywire": "--c1", "fly_res_shuffled": "--c6",
              "fly_res_flyhash": "--c3", "stream_flywire_gated": "--c1", "stream_flywire_ungated": "--c1",
              "stream_flyhash_gated": "--c3", "stream_flyhash_ungated": "--c3", "pca": "--c2",
              "autoencoder": "--b2", "knn": "--b3", "iforest": "--neutral", "zscore": "--neutral",
              "nbm_rc": "--neutral", "nbm_gbm": "--neutral"}
KC_BLOCK = 20
log = logging.getLogger(__name__)


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
        "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 1.6,
    })


def wiring_variants(wiring: connectome.Wiring, seed: int = 0) -> dict:
    """Binary KC x PN matrices of the real, shuffled and FlyHash wiring (as built for the detector)."""
    rng = np.random.default_rng([seed, EXPANSIONS.index("shuffled")])
    shuffled = degree_preserving_shuffle(wiring.matrix, rng)
    claws = int(round(wiring.matrix.getnnz(axis=1).mean()))
    flyhash = flyhash_matrix(wiring.n_kc, wiring.n_pn, claws,
                             np.random.default_rng([seed, EXPANSIONS.index("flyhash")]))
    return {k: (m > 0).astype(np.float32).toarray()
            for k, m in (("flywire", wiring.matrix), ("shuffled", shuffled), ("flyhash", flyhash))}


def _blocks(binary: np.ndarray) -> np.ndarray:
    """Share of the KCs of every block of ``KC_BLOCK`` connected to each PN, (PN, block)."""
    n_kc = binary.shape[0]
    edges = np.arange(0, n_kc, KC_BLOCK)
    return np.add.reduceat(binary, edges, axis=0).T / np.diff(np.r_[edges, n_kc])


def fig_wiring(wiring: connectome.Wiring, out: Path) -> dict:
    variants = wiring_variants(wiring)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), gridspec_kw={"height_ratios": [2.2, 1]})
    titles = {"flywire": "FlyWire (right mushroom body)", "shuffled": "Degree-preserving shuffle",
              "flyhash": "FlyHash (4 random PNs per KC)"}
    vmax = max(_blocks(b).max() for b in variants.values())
    classes = wiring.kc["kc_class"].to_numpy()
    for ax, (k, b) in zip(axes[0], variants.items()):
        ax.imshow(_blocks(b), aspect="auto", cmap="Blues", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(titles[k])
        ax.grid(False)
        for c in np.flatnonzero(classes[1:] != classes[:-1]) + 1:
            ax.axvline(c / KC_BLOCK - 0.5, color=INK2, lw=0.8, ls=":")
        ax.set_xlabel("KC blocks of 20 (a'/b' | a/b | gamma)")
        ax.set_ylabel("projection neuron")
    real_deg = np.bincount(variants["flywire"].sum(axis=1).astype(int))
    axes[1, 0].bar(np.arange(len(real_deg)), real_deg, color=COLORS["fly_flywire"],
                   label="FlyWire = shuffled")
    fh = int(variants["flyhash"].sum(axis=1)[0])
    axes[1, 0].annotate(f"FlyHash: all {wiring.n_kc:,} KCs have {fh}", xy=(fh, real_deg[fh]),
                        xytext=(6.2, real_deg.max() * 0.9), color=COLORS["fly_flyhash"], fontsize=8,
                        arrowprops={"arrowstyle": "->", "color": COLORS["fly_flyhash"]})
    for k, b in variants.items():
        out_deg = np.sort(b.sum(axis=0))[::-1]
        axes[1, 1].plot(np.arange(1, len(out_deg) + 1), out_deg, color=COLORS[f"fly_{k}"],
                        ls="--" if k == "shuffled" else "-", label=titles[k].split(" (")[0])
    axes[1, 0].set_xlabel("PN partners per KC")
    axes[1, 0].set_ylabel("KCs")
    axes[1, 0].set_ylim(0, real_deg.max() * 1.18)  # headroom: the legend sat on the tallest bar
    axes[1, 0].legend(fontsize=8, loc="upper left")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_xlabel("PN rank")
    axes[1, 1].set_ylabel("KC partners per PN")
    syn = wiring.matrix.data
    axes[1, 2].hist(syn, bins=np.arange(5, syn.max() + 2, 2), color=COLORS["fly_flywire"])
    axes[1, 2].set_xlabel("synapses per PN-KC connection (FlyWire)")
    axes[1, 2].set_ylabel("connections")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    pn = wiring.pn
    labels = [f"{g} {i}" for i, g in zip(pn["node"], pn["glomerulus"].str.slice(0, 14))]
    edges = np.flatnonzero(classes[1:] != classes[:-1]) + 1
    bounds = np.r_[0, edges, len(classes)] / KC_BLOCK
    names = {"alpha'/beta'": "α'/β'", "alpha/beta": "α/β", "gamma": "γ"}
    ticks = [[float((a + b) / 2), names.get(classes[int(a * KC_BLOCK)], "")] for a, b in zip(bounds[:-1], bounds[1:])]
    return {"matrices": {k: np.round(_blocks(b), 3).tolist() for k, b in variants.items()},
            "labels": {"flywire": "FlyWire (real)", "shuffled": "Shuffled (degrees kept)", "flyhash": "FlyHash"},
            "pn_labels": labels, "kc_bins": list(range(_blocks(variants["flywire"]).shape[1])),
            "class_edges": [float(e / KC_BLOCK) for e in edges], "class_ticks": ticks, "zmax": float(vmax),
            "in_degree": {k: np.bincount(b.sum(axis=1).astype(int)).tolist() for k, b in variants.items()},
            "out_degree": {k: np.sort(b.sum(axis=0))[::-1].astype(int).tolist() for k, b in variants.items()}}


def _variant(method: str) -> str:
    """``fly_flywire_s1`` -> ``fly_flywire``; other names unchanged."""
    return method.rsplit("_s", 1)[0] if method.startswith("fly_") and method[-1].isdigit() else method


def method_table(summary: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per method (fly variants averaged over seeds) at the main threshold, all farms."""
    s = summary[(summary["farm"] == "all") & (summary["quantile"] == care.SCORE_QUANTILE)].copy()
    s = s[~s["method"].str.startswith("grid_")]
    s["key"] = s["method"].map(_variant)
    rows = []
    for key, g in s.groupby("key", sort=False):
        rows.append({"key": key, "label": LABELS.get(key, key), "care": g["care"].mean(),
                     "care_min": g["care"].min(), "care_max": g["care"].max(),
                     "seeds": g["care"].round(4).tolist() if len(g) > 1 else [],
                     "detected": g["events_detected"].mean(), "normal_flagged": g["normal_events_flagged"].mean(),
                     "lead": g["median_lead_days"].mean(), "false_days": g["false_flag_days_per_dataset"].mean(),
                     "coverage": g["coverage"].mean(), "accuracy": g["accuracy"].mean(),
                     "reliability": g["reliability"].mean(), "earliness": g["earliness"].mean(),
                     "skill": (g["events_detected"] / g["anomaly_events"]
                               - g["normal_events_flagged"] / g["normal_events"]).mean()})
    if reference is not None:
        for _, r in reference[reference["farm"] == "all"].iterrows():
            rows.append({"key": r["method"], "label": LABELS.get(r["method"], r["method"]), "care": r["care"],
                         "care_min": r["care"], "care_max": r["care"], "seeds": [],
                         "detected": r["events_detected"], "normal_flagged": r["normal_events_flagged"],
                         "lead": r["median_lead_days"], "false_days": np.nan, "coverage": r["coverage"],
                         "accuracy": r["accuracy"], "reliability": r["reliability"], "earliness": r["earliness"],
                         "skill": r["events_detected"] / r["anomaly_events"]
                         - r["normal_events_flagged"] / r["normal_events"]})
    order = {k: i for i, k in enumerate(LABELS)}
    return pd.DataFrame(rows).sort_values("key", key=lambda c: c.map(order)).reset_index(drop=True)


def bootstrap_difference(events: pd.DataFrame, a: list[str], b: list[str], n: int = 2000,
                         seed: int = 0) -> dict:
    """CARE score of methods ``a`` minus methods ``b`` (each averaged over its seeds) with a 95 % interval from
    resampling the datasets (anomaly and normal datasets resampled separately)."""
    from .care_score import WEIGHTS, fbeta

    rng = np.random.default_rng(seed)
    keys = events[["farm", "event_id", "label"]].drop_duplicates().reset_index(drop=True)
    anomaly = keys["label"].to_numpy() == "anomaly"
    cols = ("detected", "coverage", "accuracy", "earliness")
    arrays = {}  # method -> per-dataset arrays in the order of ``keys``
    for m in a + b:
        g = events[events["method"] == m].set_index(["farm", "event_id"]).reindex(
            pd.MultiIndex.from_frame(keys[["farm", "event_id"]]))
        arrays[m] = {c: g[c].to_numpy(dtype=float) for c in cols}

    def care(m, rows):
        """:func:`care_score.care_score` on the datasets ``rows`` (with repeats), from arrays."""
        x, anom = arrays[m], anomaly[rows]
        det = x["detected"][rows] > 0
        tp, fp, fn = np.sum(det & anom), np.sum(det & ~anom), np.sum(~det & anom)
        acc = x["accuracy"][rows][~anom].mean()
        if tp + fp == 0:
            return 0.0
        if acc <= 0.5:
            return acc
        parts = {"coverage": x["coverage"][rows][anom].mean(), "earliness": x["earliness"][rows][anom].mean(),
                 "reliability": fbeta(tp, fp, fn), "accuracy": acc}
        return sum(WEIGHTS[k] * parts[k] for k in WEIGHTS) / sum(WEIGHTS.values())

    def score(methods, rows):
        return np.mean([care(m, rows) for m in methods])

    every = np.arange(len(keys))
    groups = [np.flatnonzero(anomaly), np.flatnonzero(~anomaly)]
    point = score(a, every) - score(b, every)
    diffs = []
    for _ in range(n):
        rows = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        diffs.append(score(a, rows) - score(b, rows))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"a": "+".join(a), "b": "+".join(b), "difference": float(point), "ci_low": float(lo),
            "ci_high": float(hi), "share_above_zero": float(np.mean(np.array(diffs) > 0))}


def fig_care(methods: pd.DataFrame, out: Path) -> None:
    m = methods[~methods["key"].str.startswith("stream_")].iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(m) + 1.2))
    colors = [COLORS.get(k, "#a8a69f") for k in m["key"]]
    ax.barh(m["label"], m["care"], color=colors)
    for i, (lo, hi) in enumerate(zip(m["care_min"], m["care_max"])):
        if hi > lo:
            ax.plot([lo, hi], [i, i], color=INK, lw=1.2)
    # values in a fixed column right of every bar, whisker and the 0.5 line, so nothing crosses them
    col = max(0.55, float(np.nanmax(np.r_[m["care"], m["care_max"]])) + 0.04)
    for i, (c, d, f) in enumerate(zip(m["care"], m["detected"], m["normal_flagged"])):
        ax.text(col, i, f"{c:.3f}   {d:.3g}/45 faults, {f:.3g}/50 normal", va="center", fontsize=8, color=INK2)
    ax.axvline(0.5, color=INK2, ls="--", lw=1)
    ax.set_xlim(0, col + 0.42)
    ax.set_xticks(np.arange(0, 0.61, 0.1))
    ax.set_xlabel("CARE score (0.5 = random detector)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_sensitivity(summary: pd.DataFrame, out: Path) -> None:
    s = summary[(summary["farm"] == "all") & (summary["quantile"] == care.SCORE_QUANTILE)].set_index("method")
    main = (care.MAIN_K, care.MAIN_DELTA, care.MAIN_EPS)

    def value(kind, k, d, e):
        name = f"fly_{kind}_s0" if (k, d, e) == main else care.grid_name(kind, k, d, e)
        return s.loc[name, "care"] if name in s.index else np.nan

    fig, axes = plt.subplots(1, len(care.GRID_KINDS) + 1, figsize=(14, 3.5),
                             gridspec_kw={"width_ratios": [1] * len(care.GRID_KINDS) + [1.3]})
    for ax, kind in zip(axes, care.GRID_KINDS):
        grid = np.array([[value(kind, k, d, care.MAIN_EPS) for d in care.DELTA_GRID] for k in care.K_GRID])
        ax.imshow(grid, cmap="Blues", vmin=0.3, vmax=0.7, origin="lower")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=9,
                        color="white" if grid[i, j] > 0.58 else INK)
        ax.set_xticks(range(len(care.DELTA_GRID)), [str(d) for d in care.DELTA_GRID])
        ax.set_yticks(range(len(care.K_GRID)), [f"{int(k * 100)} %" for k in care.K_GRID])
        ax.set_xlabel("decay per stored row (delta)")
        ax.set_title(LABELS[f"fly_{kind}"])
        ax.grid(False)
    axes[0].set_ylabel("active Kenyon cells (k)")
    eps_values = sorted(set(care.EPS_GRID) | {care.MAIN_EPS})
    x = np.arange(len(eps_values))
    for kind in care.GRID_KINDS:
        axes[-1].plot(x, [value(kind, care.MAIN_K, care.MAIN_DELTA, e) for e in eps_values], marker="o",
                      color=COLORS[f"fly_{kind}"], label=LABELS[f"fly_{kind}"])
    axes[-1].set_xticks(x, ["0 (none)" if e == 0 else f"{e:.0e}" for e in eps_values])
    axes[-1].set_xlabel("recovery per stored row (eps)")
    axes[-1].set_ylabel("CARE score")
    axes[-1].axhline(0.5, color=INK2, ls="--", lw=1)
    axes[-1].legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


FAULT_LABELS = {"none": "clean (false alarm)", "step_5K": "step +5 K", "step_10K": "step +10 K",
                "drift_10K": "drift to +10 K", "stuck": "frozen sensor"}


def injection_table(inj: pd.DataFrame) -> pd.DataFrame:
    """Detected share per method variant (seeds pooled) and fault."""
    inj = inj.assign(key=inj["method"].map(_variant))
    return inj.groupby(["key", "fault"])["detected"].mean().unstack()[list(FAULT_LABELS)]


def fig_injection(inj: pd.DataFrame, out: Path) -> pd.DataFrame:
    tab = injection_table(inj)
    order = [k for k in LABELS if k in tab.index]
    tab = tab.loc[order]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    width = 0.8 / len(tab)
    x = np.arange(len(FAULT_LABELS))
    for i, (key, row) in enumerate(tab.iterrows()):
        ax.bar(x + (i - len(tab) / 2 + 0.5) * width, row.to_numpy(), width, color=COLORS.get(key, "#a8a69f"),
               label=LABELS.get(key, key), alpha=1.0 if key.startswith("fly_") else 0.55,
               hatch=None if key.startswith("fly_") else "//", edgecolor=SURFACE)
    ax.set_xticks(x, list(FAULT_LABELS.values()))
    ax.set_ylabel("detected (12 h of alarms)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return tab


def injection_skill(inj: pd.DataFrame) -> pd.DataFrame:
    """Per dataset and method variant (seeds pooled): share of the injected faults detected minus the share of the
    clean runs with a detection."""
    inj = inj.assign(key=inj["method"].map(_variant))
    clean = inj[inj["fault"] == "none"].groupby(["key", "farm", "event_id"])["detected"].mean()
    faulty = inj[inj["fault"] != "none"].groupby(["key", "farm", "event_id"])["detected"].mean()
    return (faulty - clean).rename("skill").reset_index()


def injection_differences(inj: pd.DataFrame, pairs: list[tuple[str, str]], n: int = 2000,
                          seed: int = 0) -> pd.DataFrame:
    """Mean per-dataset skill of variant ``a`` minus ``b`` with a 95 % interval from resampling the datasets."""
    skill = injection_skill(inj).pivot_table(index=["farm", "event_id"], columns="key", values="skill")
    rng = np.random.default_rng(seed)
    rows = []
    for a, b in pairs:
        d = (skill[a] - skill[b]).dropna().to_numpy()
        boots = [rng.choice(d, len(d), replace=True).mean() for _ in range(n)]
        lo, hi = np.quantile(boots, [0.025, 0.975])
        rows.append({"a": a, "b": b, "datasets": len(d), "skill_a": float(skill[a].mean()),
                     "skill_b": float(skill[b].mean()), "difference": float(d.mean()), "ci_low": float(lo),
                     "ci_high": float(hi), "share_of_datasets_a_better": float(np.mean(d > 0)),
                     "share_of_datasets_b_better": float(np.mean(d < 0))})
    return pd.DataFrame(rows)


POLICY_LABELS = {"frozen": "frozen", "gated": "streaming, gated", "ungated": "streaming, always learns",
                 "gated_ack": "gated + operator ack."}


SCENARIO_LABELS = {"none": "no change (control)", "new_normal": "change of normal (+5 K step)",
                   "fault": "slow fault (drift to +10 K)"}


def fig_service(svc: pd.DataFrame, out: Path) -> pd.DataFrame:
    tab = svc.groupby(["scenario", "policy"]).agg(
        flagged=("flagged_share", "mean"), detected=("detected", "mean"),
        cleared_days=("cleared_after_days", "median"), n=("detected", "size")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    colors = ["#a8a69f", "#2a78d6", "#eb6834", "#1baf7a"]
    x = np.arange(len(SCENARIO_LABELS))
    width = 0.8 / len(POLICY_LABELS)
    for ax, col, ylabel in zip(axes, ("detected", "flagged"), ("alarm within the test half (12 h net)",
                                                               "share of time in alarm")):
        for i, (policy, label) in enumerate(POLICY_LABELS.items()):
            g = tab[tab["policy"] == policy].set_index("scenario").reindex(list(SCENARIO_LABELS))
            ax.bar(x + (i - 1.5) * width, g[col], width, color=colors[i], label=label)
        ax.set_xticks(x, [v.replace(" (", "\n(") for v in SCENARIO_LABELS.values()], fontsize=8.5)
        ax.set_ylim(0, 1)
        ax.set_ylabel(ylabel)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return tab


EVENT_METHODS = ("fly_flywire_s0", "fly_shuffled_s0", "fly_flyhash_s0", "stream_flywire_gated", "pca",
                 "autoencoder", "knn", "zscore")


def event_flags(scores_dir: Path, farm: str, event_id: int, methods=EVENT_METHODS) -> pd.DataFrame:
    """Hourly share of the prediction section's producing rows that each method flags (alarm score above its
    calibration threshold, as in the evaluation)."""
    s = pd.read_parquet(scores_dir / f"{farm}_{event_id}.parquet")
    pred = s["pred"].to_numpy()
    out = {}
    for m in methods:
        thr = care._threshold(s[m].to_numpy(float), s["cal"].to_numpy())
        flags = (s[m].to_numpy(float) > thr).astype(float)
        out[m] = np.where(s["usable"].to_numpy(), flags, np.nan)
    frame = pd.DataFrame(out, index=s.index)[pred]
    hourly = frame.resample("1h").mean()
    window = s.index[s["in_window"].to_numpy() & pred]
    hourly.attrs["window"] = (window[0], window[-1]) if len(window) else None
    return hourly


def fig_events(scores_dir: Path, picks: list[tuple[str, int, str]], out: Path) -> None:
    fig, axes = plt.subplots(len(picks), 1, figsize=(10, 2.3 * len(picks)))
    labels = [LABELS.get(_variant(m), m) for m in EVENT_METHODS]
    for ax, (farm, event_id, title) in zip(np.atleast_1d(axes), picks):
        h = event_flags(scores_dir, farm, event_id)
        x = matplotlib.dates.date2num(h.index.to_pydatetime())
        ax.imshow(h.T.to_numpy(), aspect="auto", cmap="Reds", vmin=0, vmax=1, interpolation="nearest",
                  extent=[x[0], x[-1] + 1 / 24, len(EVENT_METHODS) - 0.5, -0.5])
        if h.attrs["window"] is not None:
            w = matplotlib.dates.date2num([h.attrs["window"][0].to_pydatetime(), h.attrs["window"][1].to_pydatetime()])
            for edge in w:
                ax.axvline(edge, color=INK, lw=1.2, ls="--")
        ax.xaxis_date()
        loc = matplotlib.dates.AutoDateLocator(maxticks=8)  # fewer, shorter dates: they overlapped
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(matplotlib.dates.ConciseDateFormatter(loc))
        ax.set_yticks(range(len(EVENT_METHODS)), labels, fontsize=8)
        ax.grid(False)
        ax.set_title(title, loc="left")
    np.atleast_1d(axes)[-1].set_xlabel("prediction section; dashed: labelled window before the fault; red: alarm")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


CSS_FALLBACK = {"pca": "#eb6834", "autoencoder": "#8c6d31", "knn": "#d03b3b", "iforest": "#7d7b75",
                "zscore": "#a8a69f"}


def mechanism_trace(zip_path: Path, conn_dir: Path, farm: str, event_id: int) -> pd.DataFrame:
    """Why the frozen fly stays quiet: per row of the calibration and prediction span, how often the rarest KC of
    the row's tag was active while storing the training year, next to the PCA residual against its threshold."""
    import zipfile

    from . import baselines
    from .encoding import PopulationEncoder
    from .expansion import build
    from .novelty import tags, trailing_median

    with zipfile.ZipFile(zip_path) as z:
        info = care.event_info(z)
        info = info[(info["farm"] == farm) & (info["event_id"] == event_id)].iloc[0]
        cand = care.state_signals(z, farm)
        df = care.load_dataset(z, farm, event_id, cand["sensor"].tolist())
    m = care.masks(df, farm, info)
    sensors = [s for s in cand["sensor"] if s in df]
    X = df[sensors].to_numpy(float)
    X = X[:, care.select_signals(X[m["fit"]])]
    wiring = connectome.load(conn_dir)
    k = max(1, int(round(care.MAIN_K * wiring.n_kc)))
    enc = PopulationEncoder().fit(X[m["fit"]])
    exp = build("flywire", wiring, enc.n_channels, 0)
    exp.fit_centre(enc.transform(X[m["fit"]]))
    span = np.arange(np.flatnonzero(m["cal"])[0], len(df))
    fit_tags, fit_valid = tags(exp, enc.transform(X[m["fit"]]), k)
    counts = np.bincount(fit_tags[fit_valid].ravel(), minlength=wiring.n_kc)
    t_rows, valid = tags(exp, enc.transform(X[span]), k)
    rarest = np.where(valid, counts[t_rows].min(axis=1), np.nan)
    usable = m["usable"][span]
    pca = baselines.PCAModel().fit(X[m["fit"]])
    spe = trailing_median(pca.score(X[span]), usable)
    thr = care._threshold(spe, m["cal"][span])
    frame = pd.DataFrame({"rarest": np.where(usable, rarest, np.nan), "pca": spe / thr,
                          "pred": m["pred"][span], "window": m["in_window"][span] & m["pred"][span]},
                         index=df.index[span])
    frame.attrs.update(farm=farm, event_id=event_id, description=str(info.get("event_description", "") or ""),
                       usage_median=float(np.median(counts[counts > 0])), never_used=int((counts == 0).sum()),
                       fit_rows=int(fit_valid.sum()))
    return frame


def fig_mechanism(trace: pd.DataFrame, out: Path) -> None:
    h = trace.resample("6h").agg({"rarest": "median", "pca": "max", "pred": "max", "window": "max"})
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    pred = h.index[h["pred"].astype(bool)]
    win = h.index[h["window"].astype(bool)]
    for ax in axes:
        if len(pred):
            ax.axvline(pred[0], color=INK2, lw=1, ls=":")
        if len(win):
            ax.axvspan(win[0], win[-1], color="#d03b3b", alpha=0.10, lw=0)
    axes[0].plot(h.index, h["pca"], color=CSS_FALLBACK["pca"])
    axes[0].axhline(1.0, color=INK2, ls="--", lw=1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("PCA residual / threshold")
    axes[1].plot(h.index, h["rarest"], color=COLORS["fly_flywire"])
    axes[1].axhline(60, color=INK2, ls="--", lw=1)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("uses of the rarest KC\nin the tag (training)")
    a = trace.attrs
    axes[0].set_title(textwrap.fill(f"Wind farm {a['farm']}, dataset {a['event_id']}: {a['description']}", 95),
                      loc="left")
    axes[1].text(0.01, 0.05, "below the dashed line (about 60 uses, 0.5^60 = 1e-18) the novelty would leave zero",
                 transform=axes[1].transAxes, fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def pick_events(events: pd.DataFrame) -> list[tuple[str, int, str]]:
    """Example datasets, chosen by a fixed rule: faults caught by the real-wiring fly detector (earliest first),
    faults it missed that PCA or the autoencoder caught, and normal datasets with and without its false alarms."""
    e = events.set_index(["farm", "event_id", "method"])
    fly = events[events["method"] == "fly_flywire_s0"]
    anomaly = fly[fly["label"] == "anomaly"]
    caught = anomaly[anomaly["detected"]].sort_values("lead_steps", ascending=False)
    picks = [(r.farm, int(r.event_id), "caught") for r in caught.head(4).itertuples()]
    missed = anomaly[~anomaly["detected"]]
    others = [r for r in missed.itertuples()
              if any(e.loc[(r.farm, r.event_id, m), "detected"] for m in ("pca", "autoencoder"))]
    picks += [(r.farm, int(r.event_id), "missed") for r in others[:2]]
    normal = fly[fly["label"] == "normal"].sort_values(["farm", "event_id"])
    flagged = normal[normal["detected"]]
    picks += [(r.farm, int(r.event_id), "false alarm") for r in flagged.head(1).itertuples()]
    picks += [(r.farm, int(r.event_id), "quiet") for r in normal[~normal["detected"]].head(1).itertuples()]
    return picks


def _title(events: pd.DataFrame, farm: str, event_id: int, kind: str) -> tuple[str, str]:
    r = events[(events["farm"] == farm) & (events["event_id"] == event_id)].iloc[0]
    desc = r["event_description"] if isinstance(r["event_description"], str) and r["event_description"] else ""
    title = f"Wind farm {farm}, dataset {event_id}: {r['label']} ({kind})"
    if r["label"] == "anomaly":
        sub = f"Logged root cause: {desc}" if desc else "No root cause given."
    else:
        sub = "Normal behaviour: every alarm is a false alarm."
    return title, sub


def _json_list(values) -> list:
    return [None if v is None or not np.isfinite(v) else round(float(v), 4) for v in values]


def dashboard_data(wiring_data: dict, methods: pd.DataFrame, events: pd.DataFrame, scores_dir: Path,
                   inj_tab: pd.DataFrame | None, cost: pd.DataFrame | None, svc_tab: pd.DataFrame | None,
                   replay: dict | None) -> dict:
    ev_list = []
    for farm, event_id, kind in pick_events(events):
        h = event_flags(scores_dir, farm, event_id)
        title, sub = _title(events, farm, event_id, kind)
        window = h.attrs["window"]
        ev_list.append({"title": title, "subtitle": sub, "t": [str(t) for t in h.index],
                        "window": [str(w) for w in window] if window is not None else None,
                        "methods": [LABELS.get(_variant(m), m) for m in EVENT_METHODS],
                        "z": [_json_list(h[m].to_numpy()) for m in EVENT_METHODS]})
    fly = methods[methods["key"].str.startswith("fly_")]
    base = methods[~methods["key"].str.startswith(("fly_", "stream_", "nbm_"))]
    best_fly, best_base = fly.loc[fly["care"].idxmax()], base.loc[base["care"].idxmax()]
    by_key = methods.set_index("key")
    tiles = [{"value": f"{best_fly['care']:.3f}", "label": f"best fly CARE score ({best_fly['label']})"},
             {"value": f"{best_base['care']:.3f}", "label": f"best standard detector ({best_base['label']})"},
             {"value": f"{by_key.loc['fly_flywire', 'care']:.3f} vs {by_key.loc['fly_shuffled', 'care']:.3f}",
              "label": "CARE score, real vs shuffled wiring"}]
    if cost is not None:
        fly_cost = cost[cost["method"].str.startswith("fly")].iloc[0]
        tiles.append({"value": f"{fly_cost['learned_bytes'] / 1e3:.0f} kB, {fly_cost['single_us']:.0f} µs",
                      "label": "learned state, time per sample (score + learn)"})
    records = [{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()}
               for r in methods.to_dict("records")]
    data = {"tiles": tiles, "wiring": wiring_data, "events": ev_list, "colors": CSS_COLORS, "methods": records}
    if inj_tab is not None:
        keys = [k for k in ("fly_flywire", "fly_shuffled", "fly_flyhash", "fly_dense", "fly_flyhash_full",
                            "fly_res_flywire", "fly_res_shuffled", "fly_res_flyhash", "pca", "autoencoder", "knn",
                            "iforest", "zscore") if k in inj_tab.index]
        data["injection"] = {"fault_labels": list(FAULT_LABELS.values()),
                             "methods": [{"key": k, "label": LABELS.get(k, k),
                                          "values": _json_list(inj_tab.loc[k].to_numpy())} for k in keys]}
    if cost is not None:
        data["cost"] = cost.to_dict("records")
    if svc_tab is not None:
        values = {}
        for p in POLICY_LABELS:
            values[p] = [float(svc_tab[(svc_tab["scenario"] == s) & (svc_tab["policy"] == p)]["detected"].iloc[0])
                         for s in SCENARIO_LABELS]
        data["service"] = {"policies": list(POLICY_LABELS), "policy_labels": list(POLICY_LABELS.values()),
                           "scenario_labels": list(SCENARIO_LABELS.values()), "values": values}
    if replay is not None:
        data["replay"] = replay
    return data


def replay_view(trace: dict) -> dict:
    """The replay trace prepared for the dashboard animation."""
    window = None
    inside = [i for i, w in enumerate(trace["in_window"]) if w]
    if inside:
        window = [trace["t"][inside[0]], trace["t"][inside[-1]]]
    offset = int(np.searchsorted(pd.to_datetime(pd.Series(trace["t"])).to_numpy(),
                                 pd.Timestamp(trace["raster_start"]).to_datetime64()))
    smooth = [v for v in trace["smooth"] if v is not None]
    desc = trace["description"] or "no root cause given"
    caption = (f"Wind farm {trace['farm']}, dataset {trace['event_id']} ({trace['label']}; {desc}). The fly "
               "detector (FlyWire wiring) is fitted on the training year, takes its threshold from the calibration "
               "rows and is then fed one 10-minute sample at a time; it keeps learning while not in alarm. Bottom: "
               "the Kenyon cells of the hourly tags whose output synapse is still strong (weight above 0.05; "
               "red = 1, untouched), the part of the tag that says 'new'.")
    return {**trace, "window": window, "raster_offset": offset, "caption": caption,
            "ymax": max(smooth + [trace["threshold"]]) * 1.1,
            "raster_days": round(trace["raster_hours"] / 24, 1)}


def report(data_dir: Path, conn_dir: Path, results_dir: Path, docs_dir: Path, replay_key: str | None = None,
           zip_path: Path | None = None) -> None:
    _style()
    care_dir = results_dir / "care"
    scores_dir = data_dir / "care" / "scores"
    (docs_dir / "img").mkdir(parents=True, exist_ok=True)
    wiring = connectome.load(conn_dir)
    wiring_data = fig_wiring(wiring, docs_dir / "img" / "pn_kc_wiring.png")
    summary = pd.read_csv(care_dir / "care_scores.csv")
    events = pd.read_csv(care_dir / "care_events.csv")
    ref_file = care_dir / "reference_wind_thermal_nbm.csv"
    methods = method_table(summary, pd.read_csv(ref_file) if ref_file.exists() else None)
    methods.to_csv(care_dir / "methods.csv", index=False)
    seeds = lambda kind: [f"fly_{kind}_s{s}" for s in care.SEEDS]  # noqa: E731
    pairs = [(seeds("flywire"), seeds("shuffled")), (seeds("flywire"), seeds("flyhash")),
             (seeds("flywire"), seeds("dense")), (seeds("flyhash_full"), seeds("flyhash")),
             (seeds("flywire"), ["pca"]), (seeds("flywire"), ["autoencoder"]),
             (["stream_flywire_gated"], seeds("flywire")), (["stream_flywire_ungated"], ["stream_flywire_gated"]),
             (seeds("res_flywire"), seeds("res_shuffled")), (seeds("res_flywire"), seeds("res_flyhash")),
             (seeds("res_flywire"), ["pca"]), (seeds("res_flywire"), seeds("flywire"))]
    pd.DataFrame([bootstrap_difference(events, a, b) for a, b in pairs]).to_csv(
        care_dir / "differences.csv", index=False)
    fig_care(methods, care_dir / "care_methods.png")
    fig_sensitivity(summary, care_dir / "sensitivity.png")
    picks = pick_events(events)
    fig_events(scores_dir, [(f, e, _title(events, f, e, k)[0]) for f, e, k in picks[:3]], care_dir / "examples.png")
    missed = [(f, e) for f, e, k in picks if k == "missed"]
    if zip_path is not None and zip_path.exists() and missed:
        trace = mechanism_trace(zip_path, conn_dir, *missed[0])
        fig_mechanism(trace, care_dir / "mechanism.png")
        pd.Series(trace.attrs).to_csv(care_dir / "mechanism.csv", header=["value"])
    inj_tab = svc_tab = cost = replay = None
    if (care_dir / "injection.csv").exists():
        inj = pd.read_csv(care_dir / "injection.csv")
        inj_tab = fig_injection(inj, care_dir / "injection.png")
        inj_tab.to_csv(care_dir / "injection_summary.csv")
        injection_differences(inj, [("fly_flywire", "fly_shuffled"), ("fly_flywire", "fly_flyhash"),
                                    ("fly_flywire", "fly_dense"), ("fly_flyhash_full", "fly_flyhash"),
                                    ("fly_flywire", "pca"), ("fly_flywire", "autoencoder"),
                                    ("fly_flywire", "knn"), ("fly_res_flywire", "fly_flywire"),
                                    ("fly_res_flywire", "fly_res_shuffled")]).to_csv(
            care_dir / "injection_differences.csv", index=False)
    if (care_dir / "service.csv").exists():
        svc_tab = fig_service(pd.read_csv(care_dir / "service.csv"), care_dir / "service.png")
        svc_tab.to_csv(care_dir / "service_summary.csv", index=False)
    if (care_dir / "cost.csv").exists():
        cost = pd.read_csv(care_dir / "cost.csv")
    replays = sorted((data_dir / "replay").glob("*.json")) if replay_key is None \
        else [data_dir / "replay" / f"{replay_key}.json"]
    if replays and replays[0].exists():
        replay = replay_view(json.loads(replays[0].read_text(encoding="utf-8")))
    data = dashboard_data(wiring_data, methods, events, scores_dir, inj_tab, cost, svc_tab, replay)
    template = resources.files("fly_novelty").joinpath("dashboard.html").read_text(encoding="utf-8")
    html = template.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"), allow_nan=False))
    (docs_dir / "index.html").write_text(html, encoding="utf-8", newline="\n")
    log.info("dashboard written: %s (%.0f kB)", docs_dir / "index.html", len(html) / 1e3)
