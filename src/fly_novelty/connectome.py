"""PN -> KC wiring of the right mushroom body, taken from the FlyWire whole-brain connectome (v783).

Sources, both CC BY 4.0:

- ``proofread_connections_783.feather`` (FlyWire Consortium, Zenodo doi:10.5281/zenodo.10676866): synapse counts
  per neuron pair and neuropil (Dorkenwald et al. 2024, Nature 634:124-138);
- ``Supplemental_file1_neuron_annotations.tsv``, release v2.1.0 of github.com/flyconnectome/flywire_annotations,
  Supplementary Data 1 of Schlegel et al. 2024, Nature 634:139-152: cell classes and types.

Kept: every antennal lobe projection neuron (``cell_class == "ALPN"``, either side, uni- and multiglomerular) with
at least ``MIN_SYNAPSES`` synapses onto a Kenyon cell of the right hemisphere (``cell_class == "Kenyon_Cell"``,
``side == "right"``), synapses summed over neuropils. Kenyon cells without such an input are left out.

The derived tables in ``connectome/`` (node lists with FlyWire root ids and types, edge list with synapse counts)
are small and committed with the repository; :func:`load` turns them into a sparse KC x PN matrix.

Usage:
    stats = build(Path("data/flywire/proofread_connections_783.feather"),
                  Path("data/flywire/Supplemental_file1_neuron_annotations_v2.1.0.tsv"), Path("connectome"))
    wiring = load(Path("connectome"))
    wiring.matrix.shape   # (n_kc, n_pn)
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

MIN_SYNAPSES = 5  # the usual FlyWire threshold for a reliable connection
ANNOTATIONS_URL = ("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/v2.1.0/"
                   "supplemental_files/Supplemental_file1_neuron_annotations.tsv")
ANNOTATIONS_FILE = "Supplemental_file1_neuron_annotations_v2.1.0.tsv"
CONNECTIONS_FILE = "proofread_connections_783.feather"
ZENODO_RECORD = "https://zenodo.org/api/records/10676866"
KC_CLASSES = {"KCg": "gamma", "KCab": "alpha/beta", "KCa'b'": "alpha'/beta'"}
log = logging.getLogger(__name__)


@dataclass
class Wiring:
    """Sparse KC x PN synapse-count matrix with the node tables (row ``i`` of ``kc`` is matrix row ``i``)."""

    matrix: sparse.csr_matrix
    pn: pd.DataFrame
    kc: pd.DataFrame

    @property
    def n_kc(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_pn(self) -> int:
        return self.matrix.shape[1]


def _download(url: str, dest: Path, size: int | None = None, md5: str | None = None) -> Path:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    done = dest.stat().st_size if dest.exists() else 0
    if size is None or done < size:
        headers = {"Range": f"bytes={done}-"} if done and size else {}
        mode = "ab" if headers else "wb"
        log.info("downloading %s", url)
        with requests.get(url, headers=headers, stream=True, timeout=120) as r, dest.open(mode) as fh:
            r.raise_for_status()
            for chunk in r.iter_content(1 << 22):
                fh.write(chunk)
    if md5:
        h = hashlib.md5()
        with dest.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != md5:
            raise IOError(f"{dest.name}: MD5 mismatch, delete the file and fetch again")
    return dest


def fetch(flywire_dir: Path, connections: bool = True) -> None:
    """Download the cell annotations (27 MB) and, if asked, the connection table (852 MB, resumable)."""
    _download(ANNOTATIONS_URL, flywire_dir / ANNOTATIONS_FILE)
    if connections:
        import requests

        files = requests.get(ZENODO_RECORD, timeout=60).json()["files"]
        info = next(f for f in files if f["key"] == CONNECTIONS_FILE)
        algo, _, digest = info["checksum"].partition(":")
        _download(info["links"]["self"], flywire_dir / CONNECTIONS_FILE, info["size"],
                  digest if algo == "md5" else None)


def _kc_class(hemibrain_type: pd.Series) -> pd.Series:
    prefix = hemibrain_type.astype(str).str.split("-").str[0]
    return prefix.map(KC_CLASSES).fillna("other")


def _glomerulus(row) -> str:
    t = str(row["hemibrain_type"]) if isinstance(row["hemibrain_type"], str) else str(row["cell_type"])
    return t.split("_")[0] if row["cell_sub_class"] == "uniglomerular" else t


def build(connections_path: Path, annotations_path: Path, out_dir: Path,
          min_synapses: int = MIN_SYNAPSES) -> dict:
    """Extract the PN -> right-KC connections and write ``pn_nodes.csv``, ``kc_nodes.csv``, ``pn_kc_edges.csv``
    and ``stats.json`` to ``out_dir``."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.feather as pf

    ann = pd.read_csv(annotations_path, sep="\t", low_memory=False)
    kc_all = ann[(ann["cell_class"] == "Kenyon_Cell") & (ann["side"] == "right")]
    pn_all = ann[ann["cell_class"] == "ALPN"]
    table = pf.read_table(connections_path, columns=["pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count"],
                          memory_map=True)
    keep = pc.and_(pc.is_in(table["post_pt_root_id"], value_set=pa.array(kc_all["root_id"].to_numpy(np.int64))),
                   pc.is_in(table["pre_pt_root_id"], value_set=pa.array(pn_all["root_id"].to_numpy(np.int64))))
    rows = table.filter(keep).to_pandas()
    in_calyx = rows.loc[rows["neuropil"] == "MB_CA_R", "syn_count"].sum() / max(rows["syn_count"].sum(), 1)
    pairs = rows.groupby(["pre_pt_root_id", "post_pt_root_id"], as_index=False)["syn_count"].sum()
    strong = pairs[pairs["syn_count"] >= min_synapses]

    pn = pn_all[pn_all["root_id"].isin(strong["pre_pt_root_id"])].copy()
    pn["glomerulus"] = pn.apply(_glomerulus, axis=1)
    pn = pn.sort_values(["cell_sub_class", "glomerulus", "root_id"], ascending=[False, True, True])
    pn = pn.rename(columns={"cell_sub_class": "sub_class", "hemibrain_type": "type"})
    pn = pn[["root_id", "type", "glomerulus", "sub_class", "side", "top_nt"]].reset_index(drop=True)
    pn.insert(0, "node", np.arange(len(pn)))

    kc = kc_all[kc_all["root_id"].isin(strong["post_pt_root_id"])].copy()
    kc["kc_class"] = _kc_class(kc["hemibrain_type"])
    kc = kc.rename(columns={"hemibrain_type": "kc_type"}).sort_values(["kc_class", "kc_type", "root_id"])
    kc = kc[["root_id", "kc_type", "kc_class"]].reset_index(drop=True)
    kc.insert(0, "node", np.arange(len(kc)))

    pn_index = pd.Series(pn["node"].to_numpy(), index=pn["root_id"].to_numpy())
    kc_index = pd.Series(kc["node"].to_numpy(), index=kc["root_id"].to_numpy())
    edges = pd.DataFrame({"pn": pn_index.reindex(strong["pre_pt_root_id"]).to_numpy(),
                          "kc": kc_index.reindex(strong["post_pt_root_id"]).to_numpy(),
                          "synapses": strong["syn_count"].to_numpy()}).sort_values(["kc", "pn"])

    out_dir.mkdir(parents=True, exist_ok=True)
    pn.to_csv(out_dir / "pn_nodes.csv", index=False, lineterminator="\n")
    kc.to_csv(out_dir / "kc_nodes.csv", index=False, lineterminator="\n")
    edges.to_csv(out_dir / "pn_kc_edges.csv", index=False, lineterminator="\n")
    in_deg = edges.groupby("kc").size()
    out_deg = edges.groupby("pn").size()
    stats = {
        "min_synapses": min_synapses,
        "right_kenyon_cells": int(len(kc_all)), "kenyon_cells_kept": int(len(kc)),
        "alpn_total": int(len(pn_all)), "pn_kept": int(len(pn)),
        "pn_kept_uniglomerular": int((pn["sub_class"] == "uniglomerular").sum()),
        "pn_kept_left": int((pn["side"] == "left").sum()),
        "edges": int(len(edges)), "synapses": int(edges["synapses"].sum()),
        "pairs_below_threshold": int(len(pairs) - len(strong)),
        "synapse_share_in_right_calyx": round(float(in_calyx), 4),
        "kc_in_degree_mean": round(float(in_deg.mean()), 3),
        "kc_in_degree_quantiles_5_50_95": [float(v) for v in in_deg.quantile([0.05, 0.5, 0.95])],
        "pn_out_degree_mean": round(float(out_deg.mean()), 3), "pn_out_degree_max": int(out_deg.max()),
        "kc_classes": kc["kc_class"].value_counts().to_dict(),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8", newline="\n")
    log.info("PN->KC wiring: %d PNs, %d KCs, %d edges, %d synapses", len(pn), len(kc), len(edges),
             stats["synapses"])
    return stats


def load(conn_dir: Path) -> Wiring:
    """The committed wiring tables as a sparse KC x PN matrix of synapse counts."""
    pn = pd.read_csv(conn_dir / "pn_nodes.csv")
    kc = pd.read_csv(conn_dir / "kc_nodes.csv")
    edges = pd.read_csv(conn_dir / "pn_kc_edges.csv")
    matrix = sparse.csr_matrix((edges["synapses"].to_numpy(np.float64),
                                (edges["kc"].to_numpy(), edges["pn"].to_numpy())), shape=(len(kc), len(pn)))
    return Wiring(matrix=matrix, pn=pn, kc=kc)
