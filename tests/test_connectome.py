import json
from pathlib import Path

import numpy as np
import pandas as pd

from fly_novelty import connectome

ROOT = Path(__file__).resolve().parents[1]


def test_build_keeps_alpn_to_right_kc_connections_above_threshold(tmp_path):
    ann = pd.DataFrame({
        "root_id": [1, 2, 3, 10, 11, 12, 20],
        "cell_class": ["ALPN", "ALPN", "ALPN", "Kenyon_Cell", "Kenyon_Cell", "Kenyon_Cell", "LHLN"],
        "cell_sub_class": ["uniglomerular", "uniglomerular", "multiglomerular", None, None, None, None],
        "cell_type": [None, None, "M_x", None, None, None, None],
        "hemibrain_type": ["DA1_lPN", "DM1_lPN", "M_lPNm11A", "KCg-m", "KCab-s", "KCa'b'-m", "x"],
        "side": ["right", "left", "right", "right", "right", "left", "right"],
        "top_nt": ["acetylcholine", "acetylcholine", "gaba", "dopamine", "dopamine", "dopamine", "gaba"],
    })
    conn = pd.DataFrame({
        "pre_pt_root_id": [1, 1, 1, 2, 3, 3, 20, 1],
        "post_pt_root_id": [10, 10, 11, 10, 11, 12, 10, 12],
        "neuropil": ["MB_CA_R", "MB_PED_R", "MB_CA_R", "MB_CA_R", "MB_CA_R", "MB_CA_L", "MB_CA_R", "MB_CA_L"],
        "syn_count": [3, 4, 9, 4, 6, 30, 50, 12],
    })
    ann.to_csv(tmp_path / "ann.tsv", sep="\t", index=False)
    conn.to_feather(tmp_path / "conn.feather")
    stats = connectome.build(tmp_path / "conn.feather", tmp_path / "ann.tsv", tmp_path / "out")
    w = connectome.load(tmp_path / "out")
    # PN 1 -> KC 10 (3 + 4 = 7 synapses over two neuropils) and -> KC 11 (9); PN 3 -> KC 11 (6);
    # PN 2 -> KC 10 has 4 synapses (dropped), KC 12 is on the left, neuron 20 is not a PN
    assert stats["edges"] == 3 and stats["synapses"] == 22 and stats["pn_kept"] == 2
    assert w.matrix.shape == (2, 2)
    assert sorted(w.matrix.data.tolist()) == [6.0, 7.0, 9.0]
    assert set(w.pn["glomerulus"]) == {"DA1", "M_lPNm11A"}
    assert set(w.kc["kc_class"]) == {"gamma", "alpha/beta"}


def test_committed_wiring_matches_its_stats():
    stats = json.loads((ROOT / "connectome" / "stats.json").read_text(encoding="utf-8"))
    w = connectome.load(ROOT / "connectome")
    assert w.matrix.shape == (stats["kenyon_cells_kept"], stats["pn_kept"]) == (2364, 149)
    assert w.matrix.nnz == stats["edges"] and w.matrix.sum() == stats["synapses"]
    assert w.matrix.data.min() >= stats["min_synapses"]
    assert np.isclose(w.matrix.getnnz(axis=1).mean(), stats["kc_in_degree_mean"], atol=1e-3)
    assert (w.matrix.getnnz(axis=1) > 0).all()
