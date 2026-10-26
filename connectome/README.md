# PN -> KC wiring derived from the FlyWire connectome

The files in this folder are **derived from**:

> FlyWire Consortium. *FlyWire Whole-brain Connectome Connectivity Data*, version 783. Zenodo, 2024.
> doi:[10.5281/zenodo.10676866](https://doi.org/10.5281/zenodo.10676866). Licence:
> [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

> Neuron annotations, release v2.1.0 of [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations),
> published as Supplementary Data 1 of Schlegel et al. 2024 (Nature, open access under CC BY 4.0).

Please cite the papers describing the data: Dorkenwald et al. 2024, *Neuronal wiring diagram of an adult brain*,
Nature 634:124-138; Schlegel et al. 2024, *Whole-brain annotation and multi-connectome cell typing of
Drosophila*, Nature 634:139-152; Buhmann et al. 2021, Nature Methods 18:771-774 (synapse detection); Heinrich et
al. 2018, MICCAI (synaptic clefts); Eckstein, Bates et al. 2024, Cell 187:2574-2594 (transmitter predictions, the
`top_nt` column).

**Changes made** (by `python -m fly_novelty connectome`, code in `src/fly_novelty/connectome.py`): from the
connection table only the columns `pre_pt_root_id`, `post_pt_root_id`, `neuropil` and `syn_count` were used, from
the annotation table `root_id`, `cell_class`, `cell_sub_class`, `cell_type`, `hemibrain_type`, `side` and `top_nt`.
Kept are the connections from antennal lobe projection neurons (`cell_class == "ALPN"`, both sides, uni- and
multiglomerular) to Kenyon cells of the right hemisphere (`cell_class == "Kenyon_Cell"`, `side == "right"`), with
synapses summed over neuropils and at least 5 synapses per pair. Glomerulus names were cut from the hemibrain type
of uniglomerular PNs (`DA1_lPN` -> `DA1`); the Kenyon cell class (gamma, alpha/beta, alpha'/beta') from the
hemibrain type prefix. This is a modelling simplification, not a curated biological dataset.

| file | content |
|---|---|
| `pn_nodes.csv` | `node` (matrix column), FlyWire `root_id` (v783), `type` (hemibrain type), `glomerulus`, `sub_class`, `side`, `top_nt` |
| `kc_nodes.csv` | `node` (matrix row), FlyWire `root_id` (v783), `kc_type` (hemibrain type), `kc_class` |
| `pn_kc_edges.csv` | `pn`, `kc` (node indices), `synapses` (count, >= 5) |
| `stats.json` | sizes and degree statistics |

The derived files are distributed under the same licence, CC BY 4.0.
