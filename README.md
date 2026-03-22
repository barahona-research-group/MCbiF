[![DOI](https://zenodo.org/badge/1148147780.svg)](https://doi.org/10.5281/zenodo.18860905)

# Multiscale Clustering Bifiltration (MCbiF)

Code for the ICLR 2026 conference paper "MCbiF: Measuring Topological Autocorrelation in Multiscale Clusterings via 2-Parameter Persistent Homology" by Juni Schindler and Mauricio Barahona: https://openreview.net/forum?id=E7D6uybODJ

## Installation
Clone the repository and open the folder in your terminal. 

```zsh
git clone https://github.com/barahona-research-group/MCbiF.git
cd MCF/
```

Then, to install the package with ``pip``, execute the following command:

```zsh
pip install .
```

To install the package with support for optimial Sankey diagram plotting execute instead:

```zsh
pip install ."[sankey]" 
```

To also install all dependencies for the experiments, execute:

```zsh
pip install ."[experiments]" 
```

Note that this package requires an installation of the the `Rivet` software for multiparameter persistent homology. Installation instructions are available here: https://rivet.readthedocs.io/en/latest/installing.html

## Usage

MCbiF can be used to analyse the topological autocorrelation of arbitrary sequences of partitions $\theta$, including non-hierarchical ones. The required input data is:

- `N` times `M` dimensional array `partitions` that stores the cluster IDs of `N` elements across `M` partitions in the sequence $\theta$.
- `M` dimensional array `filtration_indices` that stores the real-valued scales $t_1,\dots,t_M$ of $\theta$.

See our paper for technical background. The MCbiF pipeline can then be computed as follows:

```python
from mcbif import MultiscaleClusteringBifiltration

# data for toy example
partitions = [[0, 1, 2], [0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0]]
filtration_indices = [0, 1, 2, 3, 4]

# initialise MCbiF object with nerve-based computation
mcbif_obj = MultiscaleClusteringBifiltration(method="nerve")

# load partition data
mcbif_obj.load_data(partitions,filtration_indices)

# compute all MCbiF measures
mcbif_results = mcbif_obj.compute_all_measures()

# report average 0- and 1-conflict
print("\nAverage 0-conflict:", mcbif_obj.conflict_0_avg)
print("Average 1-conflict:", mcbif_obj.conflict_1_avg)

# plot 0- and 1-dimensional Hilbert functions
mcbif.plot_hilbert_function(0)
mcbif.plot_hilbert_function(1);
```

We also provide code for plotting optimised Sankey diagrams:

```python
from sankey import Sankey

# initialise Sankey object
sankey_obj = Sankey(partitions=partitions)

# optimise Sankey layout
sankey_obj.compute_omics_sankey()

# plot Sankey diagram
fig = sankey.plot_sankey() 
```

See the notebook `toy_examples.ipynb` for two illustrative examples.


## Experiments

The `\experiments` directory contains code for our three numerical experiments:

- Regression Task: Minimimal Crossing Number of Sankey Layout (`experiments\regression`)
- Classification Task: Non-Order Preserving Sequences of Partitions (`experiments\classification`)
- Application to Real-World Temporal Contact Data of Free-Ranging House Mice (`experiments\wild_mice`)

## Cite

Please cite our paper if you use this code in your own work:

```bibtex
@inproceedings{schindlerMCbiFMeasuringTopological2026,
  author = {Schindler, Juni and Barahona, Mauricio},
  title = {MCbiF: Measuring Topological Autocorrelation in Multiscale Clusterings via 2-Parameter Persistent Homology},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year = {2026},
  url = {https://openreview.net/forum?id=E7D6uybODJ}
  doi = {10.48550/arXiv.2510.14710}
}
```

## Licence

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.