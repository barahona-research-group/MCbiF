# Multiscale Clustering Bifiltration (MCbiF)

Python code for the ICLR 2026 conference paper "MCbiF: Measuring Topological Autocorrelation in Multiscale Clusterings via 2-Parameter Persistent Homology" by Juni Schindler and Mauricio Barahona: https://openreview.net/forum?id=E7D6uybODJ

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

## Experiments

The `\experiments` directory contains code for our three experiments:

- Regression Task: Minimimal Crossing Number of Sankey Layout
- Classification Task: Non-Order Preserving Sequences of Partitions
- Application to Real-World Temporal Contact Data of Free-Ranging House Mice

## Cite

Please cite our paper if you use this code in your own work:

```bibtex
@inproceedings{schindlerMCbiFMeasuringTopological2026,
  author = {Schindler, Juni and Barahona, Mauricio},
  title = {MCbiF: Measuring Topological Autocorrelation in Multiscale Clusterings via 2-Parameter Persistent Homology},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=E7D6uybODJ}
  doi = {10.48550/arXiv.2510.14710}
}
```

## Licence

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.