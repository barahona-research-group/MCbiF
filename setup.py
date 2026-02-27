"""Setup."""

from setuptools import find_namespace_packages
from setuptools import setup

__version__ = "0.0.1"


required_packages = [
    "matplotlib",
    "numpy",
    "gudhi",    
    "tqdm",
    'pyrivet @ git+https://github.com/juni-schindler/rivet-python.git',
]

sankey_packages = [
    'omicssankey @ git+https://github.com/juni-schindler/OmicsSankey.git',
    "scikit-learn",
    "plotly",
    "nbformat",
    "kaleido",
]

experiments_packages = sankey_packages + [
    'statannotations',
    'jinja2',
    'optuna',
    'tensorflow',
    'networkx',
    'torch',
    'torch_geometric',
]

setup(
    name="MCbiF",
    version=__version__,
    author="Juni Schindler",
    install_requires=required_packages,
    zip_safe=False,
    extras_require={
        "sankey": sankey_packages,
        "experiments": experiments_packages,
    },
    packages=find_namespace_packages("src"),
    include_package_data=True,
    package_dir={"": "src"},
)
