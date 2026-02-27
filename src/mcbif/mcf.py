"""Code for Multiscale Clustering Filtration (MCF)."""

import itertools
import numpy as np

from tqdm import tqdm

from gudhi import SimplexTree

from mcbif.io import load_results
from mcbif.utils import node_id_to_dict, _cluster_id_preprocessing


class MultiscaleClusteringFiltration:
    """Main class to construct MCF from a sequence of partitions and analyse
    its persistent homology."""

    def __init__(self, method="standard", max_dim=3):
        """Initialise MCF object.

        Parameters:
            max_dim (int): Maximum dimension of simplices considered
                in filtration, between 1 and 3.

            method (str): Method to construct the MCF. Both methods lead to the
                same persistent homology, see our paper.
                - 'standard': Standard method where nodes in MCF correspond
                    to points. Faster when the number of points is smaller
                    than the total number of distinct clusters.
                - 'nerve': Nerve-based method where nodes in MCF correspond to
                    clusters. Faster when the total number of distinct clusters
                    is smaller than the number of points.
        """

        # initialise sequence of partitions
        self.partitions = []
        self.filtration_indices = []

        # set max dimension of filtration
        self.max_dim = min(3, max_dim)

        # set method to construct filtration, either standard or nerve-based
        self.method = method

        # initialise for gudhi
        self.filtration_gudhi = SimplexTree()

    @property
    def n_partitions(self):
        """ "Computes number of partitions in sequence."""
        return len(self.partitions)
    
    @property
    def n_simplices(self):
        """Computes number of simplices in MCF."""
        return self.filtration_gudhi.num_simplices()

    def load_data(self, partitions, filtration_indices=None):
        """Method to load sequence of partitions and
        filtration indices."""
        self.partitions = partitions

        if filtration_indices is None:
            # if no filtration indices are given use enumeration
            self.filtration_indices = np.arange(1, self.n_partitions + 1)
        else:
            self.filtration_indices = filtration_indices

    def load_data_from_file(self, file_path):
        """Method to load data from precomputed MCF file."""

        # load results dictionary
        mcf_results = load_results(file_path)

        # unpack dictionary
        self.filtration_indices = mcf_results["filtration_indices"]
        self.max_dim = mcf_results["max_dim"]
        self.method = mcf_results["method"]           

    def _build_filtration_standard(self, tqdm_disable=False):
        """Construct MCF via standard method."""
        # initialise simplex tree
        self.filtration_gudhi = SimplexTree()

        # store all communities to later avoid repetitious computations
        all_communities = set()

        for t in tqdm(range(len(self.filtration_indices)), disable=tqdm_disable):

            # continue if partition at scale t has appeared before
            is_repetition = False
            for s in range(t - 1, -1, -1):
                if np.array_equal(self.partitions[s], self.partitions[t]):
                    is_repetition = True
                    break
            if is_repetition:
                continue

            # add communities at scale t as simplices to tree
            for community in node_id_to_dict(self.partitions[t]).values():
                # continue if community has been added before
                if community in all_communities:
                    continue
                # add community to set of all communities
                else:
                    all_communities.add(community)
                # compute size of community
                s_community = len(community)
                # cover community by max_dim-simplices when community is larger than max_dim
                for face in itertools.combinations(
                    community, min(self.max_dim + 1, s_community)
                ):
                    self.filtration_gudhi.insert(
                        list(face), filtration=self.filtration_indices[t]
                    )

    def _build_filtration_nerve(self, tqdm_disable=False):
        """Construct MCF via nerve-based method."""

        # initialise simplex tree
        self.filtration_gudhi = SimplexTree()

        # we compute cluster indices of new clusters per partition
        # and a dictionary that maps cluster indices to sets
        partitions_c_ind, ind_to_c = _cluster_id_preprocessing(self.partitions)

        # initialise simplices of different dimensions
        nodes = list()
        edges = list()
        triangles = list()

        # iterate through filtration indices
        for i, t in tqdm(
            enumerate(self.filtration_indices),
            total=len(self.filtration_indices),
            disable=tqdm_disable,
        ):
            # get new cluster indices
            c_ind_new = partitions_c_ind[i]
            # iterate through indices
            for c_ind in c_ind_new:
                c = ind_to_c[c_ind]

                # add tetrahedra
                if self.max_dim > 2:
                    for triangle_ind in triangles:
                        # get intersection of clusters corresponding to triangle
                        triangle_intersection = (
                            ind_to_c[triangle_ind[0]]
                            .intersection(ind_to_c[triangle_ind[1]])
                            .intersection(ind_to_c[triangle_ind[2]])
                        )
                        # check if new cluster intersects with triangle
                        if not c.isdisjoint(triangle_intersection):
                            tetrahedron = triangle_ind + [c_ind]
                            # insert tetrahedron into simplex tree
                            self.filtration_gudhi.insert(tetrahedron, filtration=t)

                # add triangles
                if self.max_dim > 1:
                    for edge_ind in edges:
                        # get intersection of clusters corresponding to edge
                        edge_intersection = ind_to_c[edge_ind[0]].intersection(
                            ind_to_c[edge_ind[1]]
                        )
                        # check if new cluster intersects with edge
                        if not c.isdisjoint(edge_intersection):
                            triangle = edge_ind + [c_ind]
                            # insert triangle into simplex tree
                            self.filtration_gudhi.insert(triangle, filtration=t)
                            # add triangle to set
                            triangles.append(triangle)

                # add edges
                for node_ind in nodes:
                    # get cluster corresponding to node
                    node_intersection = ind_to_c[node_ind[0]]
                    # check if new cluster intersects with node
                    if not c.isdisjoint(node_intersection):
                        edge = node_ind + [c_ind]
                        # insert edge into simplex tree
                        self.filtration_gudhi.insert(edge, filtration=t)
                        # add edge to set
                        edges.append(edge)

                # add nodes
                node = [c_ind]
                self.filtration_gudhi.insert(node, filtration=t)
                nodes.append(node)

    def build_filtration(self, tqdm_disable=False):
        """Build MCF filtration."""

        if self.method == "standard":
            self._build_filtration_standard(tqdm_disable=tqdm_disable)

        elif self.method == "nerve":
            self._build_filtration_nerve(tqdm_disable=tqdm_disable)