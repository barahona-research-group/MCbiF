"""Code to compute crossing number for a Sankey diagram using the OmicsSankey package."""

import numpy as np

from sklearn.metrics.cluster import contingency_matrix

from omics_sankey.main import run_method

def compute_sankey(partitions):

    M = len(partitions)

    nodes = []
    links = []
    levels = {}

    for m in range(M-1):
        p_m = partitions[m]
        n_m = np.max(p_m)+1
        p_m_plus_one = partitions[m+1]
        n_m_plus_one = np.max(p_m_plus_one)+1
        cm = contingency_matrix(p_m,p_m_plus_one)

        # add nodes
        if m == 0:
            for i in range(n_m):
                nodes.append({"name" : f"P{m}_C{i}"})
        for j in range(n_m_plus_one):
            nodes.append({"name" : f"P{m+1}_C{j}"})
        
        # add edges
        for i in range(n_m):
            for j in range(n_m_plus_one):
                cm_ij = cm[i,j]
                if cm_ij > 0:
                    links.append({"source" : f"P{m}_C{i}", "target" : f"P{m+1}_C{j}", "value" : int(cm_ij)})

        # add levels
        if m == 0:
            levels[m] = [f"P{m}_C{i}" for i in range(n_m)]
        levels[m+1] = [f"P{m+1}_C{j}" for j in range(n_m_plus_one)]

    # compile Sankey data
    data = {"nodes" : nodes, "links" : links, "level" : levels}
    return data

def compute_weighted_crossing_sankey(sankey_data):

    # Set fixed variables like stated in OMICS Sankey paper
    alpha1 = 0.01
    N = 100
    alpha2 = 0.2
    M = 100
    algo = "BC"
    dummy_signal = True
    cycle_signal = False

    # optimise OMICS Sankey layout
    result = run_method(
        algo,
        sankey_data,
        len(sankey_data["level"]),
        alpha1,
        alpha2,
        N,
        M,
        dummy_signal,
        cycle_signal,
        sankey_data["level"],
        )

    return result["Stage 2 WeightedCrossing"]

def compute_omics_sankey(sankey_data):

    # Set fixed variables like stated in OMICS Sankey paper
    alpha1 = 0.01
    N = 100
    alpha2 = 0.2
    M = 100
    algo = "BC"
    dummy_signal = True
    cycle_signal = False

    # optimise OMICS Sankey layout
    result = run_method(
        algo,
        sankey_data,
        len(sankey_data["level"]),
        alpha1,
        alpha2,
        N,
        M,
        dummy_signal,
        cycle_signal,
        sankey_data["level"],
        )
    
    return result
    