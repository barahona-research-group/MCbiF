import itertools
import numpy as np


def generate_partitions(elements, index, current_partition, all_partitions_list):
    """
    Function to generate all partitions
    """
    if index == len(elements):
        # If we have considered all elements in the set, add the partition to the result
        all_partitions_list.append([subset[:] for subset in current_partition])  # Append a copy of current_partition
        return

    # For each subset in the current partition, add the current element to it and recall
    for i in range(len(current_partition)):
        current_partition[i].append(elements[index])
        generate_partitions(elements, index + 1, current_partition, all_partitions_list)
        current_partition[i].pop()

    # Add the current element as a singleton subset and recall
    current_partition.append([elements[index]])
    generate_partitions(elements, index + 1, current_partition, all_partitions_list)
    current_partition.pop()

def partition_to_id(partition, N):

    partition_vector = np.zeros(N, dtype=int)

    for i, cluster in enumerate(partition):
        for x in cluster:
            partition_vector[x-1] = i

    return partition_vector


def all_partitions(elements, sort=True):
    """
    Function to generate all partitions for a given set
    """
    N = len(elements)
    all_partitions_list = []  # List to store all partitions
    current_partition = []     # Current partition
    generate_partitions(elements, 0, current_partition, all_partitions_list)

    # transform to cluster ids
    all_partitions_ids = np.array([partition_to_id(partition, N) for partition in all_partitions_list])

    if sort: # split into bins of partitions with the same number of clusters
        return [all_partitions_ids[all_partitions_ids.max(axis=1) == c] for c in range(N-1,-1,-1)]
    else:
        return all_partitions_ids
    

def generate_trajectories(N, M):
    indices = range(0, N)  # Create indices [1, 2, ..., N]
    trajectories = list(itertools.product(indices, repeat=M))  # Generate all combinations of length M
    return trajectories


def sample_trajectory(n_partitioncs_per_size, N, M, partitions_result, rng):

    # define probabilities for multinomial
    pvals = n_partitioncs_per_size/(n_partitioncs_per_size.sum())

    # draw from multinomial distribution that determines how many partitions with N-i blocks we have
    n_partitions_per_size_sample = rng.multinomial(M-2,pvals=pvals)

    # compile trajectory and start with singletons
    trajectory = [np.arange(N)]

    for i in range(N-2):
        # add random partitions with N-i blocks
        trajectory.append(partitions_result[i+1][rng.integers(0, n_partitioncs_per_size[i]-1,size=n_partitions_per_size_sample[i])])

    # end with one block partition
    trajectory.append(np.zeros(N, dtype=int))
    return np.vstack(trajectory)



def generate_cut_sequence(N, rng, p_swap=0, step=1):
    """Generate sequence of partitions by progressively cutting a set of N elements.
    Optionally, with probability p_swap, swap two elements in the partition at each step."""

    partition_0 = np.arange(N)
    partition_1 = np.zeros(N)

    M = N

    partitions = [partition_0]

    n_swaps = np.zeros(N)

    for m in range(1, M, step):

        cuts = np.sort(rng.choice(partition_0, size=N - m, replace=False))

        partition = []
        for i in range(len(cuts)):
            size = cuts[i] - (cuts[i - 1] if i > 0 else -1)
            partition += size * [i]
        partition += (N - cuts[-1] - 1) * [len(cuts)]

        if rng.random() < p_swap:
            n_swaps[m] = 1
            swap = rng.choice(N, size=2, replace=False)
            partition[swap[0]], partition[swap[1]] = (
                partition[swap[1]],
                partition[swap[0]],
            )

        partitions.append(partition)

    partitions.append(partition_1)

    return np.array(partitions, dtype=int), n_swaps