"""Code for conditional entropy calculations."""

import numpy as np
from sklearn import metrics

def entropy(labels):
    """Calculates the Entropy for a labeling.
    Parameters
    ----------
    labels : int array, shape = [n_samples]
        The labels
    Notes
    -----
    The logarithm used is the natural logarithm (base-e).
    """
    if len(labels) == 0:
        return 1.0
    label_idx = np.unique(labels, return_inverse=True)[1]
    pi = np.bincount(label_idx).astype(np.float64)
    pi = pi[pi > 0]
    pi_sum = np.sum(pi)
    return -np.sum((pi / pi_sum) * (np.log(pi) - np.log(pi_sum)))

def normalised_conditional_entropy(x, y):
    """
    H(X|Y) = H(X) - I(X,Y) and we normalise with log(N)
    """

    N = len(x)
    Ex = entropy(x)
    I = metrics.mutual_info_score(x, y)

    return (Ex - I) / np.log(N)

def nce_ttprime(partitions):
    n_partitions = len(partitions)
    NCE = np.zeros((n_partitions,n_partitions))

    for t in range(n_partitions):
        for t_prime in range(n_partitions):
            NCE[t,t_prime] = normalised_conditional_entropy(partitions[t],partitions[t_prime])

    return NCE

def nce_ttprime_mean(partitions):
    n_partitions = len(partitions)
    NCE = []
    for t in range(n_partitions):
        for t_prime in range(t+1, n_partitions):
            NCE.append(normalised_conditional_entropy(partitions[t_prime],partitions[t]))

    return np.array(NCE).mean()