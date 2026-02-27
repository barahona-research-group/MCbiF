"""Code for preparing data."""

import numpy as np

from sklearn.model_selection import train_test_split


SEED = 42


def prepare_data_vectors(N, M, exclude_zeros=True):
    """Load and preprocess data."""

    # Load data
    y = np.load(f"data/N{N}_M{M}_y_crossing.npy")
    nce = np.load(f"results/features/N{N}_M{M}_nce.npy")
    hf0 = np.load(f"results/features/N{N}_M{M}_hf0.npy")
    hf1 = np.load(f"results/features/N{N}_M{M}_hf1.npy")
    partitions = np.load(f"data/N{N}_M{M}_partitions.npy")
    ari = np.load(f"results/features/N{N}_M{M}_ari.npy")
    mod = np.load(f"results/features/N{N}_M{M}_mod.npy")

    # flatten features
    hf0 = np.array([hf0[i][np.triu_indices(hf0.shape[1])] for i in range(hf0.shape[0])])
    hf1 = np.array(
        [hf1[i][np.triu_indices(hf1.shape[1], k=1)] for i in range(hf0.shape[0])]
    )
    ari = np.array(
        [ari[i][np.triu_indices(ari.shape[1], k=1)] for i in range(ari.shape[0])]
    )
    mod = np.array(
        [mod[i][np.triu_indices(mod.shape[1], k=1)] for i in range(mod.shape[0])]
    )
    nce = nce.reshape(nce.shape[0], -1)
    partitions = partitions.reshape(partitions.shape[0], -1)

    # Compute outlier mask
    non_outliers = (y <= np.percentile(y, 99.9)) * (y >= np.percentile(y, 0.1))

    # Exclude sequences with zero crossings if specified
    if exclude_zeros:
        non_outliers = non_outliers * (y != 0)

    # Remove outliers
    y = y[non_outliers]
    nce = nce[non_outliers]
    hf0 = hf0[non_outliers]
    hf1 = hf1[non_outliers]
    partitions = partitions[non_outliers]
    ari = ari[non_outliers]
    mod = mod[non_outliers]

    # Combine features
    hf01_stacked = np.hstack([hf0, hf1])

    # Split data into training and test sets
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        np.arange(len(y)), y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_hf0_train, X_hf0_test = hf0[X_train_idx], hf0[X_test_idx]
    X_hf1_train, X_hf1_test = hf1[X_train_idx], hf1[X_test_idx]
    X_nce_train, X_nce_test = nce[X_train_idx], nce[X_test_idx]
    X_hf01_s_train, X_hf01_s_test = hf01_stacked[X_train_idx], hf01_stacked[X_test_idx]
    X_partitions_train, X_partitions_test = (
        partitions[X_train_idx],
        partitions[X_test_idx],
    )
    X_ari_train, X_ari_test = ari[X_train_idx], ari[X_test_idx]
    X_mod_train, X_mod_test = mod[X_train_idx], mod[X_test_idx]

    # Store all features in lists and dictionaries for easy access later
    feature_names = [
        "Partitions",
        "HF0",
        "HF1",
        "HF0 & HF1 Stacked",
        "NCE",
        "ARI",
        "MOD",
    ]
    feature_names_to_train = {
        "HF0": X_hf0_train,
        "HF1": X_hf1_train,
        "NCE": X_nce_train,
        "HF0 & HF1 Stacked": X_hf01_s_train,
        "Partitions": X_partitions_train,
        "ARI": X_ari_train,
        "MOD": X_mod_train,
    }
    feature_names_to_test = {
        "HF0": X_hf0_test,
        "HF1": X_hf1_test,
        "NCE": X_nce_test,
        "HF0 & HF1 Stacked": X_hf01_s_test,
        "Partitions": X_partitions_test,
        "ARI": X_ari_test,
        "MOD": X_mod_test,
    }

    return (
        y_train,
        y_test,
        feature_names,
        feature_names_to_train,
        feature_names_to_test,
    )


def prepare_data_images(N, M, exclude_zeros=True):
    """Load and preprocess data."""

    # Load data
    y = np.load(f"data/N{N}_M{M}_y_crossing.npy")
    nce = np.load(f"results/features/N{N}_M{M}_nce.npy")
    hf0 = np.load(f"results/features/N{N}_M{M}_hf0.npy")
    hf1 = np.load(f"results/features/N{N}_M{M}_hf1.npy")
    partitions = np.load(f"data/N{N}_M{M}_partitions.npy")
    ari = np.load(f"results/features/N{N}_M{M}_ari.npy")
    mod = np.load(f"results/features/N{N}_M{M}_mod.npy")

    # Make hf features symmetric
    hf1 = hf1 + np.transpose(hf1, axes=(0, 2, 1))
    hf0 = hf0 + np.transpose(np.triu(hf0, k=1), axes=(0, 2, 1))

    # Compute outlier mask
    non_outliers = (y <= np.percentile(y, 99.9)) * (y >= np.percentile(y, 0.1))

    # Exclude sequences with zero crossings if specified
    if exclude_zeros:
        non_outliers = non_outliers * (y != 0)

    # Remove outliers
    y = y[non_outliers]
    nce = nce[non_outliers]
    hf0 = hf0[non_outliers]
    hf1 = hf1[non_outliers]
    partitions = partitions[non_outliers]
    ari = ari[non_outliers]
    mod = mod[non_outliers]

    # Combine features for CNN input
    hf01_stacked = np.stack([hf0, hf1], axis=-1)

    # Split data into training and test sets
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        np.arange(len(y)), y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_hf0_train, X_hf0_test = hf0[X_train_idx], hf0[X_test_idx]
    X_hf1_train, X_hf1_test = hf1[X_train_idx], hf1[X_test_idx]
    X_nce_train, X_nce_test = nce[X_train_idx], nce[X_test_idx]
    X_hf01_s_train, X_hf01_s_test = hf01_stacked[X_train_idx], hf01_stacked[X_test_idx]
    X_partitions_train, X_partitions_test = (
        partitions[X_train_idx],
        partitions[X_test_idx],
    )
    X_ari_train, X_ari_test = ari[X_train_idx], ari[X_test_idx]
    X_mod_train, X_mod_test = mod[X_train_idx], mod[X_test_idx]

    # Store all features in lists and dictionaries for easy access later
    feature_names = [
        "HF0 & HF1 Stacked",
        "NCE",
        "HF0",
        "HF1",
        "Partitions",
        "ARI",
        "MOD",
    ]
    feature_names_to_train = {
        "HF0": X_hf0_train,
        "HF1": X_hf1_train,
        "NCE": X_nce_train,
        "HF0 & HF1 Stacked": X_hf01_s_train,
        "Partitions": X_partitions_train,
        "ARI": X_ari_train,
        "MOD": X_mod_train,
    }
    feature_names_to_test = {
        "HF0": X_hf0_test,
        "HF1": X_hf1_test,
        "NCE": X_nce_test,
        "HF0 & HF1 Stacked": X_hf01_s_test,
        "Partitions": X_partitions_test,
        "ARI": X_ari_test,
        "MOD": X_mod_test,
    }

    return (
        y_train,
        y_test,
        feature_names,
        feature_names_to_train,
        feature_names_to_test,
    )
