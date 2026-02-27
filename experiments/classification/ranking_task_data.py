import numpy as np
import sys
from pathlib import Path

# Add experiments directory to path
experiments_dir = Path(__file__).parent.parent
sys.path.insert(0, str(experiments_dir))

from tqdm import tqdm

from mcbif import MultiscaleClusteringBifiltration

from common.entropy import nce_ttprime
from common.sampling import generate_cut_sequence

N_SAMPLES = 1850
N_SWAPS = 1
N = 500
STEP = 18

if __name__ == "__main__":

    rng = np.random.default_rng(42)
   
    for i in tqdm(range(N_SAMPLES*2)):
        # generate sequences with cut only
        partitions, _ = generate_cut_sequence(N, rng, p_swap=0, step=STEP)
        if i < N_SAMPLES:
            continue
        else:
            np.save(f"data/raw/group_a/partitions_a_{i}.npy", partitions)

            # compute nce
            nce = nce_ttprime(partitions)
            np.save(f"data/raw/group_a/nce_a_{i}.npy", nce)

            # compute mcbif
            mcbif = MultiscaleClusteringBifiltration(method="nerve")
            mcbif.load_data(partitions)
            mcbif.compute_all_measures(
                file_path=f"data/raw/group_a/mcbif_a_{i}.pkl",
                tqdm_disable=True,
            )

    rng = np.random.default_rng(43)
   
    for i in tqdm(range(N_SAMPLES*2)):
        # generate sequences with cut and swap
        partitions, n_swaps = generate_cut_sequence(N, rng, p_swap=0.1, step=STEP)
        while n_swaps.sum() < 1:
            partitions, n_swaps = generate_cut_sequence(N, rng, p_swap=0.1, step=STEP)
        if i < N_SAMPLES:
            continue
        else:
            np.save(f"data/raw/group_b/partitions_b_{i}.npy", partitions)
            np.save(f"data/raw/group_b/n_swaps_b_{i}.npy", n_swaps)

            # compute nce
            nce = nce_ttprime(partitions)
            np.save(f"data/raw/group_b/nce_b_{i}.npy", nce)

            # compute mcbif
            mcbif = MultiscaleClusteringBifiltration(method="nerve")
            mcbif.load_data(partitions)
            mcbif.compute_all_measures(
                file_path=f"data/raw/group_b/mcbif_b_{i}.pkl",
                tqdm_disable=True,
            )