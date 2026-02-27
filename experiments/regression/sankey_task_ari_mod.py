import numpy as np
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score
import graph_tool.all as gt
import time

# Open a log file to store timings
with open("data/ari_mod_computation_timings.txt", "w") as log_file:
    for N in [5, 10]:
        start_time = time.time()

        partitions = np.load(f'data/N{N}_M20_partitions.npy')

        n_samples = len(partitions)
        M = partitions[0].shape[0]

        # compute pairwise ARI for each sequence of partitions
        ari = []
        for i in tqdm(range(n_samples), desc=f"Computing ARI for N={N}"):
            ari_sample = np.zeros((M, M))
            for j in range(M):
                for k in range(j, M):
                    ari_sample[j, k] = adjusted_rand_score(partitions[i][j], partitions[i][k])
                    ari_sample[k, j] = ari_sample[j, k]
            ari.append(ari_sample)
        ari = np.array(ari)

        # save array
        np.save(f"results/features/N{N}_M20_ari.npy", ari)

        end_time = time.time()
        elapsed_time = end_time - start_time
        log_file.write(f"Time taken for ARI computation for N={N}: {elapsed_time:.2f} seconds\n")

    for N in [5, 10]:
        start_time = time.time()

        partitions = np.load(f'data/N{N}_M20_partitions.npy')

        n_samples = len(partitions)
        M = partitions[0].shape[0]

        # compute pairwise MPO for each sequence of partitions
        mod = []
        for i in tqdm(range(n_samples), desc=f"Computing MPO for N={N}"):
            mpo_sample = np.zeros((M, M))
            for j in range(M):
                for k in range(j, M):
                    mpo_sample[j, k] = gt.partition_overlap(partitions[i][j], partitions[i][k])
                    mpo_sample[k, j] = mpo_sample[j, k]
            mod.append(mpo_sample)
        mod = np.array(mod)

        # save array
        np.save(f"results/features/N{N}_M20_mod.npy", mod)

        end_time = time.time()
        elapsed_time = end_time - start_time
        log_file.write(f"Time taken for MOD computation for N={N}: {elapsed_time:.2f} seconds\n")