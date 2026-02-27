import numpy as np
import sys
from pathlib import Path

# Add experiments directory to path
experiments_dir = Path(__file__).parent.parent
sys.path.insert(0, str(experiments_dir))

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from common.data_preprocessing import prepare_data_images
from common.gcn import (
    SmallGCN,
    load_or_compute_sankey_adjacencies,
    load_or_build_pyg_graphs,
    evaluate_loader,
    _build_loader,
    set_seed,
)

import torch
from torch import nn
from torch_geometric.loader import DataLoader
import optuna
import os
import json
import warnings

# ============================================================================
# Environment Configuration
# ============================================================================

# Disable TorchInductor CUDA Graphs to avoid AssertionError with dynamic PyG batches
os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "0")

# Cap BLAS threads to reduce contention in parallel Optuna workers
# This prevents CPU oversubscription when running multiple GPU trials
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# Suppress runtime warnings from matrix operations
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*mm.*")

# ============================================================================
# Training Hyperparameters
# ============================================================================

SEED = 42  # Random seed for reproducibility
M = 20  # Number of partition levels
JOBS_PER_GPU = int(os.environ.get("JOBS_PER_GPU", "2"))  # Concurrent trials per GPU
TRAIN_BATCH_SIZE = 32  # Batch size for training
MAX_EPOCHS = 150  # Maximum training epochs
EARLY_STOP_PATIENCE = 10  # Epochs without improvement before stopping
MIN_VAL_IMPROVEMENT = 1e-4  # Minimum validation loss improvement threshold

# Performance flags for RTX 6000 Ada GPUs
USE_AMP = True  # Automatic Mixed Precision for faster training
USE_TORCH_COMPILE = False  # torch.compile disabled for compatibility
PERFORMANCE_MODE = True  # Enable aggressive performance optimizations

# ============================================================================
# GPU Configuration Functions
# ============================================================================

def _configure_cuda():
    """
    Configure CUDA settings for optimal performance on RTX 6000 Ada GPUs.
    
    Enables:
    - High precision matmul (uses Tensor Cores)
    - TF32 for faster compute (Ada generation)
    - CuDNN benchmarking for optimal kernel selection
    """
    if torch.cuda.is_available():
        # Use high precision for matmul operations (Tensor Core optimization)
        torch.set_float32_matmul_precision("high")
        
        # Enable TF32 for faster operations on Ampere/Ada GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable CuDNN benchmarking for faster convolutions (non-deterministic)
        if PERFORMANCE_MODE and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

def print_gpu_info(device_index: int | None = None):
    """
    Print GPU information for debugging and monitoring.
    
    Args:
        device_index: GPU index (None for current device)
    """
    if torch.cuda.is_available():
        idx = device_index if device_index is not None else torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(idx)
        print(f"[GPU:{idx}] {prop.name} | SMs: {prop.multi_processor_count} | Mem: {prop.total_memory/1e9:.2f} GB")

def get_device():
    """
    Get the best available device for training.
    
    Priority: CUDA > MPS > CPU
    
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ============================================================================
# Data Loading Optimization
# ============================================================================

class PrefetchLoader:
    """
    Prefetch next batch to GPU on a separate CUDA stream.
    
    Overlaps CPU->GPU data transfer (H2D) with GPU compute to hide latency.
    This is a key optimization for GPU utilization on large models.
    
    Usage:
        loader = DataLoader(...)
        prefetch_loader = PrefetchLoader(loader, device)
        for batch in prefetch_loader:
            # batch is already on GPU
            outputs = model(batch)
    """
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        # Create separate CUDA stream for async data transfer
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        it = iter(self.loader)
        
        # Non-CUDA: just move batches to device
        if self.stream is None:
            for batch in it:
                yield batch.to(self.device, non_blocking=True)
            return
        
        # CUDA: prefetch next batch while processing current
        batch = next(it, None)
        if batch is None:
            return
        
        # Start prefetch of first batch
        with torch.cuda.stream(self.stream):
            batch = batch.to(self.device, non_blocking=True)
        
        while True:
            # Wait for prefetch to complete
            torch.cuda.current_stream().wait_stream(self.stream)
            next_batch = batch
            
            # Start prefetch of next batch (if any)
            batch = next(it, None)
            if batch is not None:
                with torch.cuda.stream(self.stream):
                    batch = batch.to(self.device, non_blocking=True)
            
            # Yield current batch (prefetch happens in background)
            yield next_batch
            
            if batch is None:
                break

# ============================================================================
# Optuna Training Function
# ============================================================================

def train_one_trial(trial, train_graphs, val_graphs, device):
    """
    Train a single Optuna trial with given hyperparameters.
    
    This function:
    1. Samples hyperparameters from Optuna search space
    2. Builds and trains a GCN model
    3. Tracks training history
    4. Returns best validation loss and model state
    
    Args:
        trial: Optuna trial object
        train_graphs: List of PyG training graphs
        val_graphs: List of PyG validation graphs
        device: torch.device for training
        
    Returns:
        Tuple of (best_val_loss, best_model_state, training_history)
    """
    # Sample hyperparameters from Optuna search space
    hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64, 128])
    num_layers = trial.suggest_categorical("num_layers", [1, 2, 3])
    lr = trial.suggest_categorical("lr", [0.01, 0.005, 0.001, 0.0005, 0.0001])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.25, 0.5])
    weight_decay = trial.suggest_categorical("weight_decay", [0.0, 1e-4])
    
    # Fixed hyperparameters
    batch_size = TRAIN_BATCH_SIZE
    max_epochs = MAX_EPOCHS
    patience = EARLY_STOP_PATIENCE
    
    # Build data loaders
    train_loader = _build_loader(train_graphs, batch_size, True, device)
    val_loader = _build_loader(val_graphs, len(val_graphs), False, device)

    # Wrap with prefetch for GPU efficiency
    train_prefetch = PrefetchLoader(train_loader, device)
    val_prefetch = PrefetchLoader(val_loader, device)

    # Build model with sampled hyperparameters
    model = SmallGCN(in_dim=3, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device)
    
    # Optimizer with weight decay for regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler: reduce LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.6, patience=2, min_lr=1e-5
    )
    
    criterion = nn.MSELoss()
    
    # Mixed precision training for faster compute on modern GPUs
    use_amp = USE_AMP and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # Training state tracking
    best_val, best_state, early = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": [], "val_r2": [], "lr": []}
    
    # Training loop
    for ep in range(max_epochs):
        # Training phase
        model.train()
        batch_losses = []
        for batch in train_prefetch:
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Forward pass with mixed precision
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                out = model(batch)
                loss = criterion(out, batch.y.view(-1))
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevent gradient explosion
            scaler.step(optimizer)
            scaler.update()
            
            batch_losses.append(loss.item())
        
        train_loss = float(np.mean(batch_losses)) if batch_losses else float("inf")

        # Validation phase
        val_loss, val_r2 = evaluate_loader(model, val_prefetch, device)
        scheduler.step(val_loss)
        
        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_r2"].append(val_r2)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        
        # Report to Optuna for pruning
        trial.report(val_loss, ep)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # Early stopping logic
        if val_loss < best_val - MIN_VAL_IMPROVEMENT:
            best_val = val_loss
            best_state = {
                "model": model.state_dict(),
                "hparams": {
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "lr": lr,
                    "dropout": dropout,
                    "weight_decay": weight_decay,
                },
            }
            early = 0
        else:
            early += 1
            if early >= patience:
                break  # Stop training if no improvement
    
    return best_val, best_state, history

# ============================================================================
# Optuna Study Orchestration
# ============================================================================

def run_optuna(adj_train, y_train, adj_val, y_val, adj_test, y_test, N, M):
    """
    Run Optuna hyperparameter optimization study.
    
    This function:
    1. Sets up Optuna study with TPE sampler and median pruner
    2. Runs parallel trials across multiple GPUs
    3. Saves best models and training histories
    4. Evaluates best model on train/val/test sets
    
    Args:
        adj_train, y_train: Training adjacencies/layers and targets
        adj_val, y_val: Validation adjacencies/layers and targets
        adj_test, y_test: Test adjacencies/layers and targets
        N, M: Dataset parameters
    """
    # Setup results directory
    RESULTS_DIR = "results/optuna/gcn_cuda_layers/models"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    storage_path = f"sqlite:///results/optuna/gcn_cuda_layers/251216_gcn_pyg_M{M}.db"

    # Initialize random seeds and CUDA settings
    set_seed(SEED)
    _configure_cuda()

    # GPUs available for parallel trials (as reported by gpustat)
    FREE_GPUS = [3, 6]

    # Load or build cached PyG graphs (v2 with layer feature)
    train_graphs = load_or_build_pyg_graphs(adj_train[0], adj_train[1], y_train, "train", N, M)
    val_graphs = load_or_build_pyg_graphs(adj_val[0], adj_val[1], y_val, "val", N, M)
    test_graphs = load_or_build_pyg_graphs(adj_test[0], adj_test[1], y_test, "test", N, M)

    # Study configuration
    study_version = "pyg_gcn_cuda_layers"
    study_name = f"gcn_sankey_pyg_N{N}_M{M}_{study_version}"

    # Sampler: TPE for efficient hyperparameter search
    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10)
    
    # Pruner: Stop unpromising trials early to save compute
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=5)
    
    # Create or load existing study
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage_path,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    def objective(trial):
        """
        Optuna objective function for a single trial.
        
        Assigns GPU, trains model, evaluates on all splits, saves results.
        """
        # Assign GPU for this trial (round-robin across FREE_GPUS)
        if torch.cuda.is_available() and FREE_GPUS:
            gpu_id = FREE_GPUS[trial.number % len(FREE_GPUS)]
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
            print_gpu_info(gpu_id)
            trial.set_user_attr("gpu_id", gpu_id)
        else:
            device = get_device()
            if device.type == "cuda":
                print_gpu_info()

        # Train model
        best_val, best_state, history = train_one_trial(trial, train_graphs, val_graphs, device)
        
        # Save training history
        history_path = f"{RESULTS_DIR}/history_pyg_gcn_N{N}_M{M}_trial_{trial.number}.json"
        with open(history_path, "w") as f:
            json.dump(history, f)
        trial.set_user_attr("history_path", history_path)

        # Save best model and evaluate on all splits
        if best_state is not None:
            weights_path = f"{RESULTS_DIR}/best_model_pyg_gcn_N{N}_M{M}_trial_{trial.number}.pth"
            torch.save(best_state, weights_path)
            trial.set_user_attr("weights_path", weights_path)

            # Rebuild model with best hyperparameters
            model = SmallGCN(
                in_dim=3,
                hidden_dim=best_state["hparams"]["hidden_dim"],
                num_layers=best_state["hparams"]["num_layers"],
                dropout=best_state["hparams"]["dropout"],
            ).to(device)
            model.load_state_dict(best_state["model"])

            # Build loaders for final evaluation
            train_loader = _build_loader(train_graphs, len(train_graphs), False, device)
            val_loader = _build_loader(val_graphs, len(val_graphs), False, device)
            test_loader = _build_loader(test_graphs, len(test_graphs), False, device)

            # Wrap with prefetch
            train_loader = PrefetchLoader(train_loader, device)
            val_loader = PrefetchLoader(val_loader, device)
            test_loader = PrefetchLoader(test_loader, device)

            # Evaluate on all splits
            tr_loss, tr_r2 = evaluate_loader(model, train_loader, device)
            vl_loss, vl_r2 = evaluate_loader(model, val_loader, device)
            ts_loss, ts_r2 = evaluate_loader(model, test_loader, device)

            # Record metrics in trial attributes
            trial.set_user_attr("train_loss", tr_loss)
            trial.set_user_attr("train_r2_score", tr_r2)
            trial.set_user_attr("val_loss", vl_loss)
            trial.set_user_attr("val_r2_score", vl_r2)
            trial.set_user_attr("test_loss", ts_loss)
            trial.set_user_attr("test_r2_score", ts_r2)

        return best_val

    # Run parallel optimization across GPUs
    parallel_jobs = min(len(FREE_GPUS) * JOBS_PER_GPU, 70)
    print(f"[Optuna] parallel_jobs={parallel_jobs} (FREE_GPUS={FREE_GPUS}, JOBS_PER_GPU={JOBS_PER_GPU})")
    study.optimize(objective, n_trials=100, n_jobs=parallel_jobs, show_progress_bar=True)

    # Save study results to CSV
    study.trials_dataframe().to_csv(
        f"{RESULTS_DIR}/study_results_pyg_gcn_N{N}_M{M}_{study_version}.csv", index=False
    )

    # Print best trial results
    best_trial = study.best_trial
    print(
        f"[PyG GCN] Device: {device}, N: {N}, M: {M}, Best params: {best_trial.params}, "
        f"Val Loss: {best_trial.value:.4f}, Test R2: {best_trial.user_attrs.get('test_r2_score')}"
    )

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Run experiments for different dataset sizes
    for N in [5, 10]:
        # Load preprocessed data
        (
            y_train_val,
            y_test,
            feature_names,
            feature_names_to_train,
            feature_names_to_test,
        ) = prepare_data_images(N, M)

        # Extract partition data
        partitions_train_val = feature_names_to_train["Partitions"]
        partitions_test = feature_names_to_test["Partitions"]
        
        # Split train/val with stratification
        partitions_train, partitions_val, y_train, y_val = train_test_split(
            partitions_train_val, y_train_val, test_size=0.2, random_state=SEED, stratify=y_train_val,
        )

        # Compute Sankey adjacencies and layers (with caching)
        (sankey_adjacencies_train, layers_train), \
        (sankey_adjacencies_val, layers_val), \
        (sankey_adjacencies_test, layers_test) = load_or_compute_sankey_adjacencies(
            partitions_train, partitions_val, partitions_test, N, M
        )

        # Run Optuna hyperparameter optimization
        run_optuna(
            (sankey_adjacencies_train, layers_train),
            y_train,
            (sankey_adjacencies_val, layers_val),
            y_val,
            (sankey_adjacencies_test, layers_test),
            y_test,
            N,
            M,
        )



