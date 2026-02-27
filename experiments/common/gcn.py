"""Code for GCN model."""

import os
import sys
from pathlib import Path

# Add experiments directory to path
experiments_dir = Path(__file__).parent.parent
sys.path.insert(0, str(experiments_dir))

import numpy as np
import torch
import optuna
import pickle
import networkx as nx
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import r2_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.model_selection import train_test_split

from common.data_preprocessing import prepare_data_images

# ============================================================================
# Model Definition
# ============================================================================

class SmallGCN(nn.Module):
    """
    Graph Convolutional Network for regression on Sankey diagram graphs.
    
    Architecture:
    - Multiple GCN layers with ReLU activation and dropout
    - Global mean pooling to aggregate node features
    - Two-layer MLP regressor for final prediction
    
    Args:
        in_dim: Input feature dimension (3 for our setup: constant, degree, layer)
        hidden_dim: Hidden dimension for GCN layers
        num_layers: Number of GCN layers
        dropout: Dropout probability
    """
    def __init__(self, in_dim=3, hidden_dim=64, num_layers=2, dropout=0.0):
        super().__init__()
        # Build GCN layers: first layer maps from input to hidden, rest are hidden->hidden
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim, add_self_loops=False, normalize=False))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=False, normalize=False))
        
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Regressor: hidden_dim -> hidden_dim//2 -> 1
        mid = max(hidden_dim // 2, 1)
        self.regressor = nn.Sequential(nn.Linear(hidden_dim, mid), nn.ReLU(), nn.Linear(mid, 1))

    def forward(self, data):
        """
        Forward pass through the GCN.
        
        Args:
            data: PyG Data object with x, edge_index, edge_weight, and batch
            
        Returns:
            Predicted values (1D tensor)
        """
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # Handle both batched and single-graph scenarios
        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        
        # Apply GCN layers with activation and dropout
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = self.act(x)
            x = self.dropout(x)
        
        # Global pooling: aggregate all node features per graph
        g = global_mean_pool(x, batch)
        
        # Regression head: graph embedding -> scalar prediction
        return self.regressor(g).squeeze(-1)

# ============================================================================
# Data Processing Functions
# ============================================================================

def compute_sankey(partitions):
    """
    Convert partitions to Sankey diagram structure with layer information.
    
    Creates nodes and links between consecutive partition levels, tracking
    which layer each node belongs to (used later as a node feature).
    
    Args:
        partitions: List of partition arrays, where partitions[m] contains
                   cluster assignments at level m
        
    Returns:
        Dictionary with:
        - 'nodes': List of node dicts with 'name' and 'layer' keys
        - 'links': List of link dicts with 'source', 'target', 'value' keys
        - 'level': Dict mapping layer index to list of node names
    """
    L = len(partitions)
    nodes, links, levels = [], [], {}
    
    # Process each consecutive pair of partition levels
    for m in range(L - 1):
        p_m = partitions[m]
        n_m = np.max(p_m) + 1  # Number of clusters at level m
        p_m_plus_one = partitions[m + 1]
        n_m_plus_one = np.max(p_m_plus_one) + 1  # Number of clusters at level m+1
        
        # Compute contingency matrix: how elements flow from level m to m+1
        cm = contingency_matrix(p_m, p_m_plus_one)
        
        # Create nodes for first level (only once)
        if m == 0:
            for i in range(n_m):
                nodes.append({"name": f"P{m}_C{i}", "layer": m})
        
        # Create nodes for next level
        for j in range(n_m_plus_one):
            nodes.append({"name": f"P{m+1}_C{j}", "layer": m + 1})
        
        # Create links based on non-zero flows in contingency matrix
        for i in range(n_m):
            for j in range(n_m_plus_one):
                c = cm[i, j]
                if c > 0:
                    links.append({"source": f"P{m}_C{i}", "target": f"P{m+1}_C{j}", "value": int(c)})
        
        # Track which nodes belong to each level
        if m == 0:
            levels[m] = [f"P{m}_C{i}" for i in range(n_m)]
        levels[m + 1] = [f"P{m+1}_C{j}" for j in range(n_m_plus_one)]
    
    return {"nodes": nodes, "links": links, "level": levels}

def compute_sankey_adjacency(partition):
    """
    Compute weighted adjacency matrix and node layer indices from partitions.
    
    Converts the Sankey diagram into a graph representation suitable for GCN:
    - Adjacency matrix captures connectivity and flow weights
    - Layer indices track hierarchical position (used as node feature)
    
    Args:
        partition: List of partition arrays
        
    Returns:
        Tuple of:
        - adjacency_matrix: numpy array (weighted by flow values)
        - layer_indices: numpy array of layer index per node
    """
    # Build Sankey structure
    sankey_data = compute_sankey(partition)
    node_names = [n['name'] for n in sankey_data['nodes']]
    
    # Extract layer information for each node (used as feature later)
    node_layers = np.array([
        int(n.get('layer', int(n['name'].split('_')[0][1:]))) 
        for n in sankey_data['nodes']
    ], dtype=np.int32)
    
    # Build NetworkX graph with weighted edges
    G = nx.Graph()
    for node in sankey_data['nodes']:
        G.add_node(node['name'])
    
    # Add edges with weights from Sankey links (flow values)
    for link in sankey_data['links']:
        w = float(link.get('value', 1.0))
        G.add_edge(link['source'], link['target'], weight=w)
    
    # Convert to adjacency matrix (preserves edge weights)
    A = nx.adjacency_matrix(G, nodelist=node_names, weight="weight").toarray()
    
    return A, node_layers

def load_or_compute_sankey_adjacencies(partitions_train, partitions_val, partitions_test, N, M):
    """
    Load or compute Sankey adjacencies and layer information with caching.
    
    Computes adjacency matrices for all samples in train/val/test splits.
    Uses pickle caching to avoid recomputation on subsequent runs.
    
    Args:
        partitions_train, partitions_val, partitions_test: Partition lists for each split
        N, M: Dataset parameters (used for cache filenames)
        
    Returns:
        Tuple of ((adj_train, layers_train), (adj_val, layers_val), (adj_test, layers_test))
        where each is a tuple of (list of adjacency matrices, list of layer arrays)
    """
    cache_dir = Path("cache/sankey_gcn_v2")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Define cache file paths for each split
    files = {
        "train": cache_dir / f"sankey_adj_layers_train_N{N}_M{M}.pkl",
        "val": cache_dir / f"sankey_adj_layers_val_N{N}_M{M}.pkl",
        "test": cache_dir / f"sankey_adj_layers_test_N{N}_M{M}.pkl",
    }

    # Try to load from cache
    loaded = {}
    all_exist = all(p.exists() for p in files.values())
    if all_exist:
        for split, fp in files.items():
            with open(fp, "rb") as f:
                loaded[split] = pickle.load(f)
        print(f"[Cache] Loaded Sankey adjacencies+layers from {cache_dir}")
        adj_train, layers_train = loaded["train"]["adj"], loaded["train"]["layers"]
        adj_val, layers_val = loaded["val"]["adj"], loaded["val"]["layers"]
        adj_test, layers_test = loaded["test"]["adj"], loaded["test"]["layers"]
        return (adj_train, layers_train), (adj_val, layers_val), (adj_test, layers_test)

    # Compute from scratch if cache doesn't exist
    print("[Cache] Computing Sankey adjacencies+layers (no existing pickle found)...")
    
    # Process train split
    adj_train, layers_train = [], []
    for p in partitions_train:
        A, Ls = compute_sankey_adjacency(p)
        adj_train.append(A)
        layers_train.append(Ls)
    
    # Process validation split
    adj_val, layers_val = [], []
    for p in partitions_val:
        A, Ls = compute_sankey_adjacency(p)
        adj_val.append(A)
        layers_val.append(Ls)
    
    # Process test split
    adj_test, layers_test = [], []
    for p in partitions_test:
        A, Ls = compute_sankey_adjacency(p)
        adj_test.append(A)
        layers_test.append(Ls)

    # Save to cache for future use
    for split, adj, layers in [("train", adj_train, layers_train), 
                                ("val", adj_val, layers_val), 
                                ("test", adj_test, layers_test)]:
        with open(files[split], "wb") as f:
            pickle.dump({"adj": adj, "layers": layers}, f, protocol=4)
    print(f"[Cache] Saved Sankey adjacencies+layers to {cache_dir}")
    
    return (adj_train, layers_train), (adj_val, layers_val), (adj_test, layers_test)

def adjacency_to_edge_index_weight(A: np.ndarray):
    """
    Convert adjacency matrix to PyG edge format with normalized weights.
    
    Performs:
    1. Extraction of edges from adjacency matrix
    2. Addition of self-loops
    3. Symmetric normalization: D^(-1/2) * A * D^(-1/2)
    
    Args:
        A: Adjacency matrix (numpy array, can be weighted)
        
    Returns:
        Tuple of:
        - edge_index: [2, num_edges] tensor with source/target indices
        - edge_weight: [num_edges] tensor with normalized weights
    """
    A = A.astype(np.float32)
    n = A.shape[0]
    
    # Extract edges from adjacency matrix
    rows, cols = np.nonzero(A)
    weights = A[rows, cols].astype(np.float32)
    
    # Add self-loops for better representation learning
    self_idx = np.arange(n, dtype=np.int64)
    rows = np.concatenate([rows, self_idx])
    cols = np.concatenate([cols, self_idx])
    weights = np.concatenate([weights, np.ones(n, dtype=np.float32)])
    
    # Compute degree for symmetric normalization
    deg = A.sum(axis=1).astype(np.float32) + 1.0  # +1 for self-loops
    d_inv_sqrt = np.power(deg + 1e-8, -0.5)  # D^(-1/2)
    
    # Apply symmetric normalization: w_ij * d_i^(-1/2) * d_j^(-1/2)
    norm_w = weights * d_inv_sqrt[rows] * d_inv_sqrt[cols]
    
    # Convert to PyTorch tensors in COO format
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_weight = torch.tensor(norm_w, dtype=torch.float32)
    
    return edge_index, edge_weight

def build_node_features(A: np.ndarray, layers: np.ndarray) -> torch.Tensor:
    """
    Build 3D node features: [constant_1, normalized_degree, normalized_layer].
    
    Feature design:
    1. Constant 1: Provides a learnable bias term
    2. Normalized degree: Captures node connectivity (0 to 1)
    3. Normalized layer: Encodes hierarchical position (0 to 1)
    
    Args:
        A: Adjacency matrix
        layers: Node layer indices (0 to max_layer)
        
    Returns:
        Feature tensor of shape [num_nodes, 3]
    """
    # Compute node degrees and normalize to [0, 1]
    deg = A.astype(np.float32).sum(axis=1)
    deg_norm = deg / (deg.max() + 1e-8)
    
    # Normalize layer indices to [0, 1]
    layer = layers.astype(np.float32)
    layer_norm = layer / (layer.max() + 1e-8)
    
    # Stack features: [constant, degree, layer]
    X = np.stack(
        [
            np.ones_like(deg_norm, dtype=np.float32),  # Constant feature
            deg_norm.astype(np.float32),               # Degree feature
            layer_norm.astype(np.float32),             # Layer feature
        ],
        axis=1,
    )
    return torch.tensor(X, dtype=torch.float32)

def build_pyg_graphs(adj_list, layers_list, targets):
    """
    Build PyG Data objects from adjacency matrices, layers, and targets.
    
    Converts raw graph data into PyTorch Geometric format for GCN processing.
    Each graph becomes a Data object with node features, edges, and target value.
    
    Args:
        adj_list: List of adjacency matrices
        layers_list: List of layer index arrays
        targets: Target values (regression targets)
        
    Returns:
        List of PyG Data objects
    """
    data_list = []
    for A, layers, y in zip(adj_list, layers_list, targets):
        # Build node features from adjacency and layer info
        x = build_node_features(A, layers)
        
        # Convert adjacency to edge format with normalized weights
        edge_index, edge_weight = adjacency_to_edge_index_weight(A)
        
        # Create PyG Data object
        data_list.append(
            Data(
                x=x,                                                # Node features [num_nodes, 3]
                edge_index=edge_index,                              # Edge connectivity [2, num_edges]
                edge_weight=edge_weight,                            # Edge weights [num_edges]
                y=torch.tensor([float(y)], dtype=torch.float32),   # Target value
            )
        )
    return data_list

def load_or_build_pyg_graphs(adj_list, layers_list, targets, split: str, N: int, M: int):
    """
    Load or build PyG graphs with caching (v2: includes layer feature).
    
    PyG graphs are more expensive to build than adjacencies, so we cache them
    as torch .pt files for faster loading in subsequent runs.
    
    Args:
        adj_list: List of adjacency matrices
        layers_list: List of layer index arrays
        targets: Target values
        split: Data split name ('train', 'val', or 'test')
        N, M: Dataset parameters (used for cache filenames)
        
    Returns:
        List of PyG Data objects
    """
    cache_dir = Path("cache/pyg_graphs_v2")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Cache filename includes _f3 to indicate 3-feature version
    fp = cache_dir / f"pyg_{split}_N{N}_M{M}_f3.pt"
    
    # Try to load from cache
    if fp.exists():
        try:
            data_list = torch.load(fp, weights_only=False)
            print(f"[Cache] Loaded PyG graphs ({split}) from {fp}")
            return data_list
        except Exception:
            print(f"[Cache] Failed to load {fp}, rebuilding...")
    
    # Build graphs from scratch
    data_list = build_pyg_graphs(adj_list, layers_list, targets)
    
    # Save to cache for future use
    try:
        torch.save(data_list, fp)
        print(f"[Cache] Saved PyG graphs ({split}) to {fp}")
    except Exception as e:
        print(f"[Cache] Could not save PyG graphs to {fp}: {e}")
    
    return data_list

# ============================================================================
# Utility Functions
# ============================================================================

def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across numpy, torch, and CUDA.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_loader(model, loader, device):
    """
    Evaluate model on a data loader (device-agnostic).
    
    Computes loss and R² score on the provided data loader.
    Works on any device (CPU/MPS/CUDA).
    
    Args:
        model: PyTorch model (must be in eval mode or will be set)
        loader: DataLoader or iterable yielding batches
        device: torch.device
        
    Returns:
        Tuple of (avg_loss, r2_score)
    """
    model.eval()
    criterion = nn.MSELoss()
    preds, targs, losses = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            # Move batch to device (non_blocking for performance)
            batch = batch.to(device, non_blocking=True)
            
            # Forward pass
            out = model(batch)
            y = batch.y.view(-1)
            loss = criterion(out, y)
            
            # Collect predictions and targets for R² computation
            losses.append(loss.item())
            preds.extend(out.cpu().tolist())
            targs.extend(y.cpu().tolist())
    
    # Compute metrics
    avg_loss = float(np.mean(losses)) if losses else float("inf")
    r2 = r2_score(targs, preds) if len(targs) > 1 else float("nan")
    
    return avg_loss, r2

def _build_loader(graphs, batch_size, shuffle, device):
    """
    Build DataLoader with appropriate settings for the target device.
    
    Configures pin_memory and num_workers based on device type for
    optimal data loading performance.
    
    Args:
        graphs: List of PyG Data objects
        batch_size: Batch size
        shuffle: Whether to shuffle data
        device: torch.device
        
    Returns:
        DataLoader
    """
    # Enable pin_memory for faster CPU->GPU transfer on CUDA
    pin = device.type == "cuda"
    
    # Use multiple workers only on CUDA for shuffled data to avoid overhead
    workers = 2 if pin and shuffle else (1 if pin else 0)
    
    return DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin,
        num_workers=workers,
        persistent_workers=pin and workers > 0,  # Keep workers alive between epochs
    )

# ============================================================================
# Cache Generation Function
# ============================================================================

def generate_cached_data(N: int, M: int, seed: int = 42):
    """
    Generate cached data exactly as in 251216_gcn_parallel_pyg_cuda_levels_feature.py.
    
    This function orchestrates the full data pipeline:
    1. Load raw partition data
    2. Split into train/val/test
    3. Compute and cache Sankey adjacencies + layers (pickle files)
    4. Build and cache PyG graphs (torch files)
    
    Creates:
    - cache/sankey_gcn_v2/sankey_adj_layers_{split}_N{N}_M{M}.pkl
    - cache/pyg_graphs_v2/pyg_{split}_N{N}_M{M}_f3.pt
    
    for split in {train, val, test}
    
    Args:
        N, M: Dataset parameters
        seed: Random seed for train/val split
        
    Returns:
        Dictionary with split information and graph counts
    """
    set_seed(seed)
    
    print(f"[Cache Generation] Starting for N={N}, M={M}")
    
    # Load raw data from preprocessing module
    (
        y_train_val,
        y_test,
        feature_names,
        feature_names_to_train,
        feature_names_to_test,
    ) = prepare_data_images(N, M)

    partitions_train_val = feature_names_to_train["Partitions"]
    partitions_test = feature_names_to_test["Partitions"]
    
    # Split train/val with stratification to preserve class distribution
    partitions_train, partitions_val, y_train, y_val = train_test_split(
        partitions_train_val, y_train_val, test_size=0.2, random_state=seed, stratify=y_train_val,
    )

    # Generate/cache adjacencies and layers (stage 1: pickle files)
    (adj_train, layers_train), (adj_val, layers_val), (adj_test, layers_test) = load_or_compute_sankey_adjacencies(
        partitions_train, partitions_val, partitions_test, N, M
    )

    # Generate/cache PyG graphs (stage 2: torch files)
    train_graphs = load_or_build_pyg_graphs(adj_train, layers_train, y_train, "train", N, M)
    val_graphs = load_or_build_pyg_graphs(adj_val, layers_val, y_val, "val", N, M)
    test_graphs = load_or_build_pyg_graphs(adj_test, layers_test, y_test, "test", N, M)

    # Prepare summary information
    summary = {
        "N": N,
        "M": M,
        "seed": seed,
        "num_train": len(train_graphs),
        "num_val": len(val_graphs),
        "num_test": len(test_graphs),
    }
    
    print(f"[Cache Generation] Complete for N={N}, M={M}")
    print(f"  Train: {summary['num_train']} graphs")
    print(f"  Val: {summary['num_val']} graphs")
    print(f"  Test: {summary['num_test']} graphs")
    
    return summary

# ============================================================================
# Model Loading and Inference Functions
# ============================================================================

def _mps_device():
    """Get MPS device if available, otherwise CPU (for Apple Silicon)."""
    return torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else torch.device("cpu")

def _load_study(N: int, M: int):
    """
    Load Optuna study from SQLite database.
    
    Args:
        N, M: Dataset parameters
        
    Returns:
        Optuna Study object
    """
    storage_path = f"sqlite:///results/optuna/gcn/251216_gcn_pyg_M{M}.db"
    study_version = "pyg_gcn_cuda_layers"
    study_name = f"gcn_sankey_pyg_N{N}_M{M}_{study_version}"
    return optuna.load_study(study_name=study_name, storage=storage_path)

def _load_best_model_for_study(study, device: torch.device):
    """
    Load the best model from an Optuna study.
    
    Retrieves hyperparameters and weights from the best trial,
    reconstructs the model, and loads the trained weights.
    
    Args:
        study: Optuna Study object
        device: Target device for model
        
    Returns:
        Tuple of (model, best_trial)
        
    Raises:
        FileNotFoundError: If model weights file doesn't exist
    """
    best_trial = study.best_trial
    weights_path = best_trial.user_attrs.get("weights_path", None).replace("gcn_cuda_layers","gcn")
    
    # Verify weights file exists
    if weights_path is None or not os.path.exists(weights_path):
        raise FileNotFoundError(f"Best trial weights not found. Got: {weights_path}")
    
    # Load model checkpoint
    best_state = torch.load(weights_path, map_location=device)
    h = best_state["hparams"]
    
    # Reconstruct model with saved hyperparameters
    model = SmallGCN(
        in_dim=3, 
        hidden_dim=h["hidden_dim"], 
        num_layers=h["num_layers"], 
        dropout=h["dropout"]
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(best_state["model"])
    model.eval()
    
    return model, best_trial

def _load_test_graphs_cached(N: int, M: int):
    """
    Load cached test graphs from disk.
    
    Args:
        N, M: Dataset parameters
        
    Returns:
        List of PyG Data objects
        
    Raises:
        FileNotFoundError: If cache file doesn't exist
    """
    fp = Path("cache/pyg_graphs_v2") / f"pyg_test_N{N}_M{M}_f3.pt"
    if not fp.exists():
        raise FileNotFoundError(
            f"Cached test graphs not found at {fp}. "
            "Run the training script once to populate the cache."
        )
    return torch.load(fp, weights_only=False)

def predict_test_with_best_model_mps(N: int, M: int):
    """
    Load Optuna best trial model and predict on cached test graphs (MPS/CPU).
    
    This function is designed for inference on Apple Silicon (MPS) or CPU.
    It loads the best model from Optuna hyperparameter search and evaluates
    it on the test set.
    
    Args:
        N, M: Dataset parameters
        
    Returns:
        Tuple of (preds, ys, r2):
        - preds: numpy array of predictions
        - ys: numpy array of ground truth values
        - r2: R² score on test set
    """
    # Get appropriate device (MPS or CPU)
    device = _mps_device()
    
    # Load Optuna study and best model
    study = _load_study(N, M)
    model, best_trial = _load_best_model_for_study(study, device)
    
    # Load cached test graphs
    test_graphs = _load_test_graphs_cached(N, M)

    # Build DataLoader (single batch for simplicity)
    loader = DataLoader(test_graphs, batch_size=len(test_graphs), shuffle=False)
    
    # Run inference
    preds, ys = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            preds.extend(out.detach().cpu().tolist())
            ys.extend(batch.y.view(-1).cpu().tolist())

    # Convert to numpy and compute R²
    preds = np.asarray(preds, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    r2 = r2_score(ys, preds) if len(ys) > 1 else float("nan")
    
    print(f"[Inference/MPS] N={N}, M={M}, Best trial #{best_trial.number}, Test R2={r2:.4f}")
    
    return preds, ys, r2

# Example usage:
# preds, ys, r2 = predict_test_with_best_model_mps(N=10, M=20)
# summary = generate_cached_data(N=10, M=20)