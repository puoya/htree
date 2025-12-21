import os
import math
import torch
import numpy as np
from torch import Tensor
import scipy.linalg as la
import torch.optim as optim
import scipy.sparse.linalg as spla
from typing import Optional, Any, Tuple, List

import htree.conf as conf
import htree.embedding as embedding
###########################################################################
def _project_to_hyperboloid(
    embedding: np.ndarray, 
    max_iter: int = 100,
    tol: float = 1e-100,
) -> np.ndarray:
    """
    Project points onto the hyperboloid via Newton iteration.
    
    Finds scale factor α for each point such that the hyperboloid constraint
    x₀² - ||x_spatial||² = 1 is satisfied, where x₀ is the time component.
    
    Args:
        embedding: Coordinates of shape (dim + 1, n) with time in row 0.
        max_iter: Maximum Newton iterations.
        tol: Convergence tolerance for scale factor updates.
    
    Returns:
        Projected embedding satisfying the hyperboloid constraint.
    """
    n = embedding.shape[1]
    time_target = embedding[0].copy()
    spatial_norm_sq = np.einsum("ij,ij->j", embedding[1:], embedding[1:])
    
    scale = np.ones(n, dtype=np.float64)
    active = np.ones(n, dtype=bool)
    
    for _ in range(max_iter):
        if not active.any():
            break
        
        s, t, r2 = scale[active], time_target[active], spatial_norm_sq[active]
        scaled_r2 = r2 * s * s
        lorentz = np.sqrt(1 + scaled_r2)
        residual = lorentz - t
        
        # Newton step: f(s) measures projection error, solve via f'(s)/f''(s)
        f_prime = 2 * r2 * (s * residual / lorentz + s - 1)
        f_double = 2 * r2 * (1 + residual / lorentz + scaled_r2 * t / lorentz**3)
        
        safe = np.abs(f_double) > 1e-200
        delta = np.divide(f_prime, f_double, where=safe, out=np.zeros_like(f_prime))
        
        active_idx = np.flatnonzero(active)
        scale[active_idx] -= delta
        active[active_idx[(np.abs(delta) < tol) | ~safe]] = False
    
    # Apply projection: scale spatial components, recompute time from constraint
    embedding[1:] *= scale
    embedding[0] = np.sqrt(1 + spatial_norm_sq * scale**2)
    return embedding
###########################################################################
def naive_embedding(
    dist_mat: torch.Tensor,
    dim: int,
    geometry: str = "euclidean",
) -> torch.Tensor:
    """
    Compute embedding from a distance matrix using spectral factorization.

    For Euclidean geometry: classical MDS via double-centered Gram matrix.
    Uses LinearOperator for memory-efficient partial eigendecomposition when
    dim < n. Falls back to GPU-accelerated full decomposition otherwise.

    For Hyperbolic geometry: Lorentzian Gram factorization with hyperboloid
    projection via Newton iteration. Uses float64 for numerical stability.

    Args:
        dist_mat: Pairwise distance matrix of shape (n, n).
        dim: Target embedding dimension.
        geometry: Either "euclidean" or "hyperbolic".

    Returns:
        Embedding coordinates of shape (dim, n) for Euclidean
        or (dim + 1, n) for Hyperbolic.

    Raises:
        ValueError: If geometry is not recognized or dim < 1 for hyperbolic.
    """
    if geometry not in ("euclidean", "hyperbolic"):
        raise ValueError(f"Unknown geometry: {geometry}. Use 'euclidean' or 'hyperbolic'.")
    if geometry == "hyperbolic" and dim < 1:
        raise ValueError("Target dimension must be at least 1 for hyperbolic geometry.")
    n = dist_mat.shape[0]
    device, dtype = dist_mat.device, dist_mat.dtype
    embed_dim = min(dim, n if geometry == "euclidean" else n - 1)
    if geometry == "euclidean":
        # ─────────────────────────────────────────────────────────────────────
        # Euclidean: Classical MDS via double-centered Gram matrix
        # ─────────────────────────────────────────────────────────────────────

        # Gram matrix: implicit double-centered form from squared distances
        dist_sq = dist_mat.detach().cpu().numpy()
        def gram_matvec(v):
            """LinearOperator avoids materializing G, computing G @ v on-the-fly."""
            v_centered = v - v.mean()
            result = dist_sq @ v_centered
            return -0.5 * (result - result.mean())

        gram_op = spla.LinearOperator(
            shape=(n, n), matvec=gram_matvec, rmatvec=gram_matvec, dtype=dist_sq.dtype
        )

        if embed_dim < n:
            # Partial eigendecomposition: largest magnitude eigenvalues
            eigenvalues, eigenvectors = spla.eigsh(gram_op, k=embed_dim, which="LM")
        else:
            # Full spectrum requested: materialize Gram matrix and use dense solver
            row_mean = dist_sq.mean(axis=1, keepdims=True)
            col_mean = dist_sq.mean(axis=0, keepdims=True)
            total_mean = dist_sq.mean()
            G = -0.5 * (dist_sq - row_mean - col_mean + total_mean)
            # Use symmetric dense eigendecomposition
            evals_all, evecs_all = np.linalg.eigh(G)  # ascending order
            # take largest embed_dim eigenpairs (descending)
            order = np.argsort(evals_all)[::-1]
            eigenvalues = evals_all[order][:embed_dim]
            eigenvectors = evecs_all[:, order][:, :embed_dim]

        # Sort descending and clamp negative eigenvalues to zero
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0)
        eigenvectors = eigenvectors[:, order]
        # Embedding: X = (V Λ^{1/2})^T → shape (embed_dim, n)
        embedding = torch.as_tensor(
            (eigenvectors * np.sqrt(eigenvalues)).T, dtype=dtype, device=device
        )
    else:
        # ─────────────────────────────────────────────────────────────────────
        # Hyperbolic: Lorentzian Gram factorization + hyperboloid projection
        # ─────────────────────────────────────────────────────────────────────
        # Gram matrix: Lorentzian form G_ij = -cosh(d_ij)
        dist = dist_mat.detach().cpu().to(torch.float64).numpy()
        gram = -np.cosh(dist)
        # Partial eigendecomposition: timelike (most negative) + spacelike (most positive)
        eval_time, evec_time = la.eigh(gram, subset_by_index=[0, 0])
        eval_space, evec_space = la.eigh(gram, subset_by_index=[n - embed_dim, n - 1])
        # Reorder spacelike descending and clamp negative eigenvalues to zero
        eval_space = np.maximum(eval_space[::-1], 0)
        evec_space = evec_space[:, ::-1]
        # Embedding: X = (V |Λ|^{1/2})^T → shape (dim + 1, n)
        # X[0] = time component, X[1:] = spatial components
        embedding = np.zeros((embed_dim + 1, n), dtype=np.float64)
        embedding[0] = np.sqrt(-eval_time[0]) * evec_time[:, 0]
        embedding[1 : embed_dim + 1] = np.sqrt(eval_space)[:, None] * evec_space.T
        # Ensure consistent orientation (positive time component)
        if embedding[0, 0] < 0:
            embedding = -embedding
        # Project onto hyperboloid: x₀² - ||x_spatial||² = 1
        embedding = torch.as_tensor(
            _project_to_hyperboloid(embedding), dtype=torch.float64, device=device
        )
    # Pad with zeros so the first axis is always the requested dimension
    target_rows = dim if geometry == "euclidean" else (dim + 1)
    cur_rows = embedding.shape[0]
    if cur_rows < target_rows:
        embedding = torch.cat([embedding, embedding.new_zeros(target_rows - cur_rows, n)], dim=0)

    return embedding
###########################################################################
def _setup_params(dist_mat, **kwargs):
    """Merge kwargs with defaults and initialize dynamic functions."""
    defaults = {
        "init_pts": None,
        "epochs": conf.TOTAL_EPOCHS,
        "log_fn": None,
        "lr_fn": None,
        "weight_exp_fn": None,
        "scale_fn": None,
        "lr_init": conf.INITIAL_LEARNING_RATE,
        "save_mode": conf.ENABLE_SAVE_MODE,
        "dist_cutoff": conf.MAX_RANGE,
        "time_stamp": "",
        "path": conf.OUTPUT_VIDEO_DIRECTORY,
    }
    params = {k: kwargs.get(k, v) for k, v in defaults.items()}
    if params["weight_exp_fn"] is None:
        params["weight_exp_fn"] = lambda x1, x2, x3=None: compute_weight(x1, x2)
    if params["lr_fn"] is None:
        log_distances = torch.log10(dist_mat[dist_mat > 0])
        scale = (log_distances.mean() - log_distances.min()).item()
        params["lr_fn"] = lambda x1, x2, x3: compute_lr(x1, x2, x3, scale=log_span(dist_mat).item())
    if params["save_mode"]:
        params['path'] = os.path.join(conf.OUTPUT_DIRECTORY, params["time_stamp"].strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(params['path'], exist_ok=True)
        rel_err_denom = dist_mat.clone().fill_diagonal_(1.0)
        weight_exps, lrs = [], []
    if params["scale_fn"] is None:
        params["scale_fn"] = lambda x1, x2, x3=None: compute_scale(x1, x2)
    return params
###########################################################################
def precise_embedding(dist_mat: torch.Tensor, dim: int, geometry: str, return_history: bool = False, **kwargs):
    """Compute precise embeddings in Euclidean or Hyperbolic geometry."""
    
    # Input validation
    if not isinstance(dist_mat, torch.Tensor):
        raise ValueError("The 'dist_mat' must be a torch.Tensor.")
    if geometry not in {"euclidean", "hyperbolic"}:
        raise ValueError(f"Unknown geometry: {geometry}")

    # Configuration and initialization
    params = _setup_params(dist_mat, **kwargs)
    n, device, dtype = dist_mat.size(0), dist_mat.device, dist_mat.dtype
    is_hyp = geometry == "hyperbolic"
    pad = len(str(params["epochs"]))

    # Initialize coordinates (tangent space for hyperbolic, ambient space for euclidean)
    init_pts = params.get("init_pts")
    if init_pts is None:
        x = torch.empty(dim, n, device=device, dtype=dtype).uniform_(0.0, 0.01)
    else:
        x = hyperbolic_log(init_pts) if is_hyp else init_pts.clone()
    x = x.clone().detach().requires_grad_(True)

    # Optimizer and tracking history
    optimizer = optim.Adam([x], lr=params["lr_init"])
    history = {"costs": [], "weight_exps": [], "lrs": [], "scales": []}
    scale = torch.tensor(1.0, device=device, dtype=dtype)

    # Training loop
    for epoch in range(params["epochs"]):
        optimizer.zero_grad(set_to_none=True)
        # Compute adaptive weight matrix
        weight_exp = params["weight_exp_fn"](epoch, params["epochs"], history["costs"])
        scaled_target = scale * dist_mat if is_hyp else dist_mat
        weight_mat = scaled_target.clamp(min=1e-12).pow(weight_exp).fill_diagonal_(1)
        scale_learning = params["scale_fn"](epoch, params["epochs"], history["costs"]) if is_hyp else False
        # Compute pairwise distances (geometry-specific)
        if is_hyp:
            pts = hyperbolic_exp(x)
            pts_flipped = pts.clone()
            pts_flipped[0, :] *= -1
            dist = torch.arccosh(-(pts.T @ pts_flipped).clamp(max=-1))
            if scale_learning:
                scale = (dist.pow(2).sum() / (dist * dist_mat).sum()).detach()
                scaled_target = scale * dist_mat
        else:
            pts = x.t()
            sq_norms = pts.pow(2).sum(dim=1)
            dist = torch.addmm(
                sq_norms.unsqueeze(1) + sq_norms, pts, pts.t(), beta=1.0, alpha=-2.0
            ).clamp_(min=0.0)

        # Compute normalized weighted cost
        weighted_diff = (dist - scaled_target) * weight_mat
        weighted_denom = scaled_target * weight_mat
        cost = torch.norm(weighted_diff, p="fro") ** 2 / torch.norm(weighted_denom, p="fro") ** 2
        cost.backward(retain_graph=is_hyp)
        # Sanitize gradients and step
        if x.grad is not None and x.grad.isnan().any():
            x.grad.nan_to_num_(nan=0.0)
        optimizer.step()
        # Compute RMSE
        with torch.no_grad():
            rel_err = (dist / scaled_target.clamp(min=1e-10)).sub_(1.0).pow_(2)
            rel_err.fill_diagonal_(0.0)
            tri_idx = torch.triu_indices(n, n, offset=1)
            rmse = torch.sqrt(rel_err[tri_idx[0], tri_idx[1]].mean()).item()
        history["rmse"] = history.get("rmse", [])
        history["rmse"].append(rmse)
        # Update learning rate
        lr = params["lr_fn"](epoch, params["epochs"], history["costs"]) * params["lr_init"]
        optimizer.param_groups[0]["lr"] = lr
        # Record history
        history["costs"].append(cost.item())
        history["weight_exps"].append(weight_exp)
        history["scales"].append(scale.item() if is_hyp else 1.0)
        history["lrs"].append(lr)
        # Logging
        if params["log_fn"]:
            params["log_fn"](
                f"[Epoch {epoch + 1:0{pad}d}/{params['epochs']}] "
                f"Cost: {cost.item():.8f}, Scale: {history['scales'][-1]:.8f}, "
                f"Learning Rate: {lr:.10f}, Weight Exponent: {weight_exp:.8f}, "
                f"Scale Learning: {'Yes' if scale_learning else 'No'}"
            )
        # Per-epoch relative error save
        if params["save_mode"]:
            with torch.no_grad():
                rel_err = (dist / scaled_target).sub_(1.0).pow_(2).fill_diagonal_(0.0)
                np.save(os.path.join(params["path"], f"RE_{epoch + 1}.npy"), rel_err.cpu().numpy())
    # Save all tracking histories
    if params["save_mode"]:
        for name, key in [("costs", "costs"), ("learning_rates", "lrs"),
                          ("weight_exponents", "weight_exps"), ("scales", "scales")]:
            np.save(os.path.join(params["path"], f"{name}.npy"), history[key])
    # Return final embedding (with scale for hyperbolic)
    embedding = hyperbolic_exp(x.detach()) if is_hyp else x.detach()
    if return_history:
        return (embedding, scale.detach(), history) if is_hyp else (embedding, history)
    return (embedding, scale.detach()) if is_hyp else embedding
###########################################################################
def lorentz_prod(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the Lorentzian inner product of two tensors."""
    if x.shape != y.shape:
        raise ValueError("Input tensors must have the same shape.")
    x, y = x.flatten(), y.flatten()
    return torch.dot(x[1:], y[1:]) - x[0] * y[0]
###########################################################################
def hyperbolic_log(X: torch.Tensor) -> torch.Tensor:
    """Compute the hyperbolic logarithm map for given points in hyperbolic space."""
    D, N = X.shape
    if D < 2:
        raise ValueError("Dimension of points must be at least 1 (D >= 1).")

    base = torch.zeros(D, dtype=X.dtype, device=X.device)
    base[0] = 1.0

    # Lorentz product with base simplifies to -X[0, :]
    theta = torch.acosh(torch.clamp(X[0, :], min=1.0))
    scale = torch.where(theta != 0, theta / torch.sinh(theta), torch.ones_like(theta))
    tangents = scale * (X - base[:, None] * torch.cosh(theta))

    return tangents[1:]
###########################################################################
def hyperbolic_exp(V: torch.Tensor) -> torch.Tensor:
    """Compute the hyperbolic exponential map for tangent vectors."""
    D, N = V.shape
    V = torch.cat((torch.zeros(1, N, dtype=V.dtype, device=V.device), V), dim=0)

    norm_v = torch.norm(V, dim=0)
    scale = torch.where(norm_v != 0, torch.sinh(norm_v) / norm_v, torch.ones_like(norm_v))

    exp_map = scale * V
    exp_map[0] = torch.sqrt(1 + (scale * norm_v) ** 2)

    return exp_map
###########################################################################
def log_span(matrix: torch.Tensor) -> torch.Tensor:
    """Compute the span between the mean and minimum log10 distances in a square distance matrix."""
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")
    log_distances = torch.log10(matrix[matrix > 0])

    return log_distances.mean() - log_distances.min()
###########################################################################
def compute_lr(
    epoch: int,
    total_epochs: int,
    losses: torch.Tensor,
    scale: float,
) -> float:
    """
    Compute adaptive learning rate based on loss trends and epoch progression.

    This function implements a two-phase learning rate schedule:

    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ PHASE 1: ADAPTIVE ADJUSTMENT (epochs 0 to no_weight_epochs)                 │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │ During early training, the learning rate adapts based on loss behavior      │
    │ within a sliding window:                                                    │
    │                                                                             │
    │   • If loss increases too frequently → LR decreases (training unstable)     │
    │   • If loss consistently decreases  → LR increases (can learn faster)       │
    │   • Otherwise                       → LR unchanged (stable progress)        │
    │                                                                             │
    │ Rationale: Early training benefits from reactive adjustments. If the model  │
    │ oscillates (loss spikes), we slow down. If it converges smoothly, we        │
    │ accelerate to escape shallow minima faster.                                 │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │ PHASE 2: EXPONENTIAL DECAY (epochs no_weight_epochs to total_epochs)        │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │ During later training, the learning rate follows a smooth exponential       │
    │ decay controlled by the `scale` parameter:                                  │
    │                                                                             │
    │   • Higher scale → more aggressive decay (fine-tuning regime)               │
    │   • Lower scale  → gentler decay (continued exploration)                    │
    │                                                                             │
    │ Rationale: Late-stage training requires progressively smaller steps to      │
    │ fine-tune weights and converge to a local minimum without overshooting.     │
    │ The quadratic exponent creates an accelerating decay curve, ensuring most   │
    │ refinement happens in the final epochs.                                     │
    └─────────────────────────────────────────────────────────────────────────────┘
    """
    if total_epochs <= 1 or scale is None:
        raise ValueError("Total epochs must be > 1 and scale must be provided.")

    # --- Phase boundaries and window parameters ---
    adaptive_phase_end = int(conf.NO_WEIGHT_RATIO * total_epochs)
    window_size = int(conf.WINDOW_RATIO * total_epochs)
    max_allowed_increases = int(conf.INCREASE_COUNT_RATIO * window_size)

    # --- Phase 1: Loss-trend-based adaptive adjustment ---
    lr_multiplier = 1.0
    accumulated_adjustments = []

    analysis_end = min(len(losses), adaptive_phase_end) + 1
    for step in range(1, analysis_end):
        if step < window_size:
            continue

        window = losses[step - window_size : step]
        consecutive_pairs = zip(window[:-1], window[1:])
        increase_count = sum(curr < next_val for curr, next_val in consecutive_pairs)

        if increase_count > max_allowed_increases:
            # Loss increasing too often → reduce LR to stabilize
            accumulated_adjustments.append(conf.DECREASE_FACTOR)
        elif all(curr > next_val for curr, next_val in zip(window[:-1], window[1:])):
            # Loss consistently decreasing → increase LR to accelerate
            accumulated_adjustments.append(conf.INCREASE_FACTOR)
        else:
            # Stable trend → maintain current rate
            accumulated_adjustments.append(1.0)

    # Apply all accumulated adjustments from the adaptive phase
    if accumulated_adjustments:
        lr_multiplier *= torch.prod(torch.tensor(accumulated_adjustments)).item()

    # --- Phase 2: Exponential decay for fine-grained convergence ---
    if epoch >= adaptive_phase_end:
        decay_phase_length = total_epochs - adaptive_phase_end - 1
        decay_base = torch.tensor(10 ** (-scale / decay_phase_length))

        for decay_step in range(adaptive_phase_end, epoch):
            progress = (decay_step - adaptive_phase_end) / decay_phase_length
            decay_exponent = 2 * progress * torch.log10(decay_base).item()
            lr_multiplier *= 10**decay_exponent

    return lr_multiplier
###########################################################################
def compute_weight(epoch: int, epochs: int) -> float:
    """Calculate the weight exponent based on the epoch and total epochs."""
    
    if epochs <= 1:
        raise ValueError("Total epochs must be > 1.")
    
    no_weight_epochs = int(conf.NO_WEIGHT_RATIO * epochs)
    
    return 0.0 if epoch < no_weight_epochs else -(epoch - no_weight_epochs) / (epochs - 1 - no_weight_epochs)
###########################################################################
def compute_scale(epoch: int, epochs: int) -> bool:
    """Check if scale learning should occur."""
    
    if epochs <= 1:
        raise ValueError("Total epochs must be > 1.")
    
    return epoch < int(conf.CURV_RATIO * epochs)
###########################################################################
def precise_multiembedding(dist_mats, multi_embs, geometry="hyperbolic", **kwargs):
    """
    Refine multi-embeddings via gradient descent on weighted Frobenius loss.

    Supports both Euclidean and Hyperbolic geometries:
    - Hyperbolic: Joint optimization of all trees with shared scale/curvature.
      Optimizes tangent vectors in the Lorentz model using weighted distance 
      matching with adaptive learning rate and weight scheduling.
    - Euclidean: Decoupled optimization - runs precise_embedding per tree in
      parallel (since scale=1 is fixed, trees are independent). Results are
      aggregated to match hyperbolic output format.

    Args:
        dist_mats: List of target distance matrices, each shape (n_i, n_i).
        multi_embs: List of initial embedding objects with .points (dim+1, n) for hyperbolic
                    or (dim, n) for euclidean, and .curvature (hyperbolic only).
        geometry: Either "euclidean" or "hyperbolic".
        **kwargs: Optional parameters:
            epochs: Number of optimization iterations.
            lr_init: Initial learning rate for Adam optimizer.
            log_fn: Logging callback f(message_str).
            lr_fn: Learning rate schedule f(epoch, total, costs) -> multiplier.
            weight_exp_fn: Weight exponent schedule f(epoch, total, costs) -> exponent.
            scale_fn: Scale learning toggle f(epoch, total, costs) -> bool. (hyperbolic only)
            save_mode: Whether to save training curves to disk.
            time_stamp: Timestamp for output directory naming.

    Returns:
        For Hyperbolic: Tuple of (pts_list, curvature) where:
            pts_list: List of refined embeddings, each shape (dim+1, n_i).
            curvature: Negative squared scale factor (scalar).
        For Euclidean: Tuple of (pts_list, None) where:
            pts_list: List of refined embeddings, each shape (dim, n_i).
    """
    from joblib import Parallel, delayed
    
    if geometry not in ("euclidean", "hyperbolic"):
        raise ValueError(f"Unknown geometry: {geometry}. Use 'euclidean' or 'hyperbolic'.")

    # ─────────────────────────────────────────────────────────────────────────
    # Parameter initialization
    # ─────────────────────────────────────────────────────────────────────────
    defaults = {
        'epochs': conf.TOTAL_EPOCHS, 'log_fn': None, 'lr_fn': None,
        'weight_exp_fn': None, 'scale_fn': None, 'lr_init': conf.INITIAL_LEARNING_RATE,
        'save_mode': conf.ENABLE_SAVE_MODE, 'time_stamp': ""
    }
    params = {k: kwargs.get(k, v) for k, v in defaults.items()}
    
    # Default scheduling functions
    log_span_val = log_span(dist_mats[0]).item()
    if params['weight_exp_fn'] is None:
        params['weight_exp_fn'] = lambda e, t, _: compute_weight(e, t)
    if params['scale_fn'] is None:
        params['scale_fn'] = lambda e, t, _: compute_scale(e, t)
    if params['lr_fn'] is None:
        params['lr_fn'] = lambda e, t, _: compute_lr(e, t, _, scale=log_span_val)

    # Output directory setup
    save_path = None
    if params['save_mode']:
        save_path = os.path.join(
            conf.OUTPUT_DIRECTORY, params['time_stamp'].strftime('%Y-%m-%d_%H-%M-%S')
        )
        os.makedirs(save_path, exist_ok=True)
        # Create subdirectories for per-tree data
        for tree_idx in range(len(dist_mats)):
            tree_dir = os.path.join(save_path, f"tree_{tree_idx}")
            os.makedirs(tree_dir, exist_ok=True)

    num_trees = len(dist_mats)
    num_points = [emb.points.shape[1] for emb in multi_embs]
    total_points = sum(num_points)
    epochs = params['epochs']

    # Initialize tracking variables (shared structure for both geometries)
    costs, weight_exps, lrs, scales_history = [], [], [], []
    per_tree_costs = [[] for _ in range(num_trees)]
    per_tree_rmse = [[] for _ in range(num_trees)]
    curvature = None

    if geometry == "euclidean":
        from joblib import Parallel, delayed
        
        n_jobs = min(num_trees, os.cpu_count() or 1)
        if params['log_fn']:
            params['log_fn'](f"[Euclidean] Running decoupled optimization for {num_trees} trees in parallel (n_jobs={n_jobs})")
        
        def _run_single(i):
            emb, hist = precise_embedding(
                dist_mats[i],
                dim=multi_embs[i].points.shape[0],
                geometry="euclidean",
                init_pts=multi_embs[i].points.clone(),
                epochs=epochs,
                lr_init=params['lr_init'],
                weight_exp_fn=params['weight_exp_fn'],
                lr_fn=params['lr_fn'],
                log_fn=None,  # suppress per-tree logging
                save_mode=False,
                return_history=True,
            )
            return i, emb, hist
        
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_run_single)(i) for i in range(num_trees)
        )
        
        results = sorted(results, key=lambda x: x[0])
        pts_list = [r[1] for r in results]
        histories = [r[2] for r in results]
        
        # Aggregate per-tree data
        for tree_idx in range(num_trees):
            per_tree_costs[tree_idx] = histories[tree_idx]["costs"]
            per_tree_rmse[tree_idx] = histories[tree_idx]["rmse"]
        
        # Weighted average cost
        for epoch in range(epochs):
            epoch_cost = sum(
                per_tree_costs[i][epoch] * num_points[i] 
                for i in range(num_trees)
            ) / total_points
            costs.append(epoch_cost)
        weight_exps = histories[0]["weight_exps"]
        
        for epoch in range(epochs):
            epoch_lrs = [histories[i]["lrs"][epoch] for i in range(num_trees)]
            geo_mean = np.exp(np.mean(np.log(np.array(epoch_lrs) + 1e-20)))
            lrs.append(geo_mean)
        
        scales_history = [1.0] * epochs
        
        if params['log_fn']:
            for epoch in range(0, epochs, max(1, epochs // 10)):
                per_tree_str = ", ".join([f"T{i}:{per_tree_costs[i][epoch]:.6f}" for i in range(num_trees)])
                params['log_fn'](
                    f"[Epoch {epoch + 1}/{epochs}], Geometry: euclidean, "
                    f"Avg Cost: {costs[epoch]:.8f}, Weight Exp: {weight_exps[epoch]:.8f}, Scale: 1.0, "
                    f"Per-tree: [{per_tree_str}]"
                )

    else:  # geometry == "hyperbolic"
        device, dtype = multi_embs[0].points.device, multi_embs[0].points.dtype
        dim = multi_embs[0].points.shape[0]
        
        # Pre-compute index slices
        slices = []
        idx = 0
        for n in num_points:
            slices.append((idx, idx + n))
            idx += n
            
        # Pre-compute squared distance matrices
        dist_sq_sums = [dm.pow(2).sum() for dm in dist_mats]

        # Initialize tangent vectors for hyperbolic optimization
        tangents = torch.cat([hyperbolic_log(emb.points) for emb in multi_embs], dim=1)
        tangents = tangents.clone().requires_grad_(True)
        optimizer = optim.Adam([tangents], lr=params['lr_init'])
        
        # Initialize scale from curvature
        s = torch.sqrt(torch.abs(multi_embs[0].curvature))

        # Local helper functions for hyperbolic
        def compute_distance_matrix(pts):
            """Compute hyperbolic distance matrix for points."""
            flipped_pts = pts.clone()
            flipped_pts[0, :] *= -1
            return torch.arccosh(-(pts.T @ flipped_pts).clamp(max=-1))
        
        def get_points_from_params():
            """Get hyperbolic points from tangent vectors."""
            return tangents, lambda t, start, end: hyperbolic_exp(t[:, start:end])

        def compute_cost(pts, dist_mat, weight_mat, scale):
            """Relative weighted Frobenius error, excluding diagonal."""
            dist = compute_distance_matrix(pts)
            residual = (dist - scale * dist_mat) * weight_mat
            reference = scale * dist_mat * weight_mat
            return residual.pow(2).sum() / (reference.pow(2).sum() + 1e-10)

        def compute_relative_error(pts, dist_mat, scale):
            """Compute relative error matrix: ((d_ij / (s * D_ij)) - 1)^2"""
            dist = compute_distance_matrix(pts)
            scaled_target = scale * dist_mat
            rel_err = (dist / (scaled_target + 1e-10)).sub_(1.0).pow_(2)
            rel_err.fill_diagonal_(0.0)
            return rel_err

        def compute_consensus_scale(params_tensor):
            """Compute optimal global scale via closed-form least squares."""
            param_tensor, get_pts = get_points_from_params()
            num, den = 0.0, 0.0
            with torch.no_grad():
                for (start, end), dm, dsq_sum, n in zip(
                    slices, dist_mats, dist_sq_sums, num_points
                ):
                    pts = get_pts(param_tensor, start, end)
                    dist = compute_distance_matrix(pts)
                    dist_sq = dist.pow(2)
                    num += n * dist_sq.sum() / dsq_sum
                    den += n * (dist * dm).sum() / dsq_sum
            return num / den

        # ─────────────────────────────────────────────────────────────────────────
        # Training loop (hyperbolic)
        # ─────────────────────────────────────────────────────────────────────────
        param_tensor, get_pts = get_points_from_params()
        
        for epoch in range(epochs):
            # Scheduling
            p = params['weight_exp_fn'](epoch, epochs, costs)
            weight_exps.append(p)

            # Scale update
            scale_learning = params['scale_fn'](epoch, epochs, costs)
            if scale_learning:
                s = compute_consensus_scale(param_tensor)
                current_scale = s.item() if isinstance(s, torch.Tensor) else s
            else:
                if isinstance(s, torch.Tensor) and s.requires_grad:
                    s = s.detach().item()
                current_scale = s.item() if isinstance(s, torch.Tensor) else s
            
            scales_history.append(current_scale)
            
            # Learning rate update
            lr = params['lr_fn'](epoch, epochs, costs) * params['lr_init']
            optimizer.param_groups[0]['lr'] = lr
            lrs.append(lr)
            
            # Forward pass: accumulate weighted cost
            total_cost = torch.tensor(0.0, device=device, dtype=dtype)
            
            for tree_idx, ((start, end), dm, n) in enumerate(zip(slices, dist_mats, num_points)):
                pts = get_pts(param_tensor, start, end)
                
                # Weight matrix
                weight_mat = (current_scale * dm).pow(p)
                weight_mat.fill_diagonal_(1.0)
                
                # Per-tree cost
                tree_cost = compute_cost(pts, dm, weight_mat, current_scale)
                per_tree_costs[tree_idx].append(tree_cost.item())
                
                total_cost = total_cost + tree_cost * n
                
                # Per-tree relative error
                with torch.no_grad():
                    re_mat = compute_relative_error(pts, dm, current_scale).cpu().numpy()
                    tri_idx = np.triu_indices(re_mat.shape[0], k=1)
                    tri_vals = re_mat[tri_idx[0], tri_idx[1]]
                    rel_err = np.nanmedian(tri_vals)
                    per_tree_rmse[tree_idx].append(rel_err)

            # Backward pass with NaN protection
            optimizer.zero_grad(set_to_none=True)
            total_cost.backward()
            
            grad = param_tensor.grad
            if grad is not None:
                grad.nan_to_num_(nan=0.0)
            optimizer.step()

            avg_cost = total_cost.item() / total_points
            costs.append(avg_cost)
            
            if params['log_fn']:
                per_tree_str = ", ".join([f"T{i}:{c[-1]:.6f}" for i, c in enumerate(per_tree_costs)])
                params['log_fn'](
                    f"[Epoch {epoch + 1}/{epochs}], Geometry: {geometry}, "
                    f"Scale Learning: {'Yes' if scale_learning else 'No'}, "
                    f"Avg Loss: {avg_cost:.8f}, Weight Exp: {p:.8f}, Scale: {current_scale:.8f}, "
                    f"Per-tree: [{per_tree_str}]"
                )

        pts_list = []
        with torch.no_grad():
            for start, end in slices:
                pts_list.append(hyperbolic_exp(tangents[:, start:end].detach()))

        # Compute final curvature
        s_final = s.detach().item() if isinstance(s, torch.Tensor) else s
        curvature = -s_final ** 2

    # ─────────────────────────────────────────────────────────────────────────
    # Save training curves (unified for both geometries)
    # ─────────────────────────────────────────────────────────────────────────
    if save_path:
        # Aggregate data
        np.save(os.path.join(save_path, "weight_exponents.npy"), weight_exps)
        np.save(os.path.join(save_path, "learning_rates.npy"), lrs)
        np.save(os.path.join(save_path, "costs.npy"), costs)
        np.save(os.path.join(save_path, "scales.npy"), scales_history)
        
        # Per-tree costs and rmse
        for tree_idx in range(num_trees):
            tree_dir = os.path.join(save_path, f"tree_{tree_idx}")
            np.save(os.path.join(tree_dir, "costs.npy"), per_tree_costs[tree_idx])
            np.save(os.path.join(tree_dir, "rmse.npy"), per_tree_rmse[tree_idx])
        
        # Save metadata
        metadata = {
            'num_trees': num_trees,
            'num_points': num_points,
            'total_points': total_points,
            'epochs': epochs,
            'geometry': geometry
        }
        np.save(os.path.join(save_path, "metadata.npy"), metadata)

    return pts_list, curvature
###########################################################################
def hyperbolic_PCA(X, d=None):
    """Estimate hyperbolic subspace for X."""
    D, N = X.shape
    d = d or D - 1
    e_vals, e_signs, e_vecs = j_decomposition(X @ X.T / N, d)
    p_mask = e_signs > 0
    p_vals = e_vals[p_mask]
    base = torch.squeeze(e_vecs[:, ~p_mask])
    if base.ndim > 1:
        base = base[:, torch.argmax(e_vals[~p_mask])]
    base = torch.sign(base[0]) * base
    H = torch.cat([base.reshape(D, 1), e_vecs[:, p_mask][:, torch.argsort(p_vals, descending=True)]], dim=1)
    return base, H
###########################################################################
def j_decomposition(Cx, d, power_iters=100):
    device = Cx.device
    dtype = torch.float64
    n = Cx.shape[0]  # n = d + 1

    # Represent J as a 1D tensor: J = diag([-1, 1, ..., 1])
    j_vec = torch.ones(n, dtype=dtype, device=device)
    j_vec[0] = -1

    # Helper to apply J via elementwise multiplication.
    def j_mul(x):
        return j_vec * x

    # J-norm defined as xᵀ J x.
    def J_norm(x):
        return torch.dot(x, j_mul(x))

    # Normalize the base eigenvector: force p[0] >= 0 and scale so that pᵀ J p = -1.
    def normalize_p(x):
        if x[0] < 0:
            x = -x
        factor = torch.sqrt(-J_norm(x))
        return x / factor

    # Normalize other eigenvectors: scale so that vᵀ J v = 1.
    def normalize_v(x):
        factor = torch.sqrt(J_norm(x))
        return x / factor

    # Power iteration refinement for an initial eigenvector candidate.
    def refine_eigenvector(A, x, normalize_fn, iterations=100):
        for _ in range(iterations):
            x = A @ x
            x = normalize_fn(x)
        return x

    # Save the original Frobenius norm for the stopping criterion.
    orig_frob = torch.norm(Cx, p="fro")
    threshold = orig_frob / (10**12)
    tol = 1e-6  # tolerance for imaginary parts

    evals_list = []
    signs_list = []
    evecs_list = []

    # We use an explicit diagonal matrix for J (n is small).
    J_mat = torch.diag(j_vec)
    count = 0

    # --- Base eigenpair computation: leading eigenvector of A = Cx @ J.
    A = Cx @ J_mat
    eigvals, eigvecs = torch.linalg.eig(A)
    # Choose the eigenvector with the largest (absolute) eigenvalue.
    i = torch.argmax(eigvals.abs())
    candidate = eigvecs[:, i]
    # If candidate has significant imaginary parts, refine it via power iterations.
    if torch.max(candidate.imag.abs()) > tol:
        candidate = candidate.real
        candidate = refine_eigenvector(A, candidate, normalize_p, iterations=power_iters)
    else:
        candidate = candidate.real
    p = normalize_p(candidate)
    p_eval = (j_mul(p) @ (Cx @ j_mul(p))).item()
    p_sign = torch.sign(torch.dot(p, j_mul(p))).item()

    evals_list.append(p_eval)
    signs_list.append(p_sign)
    evecs_list.append(p.clone())

    # Deflate: subtract the contribution of the base eigenpair.
    Cx = Cx - p_eval * torch.outer(p, p)
    count += 1

    # --- Compute subsequent eigenpairs until we reach d or the residual is very small.
    while count <= d and torch.norm(Cx, p="fro") >= threshold:
        A = Cx @ J_mat
        eigvals, eigvecs = torch.linalg.eig(A)
        i = torch.argmax(eigvals.abs())
        candidate = eigvecs[:, i]
        if torch.max(candidate.imag.abs()) > tol:
            candidate = candidate.real
            candidate = refine_eigenvector(A, candidate, normalize_v, iterations=power_iters)
        else:
            candidate = candidate.real
        v = normalize_v(candidate)
        v_eval = (j_mul(v) @ (Cx @ j_mul(v))).item()
        v_sign = torch.sign(torch.dot(v, j_mul(v))).item()

        evals_list.append(v_eval)
        signs_list.append(v_sign)
        evecs_list.append(v.clone())

        Cx = Cx - v_eval * torch.outer(v, v)
        count += 1

    # --- If fewer than d eigenpairs were obtained, compute the rest as zero eigenvalues.
    while count <= d:
        # Start with a random vector.
        x = torch.randn(n, dtype=dtype, device=device)
        # Remove components in the directions of already computed eigenvectors.
        for j in range(len(evecs_list)):
            u = evecs_list[j]
            # Use the stored sign; note: uᵀ J u is -1 for p and 1 for other vectors.
            s = signs_list[j]
            proj = torch.dot(x, j_mul(u))  # [x, u]
            x = x - s * proj * u
        # Normalize the resulting vector.
        x = normalize_v(x)
        # Optionally, one might refine x with power iterations—but here Cx is near zero so we simply accept it.
        v_eval = 0.0  # by construction these extra directions have eigenvalue 0.
        v_sign = torch.sign(torch.dot(x, j_mul(x))).item()
        evals_list.append(v_eval)
        signs_list.append(v_sign)
        evecs_list.append(x.clone())
        count += 1

    # Convert results to tensors and then to NumPy arrays.
    evals_tensor = torch.tensor(evals_list, dtype=dtype, device=device)
    signs_tensor = torch.tensor(signs_list, dtype=dtype, device=device)
    evecs_tensor = torch.stack(evecs_list, dim=1)  # each eigenvector as a column
    return evals_tensor, signs_tensor, evecs_tensor



