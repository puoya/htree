import os
import torch
import numpy as np
import torch.optim as optim
from torch import Tensor
from typing import Optional, Any, Tuple, List
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import eigs
from scipy.optimize import minimize

import math
from . import conf
from . import embedding
# import conf
# import embedding

##########################################################################
def lgram_to_pts(gram: torch.Tensor, dim: int) -> torch.Tensor:
    """Convert Lorentzian Gram matrix to point coordinates using partial eigen decomposition."""
    if dim < 1:
        raise ValueError("Target dimension must be at least 1.")
    
    gram_np = gram.cpu().numpy().astype(np.float64)
    min_eval, min_evec = map(lambda x: torch.tensor(x, dtype=torch.float64), spla.eigsh(gram_np, k=1, which='SA'))
    evals, evecs = map(lambda x: torch.tensor(x, dtype=torch.float64), spla.eigsh(gram_np, k=min(dim, gram.shape[0] - 1) + 1, which='LM'))
    idx = torch.argsort(evals, descending=True)
    evals, evecs = evals[idx][:-1].clamp(min=0), evecs[:, idx[:-1]]
    
    coords = torch.diag(torch.sqrt(torch.cat((min_eval[0].abs().unsqueeze(0), evals)))).to(torch.float64) @ torch.cat((min_evec, evecs), dim=1).T
    return coords if coords[0, 0] >= 0 else -coords
###########################################################################
def hyperbolic_proj(vec: torch.Tensor, tol: float = 1e-100, max_iter: int = 100) -> torch.Tensor:
    """
    Projects a vector into hyperbolic space using a Newton method with increased precision.
    """
    spatial = vec[1:].to(dtype=torch.float64)  # Use double precision
    v0 = vec[0].to(dtype=torch.float64)
    a = (spatial ** 2).sum()

    t = torch.tensor(1.0, dtype=torch.float64, device=vec.device)
    
    for _ in range(max_iter):
        L = torch.sqrt(1 + a * t * t)
        fprime = 2 * a * t * (L - v0) / L + 2 * a * (t - 1)
        fdouble = 2 * a * (1 + (L - v0) / L) + 2 * a * (a * t * t * v0) / (L ** 3)
        
        if torch.abs(fdouble) < 1e-200:  # Prevent division by near-zero values
            break
            
        t_new = t - fprime / fdouble
        if torch.abs(t_new - t) < tol:
            t = t_new
            break
        t = t_new
    
    time_comp = torch.sqrt(1 + a * t * t)
    return torch.cat([time_comp.unsqueeze(0), t * spatial])
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
    tangents = torch.zeros(X.shape, dtype=X.dtype, device=X.device)

    for n in range(N):
        x = X[:, n]
        theta = torch.acosh(torch.clamp(-lorentz_prod(x, base), min=1.0))
        scale = theta / torch.sinh(theta) if theta != 0 else 1.0
        tangents[:, n] = scale * (x - base * torch.cosh(theta))
    
    return tangents[1:]
###########################################################################
def hyperbolic_exp(V: torch.Tensor) -> torch.Tensor:
    """Compute the hyperbolic exponential map for tangent vectors."""
    D, N = V.shape
    V = torch.cat((torch.zeros(1, N, dtype=V.dtype, device=V.device), V), dim=0)

    exp_map = torch.zeros_like(V)
    for n in range(N):
        v = V[:, n]
        norm_v = torch.norm(v)
        scale = torch.sinh(norm_v) / norm_v if norm_v != 0 else 1.0
        x =  scale * v
        x[0] = torch.sqrt(1 + torch.norm(scale * v)**2)
        exp_map[:, n] = x
    
    return exp_map
###########################################################################
def log_span(matrix: torch.Tensor) -> torch.Tensor:
    """Compute the span between the mean and minimum log10 distances in a square distance matrix."""
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")
    
    log_distances = torch.log10(matrix[~torch.eye(matrix.size(0), dtype=torch.bool, device=matrix.device)])
    return log_distances.mean() - log_distances.min()
###########################################################################
def compute_lr(epoch: int, total_epochs: int, losses: torch.Tensor, scale: float) -> float:
    """Compute adaptive learning rate based on loss trends and epoch progression."""
    if total_epochs <= 1 or scale is None:
        raise ValueError("Total epochs must be > 1 and scale must be provided.")
    
    no_weight_epochs = int(conf.NO_WEIGHT_RATIO * total_epochs)
    win_size = int(conf.WINDOW_RATIO * total_epochs)
    max_incr = int(conf.INCREASE_COUNT_RATIO * win_size)
    lr, multipliers = 1.0, []
    
    for i in range(1, min(len(losses), no_weight_epochs) + 1):
        if i >= win_size:
            recent = losses[i - win_size:i]
            incr_count = sum(y > x for x, y in zip(recent[:-1], recent[1:]))
            if incr_count > max_incr:
                multipliers.append(conf.DECREASE_FACTOR)
            elif all(x > y for x, y in zip(recent[:-1], recent[1:])):
                multipliers.append(conf.INCREASE_FACTOR)
            else:
                multipliers.append(1.0)
    
    lr *= torch.prod(torch.tensor(multipliers)).item()
    
    if epoch >= no_weight_epochs:
        epoch_range = total_epochs - no_weight_epochs - 1
        p = torch.tensor(10 ** (-scale / epoch_range))
        for i in range(no_weight_epochs, epoch):
            lr *= 10 ** (2 * (i - no_weight_epochs) / epoch_range * torch.log10(p).item())
    
    return lr
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
def J_norm(tensor: torch.Tensor) -> float:
    """Compute the Lorentzian norm of a vector."""
    if tensor.dim() == 1:
        if tensor.numel() == 0:
            raise ValueError("Input vector cannot be empty.")
        return -tensor[0]**2 + torch.sum(tensor[1:]**2)
    
    elif tensor.dim() == 2:
        if tensor.size(0) == 0:
            raise ValueError("Input tensor cannot have zero rows.")
        # Each column is a vector. Compute the norm for each column:
        # Note: The operation is done along dim=0.
        return -tensor[0, :]**2 + torch.sum(tensor[1:, :]**2, dim=0)
    
    else:
        raise ValueError("Input tensor must be either 1D or 2D.")

###########################################################################
def euclidean_embedding(dist_mat: torch.Tensor, dim: int, **kwargs):
    if not isinstance(dist_mat, torch.Tensor):
        raise ValueError("The 'dist_mat' must be a torch.Tensor.")
    params = {key: kwargs.get(key, default) for key, default in {
        'init_pts': None,
        'epochs': conf.TOTAL_EPOCHS,
        'log_fn': None,
        'lr_fn': None ,
        'weight_exp_fn': None,
        'lr_init': conf.INITIAL_LEARNING_RATE,
        'save_mode': conf.ENABLE_SAVE_MODE,
        'time_stamp': ""
    }.items()}
    
    if params['init_pts'] is None:
        pts = torch.rand(dim, dist_mat.size(0),requires_grad=True).mul_(0.01)
    else:
        pts = params['init_pts'].clone().detach().requires_grad_(True)
    if params['lr_fn'] is None:
        params['lr_fn'] = lambda x1, x2, x3: compute_lr(x1, x2, x3, scale=log_span(dist_mat).item())
    if params['weight_exp_fn'] is None:
        params['weight_exp_fn'] = lambda x1, x2, x3=None: compute_weight(x1, x2)
    if params['save_mode']:
        path = os.path.join(conf.OUTPUT_DIRECTORY, params['time_stamp'].strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(path, exist_ok=True)

    def cost_fn(pts: torch.Tensor, dist_mat: torch.Tensor, weight_mat: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(pts.t(), pts.t(), p=2)
        mask = ~torch.eye(dist_mat.size(0), dtype=torch.bool, device=dist_mat.device)
        cost = torch.norm(( (dist - dist_mat)* weight_mat)[mask], p='fro')**2 / torch.norm((dist_mat*weight_mat)[mask], p='fro')**2
        if params['save_mode']:
            rel_err = ((dist / dist_mat - 1).pow(2)).fill_diagonal_(0)
        else:
            rel_err = None
        return cost, rel_err

    opt = optim.Adam([pts], lr=params['lr_init'])
    costs, weight_exps, lrs = [], [], []

    for epoch in range(params['epochs']):
        p = params['weight_exp_fn'](epoch, params['epochs'],costs)
        weight_exps.append(p)
        weight_mat = torch.pow(dist_mat, p).fill_diagonal_(1)
        
        cost, rel_err = cost_fn(pts, dist_mat, weight_mat)
        cost.backward()
        costs.append(cost.item())
        pts.grad = torch.nan_to_num(pts.grad, nan=0.0)
        opt.step()
        lrs.append(params['lr_fn'](epoch, params['epochs'], costs) * params['lr_init'])
        opt.param_groups[0]['lr'] = lrs[-1]

        params['log_fn'](f"[Epoch {epoch + 1:0{len(str(params['epochs']))}d}/{params['epochs']}] Cost: {cost.item():.8f}, LR: {lrs[-1]:.10f}, Weight Exp: {p:.8f}")
        if params['save_mode']:
            np.save(os.path.join(path, f"RE_{epoch + 1}.npy"), rel_err.detach().cpu().numpy())

    if params['save_mode']:
        for name, data in zip(["weight_exponents", "learning_rates", "costs"], [weight_exps, lrs, costs]):
            np.save(os.path.join(path, f"{name}.npy"), data)

    return pts.detach() if pts.requires_grad else pts
###########################################################################
def hyperbolic_embedding(dist_mat: torch.Tensor, dim: int, **kwargs):
    if not isinstance(dist_mat, torch.Tensor):
        raise ValueError("The 'dist_mat' must be a torch.Tensor.")
    
    params = {key: kwargs.get(key, default) for key, default in {
        'init_pts': None,
        'epochs': conf.TOTAL_EPOCHS,
        'log_fn': None,
        'lr_fn': None ,
        'weight_exp_fn': None,
        'scale_fn': None,
        'lr_init': conf.INITIAL_LEARNING_RATE,
        'save_mode': conf.ENABLE_SAVE_MODE,
        'dist_cutoff': conf.MAX_RANGE,
        'time_stamp': ""
    }.items()}

    if params['init_pts'] is not None:
        tng  = hyperbolic_log(params['init_pts']).clone().detach().requires_grad_(True)
    else:
        tng = torch.rand(dim, dist_mat.size(0), requires_grad=True).mul_(0.01)
    if params['weight_exp_fn'] is None:
        params['weight_exp_fn'] = lambda x1, x2, x3=None: compute_weight(x1, x2)
    if params['scale_fn'] is None:
        params['scale_fn'] = lambda x1, x2, x3=None: compute_scale(x1, x2)
    if params['lr_fn'] is None:
        params['lr_fn'] = lambda x1, x2, x3: compute_lr(x1, x2, x3, scale=log_span(dist_mat).item())
    if params['save_mode']:
        path = os.path.join(conf.OUTPUT_DIRECTORY, params['time_stamp'].strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(path, exist_ok=True)

    def cost_fn(pts: torch.Tensor, dist_mat: torch.Tensor, weight_mat: torch.Tensor, s: torch.Tensor = None):
        flipped_pts = pts.clone()
        flipped_pts[0, :] *= -1
        dist = torch.arccosh(-(pts.T @ flipped_pts).clamp(max=-1))
        mask = ~torch.eye(dist_mat.size(0), dtype=torch.bool, device=dist_mat.device)
        if s is None:
            s = dist.pow(2).sum() / (dist * dist_mat).sum() # doubled checked (y-sx)/ sx (y = dis, x = dist_mat) solution y^2/ yx
            #s = (dist * dist_mat).sum() / dist_mat.pow(2).sum()
        cost = torch.norm( ((dist-s*dist_mat)*weight_mat)[mask], p='fro')**2 / torch.norm( (s*dist_mat*weight_mat)[mask], p='fro')**2

        if params['save_mode']:
            rel_err = ((dist / (s * dist_mat) - 1).pow(2)).fill_diagonal_(0)
        else:
            rel_err = None
        return cost, s.detach() if s.requires_grad else s, rel_err

    opt = optim.Adam([tng], lr=params['lr_init'])
    costs, weight_exps, lrs, scales, s = [], [], [], [], torch.tensor(1)
    for epoch in range(params['epochs']):
        p = params['weight_exp_fn'](epoch, params['epochs'], costs)
        weight_exps.append(p)
        weight_mat = torch.pow(s * dist_mat, p).fill_diagonal_(1)

        pts = hyperbolic_exp(tng)
        scale_learning = params['scale_fn'](epoch, params['epochs'], costs)
        scales.append(scale_learning)
        cost, s, rel_err = cost_fn(pts, dist_mat, weight_mat, None if scale_learning else s)

        cost.backward(retain_graph=True)
        costs.append(cost.item())
        tng.grad = torch.nan_to_num(tng.grad, nan=0.0)
        opt.step()

        lrs.append(params['lr_fn'](epoch, params['epochs'], costs) * params['lr_init'])
        opt.param_groups[0]['lr'] = lrs[-1]

        params['log_fn'](f"[Epoch {epoch + 1:0{len(str(params['epochs']))}d}/{params['epochs']}] Cost: {cost.item():.8f}, Scale: {s:.8f}, Learning Rate: {lrs[-1]:.10f}, Weight Exponent: {p:.8f}, Scale Learning: {'Yes' if scale_learning else 'No'}")
        if params['save_mode']:
            np.save(os.path.join(path, f"RE_{epoch+1}.npy"), rel_err.detach().numpy())

    if params['save_mode']:
        for name, data in zip(["weight_exponents", "learning_rates", "scales", "costs"], [weight_exps, lrs, scales, costs]):
            np.save(os.path.join(path, f"{name}.npy"), data)

    return hyperbolic_exp(tng.detach()), s.detach()
###########################################################################
def _precise_hyperbolic_multiembedding(dist_mats, multi_embs, **kwargs):
    params = {key: kwargs.get(key, default) for key, default in {
        'epochs': conf.TOTAL_EPOCHS,
        'log_fn': None,
        'lr_fn': None,
        'weight_exp_fn': None,
        'scale_fn': None,
        'lr_init': conf.INITIAL_LEARNING_RATE,
        'save_mode': conf.ENABLE_SAVE_MODE,
        'time_stamp': ""
    }.items()}
    
    if params['weight_exp_fn'] is None:
        params['weight_exp_fn'] = lambda x1, x2, x3=None: compute_weight(x1, x2)
    if params['scale_fn'] is None:
        params['scale_fn'] = lambda x1, x2, x3=None: compute_scale(x1, x2)
    if params['lr_fn'] is None:
        params['lr_fn'] = lambda x1, x2, x3: compute_lr(x1, x2, x3, scale=log_span(dist_mats[0]).item())
    if params['save_mode']:
        path = os.path.join(conf.OUTPUT_DIRECTORY, params['time_stamp'].strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(path, exist_ok=True)
    
    multi_embeddings = embedding.MultiEmbedding()
    tangents_list = [hyperbolic_log(emb.points) for emb in multi_embs]
    optimizer = optim.Adam([torch.cat(tangents_list, dim=1).requires_grad_(True)], lr=params['lr_init'])
    num_points = [emb.points.shape[1] for emb in multi_embs]
    
    def cost_fn(pts, dist_mat, weight_mat, s=None):
        flipped_pts = pts.clone()
        flipped_pts[0, :] *= -1
        dist = torch.arccosh(-(pts.T @ flipped_pts).clamp(max=-1))
        mask = ~torch.eye(dist_mat.size(0), dtype=torch.bool, device=dist_mat.device)
        return (torch.norm(((dist - s * dist_mat) * weight_mat)[mask], p='fro')**2 /
                torch.norm((s * dist_mat * weight_mat)[mask], p='fro')**2)
    
    def compute_s(tangents, dist_mats, num_points):
        scale_num, scale_den, idx = 0, 0, 0
        for n, n_points in enumerate(num_points):
            idx_next = idx + n_points
            pts = hyperbolic_exp(tangents[:, idx:idx_next])
            flipped_pts = pts.clone()
            flipped_pts[0, :] *= -1
            dist = torch.arccosh(-(pts.T @ flipped_pts).clamp(max=-1))
            scale_num += n_points * dist.pow(2).sum() / dist_mats[n].pow(2).sum()
            scale_den += n_points * (dist * dist_mats[n]).sum() / dist_mats[n].pow(2).sum()
            idx = idx_next
        return scale_num / scale_den
    
    costs, weight_exps, lrs, scales, s = [], [], [], [], torch.sqrt(torch.abs(multi_embs[0].curvature))
    for epoch in range(params['epochs']):
        total_cost = 0
        p = params['weight_exp_fn'](epoch, params['epochs'], costs)
        scale_learning = params['scale_fn'](epoch, params['epochs'], costs)
        s = compute_s(optimizer.param_groups[0]['params'][0], dist_mats, num_points) if scale_learning else s.detach().item() if isinstance(s, torch.Tensor) and s.requires_grad else s
        weight_exps.append(p)
        scales.append(scale_learning)
        lrs.append(params['lr_fn'](epoch, params['epochs'], costs) * params['lr_init'])
        optimizer.param_groups[0]['lr'] = lrs[-1]
        
        idx = 0
        for n, n_points in enumerate(num_points):
            idx_next = idx + n_points
            weight_mat = torch.pow(s * dist_mats[n], p).fill_diagonal_(1)
            pts = hyperbolic_exp(optimizer.param_groups[0]['params'][0][:, idx:idx_next])
            total_cost += cost_fn(pts, dist_mats[n], weight_mat, s) * n_points
            idx = idx_next
        
        optimizer.zero_grad()
        total_cost.backward()
        optimizer.param_groups[0]['params'][0].grad = torch.nan_to_num(optimizer.param_groups[0]['params'][0].grad, nan=0.0)
        optimizer.step()
        costs.append(total_cost.item()/sum(num_points))

        params['log_fn'](f"[Epoch {epoch + 1}/{params['epochs']}], Scale Learning: {'Yes' if scale_learning else 'No'}, "
                         f"Avg Loss: {costs[-1]:.8f}, Weight Exponent: {p:.8f}, Consensus Scale: {s:.8f}")
    s = s.detach().item() if isinstance(s, torch.Tensor) and s.requires_grad else s
    idx = 0

    pts_list = []
    for n, n_points in enumerate(num_points):
        idx_next = idx + n_points
        pts_list.append(hyperbolic_exp(optimizer.param_groups[0]['params'][0][:, idx:idx_next].detach())) 
        idx = idx_next
    if params['save_mode']:
        for name, data in zip(["weight_exponents", "learning_rates", "scales", "costs"], [weight_exps, lrs, scales, costs]):
            np.save(os.path.join(path, f"{name}.npy"), data)
    
    return pts_list, -s**2
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
    while count < d and torch.norm(Cx, p="fro") >= threshold:
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
    while count < d:
        print('nooooo')
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

# def j_decomposition(Cx, d):
#     tol, ev_tol = 1e-10, 1e-20
#     D = Cx.shape[0] - 1
#     J = torch.eye(D+1, dtype=Cx.dtype, device=Cx.device)
#     J[0, 0] = -1

#     # Use lists to accumulate results and later convert to tensors.
#     evals_list = []
#     signs_list = []
#     evecs = None  # Will store eigenvectors as columns
#     ###########################################################################
#     def expected_sgn(d_val):
#         """Return -1 if d_val is 1, else return 1."""
#         return -1 if d_val == 1 else 1
#     ###########################################################################
#     def cost(v1, v2):
#         # Assuming v1 and v2 are 1D tensors.
#         dot_val = torch.dot(v1, v2).abs()
#         norm1 = torch.linalg.norm(v1)
#         norm2 = torch.linalg.norm(v2)
#         return 1 - dot_val / (norm1 * norm2)
#     ###########################################################################
#     def j_eval(v):
#         # v is assumed to be a column vector.
#         num = torch.linalg.norm(Cx @ J @ Cx @ J @ v)
#         den = torch.linalg.norm(v)
#         return torch.sqrt(num / den)
#     ###########################################################################
#     def j_orthonormalize(vec, basis, sign_arr):
#         # Ensure vec is at least 2D column vector.
#         if vec.dim() == 1:
#             vec = vec.unsqueeze(1)
#         else:
#             vec = vec.reshape(-1, 1)
#         if basis is not None and basis.numel() != 0:
#             # Make sure basis is 2D.
#             if basis.dim() == 1:
#                 basis = basis.unsqueeze(1)
#             # Multiply: basis @ diag(sign_arr) @ basis.T
#             diag_sign = torch.diag(torch.tensor(sign_arr, dtype=Cx.dtype, device=Cx.device))
#             proj = basis @ diag_sign @ basis.T
#             vec = vec - proj @ J @ vec
#         norm_val = torch.linalg.norm(vec)
#         if norm_val > tol:
#             vec = vec / norm_val
#             norm_j = torch.sqrt(torch.abs((vec.T @ J @ vec)))
#             if norm_j > tol:
#                 return (vec / norm_j).flatten(), True
#         return vec.flatten(), False
#     ###########################################################################
#     def init_vec(target_sign):
#         nonlocal evecs
#         # Compute residual matrix: if there are already computed eigenvectors, subtract projection.
#         if evecs is not None and evecs.numel() != 0:
#             # Convert evals_list to tensor for diag.
#             evals_tensor = torch.tensor(evals_list, dtype=Cx.dtype, device=Cx.device)
#             diag_evals = torch.diag(evals_tensor)
#             res = Cx - evecs @ diag_evals @ evecs.T
#         else:
#             res = Cx.clone()
#         # Scale res.
#         res = res / torch.max(torch.abs(res))
#         # Compute eigen-decomposition of (res @ J)
#         A = res @ J
#         eigvals, eigvecs = torch.linalg.eig(A)
#         # Select eigenvector corresponding to largest eigenvalue magnitude.
#         magnitudes = eigvals.abs()
#         max_index = torch.argmax(magnitudes)
#         vec = eigvecs[:, max_index]
#         # Use only the real part.
#         vec = vec.real
#         vec, ok = j_orthonormalize(vec, evecs, signs_list)
#         # Adjust until conditions are met.
#         while (not ok) or (J_norm(vec) * target_sign < 0):
#             perturb = tol * torch.randn(res.shape[0], device=Cx.device, dtype=Cx.dtype)
#             vec, ok = j_orthonormalize(vec + perturb, evecs, signs_list)
#         return vec
#     ###########################################################################
#     def rand_vec(target_sign):
#         while True:
#             vec, ok = j_orthonormalize(torch.randn(D + 1, device=Cx.device, dtype=Cx.dtype), evecs, signs_list)
#             if ok and J_norm(vec) * target_sign > 0:
#                 return vec
#     ###########################################################################
#     condition = True
#     while condition:
#         target_sign = expected_sgn(len(evals_list) + 1)
#         v = init_vec(target_sign)
#         errors = [1.0]

#         while True:
#             v2, ok = j_orthonormalize((Cx @ J @ v), evecs, signs_list)
#             if not ok:
#                 v2 = rand_vec(target_sign)
#                 break
#             err = cost(v, v2)
#             if err < ev_tol and (J_norm(v2) * target_sign) >= 0:
#                 break
#             errors.append(min(errors[-1], err))
#             v = v2  # update v for next iteration if needed

#         v, ok = j_orthonormalize(v2, evecs, signs_list)
#         if (not ok) or (J_norm(v) * target_sign < 0):
#             v = rand_vec(target_sign)
#             current_sign = J_norm(v)
#         else:
#             current_sign = torch.sign(J_norm(v))

#         v = v.reshape(-1, 1)
#         evals_list.append(j_eval(v))
#         signs_list.append(current_sign)
#         if evecs is None or evecs.numel() == 0:
#             evecs = v
#         else:
#             evecs = torch.cat((evecs, v), dim=1)

#         # Terminate when we have d positive signs and at least 1 negative sign.
#         if (sum(1 for s in signs_list if s > 0) == d) and (sum(1 for s in signs_list if s < 0) >= 1):
#             condition = False

#     # Convert lists to tensors.
#     evals = torch.tensor(evals_list, dtype=Cx.dtype, device=Cx.device)
#     signs = torch.tensor(signs_list, dtype=Cx.dtype, device=Cx.device)
#     return evals, signs, evecs