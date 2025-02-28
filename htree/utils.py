import os
import torch
import numpy as np
import torch.optim as optim
from . import conf

# import conf

from torch import Tensor
from typing import Optional, Any, Tuple

###########################################################################
###########################################################################
###########################################################################
def lgram_to_points(gram_matrix: torch.Tensor, dimension: int) -> torch.Tensor:
    """
    Convert a Lorentzian Gram matrix to points using eigen decomposition.

    Parameters:
    - gram_matrix (torch.Tensor): The Gram matrix to be decomposed.
    - dimension (int): Dimension of the target space.

    Returns:
    - torch.Tensor: Coordinates corresponding to the input Gram matrix.

    Raises:
    - ValueError: If the dimension is less than 1.
    """
    if dimension < 1:
        raise ValueError("Dimension must be at least 1.")

    min_dimension = min(dimension, gram_matrix.shape[0] - 1)

    # Perform eigen decomposition of the Gram matrix
    eigenvalues, eigenvectors = torch.linalg.eig(gram_matrix)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    # Identify the smallest eigenvalue and its index
    min_eigenvalue, min_index = torch.min(eigenvalues, dim=0)
    min_eigenvector = eigenvectors[:, min_index]

    # Remove the smallest eigenvalue and sort the rest in descending order
    eigenvalues = torch.cat((eigenvalues[:min_index], eigenvalues[min_index+1:]))
    eigenvectors = torch.cat((eigenvectors[:, :min_index], eigenvectors[:, min_index+1:]), dim=1)

    sorted_indices = torch.argsort(-eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    top_eigenvalues = eigenvalues[:dimension]
    top_eigenvectors = eigenvectors[:, :dimension]

    # Create the final set of eigenvalues for constructing the coordinate matrix
    final_eigenvalues = torch.cat((torch.abs(min_eigenvalue).unsqueeze(0), top_eigenvalues), dim=0)
    final_eigenvalues[final_eigenvalues <= 0] = 0
    final_eigenvalues = torch.sqrt(final_eigenvalues)

    # Adjust eigenvectors to match the selected eigenvalues
    final_eigenvectors = torch.cat((min_eigenvector.unsqueeze(1), top_eigenvectors), dim=1)

    # Construct the coordinate matrix
    X = torch.diag(final_eigenvalues) @ final_eigenvectors.T

    # Ensure the first coordinate is positive
    if X[0, 0] < 0:
        X = -X

    if min_dimension < dimension:
        zero_rows = torch.zeros((dimension - min_dimension, gram_matrix.shape[0]))
        X = torch.cat((X, zero_rows), dim=0)

    return X
###########################################################################
###########################################################################
########################################################################### 
def project_to_hyperbolic_space(vector: torch.Tensor, start_lr: float = 1e-2, end_lr: float = 1e-20, epochs: int = 5000) -> torch.Tensor:
    """
    Projects a given vector into hyperbolic space by optimizing the projected points using the Adam optimizer.
    The projected vector will have a hyperbolic (Lorentzian) norm exactly equal to -1.

    Parameters:
    - vector (torch.Tensor): The input vector to be projected into hyperbolic space.
    - start_lr (float): The initial learning rate for the Adam optimizer.
    - end_lr (float): The final learning rate for the Adam optimizer.
    - epochs (int): The number of optimization epochs.

    Returns:
    - torch.Tensor: The vector projected into hyperbolic space.

    Raises:
    - ValueError: If `epochs` is less than or equal to 0.
    """
    if epochs <= 0:
        raise ValueError("Number of epochs must be greater than 0.")

    # Determine the dimension of the input vector
    vector = vector.double().squeeze()
    dimension = len(vector) - 1

    # Initialize the projected points as a learnable parameter
    projected_vector = vector[1:].clone().detach().double().requires_grad_(True)

    # Set up the Adam optimizer
    optimizer = optim.Adam([projected_vector], lr=start_lr)
    previous_cost = float('inf')

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Compute the projection operator
        projected_vector_in_space = torch.cat([
            torch.sqrt(1 + torch.sum(projected_vector**2)).unsqueeze(0),
            projected_vector
        ])

        # Compute the cost function: squared difference from the original vector
        cost = torch.sum(torch.pow(projected_vector_in_space - vector, 2))

        # Backpropagate the cost and update the projected vector
        cost.backward()
        optimizer.step()

        # Update the learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = start_lr * ((end_lr / start_lr) ** (epoch / epochs))

        # Break early if the cost is sufficiently low
        if abs(cost.item() - previous_cost) < 1e-50:
            # print(cost.item(),epoch,abs(cost.item() - previous_cost))
            break
        else:
            previous_cost = cost.item()

    # Final projection with optimized projected_vector
    projected_vector_in_space = torch.cat([
        torch.sqrt(1 + torch.sum(projected_vector**2)).unsqueeze(0),
        projected_vector
    ])

    return projected_vector_in_space.detach().double()
###########################################################################
###########################################################################
########################################################################### 
def lorentzian_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the Lorentzian product between two vectors.

    Parameters:
    - x (torch.Tensor): A tensor representing the first vector, which can be of shape `[1, d]`, `[d, 1]`, or `[d]`.
    - y (torch.Tensor): A tensor representing the second vector, which can be of shape `[1, d]`, `[d, 1]`, or `[d]`.

    Returns:
    - torch.Tensor: The Lorentzian product of the two vectors.
    
    Raises:
    - ValueError: If the shapes of `x` and `y` are incompatible.
    """
    if x.shape != y.shape:
        raise ValueError("Input tensors must have the same shape.")

    # Ensure x and y are flattened to the shape [d]
    x_flat = x.flatten()
    y_flat = y.flatten()

    # Compute the Lorentzian product
    return -x_flat[0] * y_flat[0] + torch.dot(x_flat[1:], y_flat[1:])
###########################################################################
###########################################################################
###########################################################################
def hyperbolic_log(X: torch.Tensor) -> torch.Tensor:
    """
    Compute the hyperbolic logarithm map for given points in hyperbolic space.

    Parameters:
    - X (torch.Tensor): Points in hyperbolic space with shape `(D+1, N)`, where `D` is the dimension 
      and `N` is the number of points.

    Returns:
    - torch.Tensor: The tangent vectors at the base point with shape `(D-1, N)`.
    
    Raises:
    - ValueError: If the input tensor `X` does not have the required dimensions.
    """
    D,N = X.shape
    D -= 1  # Adjust D to match the context of hyperbolic space
    if D < 1:
        raise ValueError("Dimension of points must be at least 1 (D >= 1).")

    # Define the base point as [1, 0, 0, ..., 0]
    base_point = torch.zeros(D + 1, dtype=X.dtype, device=X.device)
    base_point[0] = 1.0
    
    # Initialize the result matrix for tangent vectors
    tangent_vectors = torch.zeros(X.shape, dtype=X.dtype, device=X.device)

    for n in range(N):
        x = X[:, n]
        
        # Compute the theta value
        theta = -lorentzian_product(x, base_point)
        theta = torch.maximum(theta, torch.tensor(1.0, dtype=X.dtype, device=X.device))
        theta = torch.acosh(theta)
        
        # Compute the tangent vector
        if theta != 0:
            theta_over_sinh_theta = theta / torch.sinh(theta)
        else:
            theta_over_sinh_theta = torch.tensor(1.0, dtype=X.dtype, device=X.device)
        
        tangent_vectors[:, n] = theta_over_sinh_theta * (x - base_point * torch.cosh(theta))
    return tangent_vectors[1:]
###########################################################################
###########################################################################
###########################################################################
def hyperbolic_exponential(V: torch.Tensor) -> torch.Tensor:
    """
    Compute the hyperbolic exponential map for given tangent vectors.

    Parameters:
    - V (torch.Tensor): Tangent vectors with shape `(D, N)`, where `D` is the dimension and `N` is the number of vectors.

    Returns:
    - torch.Tensor: The resulting points in hyperbolic space with shape `(D, N)`.
    
    Raises:
    - ValueError: If the resulting norm of the points is positive, which is an error case.
    """
    D, N = V.shape

    # Add a zero row to the tangent vectors to match the base point dimension
    zero_row = torch.zeros(1, N, dtype=V.dtype, device=V.device)
    V = torch.cat((zero_row, V), dim=0)

    # Initialize the result matrix for hyperbolic points
    result_matrix = torch.zeros((D + 1, N), dtype=V.dtype, device=V.device)

    # Define the base point as [1, 0, 0, ..., 0]
    base_point = torch.zeros(D + 1, dtype=V.dtype, device=V.device)
    base_point[0] = 1.0

    for n in range(N):
        v = V[:, n]
        norm_v = torch.sqrt(J_norm(v))

        if norm_v != 0:
            sinh_norm_v_over_norm_v = torch.sinh(norm_v) / norm_v
        else:
            sinh_norm_v_over_norm_v = torch.tensor(1.0, dtype=V.dtype, device=V.device)

        x = torch.cosh(norm_v) * base_point + sinh_norm_v_over_norm_v * v
        norm_x = J_norm(x)

        if norm_x > 0:
            raise ValueError('Error: Norm should not be positive')

        # Normalize the result vector
        result_matrix[:, n] = x / torch.sqrt(-norm_x)
    
    return result_matrix
###########################################################################
###########################################################################
###########################################################################
def hyperbolic_embedding(distance_matrix: torch.Tensor, dimension: int, **kwargs):
    """
    Optimize the embedding of points based on a given distance matrix.

    Parameters:
    - distance_matrix (torch.Tensor): The matrix representing distances between points (required).
    - dimension (int): The number of dimensions for the embedding space (required).
    - **kwargs: Optional parameters for the optimization process.
        - initial_tangents (torch.Tensor or None, optional): Initial tangents to start the optimization. 
          If None, random tangents are generated (default is None).
        - total_epochs (int, optional): Number of iterations for the optimization process (default is 2000).
        - log_function (callable or None, optional): Function to log optimization progress. 
          If None, logging is disabled (default is None).
        - learning_rate (callable or None, optional): Function to compute the learning rate dynamically. 
          If None, a fixed learning rate is used (default is None).
        - scale_learning (callable or None, optional): Function to determine scale learning status during optimization 
          (default is None).
        - weight_exponent (float or None, optional): Exponent to adjust weights in the optimization process 
          (default is None).
        - initial_lr (float, optional): The initial learning rate for the optimizer (default is 0.1).
        - enable_save (bool, optional): If True, intermediate states during the optimization are saved (default is False).
        - time (callable or None, optional): Function to provide a timestamp for logging purposes (default is None).

    Returns:
    - tangents (torch.Tensor): The optimized tangents resulting from the optimization process.
    - scale (float): The final scale factor used in the optimization.
    
    Raises:
    - ValueError: If the `distance_matrix` is not a torch.Tensor.
    """
    
    # Validate inputs
    if not isinstance(distance_matrix, torch.Tensor):
        raise ValueError("The 'distance_matrix' must be a torch.Tensor.")

    # Set default values for keyword arguments
    total_epochs = kwargs.get('total_epochs', conf.TOTAL_EPOCHS)
    initial_lr = kwargs.get('initial_lr', conf.INITIAL_LEARNING_RATE)
    enable_save = kwargs.get('enable_save', conf.ENABLE_SAVE_MODE)
    initial_tangents = kwargs.get('initial_tangents', None)
    log_function = kwargs.get('log_function', None)
    learning_rate = kwargs.get('learning_rate', None)
    scale_learning = kwargs.get('scale_learning', None)
    weight_exponent = kwargs.get('weight_exponent', None)
    timestamp = kwargs.get('time', None)
    
    if initial_tangents is None:
        n = distance_matrix.size(0)
        tangents = torch.rand(dimension, n, requires_grad=True)
        tangents.data.mul_(0.01)
    else:
        tangents = initial_tangents.clone().detach().requires_grad_(True)

    if learning_rate is None:
        learning_rate_fn = lambda epoch, total_epochs, loss_list=None: default_learning_rate(epoch, total_epochs, loss_list, scale_range=(distance_range(distance_matrix)).item())
    else:
        learning_rate_fn = learning_rate
    if scale_learning is None:
        scale_learning_fn = lambda epoch, total_epochs, loss_list=None: default_scale_learning(epoch, total_epochs)
    else:
        scale_learning_fn = scale_learning
    if weight_exponent is None:
        weight_exponent_fn = lambda epoch, total_epochs, loss_list=None: default_weight_exponent(epoch, total_epochs)
    else:
        weight_exponent_fn = weight_exponent
    
    loss_history = []
    scale_factor = torch.tensor(1.0, requires_grad = True)  # Initialize scale
    latest_scale_grad_free = torch.tensor(scale_factor.item())

    optimizer_tangents = optim.Adam([tangents], lr=initial_lr)
    weight_history = []
    lr_history = []
    scale_history = []
    for epoch in range(total_epochs):
        weight_exp = weight_exponent_fn(epoch, total_epochs, loss_history)
        weight_history.append(weight_exp)

        weight_matrix = torch.pow(latest_scale_grad_free * distance_matrix, weight_exp)
        weight_matrix.fill_diagonal_(1)

        is_scale_learning = scale_learning_fn(epoch, total_epochs, loss_history)
        scale_history.append(is_scale_learning)
        if is_scale_learning:
            loss, scale_factor, relative_error = hypebrolic_cost(tangents, distance_matrix, enable_scale_learning=True, weight_matrix = weight_matrix, enable_save = enable_save)
            latest_scale_grad_free = torch.tensor(scale_factor.detach().item())
        else:
            loss, scale_factor, relative_error = hypebrolic_cost(tangents, distance_matrix, enable_scale_learning=False, scale_factor=latest_scale_grad_free, weight_matrix=weight_matrix, enable_save = enable_save)

        loss.backward(retain_graph=True)
        loss_history.append(loss.detach().item())
        with torch.no_grad():
            tangents.grad = torch.nan_to_num(tangents.grad, nan=0.0)
        optimizer_tangents.step()

        lr = learning_rate_fn(epoch, total_epochs, loss_history)
        lr_history.append(lr*initial_lr)
        
        for param_group in optimizer_tangents.param_groups:
            param_group['lr'] = lr*initial_lr

        current_lr = optimizer_tangents.param_groups[0]['lr']
        
        num_digits = len(str(total_epochs))
        formatted_epoch = f"{epoch + 1:0{num_digits}d}"
        message = (f"[Epoch {formatted_epoch}/{total_epochs}] "
                   f"Loss: {loss.item():.8f}, Scale: {scale_factor.item():.8f}, Learning Rate: {current_lr:.10f}, "
                   f"Weight Exponent: {weight_exp:.8f}, Scale Learning: {'Yes' if is_scale_learning else 'No'}")

        if log_function:
            log_function(message)

        if enable_save and (relative_error is not None):
            path = f'{conf.OUTPUT_DIRECTORY}/{timestamp}'
            os.makedirs(path, exist_ok=True)
            # Save relative error to a file
            relative_error_path = os.path.join(path, f"RE_{epoch+1}.npy")
            np.save(relative_error_path, relative_error.detach().cpu().numpy())

    if tangents.requires_grad:
        tangents = tangents.detach()
    if scale_factor.requires_grad:
        scale_factor = scale_factor.detach()

    if enable_save:
        path = f'{conf.OUTPUT_DIRECTORY}/{timestamp}'
        os.makedirs(path, exist_ok=True)
        weight_path = os.path.join(path, f"weight_history.npy")
        np.save(weight_path, weight_history)
        lr_path = os.path.join(path, f"lr_history.npy")
        np.save(lr_path, lr_history)
        scale_path = os.path.join(path, f"scale_history.npy")
        np.save(scale_path, scale_history)
    
    return tangents, scale_factor
###########################################################################
###########################################################################
###########################################################################
def hyperbolic_embedding_consensus(distance_matrices, dimension, **kwargs):
    """
    Optimize the embedding of points based on a list of distance matrices and reach a consensus on the scale factor.

    Parameters:
    - distance_matrices (list of torch.Tensor): List of distance matrices to be embedded.
    - dimension (int): The number of dimensions for the embedding space.
    - **kwargs: Optional parameters for the optimization process (as in hyperbolic_embedding).
    
    Returns:
    - embeddings_list (list of torch.Tensor): The optimized tangents for each distance matrix.
    - consensus_scale (float): The consensus scale factor after averaging across all matrices.
    """
    
    # Set default values for keyword arguments
    total_epochs = kwargs.get('total_epochs', conf.TOTAL_EPOCHS)
    initial_lr = kwargs.get('initial_lr', conf.INITIAL_LEARNING_RATE)
    initial_tangents = kwargs.get('initial_tangents', None)
    log_function = kwargs.get('log_function', None)
    learning_rate = kwargs.get('learning_rate', None)
    scale_learning = kwargs.get('scale_learning', None)
    weight_exponent = kwargs.get('weight_exponent', None)
    consensus_scale = torch.tensor(1)

    scale_range = -float('inf')
    for distance_matrix in distance_matrices.values():
        scale_range = max((distance_range(distance_matrix)).item(), scale_range)

    if learning_rate is None:
        learning_rate_fn = lambda epoch, total_epochs, loss_list=None: default_learning_rate(epoch, total_epochs, loss_list, scale_range=scale_range)
    else:
        learning_rate_fn = learning_rate
    if scale_learning is None:
        scale_learning_fn = lambda epoch, total_epochs, loss_list=None: default_scale_learning(epoch, total_epochs)
    else:
        scale_learning_fn = scale_learning
    if weight_exponent is None:
        weight_exponent_fn = lambda epoch, total_epochs, loss_list=None: default_weight_exponent(epoch, total_epochs)
    else:
        weight_exponent_fn = weight_exponent
    
    tangents_dic = {}
    optimizers_dic = {}
    losses_dic = {}

    scale_history_list = []
    scale_factor_list = []
    loss_avg_list = []
    for name, distance_matrix in distance_matrices.items():
        if initial_tangents is None:
            n = distance_matrix.size(0)
            tangents = torch.rand(dimension, n, requires_grad=True)
            tangents.data.mul_(0.01)
        else:
            tangents = initial_tangents[name].clone().detach().requires_grad_(True)
        tangents_dic[name] = tangents
        optimizers_dic[name] =optim.Adam([tangents], lr=initial_lr)
        losses_dic[name]  = []
    
    for epoch in range(total_epochs):
        scale_nums = {}
        scale_dens = {}
        denominators = {}
        for name, distance_matrix in distance_matrices.items():
            loss_list = losses_dic[name]
            tangents = tangents_dic[name]
            optimizer_tangents = optimizers_dic[name]

            weight_exp = weight_exponent_fn(epoch, total_epochs, loss_list)
            is_scale_learning = scale_learning_fn(epoch, total_epochs, loss_list)
            scale_history_list.append(is_scale_learning)
            weight_matrix = torch.pow(consensus_scale * distance_matrix, weight_exp)
            weight_matrix.fill_diagonal_(1)              
            if is_scale_learning:
                loss, _, _ = hypebrolic_cost(tangents, distance_matrix, enable_scale_learning=False, use_unweighted_cost = False,
                                                scale_factor=consensus_scale, weight_matrix=weight_matrix)
                latest_scale_grad_free = consensus_scale
            else:
                consensus_scale = latest_scale_grad_free
                loss, _, _ = hypebrolic_cost(tangents, distance_matrix, enable_scale_learning=False, use_unweighted_cost = False, 
                                                scale_factor=consensus_scale, weight_matrix=weight_matrix)
 
            loss.backward()
            with torch.no_grad():
                tangents.grad = torch.nan_to_num(tangents.grad, nan=0.0)
            optimizer_tangents.step()

            lr = learning_rate_fn(epoch, total_epochs, loss_list)
            for param_group in optimizer_tangents.param_groups:
                param_group['lr'] = lr * initial_lr

            embeddings = hyperbolic_exponential(tangents)
            flipped_embeddings = embeddings.clone()
            flipped_embeddings[0, :] *= -1

            gram_matrix = torch.matmul(embeddings.t(), flipped_embeddings).clamp(max=-1)
            param_dist_matrix = torch.arccosh(-gram_matrix)

            loss_list.append(loss.item())
            scale_nums[name] =  ((param_dist_matrix * distance_matrix).sum()).item() 
            scale_dens[name] =  (distance_matrix.pow(2).sum()).item() 
            denominators[name] =  (torch.norm(consensus_scale * distance_matrix * weight_matrix, p='fro') ** 2).item()
            tangents_dic[name] = tangents
            losses_dic[name] = loss_list

            
        avg_loss = 0
        denom_sum = 0
        
        tmp_num = 0
        tmp_den = 0
        for name, distance_matrix in distance_matrices.items():
            loss_list = losses_dic[name]
            avg_loss = avg_loss + denominators[name]*loss_list[epoch]
            denom_sum = denom_sum + denominators[name]
            
            tmp_num = tmp_num + scale_nums[name]
            tmp_den = tmp_den + scale_dens[name]

        avg_loss /= denom_sum
        if is_scale_learning:
            consensus_scale = tmp_num / tmp_den
            
        scale_factor_list.append(consensus_scale)
        scale_history_list.append(is_scale_learning)
        loss_avg_list.append(avg_loss)
    
        if log_function:
            num_digits = len(str(total_epochs))
            formatted_epoch = f"{epoch + 1:0{num_digits}d}"
            message = (f"[Epoch {formatted_epoch}/{total_epochs}], Scale Learning: {'Yes' if is_scale_learning else 'No'},"
                       f"Avg Loss: {avg_loss:.8f},Weight Exponent: {weight_exp:.8f}, Consensus Scale: {consensus_scale:.8f}")
            log_function(message)
            print(message)
        
    for name, tangents in tangents_dic.items():
        tangents_dic[name] = tangents.detach()
    return tangents_dic, consensus_scale
###########################################################################
###########################################################################
###########################################################################
def distance_range(matrix: torch.Tensor) -> torch.Tensor:
    """
    Calculate the range of distances in a given distance matrix.

    Parameters:
    - matrix (torch.Tensor): A square tensor representing the distance matrix.

    Returns:
    - torch.Tensor: The calculated scale range, based on the log10 of the distance values.
    
    Raises:
    - ValueError: If the input is not a square matrix.
    """
    if matrix.size(0) != matrix.size(1):
        raise ValueError("The input matrix must be square.")

    off_diagonals = ~torch.eye(matrix.size(0), dtype=torch.bool, device=matrix.device)
    log_distances = torch.log10(matrix[off_diagonals])
    
    return torch.mean(log_distances) - torch.min(log_distances)
###########################################################################
###########################################################################
###########################################################################
def default_learning_rate(
    epoch: int, 
    total_epochs: int, 
    loss_list: torch.Tensor, 
    scale_range: float = None
) -> float:
    """
    Calculate the learning rate for the current epoch.

    Parameters:
    - epoch (int): The current epoch of the training process.
    - total_epochs (int): The total number of epochs in the training process.
    - loss_list (torch.Tensor): A tensor containing loss values from previous epochs.
    - scale_range (float, optional): The range for scaling the learning rate. Must be provided.

    Returns:
    - float: The calculated learning rate for the current epoch.
    
    Raises:
    - ValueError: If `total_epochs` is less than or equal to 1, or if `scale_range` is not provided.
    """
    if total_epochs <= 1:
        raise ValueError("Total epochs must be greater than 1.")
    if scale_range is None:
        raise ValueError("Scale range must be provided.")

    NO_WEIGHT_EPOCHS = int(conf.NO_WEIGHT_RATIO * total_epochs)
    
    if epoch >= NO_WEIGHT_EPOCHS:
        relevant_losses = loss_list[:NO_WEIGHT_EPOCHS]
    else:
        relevant_losses = loss_list

    lr = calculate_multipliers(relevant_losses, total_epochs)
    
    if epoch >= NO_WEIGHT_EPOCHS:
        for i in range(NO_WEIGHT_EPOCHS, epoch):
            r = (i - NO_WEIGHT_EPOCHS) / (total_epochs - 1 - NO_WEIGHT_EPOCHS)
            p = torch.tensor(10 ** (-scale_range / (total_epochs - 1 - NO_WEIGHT_EPOCHS)))
            lr *= 10 ** (2 * r * torch.log10(p).item())
    
    return lr
###########################################################################
###########################################################################
###########################################################################
def calculate_multipliers(loss_list: torch.Tensor, total_epochs: int) -> float:
    """
    Calculate learning rate multipliers based on loss trends over time.

    Parameters:
    - loss_list (torch.Tensor): A tensor containing the list of loss values across epochs.
    - total_epochs (int): The total number of epochs in the training process.

    Returns:
    - float: The product of the calculated multipliers that adjust the learning rate.
    
    Raises:
    - ValueError: If `total_epochs` is less than or equal to 1.
    """
    if total_epochs <= 1:
        raise ValueError("Total epochs must be greater than 1.")
    
    WINDOW_SIZE = int(conf.WINDOW_RATIO * total_epochs)
    INCREASE_COUNT_MAX = int(conf.INCREASE_COUNT_RATIO * WINDOW_SIZE)
    
    multipliers = []
    for i in range(1, len(loss_list) + 1):
        if i < WINDOW_SIZE:
            multipliers.append(1.0)
            continue
        
        recent_losses = loss_list[i - WINDOW_SIZE:i]
        increase_count = sum(1 for x, y in zip(recent_losses[:-1], recent_losses[1:]) if y > x)
        
        if increase_count > INCREASE_COUNT_MAX:
            multipliers.append(conf.DECREASE_FACTOR)
        elif all(x > y for x, y in zip(recent_losses[:-1], recent_losses[1:])):
            multipliers.append(conf.INCREASE_FACTOR)
        else:
            multipliers.append(1.0)
    
    return torch.prod(torch.tensor(multipliers)).item()
###########################################################################
###########################################################################
###########################################################################
def default_scale_learning(epoch: int, total_epochs: int) -> bool:
    """
    Determine whether scale learning should occur based on the current epoch and total number of epochs.

    Parameters:
    - epoch (int): The current epoch in the training process.
    - total_epochs (int): The total number of epochs in the training process.

    Returns:
    - bool: `True` if scale learning should occur, `False` otherwise.
    
    Raises:
    - ValueError: If `total_epochs` is less than or equal to 1.
    """
    if total_epochs <= 1:
        raise ValueError("Total epochs must be greater than 1.")

    return epoch < int(conf.CURV_RATIO * total_epochs)
###########################################################################
###########################################################################
###########################################################################
def default_weight_exponent(epoch: int, total_epochs: int) -> float:
    """
    Calculate the weight exponent based on the current epoch and total number of epochs.

    Parameters:
    - epoch (int): The current epoch in the training process.
    - total_epochs (int): The total number of epochs in the training process.

    Returns:
    - float: The calculated weight exponent for the current epoch.
    
    Raises:
    - ValueError: If `total_epochs` is less than or equal to 1.
    """
    if total_epochs <= 1:
        raise ValueError("Total epochs must be greater than 1.")

    no_weight_epochs = int(conf.NO_WEIGHT_RATIO * total_epochs)
    
    if epoch < no_weight_epochs:
        return 0.0
    else:
        return -(epoch - no_weight_epochs) / (total_epochs - 1 - no_weight_epochs)
###########################################################################
###########################################################################
###########################################################################
def hypebrolic_cost(tangent_vectors: Tensor, distance_matrix: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
    """
    Compute the cost function for hyperbolic embeddings.

    Args:
        tangent_vectors (Tensor): Tangent vectors for optimization.
        distance_matrix (Tensor): The matrix of distances between points.
        **kwargs: Optional keyword arguments:
            - enable_scale_learning (bool): Flag to indicate if scale learning is enabled. Default is False.
            - scale_factor (Optional[Tensor]): Current scale factor. Default is None.
            - use_unweighted_cost (bool): Flag to indicate if the cost should be unweighted. Default is True.
            - weight_matrix (Optional[Tensor]): Weight matrix for weighted cost computation. Default is None.

    Returns:
        Tuple[Tensor, Tensor]: Computed cost value and updated scale factor.
    """
    enable_scale_learning = kwargs.get('enable_scale_learning', False)
    scale_factor = kwargs.get('scale_factor', torch.tensor(1.0, dtype=distance_matrix.dtype, device=distance_matrix.device))
    use_unweighted_cost = kwargs.get('use_unweighted_cost', True)
    weight_matrix = kwargs.get('weight_matrix', None)
    enable_save = kwargs.get('enable_save', conf.ENABLE_SAVE_MODE)
    
    embeddings = hyperbolic_exponential(tangent_vectors)
    flipped_embeddings = embeddings.clone()
    flipped_embeddings[0, :] *= -1

    gram_matrix = torch.matmul(embeddings.t(), flipped_embeddings).clamp(max=-1)
    param_dist_matrix = torch.arccosh(-gram_matrix)
    off_diagonal_mask = ~torch.eye(distance_matrix.size(0), dtype=torch.bool, device=distance_matrix.device)

    if enable_scale_learning:
        scale_factor = (param_dist_matrix * distance_matrix).sum() / (distance_matrix.pow(2).sum() )                      

    if use_unweighted_cost:
        norm_diff = torch.norm(param_dist_matrix[off_diagonal_mask] - scale_factor * distance_matrix[off_diagonal_mask], p='fro') ** 2
        norm_scale = torch.norm(scale_factor * distance_matrix[off_diagonal_mask], p='fro') ** 2
    elif weight_matrix is not None:
        weighted_param_dist_matrix = param_dist_matrix * weight_matrix
        weighted_distance_matrix = distance_matrix * weight_matrix

        norm_diff = torch.norm(weighted_param_dist_matrix[off_diagonal_mask] - scale_factor * weighted_distance_matrix[off_diagonal_mask], p='fro') ** 2
        norm_scale = torch.norm(scale_factor * weighted_distance_matrix[off_diagonal_mask], p='fro') ** 2
        
    else:
        norm_diff = torch.tensor(0.0, dtype=distance_matrix.dtype, device=distance_matrix.device)
        norm_scale = torch.tensor(1.0, dtype=distance_matrix.dtype, device=distance_matrix.device)

    cost = norm_diff / norm_scale

    if enable_save:
        numerator = param_dist_matrix.clone()
        numerator.fill_diagonal_(1)
        
        denominator = scale_factor * distance_matrix.clone()
        denominator.fill_diagonal_(1)
        
        relative_error = torch.pow( torch.div(numerator,denominator )-1 ,2)
        relative_error[torch.eye(distance_matrix.size(0), dtype=torch.bool, device=distance_matrix.device)] = 0

        return cost, scale_factor, relative_error

    return cost, scale_factor, None
###########################################################################
###########################################################################
###########################################################################
def J_norm(vector: torch.Tensor) -> float:
    """
    Compute the norm of a vector under the Lorentzian metric.

    Parameters:
    - vector (torch.Tensor): The input vector of shape (d,), (d, 1), or (1, d).

    Returns:
    - float: The Lorentzian norm of the vector.

    Raises:
    - ValueError: If the input vector is empty.
    """
    vector = vector.squeeze()  # Ensure the vector is of shape (d,)
    
    if vector.numel() == 0:
        raise ValueError("Input vector cannot be empty.")

    # Compute the Lorentzian norm
    norm2 = -vector[0]**2 + torch.sum(vector[1:]**2)
    
    return norm2
###########################################################################
###########################################################################
########################################################################### 
def euclidean_embedding(distance_matrix: torch.Tensor, dimension: int, **kwargs):
    """
    Optimize the embedding of points based on a given distance matrix.

    Parameters:
    - distance_matrix (torch.Tensor): The matrix representing distances between points (required).
    - dimension (int): The number of dimensions for the embedding space (required).
    - **kwargs: Optional parameters for the optimization process.
        - initial_points (torch.Tensor or None, optional): Initial points to start the optimization. 
          If None, random points are generated (default is None).
        - total_epochs (int, optional): Number of iterations for the optimization process (default is 2000).
        - log_function (callable or None, optional): Function to log optimization progress. 
          If None, logging is disabled (default is None).
        - learning_rate (callable or None, optional): Function to compute the learning rate dynamically. 
          If None, a fixed learning rate is used (default is None).
        - weight_exponent (float or None, optional): Exponent to adjust weights in the optimization process 
          (default is None).
        - initial_lr (float, optional): The initial learning rate for the optimizer (default is 0.1).
        - enable_save (bool, optional): If True, intermediate states during the optimization are saved (default is False).
        - time (callable or None, optional): Function to provide a timestamp for logging purposes (default is None).

    Returns:
    - points (torch.Tensor): The optimized points resulting from the optimization process.
    
    Raises:
    - ValueError: If the `distance_matrix` is not a torch.Tensor.
    """
    
    # Validate inputs
    if not isinstance(distance_matrix, torch.Tensor):
        raise ValueError("The 'distance_matrix' must be a torch.Tensor.")

    # Set default values for keyword arguments
    total_epochs = kwargs.get('total_epochs', conf.TOTAL_EPOCHS)
    initial_lr = kwargs.get('initial_lr', conf.INITIAL_LEARNING_RATE)
    enable_save = kwargs.get('enable_save', conf.ENABLE_SAVE_MODE)
    initial_points = kwargs.get('initial_points', None)
    log_function = kwargs.get('log_function', None)
    learning_rate = kwargs.get('learning_rate', None)
    weight_exponent = kwargs.get('weight_exponent', None)
    timestamp = kwargs.get('time', None)
    
    if initial_points is None:
        n = distance_matrix.size(0)
        points = torch.rand(dimension, n, requires_grad=True)
        points.data.mul_(0.01)
    else:
        points = initial_points.clone().detach().requires_grad_(True)

    if learning_rate is None:
        learning_rate_fn = lambda epoch, total_epochs, loss_list=None: default_learning_rate(epoch, total_epochs, loss_list, scale_range=(distance_range(distance_matrix)).item())
    if weight_exponent is None:
        weight_exponent_fn = lambda epoch, total_epochs, loss_list=None: default_weight_exponent(epoch, total_epochs)
    
    loss_history = []
    
    optimizer_points = optim.Adam([points], lr=initial_lr)
    weight_history = []
    lr_history = []
    scale_history = []
    for epoch in range(total_epochs):
        weight_exp = weight_exponent_fn(epoch, total_epochs, loss_history)
        weight_history.append(weight_exp)

        weight_matrix = torch.pow(distance_matrix, weight_exp)
        weight_matrix.fill_diagonal_(1)
                
        loss, relative_error = euclidean_cost(points, distance_matrix, weight_matrix=weight_matrix, enable_save = enable_save)
        loss.backward()
        loss_history.append(loss.detach().item())
        with torch.no_grad():
            points.grad = torch.nan_to_num(points.grad, nan=0.0)
        optimizer_points.step()

        lr = learning_rate_fn(epoch, total_epochs, loss_history)
        lr_history.append(lr*initial_lr)
        
        for param_group in optimizer_points.param_groups:
            param_group['lr'] = lr*initial_lr

        current_lr = optimizer_points.param_groups[0]['lr']
        num_digits = len(str(total_epochs))
        formatted_epoch = f"{epoch + 1:0{num_digits}d}"
        message = (f"[Epoch {formatted_epoch}/{total_epochs}] "
                   f"Loss: {loss.item():.8f}, Learning Rate: {current_lr:.10f}, "
                   f"Weight Exponent: {weight_exp:.8f}")

        if log_function:
            log_function(message)
        else:
            print(message)

        if enable_save and (relative_error is not None):
            path = f'{conf.OUTPUT_DIRECTORY}/{timestamp}'
            os.makedirs(path, exist_ok=True)
            # Save relative error to a file
            relative_error_path = os.path.join(path, f"RE_{epoch+1}.npy")
            np.save(relative_error_path, relative_error.detach().cpu().numpy())

    if points.requires_grad:
        points = points.detach()

    if enable_save:
        path = f'{conf.OUTPUT_DIRECTORY}/{timestamp}'
        os.makedirs(path, exist_ok=True)
        weight_path = os.path.join(path, f"weight_history.npy")
        np.save(weight_path, weight_history)
        lr_path = os.path.join(path, f"lr_history.npy")
        np.save(lr_path, lr_history)
    
    return points
###########################################################################
###########################################################################
########################################################################### 
def euclidean_cost(points: Tensor, distance_matrix: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
    """
    Compute the cost function for euclidean embeddings.

    Args:
        points (Tensor): points for optimization.
        distance_matrix (Tensor): The matrix of distances between points.
        **kwargs: Optional keyword arguments:
            - use_unweighted_cost (bool): Flag to indicate if the cost should be unweighted. Default is True.
            - weight_matrix (Optional[Tensor]): Weight matrix for weighted cost computation. Default is None.

    Returns:
        Tuple[Tensor, Tensor]: Computed cost value and updated scale factor.
    """
    
    use_unweighted_cost = kwargs.get('use_unweighted_cost', True)
    weight_matrix = kwargs.get('weight_matrix', None)
    enable_save = kwargs.get('enable_save', conf.ENABLE_SAVE_MODE)
    
    squared_norms = torch.sum(points ** 2, dim=0, keepdim=True)
    
    gram_matrix = torch.matmul(points.t(), points)
    param_dist_matrix = squared_norms - 2 * gram_matrix + squared_norms.t()
    off_diagonal_mask = ~torch.eye(distance_matrix.size(0), dtype=torch.bool, device=distance_matrix.device)

    
    if use_unweighted_cost:
        norm_diff = torch.norm(param_dist_matrix[off_diagonal_mask] - distance_matrix[off_diagonal_mask], p='fro') ** 2
        norm_scale = torch.norm(distance_matrix[off_diagonal_mask], p='fro') ** 2
    elif weight_matrix is not None:
        weighted_param_dist_matrix = param_dist_matrix * weight_matrix
        weighted_distance_matrix = distance_matrix * weight_matrix

        norm_diff = torch.norm(weighted_param_dist_matrix[off_diagonal_mask] - weighted_distance_matrix[off_diagonal_mask], p='fro') ** 2
        norm_scale = torch.norm( weighted_distance_matrix[off_diagonal_mask], p='fro') ** 2
        
    else:
        norm_diff = torch.tensor(0.0, dtype=distance_matrix.dtype, device=distance_matrix.device)
        norm_scale = torch.tensor(1.0, dtype=distance_matrix.dtype, device=distance_matrix.device)

    cost = norm_diff / norm_scale

    if enable_save:
        numerator = param_dist_matrix.clone()
        numerator.fill_diagonal_(1)
        
        denominator = distance_matrix.clone()
        denominator.fill_diagonal_(1)
        
        relative_error = torch.pow( torch.div(numerator,denominator )-1 ,2)
        relative_error[torch.eye(distance_matrix.size(0), dtype=torch.bool, device=distance_matrix.device)] = 0

        return cost, relative_error

    return cost, None
###########################################################################
###########################################################################
###########################################################################