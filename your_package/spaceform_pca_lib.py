import numpy as np
import logging
from scipy.sparse.linalg import eigs
import config

class Subspace:
    """
    Represents a mathematical subspace with a base point and tangent subspace properties.

    Attributes:
    ----------
    H : float
        Represents the combined point and tangent subspace.
    Hp : float
        Represents the tangent subspace.
    p : ndarray
        The base point of the subspace.
    """
    def __init__(self, H=None, Hp=None, p=None):
        """
        Initialize a Subspace instance.

        Parameters:
        ----------
        H : float, optional
            Combined point and tangent subspace. Default is None.
        Hp : float, optional
            Tangent subspace. Default is None.
        p : ndarray, optional
            Base point of the subspace. Default is None.
        """
        self.H = H
        self.Hp = Hp
        self.p = p


def prod(vector1, vector2):
    """
    Compute the Lorentzian inner product of two vectors using the Lorentzian metric.

    Parameters:
    ----------
    vector1 : ndarray
        The first input vector.
    vector2 : ndarray
        The second input vector.

    Returns:
    -------
    float
        The Lorentzian inner product of the two vectors.
    """
    modified_vector1 = vector1.copy()
    modified_vector1[0] = -modified_vector1[0]  # Apply the -1 metric to the time-like component

    return np.dot(modified_vector1.ravel(), vector2.ravel())


def estimate_hyperbolic_subspace(data):
    """
    Estimate the hyperbolic subspace from the input data.

    Parameters:
    data (ndarray): Input data matrix of shape (D+1, N).

    Returns:
    Subspace: Object containing hyperbolic subspace properties.
    """
    dim = data.shape[0] - 1  # Dimension in hyperbolic space
    Cx = np.dot(data, data.T)  # Covariance matrix
    eigvals, eig_signs, eigvecs = evd(Cx)

    # Identify and sort positive eigenvalues
    pos_indices = eig_signs >= 0
    pos_eigvals = eigvals[pos_indices]
    sorted_indices = np.argsort(pos_eigvals)[::-1]

    # Determine base point (eigenvector with negative eigenvalue)
    base_vec = np.squeeze(eigvecs[:, ~pos_indices])
    if base_vec.ndim > 1:
        base_vec = base_vec[:, max_index(eigvals[~pos_indices])]

    # Ensure base point's first component is positive
    if base_vec[0] < 0:
        base_vec = -base_vec

    # Construct subspace
    base_vec = base_vec.reshape(dim + 1, 1)
    Hp = eigvecs[:, pos_indices][:, sorted_indices]

    subspace = Subspace()
    subspace.H = np.concatenate((base_vec, Hp), axis=1)
    subspace.Hp = Hp
    subspace.p = base_vec
    return subspace

def normalize(vec):
    """
    Normalize a vector with checks for hyperbolic normalization based on a predefined threshold.

    Parameters:
    ----------
    vec : ndarray
        The input vector to be normalized.

    Returns:
    -------
    tuple
        A tuple containing the normalized vector and a boolean indicating whether additional hyperbolic normalization was performed.
    """
    valid = False
    threshold = config.atol ** 2

    # Compute the Euclidean norm of the vector
    euclidean_norm = np.linalg.norm(vec)

    if euclidean_norm > threshold:
        vec /= euclidean_norm
        hyperbolic_norm = np.sqrt(np.abs(prod(vec, vec)))
        if hyperbolic_norm > threshold:
            vec /= hyperbolic_norm
            valid = True

    return vec, valid

def max_index(array):
    """
    Get the first index of the maximum value in an array.

    Parameters:
    ----------
    array : ndarray
        Input array.

    Returns:
    -------
    int
        The first index of the maximum value, or -1 if the array is empty.
    """
    if array.size == 0:
        return -1  # Return -1 for empty array

    return np.argmax(array)

def orthonormalize(vec, eigvecs, eigsigns):
    """
    J-orthonormalize a vector relative to a subspace.

    Parameters:
    ----------
    vec : ndarray
        Vector to be orthonormalized.
    eigvecs : ndarray
        Basis vectors of the subspace.
    eigsigns : list
        Eigenvalue signs for projection adjustments.

    Returns:
    -------
    ndarray
        The J-orthonormalized vector.
    """
    dim = len(vec) - 1
    vec = np.reshape(vec, (dim + 1, 1))
    if len(eigsigns) > 0 :
        J = np.eye(dim + 1)
        J[0, 0] = -1
        proj_matrix = np.eye(dim + 1) - eigvecs @ np.diag(eigsigns) @ eigvecs.T @ J
        vec = proj_matrix @ vec

    return vec

def exp_sign(cnt):
    """
    Determine the expected sign based on count.

    Parameters:
    ----------
    cnt : int
        Input count.

    Returns:
    -------
    int
        -1 if count is 1, otherwise 1.
    """
    return -1 if cnt == 1 else 1


def compute_residual(Cx, eigvals, eigvecs):
    """
    Calculate the normalized residual matrix.

    Parameters:
    Cx (ndarray): Original matrix (DxD).
    eigvals (ndarray): Eigenvalues array.
    eigvecs (ndarray): Matrix of eigenvectors (DxD).

    Returns:
    ndarray: Residual matrix, normalized by maximum absolute value.
    """
    # Compute the contribution matrix as the weighted sum of outer products
    contributions = 0
    if len(eigvals) > 0:
        contributions = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    residual = Cx - contributions
    residual /= np.mean(np.abs(residual))
    return residual


def init_eigvec(Cx, target_sign, eigvals, eigvecs, eigsigns):
    """
    Compute an initial J-eigenvector for a residual matrix.

    Parameters:
    ----------
    Cx : ndarray
        Covariance matrix (DxD).
    target_sign : float
        Expected sign of the eigenvector.
    evals : ndarray
        Eigenvalues of the covariance matrix.
    evecs : ndarray
        Previously computed eigenvectors (DxD).
    esigns : ndarray
        Signs of eigenvalues.

    Returns:
    -------
    ndarray
        Computed J-eigenvector.
    """
    res_Cx = compute_residual(Cx, eigvals, eigvecs)
    dim = res_Cx.shape[0] - 1
    tol = config.atol
    J = np.eye(dim + 1)
    J[0, 0] = -1

    # Compute leading eigenvector of J-transformed residual matrix
    _, vec = eigs(res_Cx @ J, k=1, which='LM')
    vec = np.real(vec)

    while True:
        noise = tol * np.random.randn(dim + 1, 1)
        vec = orthonormalize(vec + noise, eigvecs, eigsigns)
        vec, valid = normalize(vec)
        if valid and prod(vec, vec) * target_sign >= 0:
            return vec


def random_eigvec(dim, target_sign, eigvecs, eigsigns):
    """
    Generate a valid random J-eigenvector.

    Parameters:
    dim (int): Vector dimension.
    target_sign (float): Desired eigenvector sign.
    eigvecs (ndarray): Precomputed eigenvector matrix (dim x dim).
    eigsigns (ndarray): Signs of eigenvalues.

    Returns:
    ndarray: Valid random J-eigenvector.
    """
    while True:
        vec = np.random.randn(dim + 1, 1)
        vec = orthonormalize(vec, eigvecs, eigsigns)
        vec, valid = normalize(vec)
        if valid and prod(vec, vec) * target_sign >= 0:
            return vec


def clean_mul(Cxj, v, eigvecs, eigsigns):
    """
    Performs one iteration of matrix-vector multiplications with normalization and orthonormalization steps.
    
    Parameters:
        Cxj (np.ndarray): The matrix used for multiplication.
        v (np.ndarray): The input vector to be multiplied.
        accuracy_factor (int): Number of times orthonormalization is applied.
        eigvecs (np.ndarray): The set of eigenvectors for orthonormalization.
        eigsigns (np.ndarray): Signs of the eigenvalues for orthonormalization.

    Returns:
        np.ndarray: The resulting vector after the iteration.
    """

    # Step 1: Matrix-vector multiplication
    v_out = np.matmul(Cxj, v)
    v_out, _ = normalize(v_out)
    v_out = orthonormalize(v_out, eigvecs, eigsigns)
    v_out, _ = normalize(v_out)

    return v_out


def eigenval(Cx, vec):
    """
    Computes the j-th eigenvalue for given matrices and eigenvector.
    
    Parameters:
        Cx (np.ndarray): The covariance matrix.
        vec (np.ndarray): The j-eigenvector.

    Returns:
        float: The computed j_eigenvalue.
    """
    dim = len(vec) - 1
    J = np.eye(dim + 1)
    J[0, 0] = -1
    
    result_vector = Cx @ J @ Cx @ J @ vec
    return np.sqrt(np.linalg.norm(result_vector) / np.linalg.norm(vec))

def evd(Cx):
    """
    Perform eigenvalue decomposition (EVD) for a given matrix `Cx`.

    The function iteratively computes eigenvalues, eigenvectors, and signs of eigenvalues 
    until all eigenvalues are extracted, ensuring stability and accuracy using orthonormalization.

    Parameters:
        Cx (numpy.ndarray): Input matrix for decomposition.

    Returns:
        tuple: A tuple containing:
            - eigvals (numpy.ndarray): Array of eigenvalues.
            - eigsigns (numpy.ndarray): Array of eigenvalue signs.
            - eigvecs (numpy.ndarray): Matrix of eigenvectors.
    """
    dim = Cx.shape[0] - 1
    J = np.eye(dim + 1)
    J[0, 0] = -1
    Cxj = Cx @ J

    # Constants
    ev_thresh = config.atol ** 3
    max_iters = int(1000*np.sqrt(dim))

    eigvals, eigsigns, eigvecs = [], [], []
    iteration = 0
    condition = True

    while condition:
        iteration += 1
        target_sign = exp_sign(iteration)
        v = init_eigvec(Cx, target_sign, eigvals, eigvecs, eigsigns)
        
        v_next = clean_mul(Cxj, v, eigvecs, eigsigns)
        for _ in range(max_iters):
            v_next = clean_mul(Cxj, v_next, eigvecs, eigsigns)
            
        v = orthonormalize(v_next, eigvecs, eigsigns)
        v, valid = normalize(v)

        if (np.linalg.norm(v) < ev_thresh or prod(v, v) * target_sign < 0) or (not valid):
            v = random_eigvec(dim, target_sign, eigvecs, eigsigns)

        sign = prod(v, v)
        v = v.reshape(-1, 1)
        eigvecs = np.concatenate((eigvecs, v), axis=1) if len(eigvals)>0 else v
        eigvals.append(eigenval(Cx, v))
        eigsigns.append(sign)
        
        # Update stopping condition
        pos_signs = np.sum(np.array(eigsigns) >= 0)
        neg_signs = np.sum(np.array(eigsigns) < 0)
        condition = not (pos_signs == dim and neg_signs >= 1)
    return np.array(eigvals), np.array(eigsigns), np.array(eigvecs)