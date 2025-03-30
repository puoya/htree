import numpy as np
from datetime import datetime
import torch
# from logger import get_logger, logging_enabled, get_time
# import utils
from .logger import get_logger, logging_enabled, get_time
from . import utils

class PCA:
    """
    Principal Component Analysis (PCA) for different embedding geometries.

    This class performs PCA on a given embedding object and computes a reusable mapping
    for dimensionality reduction. It supports both 'euclidean' and 'hyperbolic' geometries.

    Attributes:
        embedding: The input embedding object which should have attributes:
            - geometry (str): Either 'euclidean' or 'hyperbolic'.
            - points (torch.Tensor): A tensor of points with shape [dim, n_points].
            - model (str): Model of the embedding (e.g., 'loid', 'cartesian').
            - centroid (callable): A method returning the centroid of the embedding points.
            - switch_model (callable): A method to switch the model (e.g., to 'loid') if needed.
        mean (torch.Tensor): The computed mean/centroid of the embedding points.
        subspace (torch.Tensor): The PCA mapping matrix (basis of principal components).
    """

    def __init__(self, embedding):
        """
        Initializes the PCA instance and computes the mapping.

        The initialization process validates the embedding and immediately computes the PCA mapping,
        based on the geometry of the embedding.

        Args:
            embedding: An embedding object containing at least the attributes:
                - geometry: 'euclidean' or 'hyperbolic'
                - points: A torch.Tensor of data points
                - model: A string identifier for the embedding model (e.g., 'loid')
                - centroid(): A method that computes the centroid of the points
                - switch_model(): (for hyperbolic) A method to switch the model if necessary

        Raises:
            ValueError: If the embedding points are empty.
        """
        self.embedding = embedding
        self._geometry = embedding.geometry
        self._points = embedding.points
        self.mean = None
        self.subspace = None

        self._log_info("Initializing PCA")
        self._validate_embedding()
        self._compute_mapping()

    def _log_info(self, message: str):
        """
        Logs an informational message if global logging is enabled.

        Args:
            message (str): The log message.
        """
        if logging_enabled(): get_logger().info(message) 

    def _validate_embedding(self):
        """
        Validates the input embedding.

        Ensures that the embedding contains data points and logs the validation status.

        Raises:
            ValueError: If the embedding points tensor is empty.
        """
        if self._points.size == 0:
            raise ValueError("Embedding points are empty. PCA cannot be performed.")
        self._log_info("Embedding validation complete.")

    def _compute_mapping(self):
        """
        Computes the PCA mapping matrix based on the embedding geometry.

        For 'euclidean' geometry:
            - Computes the centroid using the embedding's centroid() method.
            - Centers the data and computes the Gram matrix.
            - Performs eigenvalue decomposition to obtain the principal components.
        For 'hyperbolic' geometry:
            - If the embedding model is not 'loid', it switches to the 'loid' model.
            - Computes the hyperbolic centroid and mapping via a specialized utility function.
        
        Raises:
            ValueError: If the embedding geometry is neither 'euclidean' nor 'hyperbolic'.
        """
    
        if self._geometry == 'euclidean':
            c = self.embedding.centroid()
            self.mean = c
            centered = self._points - c.view(-1, 1)
            gram = centered @ centered.T
            eigvals, eigvecs = torch.linalg.eigh(gram)
            self.subspace = eigvecs[:, torch.argsort(eigvals, descending=True)].T
        elif self._geometry == 'hyperbolic':
            # Switch to Loid model if not already in it
            if self.embedding.model != 'loid':
                self.embedding = self.embedding.switch_model()
                self._points = self.embedding.points
            # Compute the centroid of the points in hyperbolic space
            base, H = utils.hyperbolic_PCA(self._points)
            self.subspace = H
            self.mean = base
        else:
            raise ValueError(f"Unsupported geometry: {self.geometry}. "
                             "Valid options are 'euclidean' or 'hyperbolic'.")

        self._log_info("Mapping matrix and subspace computed.")

    def map_to(self, dim):
        """
        Reduces the dimensionality of the embedding using the computed PCA mapping.

        Depending on the geometry, this function applies the PCA mapping to reduce the input 
        embedding's dimensions to the target 'dim'. It returns a new embedding instance with 
        the reduced data.

        Args:
            dim (int): The target dimension for the reduced embedding. Must be less than or equal
                       to the number of principal components available.

        Returns:
            A new embedding object with its 'points' attribute updated to the reduced data.

        Raises:
            ValueError: If the requested target dimension exceeds the computed subspace dimensions.
            ValueError: If the geometry of the embedding is unsupported.
        """
        if dim > self.subspace.shape[1]:
            raise ValueError("Target dimension exceeds computed subspace dimensions.")

        if self._geometry == 'euclidean':
            c = self.mean
            centered = self._points - c.reshape(-1, 1)
            reduced = self.subspace[:, :dim].T @ centered

        elif self._geometry == 'hyperbolic':
            J = torch.eye(self._points.shape[0], device=self._points.device, dtype=self._points.dtype)
            J[0, 0] = -1

            H, Jk = self.subspace[:, :dim + 1], torch.eye(dim + 1, device=self._points.device, dtype=self._points.dtype)
            Jk[0, 0] = -1

            reduced = Jk @ H.T @ J @ self._points
            reduced /= torch.sqrt(-utils.J_norm(reduced)).unsqueeze(0)

        else:
            raise ValueError(f"Unsupported geometry: {self._geometry}. Options: 'euclidean' or 'hyperbolic'.")
        new_embedding = self.embedding.copy()
        new_embedding.points = reduced
        return new_embedding

    def __repr__(self):
        """
        Returns a string representation of the PCA instance.

        The representation includes the geometry type, original dimension of the embedding,
        and status of mean and subspace computation.

        Returns:
            str: A summary string for the PCA instance.
        """
        return (f"PCA(geometry={self.geometry}, "
                f"original_dimension={self.points.shape[0]}, "
                f"mean_computed={self.mean is not None}, "
                f"subspace_computed={self.subspace is not None})")