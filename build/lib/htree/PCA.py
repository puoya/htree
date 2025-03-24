import numpy as np
import logging
from datetime import datetime
import torch
import spaceform_pca_lib as sfpca


class PCA:
    """
    A class to perform Principal Component Analysis (PCA) on embeddings and maintain a reusable mapping.

    Attributes:
        embedding: The input embedding to perform PCA on.
        geometry (str): Geometry of the embedding ('hyperbolic' or 'euclidean').
        model (str): Model of the embedding (e.g., 'loid', 'cartesian').
        _mapping_matrix (np.ndarray): Matrix that defines the PCA mapping.
        mean (np.ndarray): Mean of the input points.
        subspace (np.ndarray): Principal components (subspace) of the input points.
    """

    def __init__(self, embedding, enable_logging=False):
        """
        Initializes the PCA instance.

        Args:
            embedding: The embedding on which PCA is performed.
            enable_logging (bool): If True, enables logging. Default is False.
        """
        self.embedding = embedding
        self._geometry = embedding.geometry
        self._points = embedding.points
        self.mean = None
        self._mapping_matrix = None

        if enable_logging:
            self._setup_logging()
        self._log_info("Initializing PCA")
        self._validate_embedding()
        self._compute_mapping()

    def _log_info(self, message: str) -> None:
        """
        Logs an informational message.

        Args:
            message (str): The message to log.
        """
        if logging_enabled():  # Check if logging is globally enabled
            get_logger().info(message)

    def _validate_embedding(self):
        """Validate the input embedding."""
        if self._points.size == 0:
            raise ValueError("Embedding points are empty. PCA cannot be performed.")
        self._log_info("Embedding validation complete.")

    def _compute_mapping(self):
        """
        Compute the PCA mapping matrix for the embedding.
        The method differs based on the geometry of the embedding space.
        """
        from sklearn.decomposition import PCA

        if self._geometry == 'euclidean':
            # Compute the centroid of the points

            centroid = self.embedding.centroid()
            self.mean = centroid

            centered_points = self._points - centroid.reshape(-1, 1)
            gram_matrix = np.dot(centered_points, centered_points.T)

            eigenvalues, eigenvectors = np.linalg.eigh(gram_matrix)

            sorted_indices = np.argsort(eigenvalues)[::-1]
            sorted_eigenvectors = eigenvectors[:, sorted_indices]
            components = sorted_eigenvectors.T

            self._mapping_matrix = torch.tensor(components)
            
        elif self._geometry == 'hyperbolic':
            # Switch to Loid model if not already in it
            if self.embedding.model != 'loid':
                self.embedding = self.embedding.switch_model()
                self._points = self.embedding.points
            # Compute the centroid of the points in hyperbolic space
            centroid = self.embedding.centroid()
            self.mean = centroid
            subspace = sfpca.estimate_hyperbolic_subspace(self._points)
            self._mapping_matrix = subspace.H
        else:
            raise ValueError(f"Unsupported geometry: {self.geometry}. "
                             "Valid options are 'euclidean' or 'hyperbolic'.")

        self._log_info("Mapping matrix and subspace computed.")


    def map_to_dimension(self, target_dimension):
        """
        Map the embedding to a lower-dimensional space using the precomputed mapping.

        Args:
            target_dimension (int): The target number of dimensions.

        Returns:
            A new embedding with reduced dimensions, retaining the original geometry.
        """
        if target_dimension > self._mapping_matrix.shape[1]:
            raise ValueError("Target dimension exceeds the dimensions of the computed subspace.")


        if self._geometry == 'euclidean':
            centroid = self.mean
            # Center the points by subtracting the centroid
            centered_points = self._points - centroid.reshape(-1, 1)
            reduced_points = self._mapping_matrix[:, :target_dimension].T @ centered_points
        elif self._geometry == 'hyperbolic':
            dim, N = np.shape(self._points)
            dim -= 1

            J = np.eye(dim + 1)
            J[0, 0] = -1

            H = self._mapping_matrix[:, :target_dimension + 1]
            Jk = np.eye(target_dimension + 1)
            Jk[0, 0] = -1

            # projection =  H @ Jk @ H.T @  J 
            projection =  Jk @ H.T @  J 
            reduced_points = np.matmul(projection,self._points).numpy()
            for n in range(N):
                x = reduced_points[:, n]
                reduced_points[:, n] = x / np.sqrt(-sfpca.prod(x,x))

        else:
            raise ValueError(f"Unsupported geometry: {self.geometry}. "
                             "Valid options are 'euclidean' or 'hyperbolic'.")


        return self._create_embedding(reduced_points)

    def _create_embedding(self, points):
        """Create a new embedding instance with the given points."""
        new_embedding = self.embedding.copy()
        new_embedding.points = points
        return new_embedding

    def get_mean(self):
        """Return the mean of the input points."""
        return self.mean

    def get_subspace(self):
        """Return the principal components (subspace)."""
        return self._mapping_matrix

    def __repr__(self):
        return (f"PCA(geometry={self.geometry}, "
                f"original_dimension={self.points.shape[0]}, "
                f"mean_computed={self.mean is not None}, "
                f"subspace_computed={self.subspace is not None})")
