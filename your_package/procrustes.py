import numpy as np
import torch
import pickle
import logging
import os
import copy
from typing import Optional, Union, List
from datetime import datetime
import config
from torch.optim import Adam

class HyperbolicProcrustes:
    """
    A class for performing Hyperbolic orthogonal Procrustes analysis, mapping one embedding to another.

    Attributes:
        cost (float): The cost associated with the mapping, inversely related to its quality.
        _mapping_matrix (torch.Tensor): Transformation matrix that maps the source embedding to the target embedding.
        source_embedding (Embedding): Source embedding instance.
        target_embedding (Embedding): Target embedding instance.
        logger (logging.Logger): Logger for recording class activities if enabled.
    """
    
    def __init__(self, 
                 source_embedding: 'Embedding', 
                 target_embedding: 'Embedding', 
                 mode: str = 'default', 
                 enable_logging: bool = False):
        """
        Initializes the HyperbolicProcrustes instance.

        Args:
            source_embedding (Embedding): The source embedding to map from.
            target_embedding (Embedding): The target embedding to map to.
            mode (str): Mode of computation, either 'default' or 'accurate'.
            enable_logging (bool): If True, enables logging. Default is False.
        """
        self._logger = None
        self._current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        if enable_logging:
            self._setup_logging()

        self.source_embedding = source_embedding
        self.source_model = source_embedding.model
        self.target_embedding = target_embedding
        self.target_model = target_embedding.model
        self._mapping_matrix = None
        

        self._log_info("Initializing HyperbolicProcrustes")
        self._validate_embeddings()
        self._compute_mapping(mode = mode)

    def _setup_logging(self, log_dir: str = config.LOG_DIR, log_level: int = logging.INFO, log_format: str = '%(asctime)s - %(levelname)s - %(message)s') -> None:
        """
        Configures logging for the class.

        Args:
            log_dir (str): Directory for saving log files. Defaults to 'log'.
            log_level (int): Logging level. Defaults to logging.INFO.
            log_format (str): Format for log messages. Defaults to '%(asctime)s - %(levelname)s - %(message)s'.
        """
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"HyperbolicProcrustes_{self._current_time}.log")
        logging.basicConfig(filename=log_file, level=log_level, format=log_format)
        self._logger = logging.getLogger(__name__)
        self._log_info("Logging setup complete.")

    def _log_info(self, message: str) -> None:
        """
        Logs an informational message.

        Args:
            message (str): The message to be logged.
        """
        if self._logger:
            self._logger.info(message)

    def _validate_embeddings(self) -> None:
        """
        Validates that the source and target embeddings have compatible dimensions and curvatures.
        """
        self._log_info("Validating embeddings")

        # Ensure embeddings are in compatible models
        if self.source_embedding.model == 'poincare':
            self.source_embedding = self.source_embedding.switch_model()
        if self.target_embedding.model == 'poincare':
            self.target_embedding = self.target_embedding.switch_model()

        # Check for matching shapes
        if self.source_embedding.dimension != self.target_embedding.dimension:
            self._log_info("Source and target embeddings must have the same shape")
            raise ValueError("Source and target embeddings must have the same shape")
        
        # Check for matching curvatures
        if not torch.isclose(self.source_embedding.curvature, self.target_embedding.curvature, config.atol):
            self._log_info("Source and target curvatures must be equal")
            raise ValueError("Source and target curvatures must be equal")
        
        self._log_info("Validation successful")

    def project_to_orthogonal(self, R: torch.Tensor) -> torch.Tensor:
        """
        Projects the given matrix R to the nearest orthogonal matrix using Singular Value Decomposition (SVD).

        Args:
            R (torch.Tensor): The matrix to be projected.

        Returns:
            torch.Tensor: The orthogonal matrix closest to R.
        """
        # Perform Singular Value Decomposition
        U, _, V = torch.svd(R)

        # Compute the nearest orthogonal matrix and ensure the output has the same dtype as R
        return (U @ V.T).to(R.dtype)

    def matrix_sqrtm(self, A: torch.Tensor) -> torch.Tensor:
        """
        Computes the matrix square root of a positive definite matrix with eigenvalue thresholding.

        Args:
            A (torch.Tensor): A positive definite matrix.

        Returns:
            torch.Tensor: The matrix square root of A.
        """
        # Compute the eigenvalues and eigenvectors
        eigvals, eigvecs = torch.linalg.eigh(A)

        # Ensure eigenvalues are non-negative by clamping
        eigvals = torch.clamp(eigvals, min=0)

        # Take the square root of the eigenvalues while preserving the dtype of A
        sqrt_eigvals = torch.diag(torch.sqrt(eigvals).to(A.dtype))

        # Reconstruct the matrix and return it with the same dtype as A
        return (eigvecs @ sqrt_eigvals @ eigvecs.T).to(A.dtype)


    def map_to_translation(self, b: torch.Tensor) -> torch.Tensor:
        """
        Constructs a translation matrix for the given vector b.

        Args:
            b (torch.Tensor): The translation vector.

        Returns:
            torch.Tensor: The translation matrix.
        """
        b = b[1:].flatten()

        D = len(b)
        norm_b = torch.norm(b)
        I = torch.eye(D, device=b.device, dtype=b.dtype)
        Rb = torch.zeros((D + 1, D + 1), device=b.device, dtype=b.dtype)
        Rb[0, 0] = torch.sqrt(1 + norm_b**2)
        Rb[0, 1:] = b.view(1, -1)
        Rb[1:, 0] = b.view(-1, 1).squeeze()
        Rb[1:, 1:] = self.matrix_sqrtm(I + torch.outer(b.view(-1), b.view(-1)))
        return Rb

    def map_to_rotation(self, U: torch.Tensor) -> torch.Tensor:
        """
        Constructs a rotation matrix from the given matrix U.

        Args:
            U (torch.Tensor): The rotation matrix.

        Returns:
            torch.Tensor: The rotation matrix in augmented space.
        """
        D = U.size(0)
        Ru = torch.eye(D + 1, device=U.device, dtype=U.dtype)
        Ru[1:, 1:] = U
        return Ru

    def _compute_mapping(self, mode: str = 'default') -> None:
        """
        Computes the Hyperbolic orthogonal Procrustes mapping and associated cost.

        Args:
            mode (str): Computation mode, 'default' for a basic approach or 'accurate' for refined optimization.
        """
        self._log_info("Computing mapping")

        D = self.source_embedding.dimension

        # Center the source and target embeddings
        src_embedding = self.source_embedding.copy()
        target_embedding = self.target_embedding.copy()


        # Filter points by intersecting labels
        source_labels = set(src_embedding._labels)
        target_labels = set(target_embedding._labels)
        common_labels = list(source_labels & target_labels)
        if not common_labels:
            raise ValueError("No matching labels found between source and target embeddings.")
        
        src_indices = [src_embedding._labels.index(label) for label in common_labels]
        target_indices = [target_embedding._labels.index(label) for label in common_labels]

        src_embedding._points = src_embedding._points[:, src_indices]
        target_embedding._points = target_embedding._points[:, target_indices]


        src_center = src_embedding.centroid()
        src_embedding.center()
        target_center = target_embedding.centroid()
        target_embedding.center()
        # Compute optimal rotation matrix using SVD
        target_points_loid = target_embedding.points
        src_points_loid = src_embedding.points
        U, _, Vt = torch.svd(torch.mm(target_points_loid[1:], src_points_loid[1:].T))
        R = self.map_to_rotation(torch.mm(U, Vt.T))
        src_embedding.rotate(R)

        # Compute the transformation matrix in Lorentzian space
        transformation = (
            self.map_to_translation(target_center) @ 
            R @ 
            self.map_to_translation(-src_center)
        )
        self._mapping_matrix = transformation  # Default mode
        
        srouce_embedding = self.source_embedding.copy()
        srouce_embedding._points = srouce_embedding._points[:, src_indices]
        srouce_embedding._points = self._mapping_matrix @srouce_embedding._points
        
        target_embedding = self.target_embedding.copy()
        target_embedding._points = target_embedding._points[:, target_indices]


        if mode == 'accurate':
            src_points = srouce_embedding._points
            b = torch.zeros(D, requires_grad=True,dtype=src_points.dtype)
            R = torch.eye(D, requires_grad=True,dtype=src_points.dtype)
            optimizer = Adam([b, R], lr=0.001)
            for epoch in range(1000):
                optimizer.zero_grad()
                norm_b = torch.norm(b)
                b_new = torch.cat([torch.sqrt(1 + norm_b**2).unsqueeze(0), b])
                # Apply the current rotation and translation to the source points
                srouce_embedding._points = self.map_to_rotation(R) @ src_points.clone()
                srouce_embedding.translate(b_new)
                # Compute the cost function
                cost = sum(
                    (srouce_embedding.poincare_distance(srouce_embedding._points[:, n], target_embedding._points[:, n]))**2 
                    for n in range(srouce_embedding.n_points)
                )
                cost.backward(retain_graph=True)
                # Orthogonal projection of rotation matrix
                with torch.no_grad():
                    R.data.copy_(self.project_to_orthogonal(R))

                # Handle NaN gradients
                with torch.no_grad():
                    for param in optimizer.param_groups[0]['params']:
                        if param.grad is not None:
                            param.grad = torch.nan_to_num(param.grad, nan=0.0)
                optimizer.step()

                self._log_info(f"Epoch {epoch}, Cost: {cost.item()}, b: {b.detach().numpy()}, R: {R.detach().numpy()}")
            
            # Final orthogonal projection of R
            R = self.project_to_orthogonal(R.detach())
            b = b.detach()
            norm_b = torch.norm(b)
            b_new = torch.cat([torch.sqrt(1 + norm_b**2).unsqueeze(0), b])
            self._mapping_matrix = (
                self.map_to_translation(b_new) @ 
                self.map_to_rotation(R) @ 
                transformation.double()
            )

    def map(self, source_embedding: 'Embedding') -> 'Embedding':
        """
        Applies the computed mapping matrix to transform the given embedding.

        Args:
            source_embedding (Embedding): The embedding to be transformed.

        Returns:
            Embedding: The transformed embedding.
        """
        self._log_info("Mapping Embedding")

        if source_embedding.model == 'poincare':
            source_embedding = source_embedding.switch_model()
        target_embedding = source_embedding.copy()
        
        target_embedding.points = (self._mapping_matrix) @ source_embedding.points
        if target_embedding.model != self.target_model:
            target_embedding = target_embedding.switch_model()
        
        self._log_info(f"Mapped points with shape: {target_embedding.points.shape}")
        return target_embedding


class EuclideanProcrustes:
    """
    A class for performing Euclidean orthogonal Procrustes analysis, mapping one embedding to another.

    Attributes:
        _mapping_matrix (torch.Tensor): Transformation matrix that maps the source embedding to the target embedding.
        source_embedding (Embedding): Source embedding instance.
        target_embedding (Embedding): Target embedding instance.
        logger (logging.Logger): Logger for recording class activities if enabled.
    """
    
    def __init__(self, 
                 source_embedding: 'Embedding', 
                 target_embedding: 'Embedding', 
                 enable_logging: bool = False):
        """
        Initializes the EuclideanProcrustes instance.

        Args:
            source_embedding (Embedding): The source embedding to map from.
            target_embedding (Embedding): The target embedding to map to.
            mode (str): Mode of computation, either 'default' or 'accurate'.
            enable_logging (bool): If True, enables logging. Default is False.
        """
        self._current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._logger = None
        if enable_logging:
            self._setup_logging()

        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self._mapping_matrix = None

        self._log_info("Initializing EuclideanProcrustes")
        self._validate_embeddings()
        self._compute_mapping()

    def _setup_logging(self, log_dir: str = config.LOG_DIR, log_level: int = logging.INFO, log_format: str = '%(asctime)s - %(levelname)s - %(message)s') -> None:
        """
        Configures logging for the class.

        Args:
            log_dir (str): Directory for saving log files. Defaults to 'log'.
            log_level (int): Logging level. Defaults to logging.INFO.
            log_format (str): Format for log messages. Defaults to '%(asctime)s - %(levelname)s - %(message)s'.
        """
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"EuclideanProcrustes_{self._current_time}.log")
        logging.basicConfig(filename=log_file, level=log_level, format=log_format)
        self._logger = logging.getLogger(__name__)
        self._log_info("Logging setup complete.")

    def _log_info(self, message: str) -> None:
        """
        Logs an informational message.

        Args:
            message (str): The message to be logged.
        """
        if self._logger:
            self._logger.info(message)

    def _validate_embeddings(self) -> None:
        """
        Validates that the source and target embeddings have compatible dimensions.
        """
        self._log_info("Validating embeddings")

        # Check for matching shapes
        if self.source_embedding.dimension != self.target_embedding.dimension:
            self._log_info("Source and target embeddings must have the same shape")
            raise ValueError("Source and target embeddings must have the same shape")
        
        self._log_info("Validation successful")

    # def project_to_orthogonal(self, R: torch.Tensor) -> torch.Tensor:
    #     """
    #     Projects the given matrix R to the nearest orthogonal matrix using Singular Value Decomposition (SVD).

    #     Args:
    #         R (torch.Tensor): The matrix to be projected.

    #     Returns:
    #         torch.Tensor: The orthogonal matrix closest to R.
    #     """
    #     U, _, V = torch.svd(R)
    #     return (U @ V.T).double()

    def _compute_mapping(self) -> None:
        """
        Computes the Euclidean orthogonal Procrustes mapping and associated cost.

        Args:
            mode (str): Computation mode, 'default' for a basic approach or 'accurate' for refined optimization.
        """
        self._log_info("Computing mapping")

        dimension = self.source_embedding.dimension
        # Center the source and target embeddings
        src_embedding = self.source_embedding.copy()
        src_center = src_embedding.centroid()
        src_embedding.center()


        trg_embedding = self.target_embedding.copy()
        trg_center = trg_embedding.centroid()
        trg_embedding.center()

        # Compute optimal rotation matrix using SVD
        U, _, Vt = torch.svd(torch.mm(trg_embedding.points, src_embedding.points.T))
        R_init = torch.mm(U, Vt.T)
        src_embedding.rotate(R_init)

        b_init = trg_center - R_init @ src_center

        # Compute the transformation matrix
        self._mapping_matrix = torch.eye(dimension+1, device=R_init.device, dtype=R_init.dtype)
        self._mapping_matrix[:-1, :-1] = R_init
        self._mapping_matrix[:-1, -1] = b_init

        # src_embedding = self.source_embedding.copy()
        # trg_embedding = self.target_embedding.copy()
        # if mode == 'accurate':
        #     src_points = src_embedding._points
        #     # b = torch.zeros(D, requires_grad=True,dtype=src_points.dtype)
        #     # R = torch.eye(D, requires_grad=True,dtype=src_points.dtype)

        #     b = torch.tensor(b_init, requires_grad=True, dtype=src_points.dtype)
        #     R = torch.tensor(R_init, requires_grad=True, dtype=src_points.dtype)
        #     optimizer = Adam([b, R], lr=0.001)

        #     for epoch in range(1000):
        #         optimizer.zero_grad()

        #         # Apply the current rotation and translation to the source points
        #         src_embedding.points = R @ src_points.clone()
        #         src_embedding.translate(b)

        #         # Compute the cost function
        #         cost = sum(
        #             (torch.norm(src_embedding.points[:, n]- trg_embedding.points[:, n]))**2 
        #             for n in range(src_embedding.n_points)
        #         )
        #         cost.backward(retain_graph=True)
                
        #         # Orthogonal projection of rotation matrix
        #         with torch.no_grad():
        #             R.data.copy_(self.project_to_orthogonal(R))

        #         # Handle NaN gradients
        #         with torch.no_grad():
        #             for param in optimizer.param_groups[0]['params']:
        #                 if param.grad is not None:
        #                     param.grad = torch.nan_to_num(param.grad, nan=0.0)
        #         optimizer.step()

        #         self._log_info(f"Epoch {epoch}, Cost: {cost.item()}, b: {b.detach().numpy()}, R: {R.detach().numpy()}")
            
        #     # Final orthogonal projection of R
        #     R = self.project_to_orthogonal(R.detach())
        #     self._mapping_matrix[:-1, :-1] = R
        #     self._mapping_matrix[:-1, -1] = b.detach()

    def map(self, source_embedding: 'Embedding') -> 'Embedding':
        """
        Applies the computed mapping matrix to transform the given embedding.

        Args:
            source_embedding (Embedding): The embedding to be transformed.

        Returns:
            Embedding: The transformed embedding.
        """
        self._log_info("Mapping Embedding")
        target_embedding = source_embedding.copy()
        target_embedding.points = (self._mapping_matrix[:-1, :-1] @ source_embedding.points 
                                   + self._mapping_matrix[:-1, -1].view(-1, 1))
        self._log_info(f"Mapped points with shape: {target_embedding.points.shape}")
        return target_embedding
