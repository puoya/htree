import os
import copy
import torch
import pickle
import logging
import numpy as np

from typing import Optional, Union, List,Callable
import scipy.sparse.linalg as spla
from datetime import datetime

import htree.conf as conf
import htree.utils as utils
import htree.embedding as embedding
import htree.procrustes as procrustes
from htree.logger import get_logger, logging_enabled, get_time
################################################################################################
class Embedding:
    """
    A class representing an abstract embedding.

    Attributes:
        geometry (str): The geometry of the space (e.g., 'euclidean', 'hyperbolic').
        points (torch.Tensor): A PyTorch tensor representing the points in the space.
        labels (list): A list of labels corresponding to the points in the space.
    """

    def __init__(self, 
                 geometry: Optional[str] = 'hyperbolic', 
                 points: Optional[Union[np.ndarray, torch.Tensor]] = None, 
                 labels: Optional[List[Union[str, int]]] = None):
        """
        Initializes the Embedding.

        Args:
            geometry (str): The geometry of the space. Default is 'hyperbolic'.
            points (Optional[Union[np.ndarray, torch.Tensor]]): A NumPy array or PyTorch tensor of points. Default is None.
            labels (Optional[List[Union[str, int]]]): A list of labels corresponding to the points. Default is None.
        """
        self._current_time = get_time() or datetime.now()  # Uses a global timestamp function
        if geometry not in {'euclidean', 'hyperbolic'}:
            self._log_info("Invalid geometry type: %s", geometry)
            raise ValueError("Invalid geometry type. Choose either 'euclidean' or 'hyperbolic'.")

        if geometry not in {'euclidean', 'hyperbolic'}:
            self._log_info(f"Invalid geometry type: {geometry}")
            raise ValueError("Invalid geometry type. Choose either 'euclidean' or 'hyperbolic'.")

        self._geometry = geometry
        self._points = self._convert_value(points) if points is not None else torch.empty((0, 0))
        self._labels = labels if labels is not None else list(range(self._points.shape[1]))
        self._log_info(f"Initialized Embedding with geometry={self._geometry}")
    ################################################################################################
    def _log_info(self, message: str) -> None:
        """
        Logs an informational message.

        Args:
            message (str): The message to log.
        """
        if logging_enabled():  # Check if logging is globally enabled
            get_logger().info(message)
    ################################################################################################
    def _convert_value(self, value: Union[np.ndarray, torch.Tensor, list, int, float]) -> torch.Tensor:
        """
        Converts the points to a PyTorch tensor with double precision.

        Args:
            value (Union[np.ndarray, torch.Tensor, list, int, float]): The points to convert.

        Returns:
            torch.Tensor: The converted points with double precision.
        """
        if not isinstance(value, (list, np.ndarray, torch.Tensor, int, float)):
            if logging_enabled():
                self._log_info("Points must be a list, scalar, NumPy array, or PyTorch tensor, got: %s", type(value))
            raise TypeError("Points must be a list, scalar, NumPy array, or PyTorch tensor")

        if isinstance(value, list):  # Convert lists to NumPy array first
            value = np.array(value, dtype=np.float64)
        if isinstance(value, np.ndarray):
            return torch.tensor(value, dtype=torch.float64)
        elif isinstance(value, torch.Tensor):
            return value.to(dtype=torch.float64)
        return torch.tensor(value, dtype=torch.float64)
    ################################################################################################
    @property
    def geometry(self) -> str:
        """Gets the geometry of the space."""
        return self._geometry   
    ################################################################################################
    @property
    def points(self) -> torch.Tensor:
        """
        Gets the points in the space.

        Returns:
            torch.Tensor: The points in the space.
        """
        return self._points
    ################################################################################################
    @points.setter
    def points(self, value: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Sets the points in the space and checks norm constraints.

        Args:
            value (Union[np.ndarray, torch.Tensor]): The new points to set.

        Raises:
            ValueError: If the norm constraints are violated by the new points.
        """
        if isinstance(value, np.ndarray):  # If NumPy array, convert it
            value = torch.tensor(value, dtype=torch.float64)
        elif isinstance(value, list):  # If list, convert to NumPy first, then Tensor
            value = torch.tensor(np.array(value, dtype=np.float64), dtype=torch.float64)
        elif isinstance(value, torch.Tensor):  # If Tensor, enforce dtype
            value = value.to(dtype=torch.float64)
        else:  # Scalars (int, float)
            value = torch.tensor(value, dtype=torch.float64)

        self._points = value  # Store as torch.Tensor
        self._update_dimensions()
        if self._geometry == 'hyperbolic':
            self._validate_norms() 
        self._log_info(f"Updated points with shape={self._points.shape}")
    ################################################################################################
    @property
    def labels(self) -> List[Union[str, int]]:
        """Gets the labels corresponding to the points."""
        return self._labels
    ################################################################################################
    @labels.setter
    def labels(self, value: List[Union[str, int]]) -> None:
        """
        Sets the labels corresponding to the points.

        Args:
            value (List[Union[str, int]]): The new labels to set.

        Raises:
            ValueError: If the number of labels does not match the number of points.
        """
        if len(value) != self._points.shape[1]:
            self._log_info(f"The number of labels must match the number of points, got: {len(value)} labels for {self._points.shape[1]} points")
            raise ValueError("The number of labels must match the number of points")
        self._labels = value
        self._log_info(f"Updated labels with length={len(self._labels)}")
    ################################################################################################
    def _update_dimensions(self) -> None:
        """Updates the dimension based on the points. Must be implemented by a subclass."""
        raise NotImplementedError("update_dimensions must be implemented by a subclass")
    ################################################################################################
    def _validate_norms(self) -> None:
        """Validates that all points are within the norm constraints for the embedding."""
        raise NotImplementedError("_validate_norms must be implemented by a subclass")
    ################################################################################################
    def distance_matrix(self) -> torch.Tensor:
        """
        Computes the distance matrix for points in the embedding space.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError("distance_matrix must be implemented by a subclass")
    ################################################################################################
    def save(self, filename: str) -> None:
        """
        Saves the Embedding instance to a file using pickle.

        Args:
            filename (str): The file to save the instance to.

        Raises:
            Exception: If there is an issue with saving the file.
        """
        try:
            with open(filename, 'wb') as file:
                pickle.dump(self, file)
                self._log_info(f"Saved Embedding to {filename}")
        except Exception as e:
            self._log_info("Failed to save Embedding: %s", e)
            raise
    ################################################################################################
    @staticmethod
    def load(filename: str) -> 'Embedding':
        """
        Loads an Embedding instance from a file using pickle.

        Args:
            filename (str): The file to load the instance from.

        Returns:
            Embedding: The loaded Embedding instance.

        Raises:
            Exception: If there is an issue with loading the file.
        """
        try:
            with open(filename, 'rb') as file:
                instance = pickle.load(file)
            if hasattr(instance, '_logger') and instance._logger:
                instance._log_info(f"Loaded Embedding from {filename}")
            return instance
        except Exception as e:
            self._log_info(f"Failed to load Embedding: {e}")
            raise
    ################################################################################################
    def copy(self) -> 'Embedding':
        """
        Create a deep copy of the Embedding object.
        """
        
        Embedding_copy = copy.deepcopy(self)
        self._log_info(f"Embedding copied successfully.")
        return Embedding_copy
    ################################################################################################
    def __repr__(self) -> str:
        """Returns a string representation of the Embedding."""
        return (f"Embedding(geometry={self._geometry}, points_shape={list(self._points.shape)})")
####################################################################################################
class HyperbolicEmbedding(Embedding):
    """
    A class representing a hyperbolic embedding.

    Attributes:
        curvature (torch.Tensor): The curvature of the hyperbolic space.
        model (str): The model used ('poincare' or 'loid').
        norm_constraint (tuple): Constraints for point norms in the hyperbolic space.
    """

    def __init__(
        self,
        curvature: Optional[float] = -1,
        model: Optional[str] = 'loid',
        points: Optional[Union[np.ndarray, torch.Tensor]] = None,
        labels: Optional[List[Union[str, int]]] = None
    ):
        """
        Initializes the HyperbolicEmbedding.

        Args:
            curvature (Optional[float]): The curvature of the space. Must be negative. Default is -1.
            model (Optional[str]): The model of the space ('poincare' or 'loid'). Default is 'poincare'.
            points (Optional[Union[np.ndarray, torch.Tensor]]): A NumPy array or PyTorch tensor of points. Default is None.
            labels (Optional[List[Union[str, int]]]): A list of labels corresponding to the points. Default is None.

        Raises:
            ValueError: If the curvature is not negative.
        """
        if curvature >= 0:
            raise ValueError("Curvature must be negative for hyperbolic space.")

        super().__init__(geometry='hyperbolic', points=points, labels=labels)
        self._curvature = self._convert_value(curvature)
        self.model = model
        self._log_info(f"Initialized HyperbolicEmbedding with curvature={self._curvature}")
    ################################################################################################
    @property
    def curvature(self) -> torch.Tensor:
        """Gets the curvature of the space."""
        return self._curvature
    ################################################################################################
    @curvature.setter
    def curvature(self, value: float) -> None:
        """Sets the curvature of the space.

        Args:
            value (float): The new curvature value.
        """
        self._curvature = self._convert_value(value) 
        self._log_info(f"Updated curvature to {self._curvature}")
    ################################################################################################
    def switch_model(self) -> 'HyperbolicEmbedding':
        """
        Switches between Poincare and Loid models.

        Returns:
            HyperbolicEmbedding: A new instance of HyperbolicEmbedding with the switched model.

        Raises:
            ValueError: If there are no points to switch model.
        """
        if not self._points.numel():
            raise ValueError("No points to switch model.")

        self._log_info(f"Switching model from {self.model}")

        if self.model == 'poincare':
            norm_points = torch.norm(self._points, dim=0)
            new_points = torch.cat((
                (1 + norm_points**2) / (1 - norm_points**2).unsqueeze(0),
                (2 * self._points) / (1 - norm_points**2)
            ), dim=0)
            new_space = LoidEmbedding(self._curvature, new_points, self.labels)
            self._log_info("Switched to LoidEmbedding model.")
        elif self.model == 'loid':
            new_points = self._points[1:] / (self._points[0] + 1)
            new_space = PoincareEmbedding(self._curvature, new_points, self.labels)
            self._log_info("Switched to PoincareEmbedding model.")
        else:
            if logging_enabled():
                self._log_info(f"Unknown model type: {self.model}")
            raise ValueError("Unknown model type.")
        return new_space
    ################################################################################################
    def poincare_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the Poincare distance between points x and y.

        Args:
            x (torch.Tensor): Point(s) in the Poincare ball.
            y (torch.Tensor): Point(s) in the Poincare ball.

        Returns:
            torch.Tensor: Poincare distance between x and y.
        """
        norm_x = torch.sum(x**2, dim=0, keepdim=True)
        norm_y = torch.sum(y**2, dim=0, keepdim=True)
        diff_norm = torch.sum((x - y)**2, dim=0, keepdim=True)
        denominator = (1 - norm_x) * (1 - norm_y)
        distance = torch.acosh(1 + 2 * diff_norm / denominator)
        return distance
    ################################################################################################
    def to_poincare(self, vectors: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Transforms vectors from Loid to Poincare model.

        Args:
            vectors (Union[np.ndarray, torch.Tensor]): Input vectors (columns are vectors, or a single vector).

        Returns:
            torch.Tensor: Transformed vectors in Poincare model.

        Raises:
            TypeError: If input vectors are not a NumPy array or a PyTorch tensor.
        """
        if isinstance(vectors, (np.ndarray, torch.Tensor)):
            vectors = torch.tensor(vectors, dtype=self._points.dtype) if isinstance(vectors, np.ndarray) else vectors.to(dtype=self._points.dtype, non_blocking=True)
        else:
            raise TypeError("Input vectors must be a NumPy array or a PyTorch tensor.")

        if vectors.dim() == 1:
            vectors = vectors.unsqueeze(1)
        new_points = vectors[1:, :] / (1 + vectors[0, :])
        return new_points
    ################################################################################################
    def to_loid(self, vectors: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Transforms vectors from Poincare to Loid model.

        Args:
            vectors (Union[np.ndarray, torch.Tensor]): Input vectors (columns are vectors, or a single vector).

        Returns:
            torch.Tensor: Transformed vectors in Loid model.

        Raises:
            TypeError: If input vectors are not a NumPy array or a PyTorch tensor.
        """
        if isinstance(vectors, (np.ndarray, torch.Tensor)):
            vectors = torch.tensor(vectors, dtype=self._points.dtype) if isinstance(vectors, np.ndarray) else vectors.to(dtype=self._points.dtype, non_blocking=True)
        else:
            raise TypeError("Input vectors must be a NumPy array or a PyTorch tensor.")

        if vectors.dim() == 1:
            vectors = vectors.unsqueeze(1)

        norm_points = torch.norm(vectors, dim=0)
        new_points = torch.zeros((vectors.shape[0] + 1, vectors.shape[1]))
        new_points[0] = (1 + norm_points**2) / (1 - norm_points**2)
        new_points[1:] = (2 * vectors) / (1 - norm_points**2)        
        return new_points
    ################################################################################################
    def matrix_sqrtm(self, A: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Computes the matrix square root of a positive definite matrix using eigenvalue decomposition.

        Args:
            A (Union[np.ndarray, torch.Tensor]): A symmetric positive definite matrix (NumPy array or PyTorch tensor).

        Returns:
            torch.Tensor: The matrix square root of A.
        
        Raises:
            TypeError: If the input is neither a NumPy array nor a PyTorch tensor.
        """
        # Convert NumPy array to PyTorch tensor, ensuring correct precision
        if isinstance(A, (np.ndarray, torch.Tensor)):
            A = torch.tensor(A, dtype=self._points.dtype) if isinstance(A, np.ndarray) else A.to(dtype=self._points.dtype, non_blocking=True)
        else:
            raise TypeError("Input must be a NumPy array or a PyTorch tensor.")

        eigvals, eigvecs = torch.linalg.eigh(A)
        eigvals = torch.clamp(eigvals, min=0)
        return eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T
    ################################################################################################
    def __repr__(self) -> str:
        """Returns a string representation of the HyperbolicEmbedding."""
        return (f"HyperbolicEmbedding(curvature={self._curvature.item():.2f}, model={self.model}, points_shape={list(self._points.shape)})")
####################################################################################################
class PoincareEmbedding(HyperbolicEmbedding):
    """
    A class representing a Poincare hyperbolic embedding space.

    Inherits from:
        HyperbolicEmbedding

    Attributes:
        norm_constraint (tuple): Constraints for point norms in the Poincare model.
    """

    def __init__(self, 
                 curvature: Optional[float] = -1, 
                 points: Optional[Union[np.ndarray, torch.Tensor]] = None, 
                 labels: Optional[List[Union[str, int]]] = None
                 ) -> None:
        """
        Initializes the PoincareEmbedding.

        Args:
            curvature (Optional[float]): The curvature of the space. Must be negative.
            points (Optional[Union[np.ndarray, torch.Tensor]]): A NumPy array or PyTorch tensor of points. Default is None.
            labels (Optional[List[Union[str, int]]]): A list of labels corresponding to the points. Default is None.
        """
        super().__init__(curvature=curvature, points=points, labels=labels)
        self.model = 'poincare'
        self.norm_constraint = conf.POINCARE_DOMAIN
        self._update_dimensions()
        self._validate_norms()
        self._log_info(f"Initialized PoincareEmbedding with curvature={self.curvature} and checked point norms")
    
    def _validate_norms(self) -> None:
        """Validates that all points are within the norm constraints for the Poincare model."""
        norm2 = self._norm2()
        min_norm, max_norm = self.norm_constraint
        if torch.any(norm2 < min_norm) or torch.any(norm2 >= max_norm):
            if logging_enabled():
                self._log_info(f"Points norm constraint violated: norms={norm2}, constraint=({min_norm}, {max_norm})")
            raise ValueError(f"Points norm constraint violated: norms must be in range ({min_norm}, {max_norm})")

    def _update_dimensions(self) -> None:
        """Updates the dimension and number of points based on the Poincare model."""
        self.dimension = self._points.size(0) if self._points.numel() > 0 else 0
        self.n_points = self._points.size(1) if self._points.numel() > 0 else 0
        self._log_info(f"Updated dimensions to {self.dimension}")
        self._log_info(f"Updated n_points to {self.n_points}")

    def _norm2(self) -> torch.Tensor:
        """Computes the squared L2 norm of the points."""
        norm2 = torch.norm(self._points, dim=0)**2
        self._log_info(f"Computed squared L2 norms: {norm2}")
        return norm2

    def distance_matrix(self) -> torch.Tensor:
        """
        Computes the distance matrix for points in the Poincare model.

        Returns:
            torch.Tensor: The distance matrix.
        """
        G = torch.matmul(self._points.T, self._points)
        diag_vec = torch.diag(G)
        
        EDM = -2*G + diag_vec.view(1, -1) + diag_vec.view(-1, 1)
        EDM = torch.relu(EDM)
        EDM = EDM / (1 - diag_vec.view(-1, 1))
        EDM = EDM / (1 - diag_vec.view(1, -1))
        distance_matrix = (1 / torch.sqrt(torch.abs(self.curvature))) * torch.arccosh(1 + 2 * EDM)        
        self._log_info(f"Computed distance matrix with shape distance_matrix.shape")

        return distance_matrix.fill_diagonal_(0), self.labels

    def centroid(self, 
                 mode: str = 'default', 
                 lr: float = conf.FRECHET_LEARNING_RATE, 
                 max_iter: int = conf.FRECHET_MAX_EPOCHS, 
                 tol: float = conf.FRECHET_ERROR_TOLERANCE) -> torch.Tensor:
        """
        Compute the centroid of the points in the Poincare space.
        
        Args:
            mode (str): The mode to compute the centroid. 
            lr (float): Learning rate for the optimizer. Default is conf.FRECHET_LEARNING_RATE.
            max_iter (int): Maximum number of iterations. Default is conf.FRECHET_MAX_EPOCHS.
            tol (float): Tolerance for stopping criterion. Default is conf.FRECHET_ERROR_TOLERANCE.
        
        Returns:
            torch.Tensor: The centroid of the points.
        """
        if mode == 'default':
            X = self.to_loid(self._points)
            centroid = X.mean(dim=1, keepdim=True)
            norm2 = -centroid[0]**2 + torch.sum(centroid[1:]**2, dim=0)
            
            centroid = 1 / torch.sqrt(-norm2) * centroid
            centroid[0] = torch.sqrt(1 + torch.sum(centroid[1:]**2))
            centroid = centroid[1:] / (centroid[0] + 1)

            # Ensure output matches the precision of input X
            return centroid.to(self._points.dtype)

        elif mode == 'Frechet':
            centroid = self.points.mean(dim=1, keepdim=True).clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([centroid], lr=lr)

            for _ in range(max_iter):
                optimizer.zero_grad()
                distances = self.poincare_distance(self._points, centroid)
                loss = torch.sum(distances**2)
                loss.backward(retain_graph=True)  # Retain graph for multiple backward passes if needed
                optimizer.step()

                if torch.norm(centroid).item() >= 1:
                    centroid.data = centroid.data / torch.norm(centroid).item() * (1-conf.ERROR_TOLERANCE)

                if torch.norm(centroid.grad) < tol:
                    break
            # Ensure output matches the precision of input points
            return centroid.detach().to(self._points.dtype)
        else:
            raise NotImplementedError(f"Mode '{mode}' is not implemented.")

    def translate(self, vector: Optional[Union[np.ndarray, torch.Tensor]]) -> None:
        """
        Translates the points by a given vector in the Poincare model.

        Args:
            vector (Optional[Union[np.ndarray, torch.Tensor]]): The translation vector.
        """
        vector = torch.as_tensor(vector, dtype=self._points.dtype)

        vector = vector.view(-1, 1)
        norm2 = torch.norm(vector, dim=0)**2
        min_norm, max_norm = self.norm_constraint
        if torch.any(norm2 < min_norm) or torch.any(norm2 >= max_norm):
            if logging_enabled():
                self._log_info("In Poincare model, the L2 norm of the points must be strictly less than 1. Invalid norms: %s", norm2)
            raise ValueError("In Poincare model, the L2 norm of the points must be strictly less than 1.")

        self._log_info(f"Translating points by vector with shape {vector.shape}")
        # Translate points, ensuring that the addition preserves the dtype of self.points
        self.points = self._add(vector, self._points)
        self._log_info(f"Points translated. New points shape: {self._points.shape}")

    def _add(self, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Adds a vector to the points in the Poincare model.

        Args:
            b (torch.Tensor): The translation vector.
            x (torch.Tensor): The points.

        Returns:
            torch.Tensor: The translated points.
        """
        b = b.view(-1, 1)
        
        # Determine the more precise dtype between b and x
        more_precise_dtype = torch.promote_types(b.dtype, x.dtype)

        # Convert both b and x to the more precise dtype
        b = b.to(more_precise_dtype)
        x = x.to(more_precise_dtype)

        if x.shape[0] != b.shape[0]:
            if logging_enabled():
                self._log_info("Dimension mismatch between points (%s) and vector (%s)", x.shape, b.shape)
            raise ValueError("Dimension mismatch between points and vector")  

        norm_x_sq = torch.sum(x ** 2, dim=0, keepdim=True)
        norm_b_sq = torch.sum(b**2, dim=0, keepdim=True)
        dot_product = 2 * torch.matmul(x.T, b).view(1, -1)

        denominator = (1 + dot_product + norm_x_sq * norm_b_sq).view(1, -1)
        numerator_x = x * (1 - norm_b_sq).view(1, -1)
        numerator_b = b * (1 + dot_product + norm_x_sq).view(1, -1)
        numerator = numerator_x + numerator_b

        result = numerator / denominator
        return result

    def rotate(self, R: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Rotates the points by a given matrix in the Poincare model.

        Args:
            R (Union[np.ndarray, torch.Tensor]): The rotation matrix.
        """
        if isinstance(R, np.ndarray):
            # Convert NumPy array to torch tensor
            R = torch.from_numpy(R).to(self._points.dtype)
        else:
            # If it's already a torch tensor, convert to the correct dtype
            R = R.to(self._points.dtype)
        
        I = torch.eye(R.shape[0], dtype=R.dtype, device=R.device)
        if not torch.allclose(R.T @ R, I, atol=conf.ERROR_TOLERANCE):
            self._log_info("The provided matrix is not a valid rotation matrix. Attempting to orthogonalize.")
            R = R @ torch.linalg.inv(self.matrix_sqrtm(R.T @ R))
        
        self._log_info(f"Rotating points with matrix shape {R.shape}")
        self.points = R @ self._points
        self._log_info(f"Points rotated. New points shape: {self._points.shape}")

    def center(self,mode = 'default') -> None:
        """Centers the points by translating them to the centroid."""
        centroid = self.centroid(mode = mode)
        self._log_info(f"Centroid computed: {centroid}")
        self.translate(-centroid)
        self._log_info(f"Points centered. New points shape: {self._points.shape}")
####################################################################################################
####################################################################################################
####################################################################################################    
class LoidEmbedding(HyperbolicEmbedding):
    """
    A class representing the Loid model in hyperbolic space.

    Inherits from:
        HyperbolicEmbedding

    Attributes:
        curvature (float): The curvature of the hyperbolic space.
        points (torch.Tensor): The points in the Loid space.
        labels (List[Union[str, int]]): Optional labels for the points.
    """

    def __init__(
        self,
        curvature: Optional[float] = -1,
        points: Optional[Union[np.ndarray, torch.Tensor]] = None,
        labels: Optional[List[Union[str, int]]] = None
    ) -> None:
        """
        Initializes the LoidEmbedding with the given parameters.

        Args:
            curvature (float, optional): The curvature of the space. Defaults to -1.
            points (Union[np.ndarray, torch.Tensor], optional): Initial points in the space. Defaults to None.
            labels (List[Union[str, int]], optional): Labels for the points. Defaults to None.
        """
        super().__init__(curvature=curvature, points=points, labels=labels)
        self.model = 'loid'
        self.norm_constraint = conf.LOID_DOMAIN
        self._update_dimensions()
        self._validate_norms()
        self._log_info(f"Initialized LoidEmbedding with curvature={self.curvature} and checked point norms")

    def _validate_norms(self) -> None:
        """
        Validates that all points are within the norm constraints for the Loid model.

        Raises:
            ValueError: If any point's norm is outside the specified constraint range.
        """
        norm2 = self._norm2()
        min_norm, max_norm = self.norm_constraint
        if torch.any(norm2 <= min_norm) or torch.any(norm2 > max_norm):
            if logging_enabled():
                self._log_info(f"Points norm constraint violated: norms={norm2}, constraint=({min_norm}, {max_norm})")
            points = self._points
            for n in range(self.n_points):
                points[0,n] = torch.sqrt(1+torch.sum(points[1:,n]**2))
            self._points = points

    def _update_dimensions(self) -> None:
        """
        Updates the dimensions of the space based on the current points.
        """
        self.dimension = self._points.size(0) - 1 if self._points.numel() > 0 else 0
        self.n_points = self._points.size(1) if self._points.numel() > 0 else 0
        self._log_info(f"Updated dimensions to {self.dimension}")
        self._log_info(f"Updated n_points to {self.n_points}")

    def _norm2(self) -> torch.Tensor:
        """
        Computes the Lorentzian norm squared of the points.

        Returns:
            torch.Tensor: The squared norms of the points.
        """
        if len(self._points) != 0:
            norm2 = -(self._points[0,:])**2 + torch.sum(self._points[1:,:]**2, dim=0)
            self._log_info(f"Computed Lorentzian norms: {norm2}")
            return norm2
        else: 
            return torch.tensor([])
    
    def distance_matrix(self) -> torch.Tensor:
        """
        Computes the distance matrix for points in the Loid model.

        Returns:
            torch.Tensor: The distance matrix of shape (n_points, n_points).
        """
        J = torch.eye(self.dimension+1, dtype=self._points.dtype)
        J[0,0] = -1

        G = torch.matmul(torch.matmul((self._points).T,J), self._points)
        G = torch.where(G > -1, torch.tensor(-1.0, dtype=G.dtype, device=G.device), G)
        distance_matrix = (1/torch.sqrt(torch.abs(self.curvature))) * torch.arccosh(-G)        
        self._log_info(f"Computed distance matrix with shape {distance_matrix.shape}")

        return distance_matrix.fill_diagonal_(0), self.labels

    def rotate(self, R: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Rotates the points by a given matrix in the Loid model.

        Args:
            R (Union[np.ndarray, torch.Tensor]): The rotation matrix.

        Raises:
            ValueError: If R is not a valid rotation matrix.
        """
        if isinstance(R, np.ndarray):
            # Convert NumPy array to torch tensor
            R = torch.from_numpy(R).to(self._points.dtype)
        else:
            # If it's already a torch tensor, convert to the correct dtype
            R = R.to(self._points.dtype)

        # Check if R is a rotation matrix (orthogonal for real matrices)
        I = torch.eye(R.shape[0], dtype=R.dtype, device=R.device)
        cond1 = not torch.allclose(R.T @ R, I, atol=conf.ERROR_TOLERANCE)
        cond2 = not torch.isclose(R[0, 0], torch.tensor(1.0, dtype=R.dtype), atol=conf.ERROR_TOLERANCE)
        cond3 = not torch.allclose(R[0, 1:], torch.zeros_like(R[0, 1:]), atol=conf.ERROR_TOLERANCE)
        cond4 = not torch.allclose(R[1:, 0], torch.zeros_like(R[1:, 0]), atol=conf.ERROR_TOLERANCE)
        if cond1 or cond2 or cond3 or cond4:
            self._log_info("The provided matrix is not a valid rotation matrix.")
            raise ValueError("The provided matrix is not a valid rotation matrix.")

        self._log_info(f"Rotating points with matrix of shape {R.shape}")
        self.points = R @ self._points
        self._log_info(f"Points rotated. New points shape: {self._points.shape}")

    def translate(self, vector: Optional[Union[np.ndarray, torch.Tensor]]) -> None:
        """
        Translates the points by a given vector in the Loid model.

        Args:
            vector (Optional[Union[np.ndarray, torch.Tensor]]): The translation vector.

        Raises:
            ValueError: If the J-norm of the vector is not exactly -1.
        """
        # Convert vector to a torch tensor and ensure it matches the precision of self.points
        vector = torch.as_tensor(vector, dtype=self._points.dtype)
        
        vector = vector.view(-1,1)
        norm2 = -vector[0]**2 + torch.sum(vector[1:]**2)
        min_norm, max_norm = self.norm_constraint
        if torch.any(norm2 < min_norm) or torch.any(norm2 >= max_norm):
            if logging_enabled():
                self._log_info("In Loid model, the J-norm of the points must be exactly equal to -1. Invalid norms: %s", norm2)
            raise ValueError("In Loid model, the J-norm of the points must be exactly equal to -1.")

        
        self._log_info(f"Translating points by vector with shape {vector.shape}")
        self._points = self._add(vector,self._points)
        self._log_info(f"Points translated. New points shape: {self._points.shape}")

    def _add(self, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Adds a vector to the points in the Loid model.

        Args:
            b (torch.Tensor): The vector to add.
            x (torch.Tensor): The current points.

        Returns:
            torch.Tensor: The updated points after addition.

        Raises:
            ValueError: If the hyperbolic norm of the vector is not -1 or if dimensions mismatch.
        """
        # Ensure b is a column vector
        b = b.view(-1, 1)

        # Determine the more precise dtype between b and x
        more_precise_dtype = torch.promote_types(b.dtype, x.dtype)

        # Convert both b and x to the more precise dtype
        b = b.to(more_precise_dtype)
        x = x.to(more_precise_dtype)

        # Calculate the hyperbolic norm
        norm2 = -b[0]**2 + torch.sum(b[1:]**2)
        min_norm, max_norm = self.norm_constraint
        if torch.any(norm2 < min_norm) or torch.any(norm2 >= max_norm):
            if logging_enabled():
                self._log_info("In Loid model, the J-norm of the points must be exactly equal to -1. Invalid norms: %s", norm2)
            raise ValueError("In Loid model, the J-norm of the points must be exactly equal to -1.")

        if x.shape[0] != b.shape[0]:
            if logging_enabled():
                self._log_info("Dimension mismatch between points (%s) and vector (%s)", x.shape, b.shape)
            raise ValueError("Dimension mismatch between points and vector")

        b_ = b[1:]
        norm_b = torch.norm(b_)
        I = torch.eye(self.dimension, device=b.device, dtype=b.dtype)
        
        Rb = torch.zeros((self.dimension + 1, self.dimension + 1), device=b.device, dtype=b.dtype)
        Rb[0, 0] = torch.sqrt(1 + norm_b**2)
        Rb[0, 1:] = b_.view(1, -1)
        Rb[1:, 0] = b_.view(-1, 1).squeeze()
        Rb[1:, 1:] = self.matrix_sqrtm(I + torch.outer(b_.view(-1), b_.view(-1)))

        return Rb @ x

    def centroid(self, 
                 mode: str = 'default', 
                 lr: float = conf.FRECHET_LEARNING_RATE, 
                 max_iter: int = conf.FRECHET_MAX_EPOCHS, 
                 tol: float = conf.FRECHET_ERROR_TOLERANCE) -> torch.Tensor:
        """
        Compute the centroid of the points in the Loid space.
        
        Args:
            mode (str): The mode to compute the centroid. 
            lr (float): Learning rate for the optimizer. Default is conf.FRECHET_LEARNING_RATE.
            max_iter (int): Maximum number of iterations. Default is conf.FRECHET_MAX_EPOCHS.
            tol (float): Tolerance for stopping criterion. Default is conf.FRECHET_ERROR_TOLERANCE.
        
        Returns:
            torch.Tensor: The centroid of the points.
        """
        if mode == 'default':
            X = self._points
            centroid = X.mean(dim=1, keepdim=True)
            norm2 = -centroid[0]**2 + torch.sum(centroid[1:]**2, dim=0)
            
            centroid = 1/torch.sqrt(-norm2)*centroid
            centroid[0]= torch.sqrt(1+torch.sum(centroid[1:]**2))
            return centroid.to(self._points.dtype)

        elif mode == 'Frechet':
            X = self.to_poincare(self._points)
            centroid = X.mean(dim=1, keepdim=True).clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([centroid], lr=lr)

            for _ in range(max_iter):
                optimizer.zero_grad()
                distances = self.poincare_distance(X, centroid)
                loss = torch.sum(distances**2)
                loss.backward(retain_graph=True)  # Retain graph for multiple backward passes if needed
                optimizer.step()

                if torch.norm(centroid).item() >= 1:
                    centroid.data = centroid.data / torch.norm(centroid).item() * (1-conf.ERROR_TOLERANCE)

                if torch.norm(centroid.grad) < tol:
                    break
            return self.to_loid(centroid).detach().to(self._points.dtype)
        else:
            raise NotImplementedError(f"Mode '{mode}' is not implemented.")

    def center(self,mode = 'default') -> None:
        """Centers the points by translating them to the centroid."""
        centroid = self.centroid(mode = mode)
        self._log_info(f"Centroid computed: {centroid}")
        
        _centroid = -centroid
        _centroid[0] = centroid[0]
        self.translate(_centroid)
        self._log_info(f"Points centered. New points shape: {self._points.shape}")
#############################################################################################
class EuclideanEmbedding(Embedding):
    """
    A class representing an embedding in Euclidean space.

    Attributes:
        curvature (float): The curvature of the Euclidean space (always 0).
    """

    def __init__(self, 
                 points: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 labels: Optional[List[Union[str, int]]] = None):
        """
        Initializes the EuclideanEmbedding.

        Args:
            points (Optional[Union[np.ndarray, torch.Tensor]]): A NumPy array or PyTorch tensor of points. Default is None.
        """
        super().__init__(geometry='euclidean', points=points, labels=labels)
        self.curvature = torch.tensor(0)
        self._update_dimensions()
        self.model = 'descartes'
        self._log_info(f"Initialized EuclideanEmbedding ")

    def __repr__(self) -> str:
        """Returns a string representation of the EuclideanEmbedding."""
        return (f"EuclideanEmbedding(points_shape={list(self._points.shape)})")
        
    def _update_dimensions(self) -> None:
            """Updates the dimension and number of points in Euclidean space."""
            self.dimension = self._points.size(0) if self._points.numel() > 0 else 0
            self.n_points = self._points.size(1) if self._points.numel() > 0 else 0
            self._log_info(f"Updated dimensions to {self.dimension}")
            self._log_info(f"Updated n_points to {self.n_points}")
        

    def translate(self, vector: Optional[Union[np.ndarray, torch.Tensor]] = None) -> None:
        """
        Translates the points by a given vector in Euclidean space.

        Args:
            vector (Optional[Union[np.ndarray, torch.Tensor]]): The translation vector.
        
        Raises:
            ValueError: If the dimension of the translation vector is incorrect.
        """
        if vector is None:
            raise ValueError("Translation vector cannot be None.")

        vector = torch.as_tensor(vector, dtype=self._points.dtype)
        
        if vector.shape[0] != self.dimension:
            self._log_info(f"Invalid translation vector dimension. Expected {self.dimension}, got {vector.shape[0]}")
            raise ValueError("Dimension of the translation vector is incorrect.")
        
        self.points += vector.view(self.dimension, 1)
        self._log_info(f"Translated points by vector with shape {vector.shape}")

    def rotate(self, R: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Rotates the points by a given matrix in Euclidean space.

        Args:
            R (Union[np.ndarray, torch.Tensor]): The rotation matrix.
        """
        # Convert R to a PyTorch tensor if it's a numpy array
        if isinstance(R, np.ndarray):
            R = torch.tensor(R, dtype=self.points.dtype, device=self.points.device)
        else:
            R = R.to(self.points.dtype)

        I = torch.eye(R.shape[0], dtype=R.dtype, device=R.device)
        
        if not torch.allclose(R.T @ R, I, atol=1e-6):
            self._log_info("Provided matrix is not a valid rotation matrix. Attempting to orthogonalize.")
            R = R @ torch.linalg.inv(torch.sqrtm(R.T @ R))
        
        self.points = R @ self.points
        self._log_info(f"Rotated points with matrix of shape {R.shape}")

    def center(self) -> None:
        """
        Centers the points by subtracting the centroid from each point.
        """
        centroid = self.centroid()
        self.points -= centroid.view(-1, 1)
        self._log_info("Centered points by subtracting the centroid.")
        
    def centroid(self) -> torch.Tensor:
        """
        Computes the centroid of the points.

        Returns:
            torch.Tensor: The centroid of the points.
        """
        centroid = torch.mean(self._points, dim=1)
        self._log_info(f"Computed centroid with shape {centroid.shape}")
        return centroid.to(self._points.dtype)
        

    def distance_matrix(self) -> torch.Tensor:
        """
        Computes the distance matrix for points in the Euclidean geometry.

        Returns:
            torch.Tensor: The distance matrix.
        """
        G = torch.matmul(self._points.T, self._points)
        diag_vec = torch.diag(G)
        
        EDM = -2 * G + diag_vec.view(1, -1) + diag_vec.view(-1, 1)
        EDM = torch.relu(EDM)
        self._log_info(f"Computed distance matrix with shape distance_matrix.shape")

        return torch.sqrt(EDM).fill_diagonal_(0), self.labels
#############################################################################################
#############################################################################################
#############################################################################################
class MultiEmbedding:
    """
    A class representing a collection of embeddings with functionality 
    for managing and aligning multiple embeddings.

    Attributes:
        embeddings (List['Embedding']): A list of embedding instances.
        _logger (logging.Logger): A logger for the class if logging is enabled.
    """

    def __init__(self):
        """
        Initializes the MultiEmbedding with an empty list of embeddings.
        """
        self.embeddings = []
        self._current_time = get_time() or datetime.now()  # Uses a global timestamp function
        self._log_info("Initialized MultiEmbedding with an empty list of embeddings.")
        self.curvature = None
        self.dimension = None

    def _log_info(self, message: str) -> None:
        """
        Logs an informational message.

        Args:
            message (str): The message to log.
        """
        if logging_enabled():  # Check if logging is globally enabled
            get_logger().info(message)

    def append(self, embedding: 'Embedding') -> None:
        """
        Adds an embedding to the collection only if its curvature and dimension match.

        Args:
            embedding (Embedding): The embedding instance to be added.
        """
        if self.curvature is None and self.dimension is None:
            # First embedding sets the curvature and dimension
            self.curvature = embedding.curvature
            self.dimension = embedding.dimension
        elif embedding.curvature != self.curvature or embedding.dimension != self.dimension:
            print(embedding.dimension, embedding._points.shape, self.dimension)
            print("Embedding not added due to curvature or dimension mismatch.")
            self._log_info("Embedding not added due to curvature or dimension mismatch.")
            return  # Do not append if they do not match
        
        self.embeddings.append(embedding)
        self._log_info("Added embedding.")
    ################################################################################################
    def labels(self) -> List[str]:
        all_labels = sorted({label for embedding in self.embeddings for label in embedding.labels})
        self._log_info(f"Retrieved {len(all_labels)} labels")
        return all_labels
    ################################################################################################
    def save(self, filename: str) -> None:
        """
        Saves the MultiEmbedding instance to a file using pickle.

        Args:
            filename (str): The file to save the instance to.

        Raises:
            Exception: If there is an issue with saving the file.
        """
        try:
            with open(filename, 'wb') as file:
                pickle.dump(self, file)
                self._log_info(f"Saved MultiEmbedding to {filename}")
        except Exception as e:
            self._log_info("Failed to save MultiEmbedding: %s", e)
            raise
    
    @staticmethod
    def load(filename: str) -> 'MultiEmbedding':
        """
        Loads a MultiEmbedding instance from a file using pickle.

        Args:
            filename (str): The file to load the instance from.

        Returns:
            MultiEmbedding: The loaded MultiEmbedding instance.

        Raises:
            Exception: If there is an issue with loading the file.
        """
        try:
            with open(filename, 'rb') as file:
                instance = pickle.load(file)
            if hasattr(instance, '_log_info') and instance._log_info:
                instance._log_info(f"Loaded MultiEmbedding from {filename}")
            return instance
        except Exception as e:
            self._log_info("Failed to load MultiEmbedding: %s", e)
            raise

    def copy(self) -> 'MultiEmbedding':
        """
        Create a deep copy of the MultiEmbedding object.
        """
        MultiEmbedding_copy = copy.deepcopy(self)
        self._log_info("MultiEmbedding copied successfully.")
        return MultiEmbedding_copy

    def __repr__(self) -> str:
        """
        Return a string representation of the MultiEmbedding object.
        """
        repr_str = f"MultiEmbedding({len(self.embeddings)} embeddings)"
        return repr_str

    def __iter__(self):
        """Allows iteration over embeddings."""
        return iter(self.embeddings)

    def __len__(self) -> int:
        """
        Return the number of embeddings.

        Returns:
        int: The number of Embedding objects in the MultiEmbedding.
        """
        length = len(self.embeddings)
        self._log_info(f"Number of embeddings: {length}")
        return length

    def __getitem__(self, index):
        """
        Allows indexing and slicing of embeddings.
        """
        if isinstance(index, slice):
            new_multiembedding = MultiEmbedding()
            new_multiembedding.embeddings = self.embeddings[index]
            return new_multiembedding
        return self.embeddings[index]


    def align(self, **kwargs) -> None:
        """
        Aligns all embeddings by averaging their distance matrices and adjusting
        each embedding to match the reference embedding.
        """

        if not self.embeddings:
            self._log_info("No embeddings to align.")
            return

        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'p'}
        reference_embedding = self.reference_embedding(**filtered_kwargs)

        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'func'}
        if self.curvature < 0:
            for i, embedding in enumerate(self.embeddings):
                model = procrustes.HyperbolicProcrustes(embedding, reference_embedding,**filtered_kwargs)
                self.embeddings[i] = model.map(embedding)
        else:
            for i, embedding in enumerate(self.embeddings):
                model = procrustes.EuclideanProcrustes(embedding, reference_embedding,**filtered_kwargs)
                self.embeddings[i] = model.map(embedding)

    def distance_matrix(self, 
                        func: Callable[[torch.Tensor], torch.Tensor] = torch.nanmean) -> torch.Tensor:
        """
        Computes the aggregated distance matrix from all embeddings, accommodating for different-sized matrices.

        Parameters:
            func (Callable[[torch.Tensor], torch.Tensor]): Function to compute the aggregate. Default is torch.nanmean.

        Returns:
            torch.Tensor: The aggregated distance matrix.
        """
        # Get all unique labels across embeddings

        all_labels = sorted({label for embedding in self.embeddings for label in embedding.labels})
        n = len(all_labels)
        
        data_type = None
        for embedding in self.embeddings:
            if data_type is None:
                data_type = embedding._points.dtype
                break

        # Prepare stacked matrices and counts
        stacked_matrices = torch.zeros((len(self.embeddings), n, n), dtype=data_type)
        stacked_counts = torch.zeros((len(self.embeddings), n, n), dtype=data_type)
        
        cnt = 0
        for embedding in self.embeddings:
            idx = torch.tensor(  [all_labels.index(label) for label in embedding.labels] )
            distance_matrix = torch.full((n, n), float('nan'),  dtype=data_type)
            distance_matrix[idx[:, None], idx] = embedding.distance_matrix()[0]
            stacked_matrices[cnt] = distance_matrix
            stacked_counts[cnt] = ~torch.isnan(distance_matrix)
            cnt = cnt + 1
        
        agg_distance_matrix = func(stacked_matrices, dim=0)
        if isinstance(agg_distance_matrix, tuple):
            agg_distance_matrix = agg_distance_matrix[0]

        # Identify indices where NaN values exist
        nan_indices = torch.nonzero(torch.isnan(agg_distance_matrix), as_tuple=False)
        # Iterate over all NaN indices and compute the replacement values
        agg_distance_matrix_ = agg_distance_matrix.clone()
        for idx in nan_indices:
            i, j = idx[0], idx[1]
            # Extract non-NaN values in the i-th row and j-th column
            row_non_nan = agg_distance_matrix_[i, ~torch.isnan(agg_distance_matrix_[i, :])]
            col_non_nan = agg_distance_matrix_[~torch.isnan(agg_distance_matrix_[:, j]), j]

            if row_non_nan.numel() == 0 and col_non_nan.numel() == 0:
                continue  # No non-NaN values available to estimate
            combined_non_nan = torch.cat((row_non_nan, col_non_nan))
            # Replace the NaN value with the estimated value
            agg_distance_matrix[i, j] = func(combined_non_nan)

        self._log_info(f"Computed distance matrix with NaN replacements.")
        return agg_distance_matrix, all_labels
    
    ################################################################################################
    def reference_embedding(self, **kwargs) -> 'Embedding':
        params = {  
            key: kwargs.get(key, default) for key, default in {
                'precise_opt': conf.ENABLE_ACCURATE_OPTIMIZATION,
                'epochs': conf.TOTAL_EPOCHS,
                'lr_init': conf.INITIAL_LEARNING_RATE,
                'dist_cutoff': conf.MAX_RANGE,
                'save_mode': conf.ENABLE_SAVE_MODE,
                'scale_fn': None,
                'lr_fn': None,
                'weight_exp_fn': None,
                'func' : torch.nanmean
            }.items()
        }

        if self.curvature < 0 :
            geometry = 'hyperbolic'
        else:
            geometry = 'euclidean'

        try:
            embedding = (self._embed_hyperbolic if geometry == 'hyperbolic' else self._embed_euclidean)(self.dimension, **params)
        except Exception as e:
            self._log_info(f"Error during embedding: {e}")
            raise

        directory = f"{conf.OUTPUT_DIRECTORY}/{self._current_time.strftime('%Y-%m-%d_%H-%M-%S')}"
        filepath = f"{directory}/{geometry}_embedding_{self.dimension}d.pkl"
        os.makedirs(directory, exist_ok=True)
        try:
            with open(filepath, 'wb') as file:
                pickle.dump(embedding, file)
            self._log_info(f"Object successfully saved to {filepath}")
        except (IOError, pickle.PicklingError, Exception) as e:
            self._log_info(f"Error while saving object: {e}")
            raise

        return embedding
    ################################################################################################
    def _embed_euclidean(self, dim: int, **params) -> 'Embedding':
        """Handle naive and precise Euclidean embeddings."""
        dist_mat = self.distance_matrix(func = params['func'])[0]
        # Naive Euclidean embedding
        self._log_info("Initiating naive Euclidean embedding.")
        points = self._naive_euclidean_embedding(dist_mat, dim)
        embeddings = EuclideanEmbedding(points=points, labels=self.labels())
        self._log_info("Naive Euclidean embedding completed.")
        if params['precise_opt']:
            self._log_info("Initiating precise Euclidean embedding.")
            embeddings.points = self._precise_euclidean_embedding(dist_mat, dim, points, **params)
            self._log_info("Precise Euclidean embedding completed.")
        return embeddings
    ################################################################################################
    def _naive_euclidean_embedding(self, dist_mat, dim):
        n = dist_mat.shape[0]
        J = torch.eye(n) - 1/n
        J = J.to(torch.float64)
        G = -0.5 * J @ dist_mat @ J
        vals, vecs = (torch.linalg.eigh(G) if dim >= n else 
                      map(torch.tensor, spla.eigsh(G.cpu().numpy(), k=dim, which='LM')))
        X = vecs[:, vals.argsort(descending=True)] * vals.clamp(min=0).sqrt()
        return X.t() if dim <= n else torch.cat([X.t(), torch.zeros(dim - n, n)], dim=0)
    ################################################################################################
    def _precise_euclidean_embedding(self, dist_mat, dim, points, **params):
        return utils.euclidean_embedding(dist_mat, dim, init_pts=points, log_fn=self._log_info, time_stamp=self._current_time, **params)
    ################################################################################################
    def _embed_hyperbolic(self, dim: int, **params) -> 'Embedding':
        """Handle naive and precise hyperbolic embeddings."""
        scale_factor = torch.sqrt(torch.abs(self.curvature))
        dist_mat = self.distance_matrix(func = params['func'])[0] * scale_factor
        # Naive hyperbolic embedding
        self._log_info("Initiating naive hyperbolic embedding.")
        points = self._naive_hyperbolic_embedding(dist_mat, dim)
        embeddings = LoidEmbedding(points=points, labels=self.labels(), curvature=-(scale_factor ** 2))
        self._log_info("Naive hyperbolic embedding completed.")
        if params['precise_opt']:
            self._log_info("Initiating precise hyperbolic embedding.")
            points, scale = self._precise_hyperbolic_embedding(dist_mat, dim, points, **params)
            embeddings.points = points
            embeddings.curvature *= scale**2
        return embeddings
    ################################################################################################
    def _naive_hyperbolic_embedding(self, dist_mat, dim):
        gramian = -torch.cosh(dist_mat)
        points = utils.lgram_to_pts(gramian, dim)
        for n in range(points.size(1)):
            points[:, n] = utils.hyperbolic_proj(points[:, n])
            points[0, n] = torch.sqrt(1 + torch.sum(points[1:, n] ** 2))
        return points
    ################################################################################################
    def _precise_hyperbolic_embedding(self, dist_mat, dim, points, **params):
        scale_fn = lambda x1, x2, x3=None: False 
        pts, scale = utils.hyperbolic_embedding(
            dist_mat, dim, init_pts=points, epochs=params['epochs'],
            log_fn=self._log_info, lr_fn=params['lr_fn'], scale_fn=scale_fn, 
            weight_exp_fn=params['weight_exp_fn'], lr_init=params['lr_init'], 
            save_mode=params['save_mode'], time_stamp=self._current_time
        )
        self._log_info("Precise hyperbolic embedding completed.")
        return pts, scale