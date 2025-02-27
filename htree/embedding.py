import numpy as np
import torch
import pickle
import logging
import os
import copy
from typing import Optional, Union, List,Callable
from datetime import datetime
from . import conf, utils, embedding, procrustes

class Embedding:
    """
    A class representing an abstract embedding.

    Attributes:
        geometry (str): The geometry of the space (e.g., 'euclidean', 'hyperbolic').
        points (torch.Tensor): A PyTorch tensor representing the points in the space.
        labels (list): A list of labels corresponding to the points in the space.
        _logger (logging.Logger): A logger for the class if logging is enabled.
    """
    
    def __init__(self, 
                 geometry: Optional[str] = 'hyperbolic', 
                 points: Optional[Union[np.ndarray, torch.Tensor]] = None, 
                 labels: Optional[List[Union[str, int]]] = None,
                 enable_logging: bool = False):
        """
        Initializes the Embedding.

        Args:
            geometry (str): The geometry of the space. Default is 'hyperbolic'.
            points (Optional[Union[np.ndarray, torch.Tensor]]): A NumPy array or PyTorch tensor of points. Default is None.
            labels (Optional[List[Union[str, int]]]): A list of labels corresponding to the points. Default is None.
            enable_logging (bool): If True, logging is enabled. Default is False.
        """
        self._logger = None
        self._current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if enable_logging:
            self._setup_logging()

        if geometry not in {'euclidean', 'hyperbolic'}:
            if self._logger:
                self._logger.error("Invalid geometry type: %s", geometry)
            raise ValueError("Invalid geometry type. Choose either 'euclidean' or 'hyperbolic'.")
        
        self._geometry = geometry
        self._points = self._convert_value(points) if points is not None else torch.empty((0, 0))
        self._labels = labels if labels is not None else list(range(self._points.shape[1]))
        self._log_info(f"Initialized Embedding with geometry={self._geometry}")

    def _setup_logging(self, log_dir: str = conf.LOG_DIRECTORY, log_level: int = logging.INFO, log_format: str = '%(asctime)s - %(levelname)s - %(message)s') -> None:
        """
        Set up logging configuration.

        Args:
            log_dir (str): Directory where log files will be saved. Default is conf.LOG_DIRECTORY.
            log_level (int): Logging level. Default is logging.INFO.
            log_format (str): Format for logging messages. Default is '%(asctime)s - %(levelname)s - %(message)s'.
        """
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"Embedding_{self._current_time}.log")
        logging.basicConfig(filename=log_file, level=log_level, format=log_format)
        self._logger = logging.getLogger(__name__)
        self._log_info("Logging setup complete.")

    def _convert_value(self, value: Union[np.ndarray, torch.Tensor, list, int, float]) -> torch.Tensor:
        """
        Converts the points to a PyTorch tensor with double precision.

        Args:
            value (Union[np.ndarray, torch.Tensor, list, int, float]): The points to convert.

        Returns:
            torch.Tensor: The converted points with double precision.
        """
        if isinstance(value, list):
            # Convert list to a torch.Tensor with double precision
            value = torch.tensor(value, dtype=torch.float64)
        elif isinstance(value, np.ndarray):
            # Convert NumPy array to a torch.Tensor with double precision
            value = torch.tensor(value, dtype=torch.float64)
        elif isinstance(value, torch.Tensor):
            # Ensure PyTorch tensor is in double precision
            value = value.to(dtype=torch.float64, non_blocking=True)
        elif isinstance(value, (int, float)):
            # Convert single scalar to a torch.Tensor with double precision
            value = torch.tensor(value, dtype=torch.float64)
        else:
            if self._logger:
                self._logger.error("Points must be a list, scalar, NumPy array, or PyTorch tensor, got: %s", type(value))
            raise TypeError("Points must be a list, scalar, NumPy array, or PyTorch tensor")

        return value
            

    def _log_info(self, message: str) -> None:
        """
        Log an informational message.

        Args:
            message (str): The message to log.
        """
        if self._logger:
            self._logger.info(message)

    @property
    def geometry(self) -> str:
        """Gets the geometry of the space."""
        return self._geometry   
    
    @property
    def points(self) -> torch.Tensor:
        """
        Gets the points in the space.

        Returns:
            torch.Tensor: The points in the space.
        """
        return self._points

    @points.setter
    def points(self, value: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Sets the points in the space and checks norm constraints.

        Args:
            value (Union[np.ndarray, torch.Tensor]): The new points to set.

        Raises:
            ValueError: If the norm constraints are violated by the new points.
        """
        self._points = self._convert_value(value)
        self._update_dimensions()
        if self.geometry == 'hyperbolic':
            self._validate_norms() 
        self._log_info(f"Updated points with shape={self._points.shape}")

    @property
    def labels(self) -> List[Union[str, int]]:
        """Gets the labels corresponding to the points."""
        return self._labels
    
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
            if self._logger:
                self._logger.error("The number of labels must match the number of points, got: %d labels for %d points", len(value), self._points.shape[1])
            raise ValueError("The number of labels must match the number of points")
        self._labels = value
        self._log_info(f"Updated labels with length={len(self._labels)}")

    def _update_dimensions(self) -> None:
        """Updates the dimension based on the points. Must be implemented by a subclass."""
        raise NotImplementedError("update_dimensions must be implemented by a subclass")

    def _validate_norms(self) -> None:
        """Validates that all points are within the norm constraints for the embedding."""
        raise NotImplementedError("_validate_norms must be implemented by a subclass")

    def distance_matrix(self) -> torch.Tensor:
        """
        Computes the distance matrix for points in the embedding space.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError("distance_matrix must be implemented by a subclass")

    
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
            if self._logger:
                self._logger.error("Failed to save Embedding: %s", e)
            raise
    
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
            logging.error("Failed to load Embedding: %s", e)
            raise

    def copy(self) -> 'Embedding':
        """
        Create a deep copy of the Embedding object.
        """
        
        Embedding_copy = copy.deepcopy(self)
        self._log_info(f"Embedding copied successfully.")
        return Embedding_copy

    def __repr__(self) -> str:
        """Returns a string representation of the Embedding."""
        return (f"Embedding(geometry={self._geometry}, points_shape={list(self._points.shape)})")
####################################################################################################
####################################################################################################
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
        model: Optional[str] = 'poincare',
        points: Optional[Union[np.ndarray, torch.Tensor]] = None,
        labels: Optional[List[Union[str, int]]] = None,
        enable_logging: bool = False
    ):
        """
        Initializes the HyperbolicEmbedding.

        Args:
            curvature (Optional[float]): The curvature of the space. Must be negative. Default is -1.
            model (Optional[str]): The model of the space ('poincare' or 'loid'). Default is 'poincare'.
            points (Optional[Union[np.ndarray, torch.Tensor]]): A NumPy array or PyTorch tensor of points. Default is None.
            labels (Optional[List[Union[str, int]]]): A list of labels corresponding to the points. Default is None.
            enable_logging (bool): If True, logging is enabled. Default is False.

        Raises:
            ValueError: If the curvature is not negative.
        """
        if curvature >= 0:
            raise ValueError("Curvature must be negative for hyperbolic space.")

        super().__init__(geometry='hyperbolic', points=points, labels=labels, enable_logging=enable_logging)
        self._curvature = self._convert_value(curvature)
        self.model = model
        self._log_info(f"Initialized HyperbolicEmbedding with curvature={self._curvature} and model={self.model}")

    @property
    def curvature(self) -> torch.Tensor:
        """Gets the curvature of the space."""
        return self._curvature

    @curvature.setter
    def curvature(self, value: float) -> None:
        """Sets the curvature of the space.

        Args:
            value (float): The new curvature value.
        """
        self._curvature = self._convert_value(value) 
        self._log_info(f"Updated curvature to {self._curvature}")

    def switch_model(self) -> 'HyperbolicEmbedding':
        """
        Switches between Poincare and Loid models.

        Returns:
            HyperbolicEmbedding: A new instance of HyperbolicEmbedding with the switched model.

        Raises:
            ValueError: If there are no points to switch model.
        """
        if self._points.numel() == 0:
            raise ValueError("No points to switch model.")

        self._log_info(f"Switching model from {self.model}")

        if self.model == 'poincare':
            norm_points = torch.norm(self._points, dim=0)
            new_points = torch.zeros((self._points.shape[0] + 1, self._points.shape[1]))
            new_points[0] = (1 + norm_points**2) / (1 - norm_points**2)
            new_points[1:] = (2 * self._points) / (1 - norm_points**2)
            new_space = LoidEmbedding(curvature=self._curvature, points=new_points, labels = self.labels, enable_logging=self._logger is not None)
            self._log_info("Switched to LoidEmbedding model.")
            return new_space
        elif self.model == 'loid':
            x1 = self._points[0]
            bar_x = self._points[1:]
            new_points = bar_x / (x1 + 1)
            new_space = PoincareEmbedding(curvature=self._curvature, points=new_points, labels = self.labels, enable_logging=self._logger is not None)
            self._log_info("Switched to PoincareEmbedding model.")
            return new_space
        else:
            if self._logger:
                self._logger.error("Unknown model type: %s", self.model)
            raise ValueError("Unknown model type.")

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

        # Eigenvalue decomposition (for symmetric matrices)
        eigvals, eigvecs = torch.linalg.eigh(A)

        # Ensure eigenvalues are non-negative by clamping any small negatives to zero
        eigvals = torch.clamp(eigvals, min=0)

        # Reconstruct the matrix square root using the eigendecomposition
        return eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T

    def __repr__(self) -> str:
        """Returns a string representation of the HyperbolicEmbedding."""
        return (f"HyperbolicEmbedding(curvature={self._curvature.item():.2f}, model={self.model}, points_shape={list(self._points.shape)})")

####################################################################################################
####################################################################################################
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
                 labels: Optional[List[Union[str, int]]] = None,
                 enable_logging: bool = False) -> None:
        """
        Initializes the PoincareEmbedding.

        Args:
            curvature (Optional[float]): The curvature of the space. Must be negative.
            points (Optional[Union[np.ndarray, torch.Tensor]]): A NumPy array or PyTorch tensor of points. Default is None.
            labels (Optional[List[Union[str, int]]]): A list of labels corresponding to the points. Default is None.
            enable_logging (bool): If True, logging is enabled. Default is False.
        """
        super().__init__(curvature=curvature, points=points, labels=labels, enable_logging=enable_logging)
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
            if self._logger:
                self._logger.error(f"Points norm constraint violated: norms={norm2}, constraint=({min_norm}, {max_norm})")
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

        return distance_matrix

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
            if self._logger:
                self._logger.error("In Poincare model, the L2 norm of the points must be strictly less than 1. Invalid norms: %s", norm2)
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
            if self._logger:
                self._logger.error("Dimension mismatch between points (%s) and vector (%s)", x.shape, b.shape)
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
        enable_logging (bool): Whether to enable logging.
    """

    def __init__(
        self,
        curvature: Optional[float] = -1,
        points: Optional[Union[np.ndarray, torch.Tensor]] = None,
        labels: Optional[List[Union[str, int]]] = None,
        enable_logging: Optional[bool] = False
    ) -> None:
        """
        Initializes the LoidEmbedding with the given parameters.

        Args:
            curvature (float, optional): The curvature of the space. Defaults to -1.
            points (Union[np.ndarray, torch.Tensor], optional): Initial points in the space. Defaults to None.
            labels (List[Union[str, int]], optional): Labels for the points. Defaults to None.
            enable_logging (bool, optional): Whether to enable logging. Defaults to False.
        """
        super().__init__(curvature=curvature, points=points, labels=labels, enable_logging=enable_logging)
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
            if self._logger:
                self._logger.error(f"Points norm constraint violated: norms={norm2}, constraint=({min_norm}, {max_norm})")
            points = self._points
            for n in range(self.n_points):
                points[:,n] = utils.project_to_hyperbolic_space(points[:,n])
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

        return distance_matrix

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
            if self._logger:
                self._logger.error("In Loid model, the J-norm of the points must be exactly equal to -1. Invalid norms: %s", norm2)
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
            if self._logger:
                self._logger.error("In Loid model, the J-norm of the points must be exactly equal to -1. Invalid norms: %s", norm2)
            raise ValueError("In Loid model, the J-norm of the points must be exactly equal to -1.")

        if x.shape[0] != b.shape[0]:
            if self._logger:
                self._logger.error("Dimension mismatch between points (%s) and vector (%s)", x.shape, b.shape)
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
                 labels: Optional[List[Union[str, int]]] = None,
                 enable_logging: bool = False):
        """
        Initializes the EuclideanEmbedding.

        Args:
            points (Optional[Union[np.ndarray, torch.Tensor]]): A NumPy array or PyTorch tensor of points. Default is None.
            enable_logging (bool): If True, logging is enabled. Default is False.
        """
        super().__init__(geometry='euclidean', points=points, labels=labels, enable_logging=enable_logging)
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
        centroid = torch.mean(self.points, dim=1)
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

        return torch.sqrt(EDM)
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

    def __init__(self, enable_logging: bool = False):
        """
        Initializes the MultiEmbedding with an empty list of embeddings.

        Args:
            enable_logging (bool): If True, logging is enabled. Default is False.
        """
        self.embeddings = []
        self._current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._logger = None
        if enable_logging:
            self._setup_logging()
        self._log_info("Initialized MultiEmbedding with an empty list of embeddings.")
        self.labels = None

    def _setup_logging(self, log_dir: str = conf.LOG_DIRECTORY, log_level: int = logging.INFO, log_format: str = '%(asctime)s - %(levelname)s - %(message)s') -> None:
        """
        Set up logging configuration.

        Args:
            log_dir (str): Directory where log files will be saved. Default is 'log'.
            log_level (int): Logging level. Default is logging.INFO.
            log_format (str): Format for logging messages. Default is '%(asctime)s - %(levelname)s - %(message)s'.
        """
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"MultiEmbedding_{self._current_time}.log")
        logging.basicConfig(filename=log_file, level=log_level, format=log_format)
        self._logger = logging.getLogger(__name__)
        self._log_info("Logging setup complete.")

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
            if self._logger:
                self._logger.error("Failed to save MultiEmbedding: %s", e)
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
            if hasattr(instance, '_logger') and instance._logger:
                instance._log_info(f"Loaded MultiEmbedding from {filename}")
            return instance
        except Exception as e:
            logging.error("Failed to load MultiEmbedding: %s", e)
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


    def _log_info(self, message: str) -> None:
        """
        Logs an informational message.

        Args:
            message (str): The message to be logged.
        """
        if self._logger:
            self._logger.info(message)

    def add_embedding(self, embedding: 'Embedding') -> None:
        """
        Adds an embedding to the collection with a specified name.

        Args:
            name (str): The label associated with the embedding.
            embedding (Embedding): The embedding instance to be added.
        """
        self.embeddings.append(embedding)
        self._log_info(f"Added embedding'.")


    def align(self, func = torch.nanmean, mode = 'accurate') -> None:
        """
        Aligns all embeddings by averaging their distance matrices and adjusting
        each embedding to match the reference embedding.
        """

        if not self.embeddings:
            self._log_info("No embeddings to align.")
            return

        dimensions = {int(embedding.dimension) for embedding in self.embeddings}
        curvatures = {embedding.curvature.item() for embedding in self.embeddings}
        geometries = {embedding.geometry for embedding in self.embeddings}

        if len(curvatures) != 1 or len(geometries) != 1:
            raise ValueError("All embeddings must have the same curvature and geometry.")
        if len(dimensions) != 1:
            raise ValueError("All embeddings must have the same dimension.")
        else:
            dimension = list(dimensions)[0]

        reference_embedding = self.reference_embedding(func=func)

        for name, embedding in enumerate(self.embeddings):
            hp = procrustes.HyperbolicProcrustes(embedding, reference_embedding,mode = mode, enable_logging = self._logger is not None )
            self.embeddings[name] = hp.map(embedding)
            # Add Procrustes alignment or other alignment logic if needed.
            self._log_info(f"Aligned embedding with name '{name}'.")

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

        self._labels = all_labels
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
            distance_matrix[idx[:, None], idx] = embedding.distance_matrix()
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
        return agg_distance_matrix


    def reference_embedding(self, 
                            func: Callable[[torch.Tensor], torch.Tensor] = torch.nanmean,
                            **kwargs) -> 'Embedding':
        """
        Create a reference embedding from the average distance matrix, supporting both hyperbolic
        and Euclidean geometries, depending on the curvature of the embeddings.

        Parameters:
        - dimension (int): Dimensionality of the embedding space.
        - func (Callable[[torch.Tensor], torch.Tensor]): Function to compute the aggregate. Default is torch.nanmean.
        - accurate (bool): Embedding accuracy: False for naive, True for advanced (default is False).
        - epochs (int): Number of epochs for optimization in precise mode (default is 2000).
        - max_diameter (float): Maximum diameter for scaling the distance matrix (default is 10).
        - learning_rate (Optional[float]): Learning rate for the optimization process (default is None).
        - scale_learning (Optional[callable]): Function for scaling during optimization (default is None).
        - weight_exponent (Optional[float]): Exponent for weights in optimization (default is None).
        - initial_lr (Optional[float]): Initial learning rate for optimization (default is 0.1).

        Returns:
        - Embedding: The resulting embedding of the average distance matrix, either hyperbolic or Euclidean.

        Raises:
        - ValueError: If the embeddings do not have the same curvature or are not all hyperbolic or Euclidean.
        - ValueError: If 'dimension' is not provided.
        - RuntimeError: For any errors during the embedding process.
        """
        # Ensure all embeddings have the same curvature and geometry
        curvatures = {embedding.curvature.item() for embedding in self.embeddings}
        geometries = {embedding.geometry for embedding in self.embeddings}
        dimensions = {int(embedding.dimension) for embedding in self.embeddings}

        if (len(curvatures) != 1) or (len(geometries) != 1) or (len(dimensions) != 1):
            raise ValueError("All embeddings must have the same curvature, geometry, and dimension.")

        dimension = torch.tensor(dimensions.pop())
        curvature = torch.tensor(curvatures.pop())
        geometry = geometries.pop()

        
        # Retrieve and validate keyword arguments
        accurate = kwargs.get('accurate', conf.ENABLE_ACCURATE_OPTIMIZATION)
        total_epochs = kwargs.get('total_epochs', conf.TOTAL_EPOCHS)
        initial_lr = kwargs.get('initial_lr', conf.INITIAL_LEARNING_RATE)
        max_diameter = kwargs.get('max_diameter', conf.MAX_RANGE)
        enable_save = kwargs.get('enable_save', conf.ENABLE_SAVE_MODE)
        learning_rate = kwargs.get('learning_rate', None)
        weight_exponent = kwargs.get('weight_exponent', None)

        def scale_learning(epoch, total_epochs, loss_list):
            return False

        distance_matrix = self.distance_matrix(func = func)
        if geometry == 'hyperbolic':
            try:
                scale_factor = torch.sqrt(-curvature)
                distance_matrix *= scale_factor
                self._log_info("Initiating hyperbolic embedding of the average distance matrix.")
                
                gramian = -torch.cosh(distance_matrix)
                points = utils.lgram_to_points(gramian, dimension).detach()
                for n in range(points.size(1)):
                    points[:,n] = utils.project_to_hyperbolic_space(points[:,n])
                    points[0,n] = torch.sqrt(1+torch.sum(points[1:,n]**2))

                embeddings = embedding.LoidEmbedding(curvature = curvature, points=points, labels = self._labels, enable_logging=self._logger is not None)
                self._log_info("Naive hyperbolic embedding completed.")

                if accurate:
                    self._log_info("Initiating precise hyperbolic embedding.")
                    initial_tangents = utils.hyperbolic_log(points)
                    tangents, _ = utils.hyperbolic_embedding(
                            distance_matrix,
                            dimension,
                            initial_tangents=initial_tangents,
                            epochs=total_epochs,
                            log_function=self._log_info,
                            learning_rate=learning_rate,
                            scale_learning=scale_learning,
                            weight_exponent=weight_exponent,
                            initial_lr=initial_lr,
                            enable_save=enable_save,
                            time=self._current_time
                            )
                    embeddings.points = utils.hyperbolic_exponential(tangents)
                    self._log_info("Precise hyperbolic embedding completed.")
            except ValueError as ve:
                self._log_info(f"Value error during hyperbolic embedding: {ve}")
                raise
            except RuntimeError as re:
                self._log_info(f"Runtime error during hyperbolic embedding: {re}")
                raise
            except Exception as e:
                self._log_info(f"Unexpected error during hyperbolic embedding: {e}")
                raise
        elif geometry == 'euclidean':
            try:
                self._log_info("Initiating Euclidean embedding of the average distance matrix.")
                n = distance_matrix.size(0)

                # Step 1: Convert the distance matrix to a Gram matrix using double centering
                J = (torch.eye(n) - torch.ones((n, n)) / n).to(distance_matrix.dtype)
                gramian = -0.5 * J @ (distance_matrix**2) @ J

                # Step 2: Perform eigen decomposition on the Gram matrix
                eigenvalues, eigenvectors = torch.linalg.eigh(gramian)

                # Step 3: Sort eigenvalues and eigenvectors in descending order
                sorted_indices = torch.argsort(eigenvalues, descending=True)
                eigenvalues = eigenvalues[sorted_indices]
                eigenvectors = eigenvectors[:, sorted_indices]

                # Step 4: Select the top "dimension" eigenvalues and corresponding eigenvectors
                top_eigenvalues = eigenvalues[:dimension].clamp(min=0)  # Clamp to ensure non-negative
                top_eigenvectors = eigenvectors[:, :dimension]

                # Step 5: Compute the coordinates in d-dimensional space
                points = top_eigenvectors * torch.sqrt(top_eigenvalues).unsqueeze(0)
                points = points.t()

                # Initialize Loid embedding
                embeddings = embedding.EuclideanEmbedding(points=points,labels = self._labels)
                self._log_info("Naive euclidean embedding completed.")

                # Precise hyperbolic embedding
                if accurate:
                    self._log_info("Initiating precise euclidean embedding.")
                    points = utils.euclidean_embedding(
                        distance_matrix**2,
                        dimension,
                        initial_points=points,
                        epochs=total_epochs,
                        log_function=self._log_info,
                        learning_rate=learning_rate,
                        weight_exponent=weight_exponent,
                        initial_lr=initial_lr,
                        enable_save=enable_save,
                        time=self._current_time
                    )
                    embeddings.points = points
                    self._log_info("Precise euclidean embedding completed.")
            except ValueError as ve:
                self._log_info(f"Value error during euclidean embedding: {ve}")
                raise
            except RuntimeError as re:
                self._log_info(f"Runtime error during euclidean embedding: {re}")
                raise
            except Exception as e:
                self._log_info(f"Unexpected error during euclidean embedding: {e}")
                raise
        else:
            raise ValueError(f"Unsupported geometry: {geometry}")

        return embeddings

