import torch
from .utils import hyperbolic_exp  # Exponential map from tangent space to hyperbolic space
from .embedding import LoidEmbedding

class HyperbolicOptimizer:
    """
    A class to perform optimization in hyperbolic space.

    Attributes:
        function (callable): The function to optimize. Takes a single (D+1, N) tensor.
        D (int): Dimension of the tangent space (D+1 in hyperbolic space).
        N (int): Number of points.
        learning_rate (float): Learning rate for optimization.
        optimizer (torch.optim.Optimizer or None): Optional optimizer, default Adam.
        max_grad_norm (float): Max gradient norm for gradient clipping.
        lr_decay_factor (float): Factor by which the learning rate decays after each epoch.
    """
    
    def __init__(self, function, D, N, optimizer=None, learning_rate=0.01, max_grad_norm=1.0, lr_decay_factor=0.99):
        """
        Initialize the HyperbolicOptimizer.

        Args:
            function (callable): The function to optimize. Takes a single (D+1, N) tensor.
            D (int): Dimension of the tangent space (D+1 in hyperbolic space).
            N (int): Number of points.
            optimizer (torch.optim.Optimizer or None): Optional optimizer, default Adam.
            learning_rate (float): Learning rate for optimization.
            max_grad_norm (float): Max gradient norm for gradient clipping.
            lr_decay_factor (float): Factor by which the learning rate decays after each epoch.
        """
        self.function = function
        self.D = D  # Tangent space dimension
        self.N = N  # Number of points
        self.learning_rate = learning_rate
        self.optimizer = optimizer  # Custom optimizer or None (default)
        self.max_grad_norm = max_grad_norm  # Gradient clipping
        self.lr_decay_factor = lr_decay_factor  # Learning rate decay factor

    def optimize(self, epochs=100):
        """
        Optimize a function in hyperbolic space.

        Args:
            epochs (int): Number of optimization steps.

        Returns:
            torch.Tensor: Optimized points in hyperbolic space (D+1, N).
        """
        # Initialize optimization variables in tangent space (D, N)
        variables = torch.nn.Parameter(torch.randn(self.D, self.N) * 0.00001)  # Slightly adjusted initialization

        # Default optimizer: Riemannian Gradient Descent
        optimizer = self.optimizer([variables]) if self.optimizer else torch.optim.Adam([variables], lr=self.learning_rate)

        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Clip the variable values before the exponential map to stabilize it
            with torch.no_grad():
                variables.data = torch.clamp(variables.data, min=-5.0, max=5.0)  # Clamping to avoid extreme values

            hyperbolic_points = hyperbolic_exp(variables)  # Convert (D, N) to (D+1, N)
            
            # Evaluate function
            loss = self.function(hyperbolic_points)  
            loss.backward()

            # Gradient clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_([variables], self.max_grad_norm)
            
            optimizer.step()

            # Apply learning rate decay after each epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] *= self.lr_decay_factor

            print(f"Epoch {epoch}: Loss = {loss.item()}")

            
        return LoidEmbedding(points=hyperbolic_exp(variables).detach(), curvature=-1)  # Return optimized points in hyperbolic space