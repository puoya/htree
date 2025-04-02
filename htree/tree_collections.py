import os
import copy
import pickle
import logging
import random
from datetime import datetime
from collections.abc import Collection
from typing import Union, Set, Optional, List, Callable, Tuple, Dict, Iterator

import torch
import numpy as np
import treeswift as ts
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
import scipy.sparse.linalg as spla
from torch.optim import Adam
from tqdm import tqdm

# import conf
# import utils
# import embedding
# from logger import get_logger, logging_enabled, get_time
# import htree.conf, htree.utils, htree.embedding
# from . import conf, utils, embedding
# from .logger import get_logger, logging_enabled, get_time


from htree.logger import get_logger, logging_enabled, get_time
import htree.conf as conf
import htree.utils as utils
import htree.embedding as embedding

# Use non-GUI backend for matplotlib
import matplotlib
matplotlib.use('Agg')
#############################################################################################
# Class for handling tree operations using treeswift and additional utilities.
#############################################################################################
class Tree:
    """
    Represents a tree structure with logging capabilities.

    This class provides methods to manipulate and analyze tree structures,
    including embedding, normalizing, copying, saving, and computing distances.

    Methods:
    --------
    __init__(self, *args, **kwargs)
        Initializes the Tree object from a file or a (name, treeswift.Tree) pair.

    update_time(self)
        Sets _current_time to the current time.

    copy(self) -> 'Tree'
        Creates a deep copy of the Tree object.

    save(self, file_path: str, format: str = 'newick') -> None
        Saves the tree to a file in the specified format.

    terminal_names(self) -> List[str]
        Retrieves terminal (leaf) names in the tree.

    distance_matrix(self) -> torch.Tensor
        Computes the pairwise distance matrix for the tree.

    diameter(self) -> torch.Tensor
        Calculates and logs the diameter of the tree.

    normalize(self) -> None
        Normalizes tree branch lengths such that the tree's diameter is 1.

    embed(self, dimension: int, geometry: str = 'hyperbolic', **kwargs) -> 'Embedding'
        Embeds the tree into a specified geometric space (hyperbolic or Euclidean).

    Attributes:
    -----------
    _current_time : datetime
        The current time for logging and saving purposes.
    name : str
        The name of the tree.
    contents : treeswift.Tree
        The contents of the tree.
    """
    def __init__(self, *args, **kwargs):
        self._current_time = get_time() or datetime.now()

        if len(args) == 1 and isinstance(args[0], str):
            self.name, self.contents = os.path.basename(args[0]), self._load_tree(args[0])
            self._log_info(f"Initialized tree from file: {args[0]}")
        elif len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], ts.Tree):
            self.name, self.contents = args
            self._log_info(f"Initialized tree with name: {self.name}")
        else:
            raise ValueError("Expected a file path or a (name, treeswift.Tree) pair.")
    ################################################################################################
    def _log_info(self, message: str):
        """Logs a message if global logging is enabled."""
        if logging_enabled(): get_logger().info(message) 
    ################################################################################################
    @classmethod
    def _from_contents(cls, name: str, contents: ts.Tree) -> 'Tree':
        """Creates a Tree instance from a treeswift.Tree object."""
        instance = cls(name, contents)
        instance._log_info(f"Tree created: {name}")
        return instance
    ################################################################################################
    def _load_tree(self, file_path: str) -> ts.Tree:
        """Loads a treeswift.Tree from a Newick file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        self._log_info(f"Loading tree from: {file_path}")
        return ts.read_tree_newick(file_path)
    ################################################################################################
    def __repr__(self) -> str:
        """Returns the string representation of the Tree object."""
        return f"Tree({self.name})"
    ################################################################################################
    def update_time(self):
        """
        Sets _current_time to the current time.

        This method updates the _current_time attribute to the current system time.

        Parameters:
        -----------
        None

        Returns:
        --------
        None

        Notes:
        ------
        - The function sets the _current_time attribute to the result of datetime.now().
        - Logs are generated to indicate that the current time has been updated.

        Examples:
        ---------
        To update the current time:
        >>> instance.update_time()
        """
        self._current_time = datetime.now()
        self._log_info("Current time updated to now.")
    ################################################################################################
    def copy(self) -> 'Tree':
        """
        Creates a deep copy of the Tree object.

        This method generates a deep copy of the current Tree object, including all its attributes and contents.

        Parameters:
        -----------
        None

        Returns:
        --------
        Tree
            A deep copy of the current Tree object.

        Notes:
        ------
        - The function uses the `copy.deepcopy` method to ensure all nested objects are copied.
        - Logs are generated to indicate the successful creation of the tree copy.

        Examples:
        ---------
        To create a deep copy of the tree:
        >>> tree_copy = instance.copy()
        """
        tree_copy = copy.deepcopy(self)
        self._log_info(f"Copied tree: {self.name}")
        return tree_copy
    ################################################################################################
    def save(self, file_path: str, format: str = 'newick') -> None:
        """
        Saves the tree to a file in the specified format.

        This method saves the tree structure to a file using the specified format.

        Parameters:
        -----------
        file_path : str
            The path where the tree file will be saved.
        format : str, optional
            The format in which to save the tree ('newick' is supported). Default is 'newick'.

        Returns:
        --------
        None

        Raises:
        -------
        ValueError
            If an unsupported format is specified.

        Notes:
        ------
        - The function currently supports saving the tree in the Newick format only.
        - Logs are generated to indicate the success or failure of the save operation.

        Examples:
        ---------
        To save the tree in Newick format:
        >>> instance.save('path/to/tree_file.newick')
        """
        """Saves the tree to a file in the specified format."""
        if format.lower() == 'newick':
            self.contents.write_tree_newick(file_path)
            self._log_info(f"Tree saved: {self.name}")
        else:
            self._log_info(f"Failed to save tree: {self.name}. Unsupported format: {format}")
            raise ValueError(f"Unsupported format: {format}")
    ################################################################################################
    def terminal_names(self) -> List[str]:
        """
        Retrieves terminal (leaf) names in the tree.

        This method returns a list of the names of all terminal (leaf) nodes in the tree.

        Parameters:
        -----------
        None

        Returns:
        --------
        List[str]
            A list of terminal (leaf) node names in the tree.

        Notes:
        ------
        - The function logs the retrieval of terminal names for reference.
        - Terminal names are obtained by traversing the tree and collecting labels of leaves.

        Examples:
        ---------
        To retrieve the terminal names:
        >>> leaf_names = instance.terminal_names()
        """
        leaf_names = list(self.contents.labels(leaves=True, internal=False))
        self._log_info(f"Retrieved terminal names for tree: {self.name}")
        return leaf_names
    ################################################################################################
    def distance_matrix(self) -> torch.Tensor:
        """
        Computes the pairwise distance matrix for the tree.

        This method calculates the pairwise distances between all terminal nodes in the tree
        and returns the distance matrix as a PyTorch tensor along with the terminal names.

        Parameters:
        -----------
        None

        Returns:
        --------
        torch.Tensor
            A tensor representing the pairwise distance matrix of the tree.
        list
            A list of terminal node names corresponding to the distance matrix.

        Notes:
        ------
        - The function creates a mapping of terminal names to indices and extracts the distance data
          into numpy arrays for efficient processing.
        - The computed distances are converted into a PyTorch tensor and reshaped accordingly.
        - Logs are generated to indicate the computation status and the number of terminals.

        Examples:
        ---------
        To compute the distance matrix:
        >>> distance_matrix, terminal_names = instance.distance_matrix()
        """
        terminal_names = self.terminal_names()
        n = len(terminal_names)
        # Create a mapping of terminal names to indices
        index_map = np.array([self.contents.distance_matrix(leaf_labels=True).get(name, {}) for name in terminal_names])
        # Extract data into numpy arrays for efficient processing
        row_labels = np.repeat(np.arange(n), n)
        col_labels = np.tile(np.arange(n), n)
        distances = np.array([index_map[i].get(terminal_names[j], 0) for i, j in zip(row_labels, col_labels)])
        # Convert directly to a PyTorch tensor
        distances = torch.tensor(distances, dtype=torch.float32).reshape(n, n)
        self._log_info(f"Distance matrix computed for tree '{self.name}' with {n} terminals.")
        return (distances,terminal_names)
    ################################################################################################
    def diameter(self) -> torch.Tensor:
        """
        Calculate and log the diameter of the tree.

        This method computes the diameter of the tree and logs the value.

        Parameters:
        -----------
        None

        Returns:
        --------
        torch.Tensor
            A tensor representing the diameter of the tree.

        Notes:
        ------
        - The diameter is computed using the tree's contents and logged for reference.
        - The method utilizes PyTorch to store the diameter as a tensor.

        Examples:
        ---------
        To calculate and log the tree diameter:
        >>> tree_diameter = instance.diameter()
        """
        tree_diameter = torch.tensor(self.contents.diameter())
        self._log_info(f"Tree diameter: {tree_diameter.item()}")
        return tree_diameter
    ################################################################################################
    def normalize(self) -> None:
        """
        Normalize tree branch lengths such that the tree's diameter is 1.

        This method scales the branch lengths of the tree so that the overall diameter
        of the tree becomes 1. If the tree's diameter is zero, normalization is not performed.

        Parameters:
        -----------
        None

        Returns:
        --------
        None

        Notes:
        ------
        - The function traverses the tree in post-order to ensure all branch lengths are scaled
          appropriately.
        - Logs are generated to indicate the normalization status and the applied scale factor.

        Examples:
        ---------
        To normalize the tree:
        >>> instance.normalize()
        """
        tree_diameter = self.contents.diameter()
        if not np.isclose(tree_diameter, 0.0):
            scale_factor = 1.0 / tree_diameter
            for node in self.contents.traverse_postorder():
                if node.get_edge_length() is not None:
                    node.set_edge_length(node.get_edge_length() * scale_factor)
            self._log_info(f"Tree normalized with scale factor: {scale_factor}")
        else:
            self._log_info("Tree diameter is zero and cannot be normalized.")
    ################################################################################################
    def embed(self, dim: int, geometry: str = 'hyperbolic', **kwargs) -> 'Embedding':
        """
        Embed the tree into a specified geometric space (hyperbolic or Euclidean).

        This method handles the embedding of a tree into either hyperbolic or Euclidean space
        based on the specified geometry and additional parameters.

        Parameters:
        -----------
        dim : int
            The dimensionality of the geometric space for the embedding.
        geometry : str, optional
            The geometric space to embed into ('hyperbolic' or 'Euclidean'). Default is 'hyperbolic'.
        **kwargs : dict
            Additional parameters for the embedding process. Expected keys:
            - 'precise_opt' (bool): If True, performs precise embedding.
            - 'epochs' (int): Number of epochs for the optimization process.
            - 'lr_init' (float): Initial learning rate for the optimization process.
            - 'dist_cutoff' (float): Maximum distance cutoff for the embedding.
            - 'export_video' (bool): If True, exports a video of the embedding process.
            - 'save_mode' (bool): If True, saves the embedding object.
            - 'scale_fn' (callable): Optional scale learning function (boolean) for the optimization process.
            - 'lr_fn' (callable): Optional learning rate function for the optimization process.
            - 'weight_exp_fn' (callable): Optional weight exponent function for the optimization process.

        Returns:
        --------
        Embedding
            An Embedding object containing the geometric embedding points and their corresponding labels.

        Raises:
        -------
        ValueError
            If the 'dim' parameter is not provided.
        Exception
            If an error occurs during the embedding process.

        Notes:
        ------
        The function will save the embedding object and potentially export a video if the corresponding
        parameters are set to True.

        Examples:
        ---------
        To embed in hyperbolic space:
        >>> embedding = instance.embed(dim=2, geometry='hyperbolic')

        To embed in Euclidean space with precise optimization:
        >>> embedding = instance.embed(dim=2, geometry='Euclidean', precise_opt=True, export_video=True)
        """
        if dim is None:
            raise ValueError("The 'dimension' parameter is required.")
        params = {  
            key: kwargs.get(key, default) for key, default in {
                'precise_opt': conf.ENABLE_ACCURATE_OPTIMIZATION,
                'epochs': conf.TOTAL_EPOCHS,
                'lr_init': conf.INITIAL_LEARNING_RATE,
                'dist_cutoff': conf.MAX_RANGE,
                'export_video': conf.ENABLE_VIDEO_EXPORT,
                'save_mode': conf.ENABLE_SAVE_MODE,
                'scale_fn': None,
                'lr_fn': None,
                'weight_exp_fn': None
            }.items()
        }
        params['save_mode'] |= params['export_video']
        params['export_video'] &= params['precise_opt']

        try:
            embedding = (self._embed_hyperbolic if geometry == 'hyperbolic' else self._embed_euclidean)(dim, **params)
        except Exception as e:
            self._log_info(f"Error during embedding: {e}")
            raise

        directory = f"{conf.OUTPUT_DIRECTORY}/{self._current_time.strftime('%Y-%m-%d_%H-%M-%S')}"
        filepath = f"{directory}/{geometry}_embedding_{dim}d.pkl"
        os.makedirs(directory, exist_ok=True)
        try:
            with open(filepath, 'wb') as file:
                pickle.dump(embedding, file)
            self._log_info(f"Object successfully saved to {filepath}")
        except (IOError, pickle.PicklingError, Exception) as e:
            self._log_info(f"Error while saving object: {e}")
            raise

        if params['export_video']:
            self._gen_video(fps=params['epochs'] // conf.VIDEO_LENGTH)

        return embedding
    ################################################################################################
    def _embed_euclidean(self, dim: int, **params) -> 'Embedding':
        """Handle naive and precise Euclidean embeddings."""
        dist_mat = self.distance_matrix()[0]
        # Naive Euclidean embedding
        self._log_info("Initiating naive Euclidean embedding.")
        points = self._naive_euclidean_embedding(dist_mat, dim)
        embeddings = embedding.EuclideanEmbedding(points=points, labels=self.terminal_names())
        self._log_info("Naive Euclidean embedding completed.")
        if params['precise_opt']:
            self._log_info("Initiating precise Euclidean embedding.")
            embeddings.points = self._precise_euclidean_embedding(dist_mat, dim, points, **params)
            self._log_info("Precise Euclidean embedding completed.")
        return embeddings
    ################################################################################################
    def _naive_euclidean_embedding(self, dist_mat, dim):
        n = len(dist_mat)
        J = torch.eye(n) - 1/n
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
        scale_factor = params['dist_cutoff'] / self.diameter()
        dist_mat = self.distance_matrix()[0] * scale_factor
        # Naive hyperbolic embedding
        self._log_info("Initiating naive hyperbolic embedding.")
        points = self._naive_hyperbolic_embedding(dist_mat, dim)
        embeddings = embedding.LoidEmbedding(points=points, labels=self.terminal_names(), curvature=-(scale_factor ** 2))
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
        pts, scale = utils.hyperbolic_embedding(
            dist_mat, dim, init_pts=points, epochs=params['epochs'],
            log_fn=self._log_info, lr_fn=params['lr_fn'], scale_fn=params['scale_fn'], 
            weight_exp_fn=params['weight_exp_fn'], lr_init=params['lr_init'], 
            save_mode=params['save_mode'], time_stamp=self._current_time
        )
        self._log_info("Precise hyperbolic embedding completed.")
        return pts, scale
    ################################################################################################
    def _gen_video(self, fps: int = 10):
        """Generate a video of RE matrices evolution without saving individual frames."""
        timestamp = self._current_time    
        base = os.path.join(conf.OUTPUT_DIRECTORY, timestamp.strftime('%Y-%m-%d_%H-%M-%S'))
        weights = -np.load(os.path.join(base, "weight_exponents.npy"))
        lrs = np.log10(np.load(os.path.join(base, "learning_rates.npy")) + conf.EPSILON)
        try:
            scales = np.load(os.path.join(base, "scales.npy"), None)
        except FileNotFoundError:
            print("File not found. Proceeding without loading scales.")
            scales = None  # Or handle appropriately
        
        re_files = sorted(
            [f for f in os.listdir(base) if f.startswith('RE') and f.endswith('.npy')],
            key=lambda f: int(f.split('_')[1].split('.')[0])
            )[:len(weights)]  # Keep only the top N files

        re_mats = [np.load(os.path.join(base, file)) for file in re_files]

        tri_idx = np.triu_indices_from(re_mats[0], k=1)
        min_re, max_re, rms_vals = float('inf'), float('-inf'), []
        for mat in re_mats:
            tri_vals = mat[tri_idx]
            min_re = min(min_re, np.nanmin(tri_vals))
            max_re = max(max_re, np.nanmax(tri_vals))
            rms_vals.append(np.sqrt(np.mean(tri_vals)))
        
        min_re, max_re = np.log10(min_re + conf.EPSILON), np.log10(max_re + conf.EPSILON)
        rms_min, rms_max = (min(rms_vals) * 0.9, max(rms_vals) * 1.1) if rms_vals else (0, 1)
        lr_min, lr_max = min(lrs) - 0.1, max(lrs) + 0.1
        log_dist = np.log10(self.distance_matrix()[0] + conf.EPSILON)
        mask = np.eye(log_dist.shape[0], dtype=bool)
        
        fig, gs = plt.figure(figsize=(12, 12), tight_layout=True), GridSpec(4, 2, height_ratios=[1, 1, 2, 2], width_ratios=[1, 1])
        ax_rms = fig.add_subplot(gs[0, :])
        ax_weights = fig.add_subplot(gs[1, 0])
        ax_lr = fig.add_subplot(gs[1, 1])
        ax_re, ax_dist = fig.add_subplot(gs[2:, 0]), fig.add_subplot(gs[2:, 1])
        im_re = ax_re.imshow(np.zeros_like(re_mats[0]), cmap='viridis', vmin=min_re, vmax=max_re)
        im_dist = ax_dist.imshow(np.where(mask, np.nan, log_dist), cmap='viridis')
        fig.colorbar(im_re, ax=ax_re, fraction=0.046, pad=0.04, label='log10(RE)')
        fig.colorbar(im_dist, ax=ax_dist, fraction=0.046, pad=0.04, label='log10(Distance)')
        
        def update(epoch):
            ax_rms.clear()
            ax_weights.clear()
            ax_lr.clear()
            ax_rms.plot(range(1, epoch + 1), rms_vals[:epoch], marker='o')
            ax_rms.set(xlim=(1, len(re_mats)), ylim=(rms_min, rms_max), xlabel='Epoch', ylabel='RMS of RE', title='Evolution of Relative Errors')
            ax_weights.plot(range(1, epoch + 1), weights[:epoch], 'bo')

            if scales is not None:
                ax_weights.plot(range(1, epoch + 1), np.where(scales[:epoch], weights[:epoch], np.nan), 'ro', label='Scale Learning Enabled')
                ax_weights.legend()
            ax_weights.set(xlim=(1, len(re_mats)), ylim=(0, 1), xlabel='Epoch', ylabel='-Weight Exponent', title='Evolution of Weights')
            
            ax_lr.plot(range(1, epoch + 1), lrs[:epoch], marker='o')
            ax_lr.set(xlim=(1, len(re_mats)), ylim=(lr_min, lr_max), xlabel='Epoch', ylabel='log10(Learning Rate)', title='Evolution of Learning Rates')
            
            im_re.set_array(np.where(mask, np.nan, np.log10(re_mats[epoch] + conf.EPSILON)))
            ax_re.set_title(f'Relative Error (RE) Matrix (Epoch {epoch})')
            for ax in (ax_re, ax_dist):
                ax.set_xticks([]), ax.set_yticks([])
                ax.add_patch(patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color='black', fill=False, linewidth=2))
        
        self._log_info(f"Video is being created. Please be patient.")
        # Now generate the animation
        ani = animation.FuncAnimation(fig, update, frames=len(re_mats), interval=1000 // fps)
        
        out_dir = os.path.join(conf.OUTPUT_VIDEO_DIRECTORY, timestamp.strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(out_dir, exist_ok=True)
        vid_path = os.path.join(out_dir, 're_dist_evo.mp4')
        ani.save(vid_path, writer='ffmpeg', fps=fps)
        
        self._log_info(f"Video created: {vid_path}")
#############################################################################################
class MultiTree:
    """
    Class MultiTree
    ---------------

    Represents a collection of tree objects with methods to manipulate and analyze them.

    Initialization:
    ---------------
    __init__(self, *source: Union[str, List[Union['Tree', 'ts.Tree']]])
        Initializes a MultiTree object.
        Parameters:
            source: str or list of Tree objects or treeswift.Tree objects.
                    - If a string (file path), trees are loaded from the file.
                    - If a list of Tree or treeswift.Tree objects, trees are wrapped in Tree instances.

    Methods:
    --------
    update_time(self)
        Updates the current time for the MultiTree object.

    copy(self) -> 'MultiTree'
        Creates a deep copy of the MultiTree object.

    save(self, file_path: str, format: str = 'newick') -> None
        Saves the MultiTree object to a file in the specified format.
        Parameters:
            file_path: str
                The file path to save the tree.
            format: str, optional (default='newick')
                The format to save the tree (e.g., 'newick').

    terminal_names(self) -> List[str]
        Retrieves terminal (leaf) names from all trees in the MultiTree object.

    common_terminals(self) -> Set[str]
        Identifies terminal (leaf) names that are common across all trees in the MultiTree object.

    distance_matrix(self) -> np.ndarray
        Computes the pairwise distance matrix for all trees in the MultiTree object.

    normalize(self) -> None
        Normalizes the branch lengths of all trees such that each tree's diameter is 1.

    embed(self, dimension: int, geometry: str = 'hyperbolic', **kwargs) -> 'Embedding'
        Embeds all trees into a specified geometric space (hyperbolic or Euclidean).
        Parameters:
            dimension: int
                The dimension of the embedding space.
            geometry: str, optional (default='hyperbolic')
                The geometry of the embedding space ('hyperbolic' or 'euclidean').
            **kwargs: additional keyword arguments for embedding.

    Attributes:
    -----------
    _current_time : datetime
        The current time for logging and saving purposes.
    name : str
        The name of the MultiTree object.
    trees : List[Tree]
        A list of Tree objects contained in the MultiTree object.
    """

    def __init__(self, *source: Union[str, List[Union['Tree', 'ts.Tree']]]):
        self._current_time, self.trees = get_time() or datetime.now(), []
        if len(source) == 1 and isinstance(source[0], str):
            self.name, file_path = os.path.basename(source[0]), source[0]
            self.trees = self._load_trees(file_path)
        elif len(source) == 2 and isinstance(source[0], str) and isinstance(source[1], list):
            self.name, tree_list = source[0], source[1]
            self.trees = (
                tree_list if all(isinstance(t, Tree) for t in tree_list)
                else [Tree(f"Tree_{i}", t) for i, t in enumerate(tree_list)]
                if all(isinstance(t, ts.Tree) for t in tree_list)
                else ValueError("List must contain only Tree or treeswift.Tree instances.")
            )
        else:
            raise ValueError("Invalid input format.")
    ################################################################################################
    def _log_info(self, message: str):
        if logging_enabled(): get_logger().info(message)
    ################################################################################################
    def update_time(self):
        """
        Updates the current time to the system's current date and time.

        This function sets the internal attribute '_current_time' to the current date
        and time using `datetime.now()`. It also logs the updated time information.

        Example:
            >>> obj = MultiTree()
            >>> obj.update_time()
            Current time updated to now.

        Attributes:
            _current_time (datetime): The current date and time.
            _log_info (function): A method that logs informational messages.

        """
        self._current_time = datetime.now()
        self._log_info("Current time updated to now.")
    ################################################################################################
    def _load_trees(self, file_path: str) -> List['Tree']:
        if not os.path.exists(file_path): raise FileNotFoundError(f"File not found: {file_path}")
        try:
            return [Tree(f'tree_{i+1}', t) for i, t in enumerate(ts.read_tree_newick(file_path))]
        except Exception as e:
            raise ValueError(f"Error loading trees: {e}")
    ################################################################################################
    def __getitem__(self, index: Union[int, slice]) -> Union['Tree', 'MultiTree']:
        """Retrieve individual trees or a sub-MultiTree."""
        return MultiTree(self.name, self.trees[index]) if isinstance(index, slice) else self.trees[index]
    ################################################################################################
    def __len__(self) -> int:
        """Return number of trees."""
        return len(self.trees)
    ################################################################################################
    def __iter__(self) -> Iterator['Tree']:
        """Iterate over trees."""
        return iter(self.trees)
    ################################################################################################
    def __contains__(self, item) -> bool:
        """Check if item exists in MultiTree."""
        return item in self.trees
    ################################################################################################
    def __repr__(self) -> str:
        """String representation of MultiTree."""
        return f"MultiTree({self.name}, {len(self.trees)} trees)"
    ################################################################################################
    def copy(self) -> 'MultiTree':
        """
        Creates a deep copy of the current MultiTree instance.

        This function generates a deep copy of the MultiTree object, ensuring that all 
        nested objects are also copied. It logs the copy action for reference.

        Example:
            >>> obj = MultiTree()
            >>> obj_copy = obj.copy()
            MultiTree 'TreeName' copied.

        Returns:
            MultiTree: A deep copy of the current instance.

        Attributes:
            name (str): The name of the MultiTree instance.
            _log_info (function): A method that logs informational messages.

        """
        self._log_info(f"MultiTree '{self.name}' copied.")
        return copy.deepcopy(self)
    ################################################################################################
    def save(self, path: str, fmt: str = 'newick') -> None:
        """
        Save all trees to a file in the specified format.

        This function saves the trees in the MultiTree instance to a file at the 
        specified path. It supports the 'newick' format for saving trees.

        Args:
            path (str): The file path where trees will be saved.
            fmt (str): The format in which to save the trees. Currently, only 'newick' 
                       format is supported. Defaults to 'newick'.

        Raises:
            ValueError: If an unsupported format is specified.
            Exception: If an error occurs while saving the trees, with the error 
                       information logged.

        Example:
            >>> obj = MultiTree()
            >>> obj.save('trees.newick')
            Saved trees to trees.newick (newick format).

        Attributes:
            trees (list): A list of tree objects to be saved.
            _log_info (function): A method that logs informational messages.

        """
        if fmt != 'newick':
            self._log_info(f"Unsupported format: {fmt}")
            raise ValueError(f"Unsupported format: {fmt}")

        try:
            with open(path, 'w') as f:
                f.writelines(tree.contents.newick() + "\n" for tree in self.trees)
            self._log_info(f"Saved trees to {path} ({fmt} format).")
        except Exception as e:
            self._log_info(f"Failed to save trees to {path}: {e}")
            raise
    ################################################################################################
    def terminal_names(self) -> List[str]:
        """
        Return sorted terminal (leaf) names from all trees.

        This function retrieves terminal (leaf) names from all trees in the MultiTree instance,
        sorts them in alphabetical order, and logs the retrieval action.

        Returns:
            List[str]: A sorted list of terminal (leaf) names from all trees.

        Example:
            >>> obj = MultiTree()
            >>> names = obj.terminal_names()
            Retrieved 20 terminal names for MultiTree 'TreeName'

        Attributes:
            trees (list): A list of tree objects containing terminal (leaf) names.
            _log_info (function): A method that logs informational messages.
            name (str): The name of the MultiTree instance.

        """
        names = sorted({name for tree in self.trees for name in tree.terminal_names()})
        self._log_info(f"Retrieved {len(names)} terminal names for {self.name}")
        return names
    ################################################################################################
    def common_terminals(self) -> List[str]:
        """
        Return sorted terminal names common to all trees.

        This function retrieves terminal (leaf) names that are common to all trees
        in the MultiTree instance. It sorts these common terminal names in alphabetical
        order and logs the retrieval action.

        Returns:
            List[str]: A sorted list of terminal names common to all trees.

        Example:
            >>> obj = MultiTree()
            >>> common_names = obj.common_terminals()
            15 common terminal names retrieved for MultiTree 'TreeName'

        Attributes:
            trees (list): A list of tree objects containing terminal (leaf) names.
            _log_info (function): A method that logs informational messages.
            name (str): The name of the MultiTree instance.

        """
        if not self.trees:
            return []
        
        common = set(self.trees[0].terminal_names())
        for tree in self.trees[1:]:
            common.intersection_update(tree.terminal_names())
        
        self._log_info(f"{len(common)} common terminal names retrieved for {self.name}")
        return sorted(common)
    ################################################################################################
    def distance_matrix(self, method: str = "agg", func: Callable[[torch.Tensor], torch.Tensor] = torch.nanmean, max_iter: int = 1000) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute the distance matrix for terminal names across all trees.

        This function calculates the distance matrix for terminal (leaf) names across all 
        trees in the MultiTree instance. It supports different aggregation methods and 
        functions for distance computation. The function logs the computation progress 
        and returns the distance matrix, confidence scores, and terminal labels.

        Args:
            method (str): The method for distance computation, e.g., "agg" or "fp" (fixed point).
                          Defaults to "agg".
            func (Callable): A function to aggregate distances, e.g., `torch.nanmean`.
                             Defaults to `torch.nanmean`.
            max_iter (int): The maximum number of iterations for fixed point computation.
                            Defaults to 1000.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], List[str]]: The distance matrix, 
                                                                   confidence scores, and 
                                                                   terminal labels.

        Raises:
            ValueError: If no trees are available for distance computation.
            Exception: If an error occurs during distance matrix computation.

        Example:
            >>> obj = MultiTree()
            >>> dist_mat, conf, labels = obj.distance_matrix()
            Distance matrix computation complete.

        Attributes:
            trees (list): A list of tree objects to be analyzed.
            _log_info (function): A method that logs informational messages.
            name (str): The name of the MultiTree instance.

        """
        if not self.trees:
            self._log_info("No trees available for distance computation.")
            raise ValueError("No trees available for distance computation.")
        
        labels = self.terminal_names()
        n, m = len(labels), len(self.trees)
        dist_mats = torch.full((m, n, n), float('nan'))
        
        for i, tree in enumerate(self.trees):
            idx = torch.tensor([labels.index(lbl) for lbl in tree.terminal_names()])
            mat = torch.full((n, n), float('nan'))
            mat[idx[:, None], idx] = tree.distance_matrix()[0]
            mat.fill_diagonal_(0)
            dist_mats[i] = mat

        valid_mask = ~torch.isnan(dist_mats)
        conf = valid_mask.float().mean(dim=0)
        sigma = 3

        if method == "fp":
            def gaussian_similarity(x1: torch.Tensor, x2: torch.Tensor, s: float) -> torch.Tensor:
                return torch.exp( - s * torch.norm(x1 - x2)**2 / torch.norm(x1)**2 )

            valid_counts = valid_mask.sum(dim=0).clamp(min=1)
            avg_mat = self.distance_matrix(func=func)[0]
            prev_w = torch.zeros(m, device=dist_mats.device)
            progress = tqdm(total=max_iter, desc="Fixed Point", unit="iter")

            for epoch in range(max_iter):
                sigma_epoch = min(2 * sigma *  epoch / max_iter, sigma)
                similarities = torch.stack([
                    gaussian_similarity(avg_mat[valid_mask[i]].flatten(), dist_mats[i][valid_mask[i]].flatten(),sigma_epoch)
                    if (valid_mask[i]).any() else torch.tensor(0, device=dist_mats.device)
                    for i in range(m)
                ])
                w = similarities / similarities.sum()
                W = torch.where(valid_mask, w.view(-1, 1, 1),torch.zeros_like(dist_mats))
                W /= W.sum(dim=0, keepdim=True)
                W *= valid_counts[None, :, :]
                avg_mat = torch.nansum(W * dist_mats, dim=0) / valid_counts
                if torch.sqrt(m*torch.norm(w - prev_w)**2) < 10**(-10):
                    break
                prev_w = w.clone()
                progress.update(1)
            progress.close()
            return avg_mat, conf, labels

        avg_mat = func(dist_mats, dim=0)[0] if isinstance(func(dist_mats, dim=0), tuple) else func(dist_mats, dim=0)
        nan_mask = torch.isnan(avg_mat)
        for i, j in torch.nonzero(avg_mat, as_tuple=False):
            non_nan_vals = torch.cat([
                avg_mat[i, ~torch.isnan(avg_mat[i, :])], 
                avg_mat[~torch.isnan(avg_mat[:, j]), j]
            ])
            if non_nan_vals.numel() > 0:
                avg_mat[i, j] = func(non_nan_vals)

        self._log_info("Distance matrix computation complete.")
        return avg_mat, conf, labels
    ################################################################################################
    def normalize(self, batch_mode=False):
        """
        Normalize the edge lengths of the trees in the MultiTree object.

        This function normalizes the edge lengths of the trees based on their distance matrices.
        It can operate in batch mode to process trees in batches for efficiency.

        Parameters
        ----------
        batch_mode : bool, optional
            If True, the function operates in batch mode, processing trees in batches for efficiency.
            The default is False.

        Returns
        -------
        torch.Tensor
            A tensor containing the scales used for normalization of each tree.

        Notes
        -----
        The function performs the following steps:
        1. Precomputes distance matrices for all trees.
        2. Handles NaN values in the distance matrices and calculates valid counts.
        3. Initializes the scales tensor for all trees.
        4. Iteratively optimizes the scales in batches or for all trees depending on the batch_mode.
        5. Updates the edge lengths of the trees by scaling them.

        Example
        -------
        >>> multi_tree = MultiTree(trees)
        >>> scales = multi_tree.normalize(batch_mode=True)
        >>> print(scales)

        """
        labels = self.terminal_names()
        n, m = len(labels), len(self.trees)
        log_lr_range = (-np.log10(n) + 1, -np.log10(n)) if batch_mode else (-np.log10(n) - 1, -np.log10(n))
        max_iters = 10 * int(np.sqrt(n) + 1) if batch_mode else 10 * n
        num_passes = int(n / np.sqrt(n) + 1) if batch_mode else 1
        batch_size = int(np.sqrt(m) + 1) if batch_mode else m

        # Precompute distance matrices
        dist_mats = torch.full((m, n, n), float('nan'))
        for i, tree in enumerate(self.trees):
            idx = torch.tensor([labels.index(lbl) for lbl in tree.terminal_names()])
            mat = torch.full((n, n), float('nan'))
            mat[idx[:, None], idx] = tree.distance_matrix()[0]
            mat.fill_diagonal_(0)
            dist_mats[i] = mat

        nan_mask = torch.isnan(dist_mats)
        valid_counts = nan_mask.logical_not().sum(dim=0).clamp(min=1)
        dist_mats = torch.nan_to_num(dist_mats, nan=0.0)
        scales = torch.ones(m, dtype=torch.float32)
        total_iters = num_passes * max_iters * len(range(0, m, batch_size)) + (max_iters if batch_mode else 0)
        progress = tqdm(total=total_iters, desc="Normalizing", unit="iter")

        def optimize_scales(batch_indices, remaining_weighted, batch_nan_mask, batch_dist_mats):
            sum_x = scales[batch_indices].sum()
            x = scales[batch_indices].clone().requires_grad_(True)
            opt = Adam([x], lr=10 ** log_lr_range[0])

            for i in range(max_iters):
                lr = 10 ** (log_lr_range[0] + (log_lr_range[1] - log_lr_range[0]) * (i / max_iters))
                opt.param_groups[0]['lr'] = lr
                
                x_norm = torch.nn.functional.softplus(x)
                x_norm = (x_norm / x_norm.sum()) * sum_x
                weighted = batch_dist_mats * x_norm[:, None, None]
                avg_mat = (weighted.sum(dim=0) + remaining_weighted) / valid_counts
                loss = ((weighted - avg_mat.unsqueeze(0)) * (~batch_nan_mask)).norm(dim=(1, 2), p='fro').pow(2).sum() / (len(batch_indices) * n**2)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                progress.update(1)
                self._log_info(f"Iter {i}: Loss={loss.item():.6f} | LR={lr:.2e}")
            
            scales[batch_indices] = x_norm.detach()
            return avg_mat

        for pass_idx in range(num_passes):
            indices = list(range(m))
            if batch_mode:
                random.shuffle(indices)
            for batch_start in range(0, m, batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                remaining_indices = list(set(indices) - set(batch_indices))
                remaining_weighted = (dist_mats[remaining_indices] * scales[remaining_indices, None, None]).sum(dim=0)
                batch_dist_mats, batch_nan_mask = dist_mats[batch_indices], nan_mask[batch_indices]
                avg_mat = optimize_scales(batch_indices, remaining_weighted, batch_nan_mask, batch_dist_mats)

        if batch_mode:
            avg_mat = optimize_scales(range(m), 0, nan_mask, dist_mats)
        scales = [scales[i].item() for i in range(m)]
        for i, tree in enumerate(self.trees):
            factor = scales[i]
            for node in tree.contents.traverse_postorder():
                if node.get_edge_length() is not None:
                    node.set_edge_length(node.get_edge_length() * factor)

        progress.close()
        return scales
    ################################################################################################
    def embed(self, dim: int, geometry: str = 'hyperbolic', **kwargs) -> 'MultiEmbedding':
        """
        Embeds multiple trees into the specified geometric space (hyperbolic or Euclidean).

        Parameters:
        -----------
        dim : int
            The dimension of the embedding space. Must be provided.
            
        geometry : str, optional
            The geometric space to use for the embedding. Can be 'hyperbolic' or 'euclidean'.
            Default is 'hyperbolic'.
            
        **kwargs : dict, optional
            Additional parameters for the embedding process. Includes:
            - precise_opt (bool): Enable accurate optimization. Default is set in conf.
            - epochs (int): Total number of training epochs. Default is set in conf.
            - lr_init (float): Initial learning rate. Default is set in conf.
            - dist_cutoff (float): Maximum distance cutoff. Default is set in conf.
            - save_mode (bool): Enable save mode. Default is set in conf.
            - scale_fn (callable): Scaling function to use. Default is None.
            - lr_fn (callable): Learning rate function to use. Default is None.
            - weight_exp_fn (callable): Weight exponent function to use. Default is None.
            - normalize (bool): Whether to normalize the embeddings. Default is False.

        Returns:
        --------
        MultiEmbedding
            An object containing the multiple embeddings generated.

        Raises:
        -------
        ValueError
            If the 'dim' parameter is not provided.

        RuntimeError
            For errors encountered during the embedding process.
        
        Example:
        --------
        To embed multiple trees in 3-dimensional hyperbolic space:
        
        >>> multi_embedding = obj.embed(dim=3, geometry='hyperbolic', epochs=100, lr_init=0.01)
        
        The results will be saved with the geometry included in the filename:
        '{output_directory}/hyperbolic_embedding_3d_space.pkl'
        
        Notes:
        ------
        - The method automatically saves the resulting embeddings to a file, with the geometry and dimension 
          included in the filename for clarity.
        - Users can adjust the various parameters by passing them as keyword arguments.
        - If normalization is required, set 'normalize' to True.
        """
        if dim is None:
            raise ValueError("The 'dimension' parameter is required.")

        # Extract and set embedding parameters
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
                'normalize': False
            }.items()
        }

        if params['normalize']:
            self.normalize(batch_mode = params['precise_opt'])

        try:
            multi_embeddings = (self._embed_hyperbolic if geometry == 'hyperbolic' else self._embed_euclidean)(dim, **params)
        except Exception as e:
            self._log_info(f"Error during multi_embedding: {e}")
            raise

        directory = f"{conf.OUTPUT_DIRECTORY}/{self._current_time.strftime('%Y-%m-%d_%H-%M-%S')}"
        filepath = f"{directory}/{geometry}_multiembedding_{dim}d.pkl"
        os.makedirs(directory, exist_ok=True)
        try:
            with open(filepath, 'wb') as file:
                pickle.dump(multi_embeddings, file)
            self._log_info(f"Object successfully saved to {filepath}")
        except (IOError, pickle.PicklingError, Exception) as e:
            self._log_info(f"Error while saving object: {e}")
            raise

        return multi_embeddings
    ################################################################################################
    def _embed_euclidean(self, dim: int, **params) -> 'MultiEmbedding':
        """
        Handle naive and precise Euclidean embeddings.
        """
        self._log_info("Starting the Euclidean embedding process.")
        multi_embeddings = embedding.MultiEmbedding()
        progress = tqdm(total=len(self.trees), desc="Embedding", unit="tree")
        for index, tree in enumerate(self.trees):
            self._log_info(f"Processing tree {index + 1}/{len(self.trees)}: {tree.name}")
            multi_embeddings.append(tree._embed_euclidean(dim, **params))
            self._log_info(f"Euclidean embedding completed for {tree.name}.")
            progress.update(1)
        progress.close()
        self._log_info("Euclidean embedding process completed for all trees.")
        return multi_embeddings
    ################################################################################################
    def _embed_hyperbolic(self, dim: int, **params) -> 'MultiEmbedding':
        """
        Handle naive and precise Hyperbolic embeddings.
        """
        self._log_info("Starting the Hyperbolic embedding process.")
        scale_factor = params['dist_cutoff'] / self.distance_matrix()[0].max()
        multi_embeddings = embedding.MultiEmbedding()
        progress = tqdm(total=len(self.trees), desc="Naive Embedding", unit="tree")
        dist_mats = []
        for index, tree in enumerate(self.trees):
            dist_mats.append(tree.distance_matrix()[0])

        for index, tree in enumerate(self.trees):
            self._log_info(f"Processing tree {index + 1}/{len(self.trees)}: {tree.name}")

            points = tree._naive_hyperbolic_embedding(tree.distance_matrix()[0]*scale_factor, dim =  dim)
            multi_embeddings.append(embedding.LoidEmbedding(points=points, labels=tree.terminal_names(), curvature=-(scale_factor ** 2)))
            self._log_info(f"Naive Hyperbolic embedding completed for {tree.name}.")
            progress.update(1)
        progress.close()

        self._log_info("Hyperbolic embedding (naive) process completed for all trees.")
        if params['precise_opt']:
            self._log_info("Refining embeddings with precise optimization.")
            pts_list, curvature = utils._precise_hyperbolic_multiembedding(dist_mats, multi_embeddings,log_fn=self._log_info, time_stamp=self._current_time, **params)
            multi_embeddings = embedding.MultiEmbedding()
            for index, tree in enumerate(self.trees):
                multi_embeddings.append(embedding.LoidEmbedding(points=pts_list[index], labels=tree.terminal_names(), curvature=curvature))
            self._log_info("Precise hyperbolic embedding process completed for all trees.")
        return multi_embeddings
    ################################################################################################