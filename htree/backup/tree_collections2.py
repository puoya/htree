import os
import copy
import pickle
import random
from datetime import datetime
from collections.abc import Collection
from typing import Union, Set, Optional, List, Callable, Tuple, Dict, Iterator
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec


import torch
import numpy as np
from tqdm import tqdm
import treeswift as ts
from torch.optim import Adam
from matplotlib import patches
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation

import htree.conf as conf
import htree.utils as utils
import htree.embedding as embedding
from htree.logger import get_logger, logging_enabled, get_time

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
            - 'scale_fn' (callable): Optional scale learning function for the optimization process.
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
            dist_mat = self.distance_matrix()[0]
            curvature = None

            # Hyperbolic-specific: scale distance matrix and compute curvature
            if geometry == 'hyperbolic':
                scale_factor = params['dist_cutoff'] / self.diameter()
                dist_mat = dist_mat * scale_factor
                curvature = -(scale_factor ** 2)

            # Naive embedding
            self._log_info(f"Initiating naive {geometry} embedding.")
            naive_dist = dist_mat if geometry == 'hyperbolic' else np.sqrt(dist_mat)
            points = utils.naive_embedding(naive_dist, dim, geometry=geometry)
            self._log_info(f"Naive {geometry} embedding completed.")

            # Precise optimization (refines initial embedding)
            if params['precise_opt']:
                self._log_info(f"Initiating precise {geometry} embedding.")
                opt_result = utils.precise_embedding(
                    dist_mat, dim, geometry=geometry, init_pts=points,
                    log_fn=self._log_info, time_stamp=self._current_time, **params
                )
                if geometry == 'hyperbolic':
                    points, scale = opt_result
                    curvature *= scale ** 2
                else:
                    points = opt_result
                self._log_info(f"Precise {geometry} embedding completed.")

            # Create appropriate embedding object
            if geometry == 'hyperbolic':
                output = embedding.LoidEmbedding(
                    points=points, labels=self.terminal_names(), curvature=curvature
                )
            else:
                output = embedding.EuclideanEmbedding(
                    points=points, labels=self.terminal_names()
                )

        except Exception as e:
            self._log_info(f"Error during embedding: {e}")
            raise

        # Save embedding
        directory = f"{conf.OUTPUT_DIRECTORY}/{self._current_time.strftime('%Y-%m-%d_%H-%M-%S')}"
        filepath = f"{directory}/{geometry}_embedding_{dim}d.pkl"
        os.makedirs(directory, exist_ok=True)
        try:
            with open(filepath, 'wb') as file:
                pickle.dump(output, file)
            self._log_info(f"Object successfully saved to {filepath}")
        except (IOError, pickle.PicklingError, Exception) as e:
            self._log_info(f"Error while saving object: {e}")
            raise

        if params['export_video']:
            self._gen_video(fps=params['epochs'] // conf.VIDEO_LENGTH)

        return output
    ################################################################################################
    def _gen_video(self, fps: int = 10):
        """Generate a video of RE matrices evolution without saving individual frames.
        
        Optimized for speed with parallel I/O, vectorized operations, and efficient rendering.
        """
        import matplotlib
        matplotlib.use('Agg')
        import subprocess
        
        # ═══════════════════════════════════════════════════════════════════════
        # AESTHETIC CONFIGURATION
        # ═══════════════════════════════════════════════════════════════════════
        # Professional dark theme with accent colors
        COLORS = {
            'background': '#1a1a2e',
            'panel': '#1e2a4a',
            'grid': '#2a2a4a',
            'text': '#e8e8e8',
            'text_secondary': '#a0a0b0',
            'accent_primary': '#00d4ff',      # Cyan
            'accent_secondary': '#ff6b6b',    # Coral
            'accent_tertiary': '#ffd93d',     # Gold
            'accent_muted': '#6c757d',        # Gray
        }
        
        # Apply style
        plt.rcParams.update({
            # Figure
            'figure.facecolor': COLORS['background'],
            'figure.edgecolor': COLORS['background'],
            
            # Axes
            'axes.facecolor': COLORS['panel'],
            'axes.edgecolor': COLORS['grid'],
            'axes.labelcolor': COLORS['text'],
            'axes.titlecolor': COLORS['text'],
            'axes.grid': True,
            'axes.grid.axis': 'both',
            'axes.axisbelow': True,
            'axes.linewidth': 0.8,
            'axes.titleweight': 'bold',
            'axes.titlesize': 11,
            'axes.labelsize': 9,
            'axes.labelweight': 'medium',
            
            # Grid
            'grid.color': COLORS['grid'],
            'grid.linewidth': 0.4,
            'grid.alpha': 0.5,
            
            # Ticks
            'xtick.color': COLORS['text_secondary'],
            'ytick.color': COLORS['text_secondary'],
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            
            # Text
            'text.color': COLORS['text'],
            'font.family': 'sans-serif',
            'font.size': 9,
            
            # Legend
            'legend.facecolor': COLORS['panel'],
            'legend.edgecolor': COLORS['grid'],
            'legend.fontsize': 8,
            'legend.framealpha': 0.9,
        })
        
        timestamp = self._current_time
        base = os.path.join(conf.OUTPUT_DIRECTORY, timestamp.strftime('%Y-%m-%d_%H-%M-%S'))
        
        # ═══════════════════════════════════════════════════════════════════════
        # DATA LOADING
        # ═══════════════════════════════════════════════════════════════════════
        weights = -np.load(os.path.join(base, "weight_exponents.npy"))
        lrs = np.log10(np.load(os.path.join(base, "learning_rates.npy")) + conf.EPSILON)
        
        try:
            scales = np.load(os.path.join(base, "scales.npy"))
        except FileNotFoundError:
            print("File not found. Proceeding without loading scales.")
            scales = None

        re_files = sorted(
            [f for f in os.listdir(base) if f.startswith('RE') and f.endswith('.npy')],
            key=lambda f: int(f.split('_')[1].split('.')[0])
        )[:len(weights)]
        
        n_frames = len(re_files)
        
        re_mats = Parallel(n_jobs=-1, prefer="threads")(
            delayed(np.load)(os.path.join(base, f)) for f in re_files
        )
        
        re_mats_3d = np.stack(re_mats, axis=0)
        del re_mats
        
        # ═══════════════════════════════════════════════════════════════════════
        # VECTORIZED STATISTICS
        # ═══════════════════════════════════════════════════════════════════════
        tri_idx = np.triu_indices(re_mats_3d.shape[1], k=1)
        all_tri_vals = re_mats_3d[:, tri_idx[0], tri_idx[1]]
        
        min_re = np.log10(np.nanmin(all_tri_vals) + conf.EPSILON)
        max_re = np.log10(np.nanmax(all_tri_vals) + conf.EPSILON)
        rms_vals = np.sqrt(np.nanmean(all_tri_vals ** 2, axis=1))
        
        del all_tri_vals
        
        rms_min = rms_vals.min() * 0.9 if len(rms_vals) > 0 else 0
        rms_max = rms_vals.max() * 1.1 if len(rms_vals) > 0 else 1
        lr_min, lr_max = lrs.min() - 0.1, lrs.max() + 0.1
        
        log_dist = np.log10(self.distance_matrix()[0] + conf.EPSILON)
        mask = np.eye(log_dist.shape[0], dtype=bool)
        masked_log_dist = np.where(mask, np.nan, log_dist)
        
        log_re_mats = np.log10(re_mats_3d + conf.EPSILON)
        log_re_mats[:, mask] = np.nan
        del re_mats_3d
        
        epochs = np.arange(1, n_frames + 1)
        
        is_hyperbolic = scales is not None and not np.all(scales == 1)

        # ═══════════════════════════════════════════════════════════════════════
        # PRECOMPUTE SCALE CHANGE DETECTION
        # ═══════════════════════════════════════════════════════════════════════
        if is_hyperbolic:
            scale_active = scales.astype(bool)
            scale_changed = np.concatenate([[True], np.diff(scales) != 0])
            scale_changing_mask = scale_active & scale_changed
            scale_unchanged_mask = scale_active & ~scale_changed
        
        # ═══════════════════════════════════════════════════════════════════════
        # SETUP OUTPUT
        # ═══════════════════════════════════════════════════════════════════════
        out_dir = os.path.join(conf.OUTPUT_VIDEO_DIRECTORY, timestamp.strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(out_dir, exist_ok=True)
        vid_path = os.path.join(out_dir, 're_dist_evo.mp4')
        
        self._log_info("Video is being created. Please be patient.")
        
        # ═══════════════════════════════════════════════════════════════════════
        # FIGURE SETUP
        # ═══════════════════════════════════════════════════════════════════════
        fig = plt.figure(figsize=(14, 12), dpi=100)
        gs = GridSpec(4, 2, height_ratios=[1, 1, 2, 2], width_ratios=[1, 1],
                      hspace=0.35, wspace=0.25)
        
        ax_rms = fig.add_subplot(gs[0, :])
        ax_weights = fig.add_subplot(gs[1, 0])
        ax_lr = fig.add_subplot(gs[1, 1])
        ax_re = fig.add_subplot(gs[2:, 0])
        ax_dist = fig.add_subplot(gs[2:, 1])
        for ax in [ax_rms, ax_weights, ax_lr, ax_re, ax_dist]:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('#4a4a6a')  # Light grayish-purple border
                spine.set_linewidth(1.5)
        
        # ─────────────────────────────────────────────────────────────────────
        # RMS Plot (top, spans both columns)
        # ─────────────────────────────────────────────────────────────────────
        line_rms, = ax_rms.plot([], [], color=COLORS['accent_primary'], 
                                linewidth=2, marker='o', markersize=5,
                                markerfacecolor=COLORS['accent_primary'],
                                markeredgecolor='white', markeredgewidth=0.1,
                                alpha=1, zorder=3)
        ax_rms.set_xlim(1, n_frames)
        ax_rms.set_ylim(rms_min, rms_max)
        ax_rms.set_yscale('log')
        ax_rms.set_xlabel('Epoch', fontweight='medium')
        ax_rms.set_ylabel('RMS of RE (log scale)', fontweight='medium')
        ax_rms.set_title('Evolution of Relative Errors', fontsize=12, pad=10)
        
        # ─────────────────────────────────────────────────────────────────────
        # Weights Plot
        # ─────────────────────────────────────────────────────────────────────
        line_weights, = ax_weights.plot([], [], color=COLORS['accent_primary'],
                                        linewidth=2, marker='o', markersize=5,
                                        markerfacecolor=COLORS['accent_tertiary'],
                                        markeredgecolor='white', markeredgewidth=0.1,
                                        alpha=1, zorder=2)
        line_weights_scaled = None
        line_weights_unchanged = None
        if is_hyperbolic:
            line_weights_scaled, = ax_weights.plot([], [], linestyle='none',
                                                   marker='o', markersize=7,
                                                   markerfacecolor='#ff3333',
                                                   markeredgecolor='white', markeredgewidth=0,
                                                   label='Scale Learning Enabled', zorder=4)
            line_weights_unchanged, = ax_weights.plot([], [], linestyle='none',
                                                      marker='o', markersize=5,
                                                      markerfacecolor=COLORS['accent_primary'],
                                                      markeredgecolor='white', markeredgewidth=0.1,
                                                      label='Scale Learning Disabled', zorder=3)
            ax_weights.legend(loc='upper right', framealpha=0.9)
        ax_weights.set_xlim(1, n_frames)
        ax_weights.set_ylim(0, 1)
        ax_weights.set_xlabel('Epoch', fontweight='medium')
        ax_weights.set_ylabel('−Weight Exponent', fontweight='medium')
        ax_weights.set_title('Weight Evolution', fontsize=11, pad=8)
        
        # ─────────────────────────────────────────────────────────────────────
        # Learning Rate Plot
        # ─────────────────────────────────────────────────────────────────────
        line_lr, = ax_lr.plot([], [], color='#50fa7b', linewidth=2,
                              marker='o', markersize=5,
                              markerfacecolor=COLORS['accent_primary'],
                              markeredgecolor='white', markeredgewidth=0.1,
                              alpha=1, zorder=3)
        ax_lr.set_xlim(1, n_frames)
        ax_lr.set_ylim(lr_min, lr_max)
        ax_lr.set_xlabel('Epoch', fontweight='medium')
        ax_lr.set_ylabel('log₁₀(Learning Rate)', fontweight='medium')
        ax_lr.set_title('Learning Rate Schedule', fontsize=11, pad=8)
        
        # ─────────────────────────────────────────────────────────────────────
        # RE Heatmap
        # ─────────────────────────────────────────────────────────────────────
        ax_re.set_facecolor('#0d0d1a')
        im_re = ax_re.imshow(log_re_mats[0], cmap='magma', vmin=min_re, vmax=max_re,
                             interpolation='nearest', aspect='equal')
        title_re = ax_re.set_title('Relative Error Matrix  ·  Epoch 0', 
                                   fontsize=11, pad=10, fontweight='bold')
        ax_re.set_xticks([])
        ax_re.set_yticks([])
        
        cbar_re = fig.colorbar(im_re, ax=ax_re, fraction=0.046, pad=0.04, 
                               shrink=0.9, aspect=25)
        cbar_re.set_label('log₁₀(RE)', fontsize=9, fontweight='medium')
        cbar_re.ax.tick_params(labelsize=8, colors=COLORS['text_secondary'])
        cbar_re.outline.set_edgecolor(COLORS['grid'])
        cbar_re.outline.set_linewidth(0.5)
        
        # ─────────────────────────────────────────────────────────────────────
        # Distance Heatmap
        # ─────────────────────────────────────────────────────────────────────
        ax_dist.set_facecolor('#0d0d1a')
        im_dist = ax_dist.imshow(masked_log_dist, cmap='viridis',
                                 interpolation='nearest', aspect='equal')
        ax_dist.set_title('Distance Matrix', fontsize=11, pad=10, fontweight='bold')
        ax_dist.set_xticks([])
        ax_dist.set_yticks([])
        
        cbar_dist = fig.colorbar(im_dist, ax=ax_dist, fraction=0.046, pad=0.04,
                                 shrink=0.9, aspect=25)
        cbar_dist.set_label('log₁₀(Distance)', fontsize=9, fontweight='medium')
        cbar_dist.ax.tick_params(labelsize=8, colors=COLORS['text_secondary'])
        cbar_dist.outline.set_edgecolor(COLORS['grid'])
        cbar_dist.outline.set_linewidth(0.5)
        
        # ─────────────────────────────────────────────────────────────────────
        # Subtle borders for heatmaps
        # ─────────────────────────────────────────────────────────────────────
        for ax in (ax_re, ax_dist):
            for spine in ax.spines.values():
                spine.set_edgecolor(COLORS['accent_primary'])
                spine.set_linewidth(1.5)
                spine.set_alpha(0.4)
        
        # Add title watermark/branding
        fig.text(0.99, 0.01, 'RE Matrix Evolution', fontsize=8, 
                 color=COLORS['text_secondary'], alpha=0.5,
                 ha='right', va='bottom', style='italic')
        
        # fig.tight_layout(pad=1.5)
        fig.subplots_adjust(left=0.06, right=0.94, top=0.95, bottom=0.06, hspace=0.35, wspace=0.25)

        
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        
        # ═══════════════════════════════════════════════════════════════════════
        # DIRECT FFMPEG PIPE
        # ═══════════════════════════════════════════════════════════════════════
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'rgba',
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            vid_path
        ]
        
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        
        # ═══════════════════════════════════════════════════════════════════════
        # RENDER FRAMES DIRECTLY TO PIPE
        # ═══════════════════════════════════════════════════════════════════════
        try:
            for epoch in range(n_frames):
                x_data = epochs[:epoch + 1]
                
                line_rms.set_data(x_data, rms_vals[:epoch + 1])
                line_weights.set_data(x_data, weights[:epoch + 1])
                line_lr.set_data(x_data, lrs[:epoch + 1])
                
                if is_hyperbolic:
                    changing_mask = scale_changing_mask[:epoch + 1]
                    line_weights_scaled.set_data(x_data[changing_mask], weights[:epoch + 1][changing_mask])
                    
                    unchanged_mask = scale_unchanged_mask[:epoch + 1]
                    line_weights_unchanged.set_data(x_data[unchanged_mask], weights[:epoch + 1][unchanged_mask])
                
                im_re.set_array(log_re_mats[epoch])
                title_re.set_text(f'Relative Error Matrix  ·  Epoch {epoch}')
                
                fig.canvas.draw()
                
                buf = fig.canvas.buffer_rgba()
                proc.stdin.write(memoryview(buf))
                
        finally:
            proc.stdin.close()
            proc.wait()
        
        plt.close(fig)
        
        # Reset rcParams to defaults
        plt.rcdefaults()
        
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
    def distance_matrix(
        self,
        method: str = "agg",
        func: Callable[[torch.Tensor], torch.Tensor] = torch.nanmean,
        max_iter: int = 1000,
        n_jobs: int = -1,
        convergence_tol: float = 1e-10,
        sigma_max: float = 3.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[str]]:
        """
        Compute the average distance matrix for terminal names across all trees.

        Aggregates distance matrices from multiple trees, handling missing values
        (when trees have different terminal sets) via the specified method.

        Args:
            method: Aggregation method - "agg" for direct aggregation, "fp" for
                    iterative fixed-point with Gaussian similarity weighting.
            func: Aggregation function (e.g., torch.nanmean, torch.nanmedian).
            max_iter: Maximum iterations for fixed-point method.
            n_jobs: Number of parallel jobs (-1 for all cores).
            convergence_tol: Convergence tolerance for fixed-point iteration.
            sigma_max: Maximum sigma for Gaussian similarity kernel.

        Returns:
            Tuple of (distance_matrix, confidence_scores, terminal_labels).
            Confidence scores indicate the fraction of trees contributing to each entry.

        Raises:
            ValueError: If no trees are available.
        """
        if not self.trees:
            self._log_info("No trees available for distance computation.")
            raise ValueError("No trees available for distance computation.")

        # Build global label set and index mapping
        labels = self.terminal_names()
        label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
        num_labels = len(labels)
        num_trees = len(self.trees)

        def compute_aligned_matrix(tree: "Tree") -> torch.Tensor:
            """Compute distance matrix for a single tree, aligned to global labels."""
            tree_labels = tree.terminal_names()
            indices = torch.tensor(
                [label_to_idx[lbl] for lbl in tree_labels], dtype=torch.long
            )
            aligned = torch.full((num_labels, num_labels), float("nan"))
            aligned[indices[:, None], indices] = tree.distance_matrix()[0]
            aligned.fill_diagonal_(0.0)
            return aligned

        def unwrap_result(result: torch.Tensor | tuple) -> torch.Tensor:
            """Extract tensor from aggregation result (handles nanmedian tuple return)."""
            return result[0] if isinstance(result, tuple) else result

        # Parallel computation of per-tree distance matrices
        aligned_matrices = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(compute_aligned_matrix)(tree) for tree in self.trees
        )
        dist_stack = torch.stack(aligned_matrices)  # (num_trees, num_labels, num_labels)

        # Precompute validity mask and confidence scores
        valid_mask = ~torch.isnan(dist_stack)
        confidence = valid_mask.float().mean(dim=0)

        if method == "fp":
            return self._fixed_point_aggregation(
                dist_stack=dist_stack,
                valid_mask=valid_mask,
                confidence=confidence,
                labels=labels,
                func=func,
                unwrap=unwrap_result,
                max_iter=max_iter,
                convergence_tol=convergence_tol,
                sigma_max=sigma_max,
            )

        # Standard aggregation
        avg_matrix = unwrap_result(func(dist_stack, dim=0))

        # Vectorized NaN interpolation using row/column means
        nan_mask = torch.isnan(avg_matrix)
        if nan_mask.any():
            row_means = unwrap_result(func(avg_matrix, dim=1))
            col_means = unwrap_result(func(avg_matrix, dim=0))
            fill_values = (row_means[:, None] + col_means[None, :]) / 2
            avg_matrix = torch.where(nan_mask, fill_values, avg_matrix)

        self._log_info("Distance matrix computation complete.")
        return avg_matrix, confidence, labels


    def _fixed_point_aggregation(
        self,
        dist_stack: torch.Tensor,
        valid_mask: torch.Tensor,
        confidence: torch.Tensor,
        labels: List[str],
        func: Callable,
        unwrap: Callable,
        max_iter: int,
        convergence_tol: float,
        sigma_max: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Fixed-point iteration with adaptive Gaussian similarity weighting.

        Iteratively refines the average matrix by weighting each tree's contribution
        based on its similarity to the current estimate.
        """
        num_trees = dist_stack.shape[0]
        device = dist_stack.device

        # Flatten spatial dimensions for vectorized similarity computation
        # Shape: (num_trees, num_labels * num_labels)
        dist_flat = dist_stack.view(num_trees, -1)
        valid_flat = valid_mask.view(num_trees, -1)

        # Precompute per-tree valid counts for normalization
        tree_valid_counts = valid_flat.sum(dim=1).clamp(min=1).float()

        # Initialize with simple aggregation
        avg_matrix = unwrap(func(dist_stack, dim=0))
        prev_weights = torch.zeros(num_trees, device=device)

        with tqdm(total=max_iter, desc="Fixed Point", unit="iter") as progress:
            for iteration in range(max_iter):
                # Adaptive sigma schedule: ramps from 0 to sigma_max
                sigma = min(2 * sigma_max * iteration / max_iter, sigma_max)

                # Vectorized Gaussian similarity computation across all trees
                avg_flat = avg_matrix.view(-1)

                # Compute squared differences only where both are valid
                diff = torch.where(valid_flat, dist_flat - avg_flat, torch.zeros_like(dist_flat))
                diff_norm_sq = (diff ** 2).sum(dim=1)

                avg_valid = avg_flat.expand(num_trees, -1)
                ref_sq = torch.where(valid_flat, avg_valid ** 2, torch.zeros_like(dist_flat))
                ref_norm_sq = ref_sq.sum(dim=1).clamp(min=1e-10)

                similarities = torch.exp(-sigma * diff_norm_sq / ref_norm_sq)
                weights = similarities / similarities.sum().clamp(min=1e-10)

                # Compute weighted average with validity-aware normalization
                # Expand weights to match dist_stack shape
                weights_expanded = weights.view(num_trees, 1, 1)
                weighted_contrib = torch.where(valid_mask, weights_expanded * dist_stack, torch.zeros_like(dist_stack))

                # Normalize by sum of weights at each position
                weight_sums = torch.where(valid_mask, weights_expanded.expand_as(dist_stack), torch.zeros_like(dist_stack))
                weight_sums = weight_sums.sum(dim=0).clamp(min=1e-10)

                avg_matrix = weighted_contrib.sum(dim=0) / weight_sums

                # Check convergence via weight stability
                weight_change = torch.sqrt(num_trees * ((weights - prev_weights) ** 2).sum())
                if weight_change < convergence_tol:
                    progress.update(max_iter - iteration)
                    break

                prev_weights = weights
                progress.update(1)

        return avg_matrix, confidence, labels
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
        list[float]
            A list containing the scales used for normalization of each tree.

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
        terminal_labels = self.terminal_names()
        num_terminals, num_trees = len(terminal_labels), len(self.trees)
        label_to_idx = {label: i for i, label in enumerate(terminal_labels)}
        
        # Hyperparameters based on problem size
        sqrt_num_terminals = np.sqrt(num_terminals)
        sqrt_num_trees = np.sqrt(num_trees)
        log_lr_start = -np.log10(num_terminals) + (1 if batch_mode else -1)
        log_lr_end = -np.log10(num_terminals)
        max_iterations = 10 * int(sqrt_num_terminals + 1) if batch_mode else 10 * num_terminals
        num_passes = int(num_terminals / sqrt_num_terminals + 1) if batch_mode else 1
        batch_size = int(sqrt_num_trees + 1) if batch_mode else num_trees

        # Precompute all distance matrices (vectorized construction)
        distance_matrices = torch.full((num_trees, num_terminals, num_terminals), float('nan'))
        for tree_idx, tree in enumerate(self.trees):
            tree_labels = tree.terminal_names()
            indices = torch.tensor([label_to_idx[lbl] for lbl in tree_labels], dtype=torch.long)
            idx_grid = indices[:, None], indices
            distance_matrices[tree_idx][idx_grid] = tree.distance_matrix()[0]
        distance_matrices.diagonal(dim1=-2, dim2=-1).fill_(0)

        # Precompute masks and valid counts
        valid_mask = ~torch.isnan(distance_matrices)
        valid_counts = valid_mask.sum(dim=0).clamp(min=1).float()
        distance_matrices = distance_matrices.nan_to_num_(0.0)
        
        # Initialize scales
        scales = torch.ones(num_trees, dtype=torch.float32)
        normalization_factor = 1.0 / (num_terminals ** 2)
        
        # Progress tracking
        num_batches = (num_trees + batch_size - 1) // batch_size
        total_iterations = num_passes * max_iterations * num_batches + (max_iterations if batch_mode else 0)
        progress = tqdm(total=total_iterations, desc="Normalizing", unit="iter")

        # Pre-compute learning rate schedule
        lr_schedule = 10 ** (log_lr_start + (log_lr_end - log_lr_start) * torch.arange(max_iterations) / max_iterations)

        def optimize_batch_scales(batch_idx, weighted_sum_excluding_batch, batch_valid_mask, batch_distances):
            """Optimize scales for a batch of trees using gradient descent."""
            batch_scale_sum = scales[batch_idx].sum()
            params = scales[batch_idx].clone().requires_grad_(True)
            optimizer = Adam([params], lr=lr_schedule[0].item())

            for iteration in range(max_iterations):
                optimizer.param_groups[0]['lr'] = lr_schedule[iteration].item()
                
                # Normalize parameters to maintain sum constraint
                normalized_params = torch.nn.functional.softplus(params)
                normalized_params = normalized_params * (batch_scale_sum / normalized_params.sum())
                
                # Compute weighted matrices and average
                weighted_batch = batch_distances * normalized_params[:, None, None]
                average_matrix = (weighted_batch.sum(dim=0) + weighted_sum_excluding_batch) / valid_counts
                
                # Frobenius norm loss (only on valid entries)
                residuals = (weighted_batch - average_matrix.unsqueeze(0)) * batch_valid_mask
                loss = residuals.pow(2).sum() * normalization_factor
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                progress.update(1)
                self._log_info(f"Iter {iteration}: Loss={loss.item():.6f} | LR={lr_schedule[iteration]:.2e}")
            
            scales[batch_idx] = normalized_params.detach()
            return average_matrix

        # Main optimization loop
        tree_indices = list(range(num_trees))
        for _ in range(num_passes):
            if batch_mode:
                random.shuffle(tree_indices)
            
            for batch_start in range(0, num_trees, batch_size):
                batch_idx = tree_indices[batch_start:batch_start + batch_size]
                other_idx = list(set(tree_indices) - set(batch_idx))
                
                weighted_sum_others = (distance_matrices[other_idx] * scales[other_idx, None, None]).sum(dim=0) if other_idx else 0
                optimize_batch_scales(batch_idx, weighted_sum_others, valid_mask[batch_idx], distance_matrices[batch_idx])

        # Final global optimization pass in batch mode
        if batch_mode:
            all_indices = list(range(num_trees))
            optimize_batch_scales(all_indices, 0, valid_mask, distance_matrices)

        progress.close()

        # Apply scales to tree edge lengths
        final_scales = scales.tolist()
        for tree_idx, tree in enumerate(self.trees):
            scale_factor = final_scales[tree_idx]
            for node in tree.contents.traverse_postorder():
                edge_length = node.get_edge_length()
                if edge_length is not None:
                    node.set_edge_length(edge_length * scale_factor)

        return final_scales
    ################################################################################################
    def _gen_video(self, fps: int = 10):
        """
        Generate a video of training metrics evolution for multi-tree hyperbolic embeddings.
        
        Optimized for large numbers of trees (100s-1000s) with smart visualization.
        Shows RMS RE evolution, weight evolution, learning rate, and cost.
        
        Args:
            fps: Frames per second for the output video.
        """
        import matplotlib
        matplotlib.use('Agg')
        import subprocess
        from matplotlib.cm import get_cmap
        from matplotlib.colors import Normalize
        
        # ═══════════════════════════════════════════════════════════════════════
        # AESTHETIC CONFIGURATION
        # ═══════════════════════════════════════════════════════════════════════
        COLORS = {
            'background': '#1a1a2e',
            'panel': '#1e2a4a',
            'grid': '#2a2a4a',
            'text': '#e8e8e8',
            'text_secondary': '#a0a0b0',
            'accent_primary': '#00d4ff',
            'accent_secondary': '#ff6b6b',
            'accent_tertiary': '#ffd93d',
            'accent_muted': '#6c757d',
        }
        
        plt.rcParams.update({
            'figure.facecolor': COLORS['background'],
            'figure.edgecolor': COLORS['background'],
            'axes.facecolor': COLORS['panel'],
            'axes.edgecolor': COLORS['grid'],
            'axes.labelcolor': COLORS['text'],
            'axes.titlecolor': COLORS['text'],
            'axes.grid': True,
            'axes.grid.axis': 'both',
            'axes.axisbelow': True,
            'axes.linewidth': 0.8,
            'axes.titleweight': 'bold',
            'axes.titlesize': 11,
            'axes.labelsize': 9,
            'axes.labelweight': 'medium',
            'grid.color': COLORS['grid'],
            'grid.linewidth': 0.4,
            'grid.alpha': 0.5,
            'xtick.color': COLORS['text_secondary'],
            'ytick.color': COLORS['text_secondary'],
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'text.color': COLORS['text'],
            'font.family': 'sans-serif',
            'font.size': 9,
            'legend.facecolor': COLORS['panel'],
            'legend.edgecolor': COLORS['grid'],
            'legend.fontsize': 8,
            'legend.framealpha': 0.9,
        })
        
        timestamp = self._current_time
        base = os.path.join(conf.OUTPUT_DIRECTORY, timestamp.strftime('%Y-%m-%d_%H-%M-%S'))
        
        # ═══════════════════════════════════════════════════════════════════════
        # LOAD METADATA AND DETERMINE NUMBER OF TREES
        # ═══════════════════════════════════════════════════════════════════════
        try:
            metadata = np.load(os.path.join(base, "metadata.npy"), allow_pickle=True).item()
            num_trees = metadata['num_trees']
        except FileNotFoundError:
            tree_dirs = sorted([d for d in os.listdir(base) if d.startswith('tree_')])
            num_trees = len(tree_dirs)
        
        # ═══════════════════════════════════════════════════════════════════════
        # DATA LOADING - AGGREGATE
        # ═══════════════════════════════════════════════════════════════════════
        weights = -np.load(os.path.join(base, "weight_exponents.npy"))
        lrs = np.log10(np.load(os.path.join(base, "learning_rates.npy")) + conf.EPSILON)
        aggregate_costs = np.load(os.path.join(base, "costs.npy"))

        
        # Load scales for scale learning visualization
        try:
            scales = np.load(os.path.join(base, "scales.npy"))
        except FileNotFoundError:
            scales = None
        
        n_frames = len(weights)
        epochs = np.arange(1, n_frames + 1)
        
        # Determine if hyperbolic (scale learning applies)
        is_hyperbolic = scales is not None and not np.all(scales == 1)
        
        # ═══════════════════════════════════════════════════════════════════════
        # PRECOMPUTE SCALE CHANGE DETECTION 
        # ═══════════════════════════════════════════════════════════════════════
        if is_hyperbolic:
            scale_active = scales.astype(bool)
            scale_changed = np.concatenate([[True], np.diff(scales) != 0])
            scale_changing_mask = scale_active & scale_changed
            scale_unchanged_mask = scale_active & ~scale_changed
        
        # ═══════════════════════════════════════════════════════════════════════
        # DATA LOADING - PER TREE (only RMS values, skip full RE matrices)
        # ═══════════════════════════════════════════════════════════════════════
        self._log_info(f"Loading data for {num_trees} trees...")
        
        all_rms_vals = []
        all_costs = []
        
        for tree_idx in range(num_trees):
            tree_dir = os.path.join(base, f"tree_{tree_idx}")
            
            # Load per-tree costs
            tree_costs = np.load(os.path.join(tree_dir, "costs.npy"))
            all_costs.append(tree_costs)
            
            # Load RMS RE values directly
            rms_vals = np.load(os.path.join(tree_dir, "rmse.npy"))
            all_rms_vals.append(rms_vals)
        
        all_rms_vals = np.array(all_rms_vals)  # Shape: (num_trees, n_frames)
        all_costs = np.array(all_costs)  # Shape: (num_trees, n_frames)
        
        # Find min and max RMS trees (based on final epoch)
        final_rms = all_rms_vals[:, -1]
        min_rms_tree = np.argmin(final_rms)
        max_rms_tree = np.argmax(final_rms)
        
        # Compute axis limits
        rms_min = np.nanmin(all_rms_vals) * 0.9
        rms_max = np.nanmax(all_rms_vals) * 1.1
        lr_min, lr_max = lrs.min() - 0.1, lrs.max() + 0.1
        weight_min, weight_max = weights.min() * 0.95, weights.max() * 1.05
        cost_min = min(np.nanmin(aggregate_costs), np.nanmin(all_costs)) * 0.9
        cost_max = max(np.nanmax(aggregate_costs), np.nanmax(all_costs)) * 1.1
        
        # ═══════════════════════════════════════════════════════════════════════
        # SETUP OUTPUT
        # ═══════════════════════════════════════════════════════════════════════
        out_dir = os.path.join(conf.OUTPUT_VIDEO_DIRECTORY, timestamp.strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(out_dir, exist_ok=True)
        vid_path = os.path.join(out_dir, 're_evolution_multi.mp4')
        
        self._log_info(f"Creating video for {num_trees} trees. Please be patient.")
        
        # ═══════════════════════════════════════════════════════════════════════
        # FIGURE SETUP - 2x2 GRID
        # ═══════════════════════════════════════════════════════════════════════
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
        ax_rms, ax_weights = axes[0]
        ax_lr, ax_cost = axes[1]
        
        # ─────────────────────────────────────────────────────────────────────
        # Color map for trees - USE PLASMA (no blue, visible on blue background)
        # ─────────────────────────────────────────────────────────────────────
        cmap = get_cmap('plasma')
        norm = Normalize(vmin=0, vmax=num_trees - 1)
        tree_colors = [cmap(norm(i)) for i in range(num_trees)]
        
        # ─────────────────────────────────────────────────────────────────────
        # RMS RE Plot (top-left)
        # ─────────────────────────────────────────────────────────────────────
        lines_rms = []
        for tree_idx in range(num_trees):
            alpha = 0.3 if tree_idx not in [min_rms_tree, max_rms_tree] else 1.0
            lw = 0.5 if tree_idx not in [min_rms_tree, max_rms_tree] else 2.0
            line, = ax_rms.plot([], [], color=tree_colors[tree_idx], linewidth=lw, alpha=alpha)
            lines_rms.append(line)
        
        ax_rms.set_xlim(1, n_frames)
        ax_rms.set_ylim(rms_min, rms_max)
        ax_rms.set_yscale('log')
        ax_rms.set_xlabel('Epoch', fontweight='medium')
        ax_rms.set_ylabel('Median RE (log scale)', fontweight='medium')
        ax_rms.set_title(f'Median Relative Error Evolution ({num_trees} Trees)', fontsize=12, pad=10)
        
        # Annotations for min/max trees with small text and background box
        annot_min = ax_rms.annotate('', xy=(0, 0), fontsize=7, fontweight='bold',
                                     color='#50fa7b',
                                     bbox=dict(boxstyle='round,pad=0.2', facecolor=COLORS['panel'],
                                               edgecolor='#50fa7b', linewidth=0.5, alpha=0.9))
        annot_max = ax_rms.annotate('', xy=(0, 0), fontsize=7, fontweight='bold',
                                     color='#ff5555',
                                     bbox=dict(boxstyle='round,pad=0.2', facecolor=COLORS['panel'],
                                               edgecolor='#ff5555', linewidth=0.5, alpha=0.9))
        
        # Add colorbar for tree index (using plasma)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_rms, fraction=0.046, pad=0.04, shrink=0.8)
        cbar.set_label('Tree Index', fontsize=9, fontweight='medium')
        cbar.ax.tick_params(labelsize=7)
        cbar.outline.set_edgecolor(COLORS['grid'])
        cbar.outline.set_linewidth(0.5)
        
        # ─────────────────────────────────────────────────────────────────────
        # Weight Evolution Plot (top-right) - WITH SCALE LEARNING INDICATORS
        # ─────────────────────────────────────────────────────────────────────
        line_weights, = ax_weights.plot([], [], color=COLORS['accent_primary'],
                                        linewidth=2, marker='o', markersize=3,
                                        markerfacecolor=COLORS['accent_tertiary'],
                                        markeredgecolor='white', markeredgewidth=0.1,
                                        alpha=1, zorder=2)
        
        line_weights_scaled = None
        line_weights_unchanged = None
        if is_hyperbolic:
            line_weights_scaled, = ax_weights.plot([], [], linestyle='none',
                                                   marker='o', markersize=5,
                                                   markerfacecolor='#ff3333',
                                                   markeredgecolor='white', markeredgewidth=0.1,
                                                   label='Scale Learning Enabled', zorder=4)
            line_weights_unchanged, = ax_weights.plot([], [], linestyle='none',
                                                      marker='o', markersize=3,
                                                      markerfacecolor=COLORS['accent_primary'],
                                                      markeredgecolor='white', markeredgewidth=0.1,
                                                      label='Scale Learning Disabled', zorder=3)
            ax_weights.legend(loc='upper right', framealpha=0.9, fontsize=7)
        
        ax_weights.set_xlim(1, n_frames)
        ax_weights.set_ylim(weight_min, weight_max)
        ax_weights.set_xlabel('Epoch', fontweight='medium')
        ax_weights.set_ylabel('− Weight Exponent', fontweight='medium')
        ax_weights.set_title('Weight Evolution', fontsize=11, pad=8)
        
        # ─────────────────────────────────────────────────────────────────────
        # Learning Rate Plot (bottom-left)
        # ─────────────────────────────────────────────────────────────────────
        line_lr, = ax_lr.plot([], [], color='#50fa7b', linewidth=2,
                              marker='o', markersize=3,
                              markerfacecolor=COLORS['accent_primary'],
                              markeredgecolor='white', markeredgewidth=0.1)
        ax_lr.set_xlim(1, n_frames)
        ax_lr.set_ylim(lr_min, lr_max)
        ax_lr.set_xlabel('Epoch', fontweight='medium')
        ax_lr.set_ylabel('log₁₀(Learning Rate)', fontweight='medium')
        ax_lr.set_title('Learning Rate Schedule', fontsize=11, pad=8)
        
        # ─────────────────────────────────────────────────────────────────────
        # Cost Evolution Plot (bottom-right) - SEMILOG Y (using plasma for trees)
        # ─────────────────────────────────────────────────────────────────────
        lines_cost = []
        for tree_idx in range(num_trees):
            alpha = 0.2
            lw = 0.4
            line, = ax_cost.plot([], [], color=tree_colors[tree_idx], linewidth=lw, alpha=alpha)
            lines_cost.append(line)
        
        line_cost_agg, = ax_cost.plot([], [], color='white', linewidth=2.5,
                                      label='Aggregate', linestyle='-')
        
        ax_cost.set_xlim(1, n_frames)
        ax_cost.set_ylim(cost_min, cost_max)
        ax_cost.set_yscale('log')
        ax_cost.set_xlabel('Epoch', fontweight='medium')
        ax_cost.set_ylabel('Cost (log scale)', fontweight='medium')
        ax_cost.set_title('Cost Evolution', fontsize=11, pad=8)
        ax_cost.legend(loc='upper right', fontsize=8)

        # Add colorbar for tree index (matching RMS plot)
        sm_cost = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm_cost.set_array([])
        cbar_cost = fig.colorbar(sm_cost, ax=ax_cost, fraction=0.046, pad=0.04, shrink=0.8)
        cbar_cost.set_label('Tree Index', fontsize=9, fontweight='medium')
        cbar_cost.ax.tick_params(labelsize=7)
        cbar_cost.outline.set_edgecolor(COLORS['grid'])
        cbar_cost.outline.set_linewidth(0.5)
        
        # Add borders (small linewidth)
        for ax in [ax_rms, ax_weights, ax_lr, ax_cost]:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('#4a4a6a')
                spine.set_linewidth(1.0)
        
        fig.tight_layout()
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        
        # ═══════════════════════════════════════════════════════════════════════
        # DIRECT FFMPEG PIPE
        # ═══════════════════════════════════════════════════════════════════════
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'rgba',
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            vid_path
        ]
        
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        
        # ═══════════════════════════════════════════════════════════════════════
        # RENDER FRAMES
        # ═══════════════════════════════════════════════════════════════════════
        try:
            for epoch in range(n_frames):
                x_data = epochs[:epoch + 1]
                
                # Update RMS lines for all trees
                for tree_idx in range(num_trees):
                    lines_rms[tree_idx].set_data(x_data, all_rms_vals[tree_idx, :epoch + 1])
                
                # Update annotations for min/max with offset to avoid overlap
                current_epoch_idx = epoch
                min_val = all_rms_vals[min_rms_tree, current_epoch_idx]
                max_val = all_rms_vals[max_rms_tree, current_epoch_idx]
                
                annot_min.set_text(f'Min: T{min_rms_tree}')
                annot_min.xy = (epoch + 1, min_val)
                annot_min.set_position((epoch + 1.5, min_val * 0.85))
                
                annot_max.set_text(f'Max: T{max_rms_tree}')
                annot_max.xy = (epoch + 1, max_val)
                annot_max.set_position((epoch + 1.5, max_val * 1.15))
                
                # Update weight line
                line_weights.set_data(x_data, weights[:epoch + 1])
                
                # Update scale learning markers (if hyperbolic)
                if is_hyperbolic:
                    changing_mask = scale_changing_mask[:epoch + 1]
                    line_weights_scaled.set_data(x_data[changing_mask], weights[:epoch + 1][changing_mask])
                    
                    unchanged_mask = scale_unchanged_mask[:epoch + 1]
                    line_weights_unchanged.set_data(x_data[unchanged_mask], weights[:epoch + 1][unchanged_mask])
                
                # Update learning rate line
                line_lr.set_data(x_data, lrs[:epoch + 1])
                
                # Update cost lines
                for tree_idx in range(num_trees):
                    lines_cost[tree_idx].set_data(x_data, all_costs[tree_idx, :epoch + 1])
                line_cost_agg.set_data(x_data, aggregate_costs[:epoch + 1])
                
                fig.canvas.draw()
                buf = fig.canvas.buffer_rgba()
                proc.stdin.write(memoryview(buf))
                
        finally:
            proc.stdin.close()
            proc.wait()
        
        plt.close(fig)
        plt.rcdefaults()
        
        self._log_info(f"Multi-tree video created: {vid_path}")
    
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
        from functools import partial
        import gc

        if dim is None:
            raise ValueError("The 'dimension' parameter is required.")

        # Extract and set embedding parameters - use tuple for faster lookup
        _defaults = (
            ('precise_opt', conf.ENABLE_ACCURATE_OPTIMIZATION),
            ('epochs', conf.TOTAL_EPOCHS),
            ('lr_init', conf.INITIAL_LEARNING_RATE),
            ('dist_cutoff', conf.MAX_RANGE),
            ('save_mode', conf.ENABLE_SAVE_MODE),
            ('export_video', conf.ENABLE_VIDEO_EXPORT),
            ('scale_fn', None),
            ('lr_fn', None),
            ('weight_exp_fn', None),
            ('normalize', False)
        )
        params = {key: kwargs.get(key, default) for key, default in _defaults}

        if params['normalize']:
            self.normalize(batch_mode=params['precise_opt'])

        params['save_mode'] |= params['export_video']
        params['export_video'] &= params['precise_opt']

        # Pre-compute values used across all trees
        n_trees = len(self.trees)
        
        # Determine optimal number of jobs based on tree count
        n_jobs = min(n_trees, os.cpu_count())

        try:
            if geometry == 'hyperbolic':
                self._log_info("Starting the Hyperbolic embedding process.")
                scale_factor = params['dist_cutoff'] / self.distance_matrix()[0].max()
                neg_curv = -(scale_factor ** 2)

                # Define worker function inside to capture scale_factor, dim, neg_curv
                def _process_hyperbolic_tree(tree_idx_and_tree):
                    idx, tree = tree_idx_and_tree
                    dist_mat = tree.distance_matrix()[0]
                    points = utils.naive_embedding(
                        dist_mat * scale_factor, 
                        dim, 
                        geometry='hyperbolic'
                    )
                    return idx, dist_mat, embedding.LoidEmbedding(
                        points=points, 
                        labels=tree.terminal_names(), 
                        curvature=neg_curv
                    )

                # Parallel processing of all trees - single pass
                self._log_info(f"Processing {n_trees} trees in parallel...")
                results = Parallel(
                    n_jobs=n_jobs,
                    backend='loky',
                    return_as='generator'  # Memory efficient - yields results as ready
                )(
                    delayed(_process_hyperbolic_tree)((i, tree)) 
                    for i, tree in enumerate(self.trees)
                )

                # Collect results maintaining order
                dist_mats = [None] * n_trees
                embeddings_list = [None] * n_trees
                
                for idx, dist_mat, emb in results:
                    dist_mats[idx] = dist_mat
                    embeddings_list[idx] = emb
                    self._log_info(f"Naive Hyperbolic embedding completed for tree {idx + 1}/{n_trees}")
                # Build MultiEmbedding efficiently
                multi_embeddings = embedding.MultiEmbedding()
                for emb in embeddings_list:
                    multi_embeddings.append(emb)
                del embeddings_list  # Free memory
                gc.collect()
                self._log_info("Hyperbolic embedding (naive) process completed for all trees.")
                
                if params['precise_opt']:
                    self._log_info("Refining embeddings with precise optimization.")
                    pts_list, curvature = utils.precise_multiembedding(
                        dist_mats, 
                        multi_embeddings, 
                        geometry="hyperbolic",
                        log_fn=self._log_info, 
                        time_stamp=self._current_time, 
                        **params
                    )
                    
                    # Rebuild MultiEmbedding with optimized results
                    multi_embeddings = embedding.MultiEmbedding()
                    tree_labels = [tree.terminal_names() for tree in self.trees]  # Cache labels
                    for pts, labels in zip(pts_list, tree_labels):
                        multi_embeddings.append(
                            embedding.LoidEmbedding(points=pts, labels=labels, curvature=curvature)
                        )
                    del pts_list, dist_mats
                    gc.collect()
                    self._log_info("Precise hyperbolic embedding process completed for all trees.")
                else:
                    del dist_mats
                    gc.collect()
            # ─────────────────────────────────────────────────────────────────────────────
            else:
                # Euclidean embedding
                self._log_info("Starting the Euclidean embedding process.")

                # Define worker function for naive Euclidean embedding
                def _process_euclidean_tree(tree_idx_and_tree):
                    idx, tree = tree_idx_and_tree
                    dist_mat = tree.distance_matrix()[0]
                    points = utils.naive_embedding(dist_mat, dim, geometry='euclidean')
                    return idx, dist_mat, embedding.EuclideanEmbedding(
                        points=points, 
                        labels=tree.terminal_names()
                    )

                # Parallel processing of all trees - naive embeddings
                self._log_info(f"Computing naive Euclidean embeddings for {n_trees} trees in parallel...")
                results = Parallel(
                    n_jobs=n_jobs,
                    backend='loky',
                    return_as='generator'
                )(
                    delayed(_process_euclidean_tree)((i, tree)) 
                    for i, tree in enumerate(self.trees)
                )

                # Collect results maintaining order
                dist_mats = [None] * n_trees
                embeddings_list = [None] * n_trees
                
                for idx, dist_mat, emb in results:
                    dist_mats[idx] = dist_mat
                    embeddings_list[idx] = emb
                    self._log_info(f"Naive Euclidean embedding completed for tree {idx + 1}/{n_trees}")

                # Build MultiEmbedding
                multi_embeddings = embedding.MultiEmbedding()
                for emb in embeddings_list:
                    multi_embeddings.append(emb)
                
                del embeddings_list
                gc.collect()
                self._log_info("Euclidean embedding (naive) process completed for all trees.")
                
                if params['precise_opt']:
                    self._log_info("Refining Euclidean embeddings with precise optimization (parallel per-tree).")
                    pts_list, _ = utils.precise_multiembedding(
                        dist_mats, 
                        multi_embeddings, 
                        geometry="euclidean",
                        log_fn=self._log_info, 
                        time_stamp=self._current_time, 
                        **params
                    )
                    
                    # Rebuild MultiEmbedding with optimized results
                    multi_embeddings = embedding.MultiEmbedding()
                    tree_labels = [tree.terminal_names() for tree in self.trees]
                    for pts, labels in zip(pts_list, tree_labels):
                        multi_embeddings.append(
                            embedding.EuclideanEmbedding(points=pts, labels=labels)
                        )
                    del pts_list, dist_mats
                    gc.collect()
                    self._log_info("Precise Euclidean embedding process completed for all trees.")
                else:
                    del dist_mats
                    gc.collect()
        except Exception as e:
            self._log_info(f"Error during multi_embedding: {e}")
            raise

        # Save results
        directory = f"{conf.OUTPUT_DIRECTORY}/{self._current_time.strftime('%Y-%m-%d_%H-%M-%S')}"
        filepath = f"{directory}/{geometry}_multiembedding_{dim}d.pkl"
        os.makedirs(directory, exist_ok=True)
        
        try:
            with open(filepath, 'wb') as file:
                pickle.dump(multi_embeddings, file, protocol=pickle.HIGHEST_PROTOCOL)
            self._log_info(f"Object successfully saved to {filepath}")
        except (IOError, pickle.PicklingError, Exception) as e:
            self._log_info(f"Error while saving object: {e}")
            raise
        if params['export_video']:
            self._gen_video(fps=params['epochs'] // conf.VIDEO_LENGTH)

        return multi_embeddings