"""
Phylogenetic tree data structures for embedding and distance computation.

This module provides two primary classes:
- Tree: Single phylogenetic tree with embedding and distance operations.
- MultiTree: Collection of trees with aggregated distance computation and
  batch embedding capabilities.

Both classes support hyperbolic and Euclidean geometric embeddings with
optional GPU-accelerated optimization.
"""

import os
import gc
import copy
import pickle
import random
import subprocess
from datetime import datetime
from typing import (
    Union, Optional, List, Callable, Tuple, Iterator
)

import numpy as np
import torch
import treeswift as ts
from tqdm import tqdm
from torch.optim import Adam
from joblib import Parallel, delayed

import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for server environments
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize

import htree.conf as conf
import htree.utils as utils
import htree.embedding as embedding
from htree.logger import get_logger, logging_enabled, get_time


# =============================================================================
# Tree: Single Phylogenetic Tree
# =============================================================================

class Tree:
    """
    Phylogenetic tree with embedding and distance computation capabilities.

    Wraps a treeswift.Tree object with additional functionality for:
    - Computing pairwise distance matrices
    - Normalizing branch lengths
    - Embedding into hyperbolic or Euclidean spaces
    - Generating optimization visualization videos

    Attributes
    ----------
    name : str
        Identifier for this tree instance.
    contents : treeswift.Tree
        Underlying tree structure.

    Examples
    --------
    >>> tree = Tree("path/to/tree.newick")
    >>> tree = Tree("my_tree", treeswift_tree_object)
    >>> dist_matrix, labels = tree.distance_matrix()
    >>> emb = tree.embed(dim=3, geometry='hyperbolic')
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a Tree from a file path or (name, treeswift.Tree) pair.

        Parameters
        ----------
        *args : str or (str, treeswift.Tree)
            Either a single file path string, or a tuple of (name, tree).

        Raises
        ------
        ValueError
            If arguments do not match expected patterns.
        FileNotFoundError
            If the specified file path does not exist.
        """
        self._timestamp = get_time() or datetime.now()

        if len(args) == 1 and isinstance(args[0], str):
            filepath = args[0]
            self.name = os.path.basename(filepath)
            self.contents = self._load_tree(filepath)
            self._log(f"Loaded tree from file: {filepath}")

        elif len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], ts.Tree):
            self.name, self.contents = args
            self._log(f"Initialized tree: {self.name}")

        else:
            raise ValueError(
                "Expected a file path (str) or (name: str, tree: treeswift.Tree) pair."
            )

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _log(self, message: str) -> None:
        """Log message if global logging is enabled."""
        if logging_enabled():
            get_logger().info(message)

    def _load_tree(self, filepath: str) -> ts.Tree:
        """Load tree from Newick file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Tree file not found: {filepath}")
        return ts.read_tree_newick(filepath)

    @classmethod
    def _from_contents(cls, name: str, contents: ts.Tree) -> 'Tree':
        """Factory method to create Tree from existing treeswift.Tree."""
        instance = cls(name, contents)
        instance._log(f"Tree created: {name}")
        return instance

    def __repr__(self) -> str:
        return f"Tree({self.name})"

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def update_time(self) -> None:
        """Update internal timestamp to current time."""
        self._timestamp = datetime.now()
        self._log("Timestamp updated.")

    def copy(self) -> 'Tree':
        """
        Create a deep copy of this tree.

        Returns
        -------
        Tree
            Independent copy with all attributes duplicated.
        """
        tree_copy = copy.deepcopy(self)
        self._log(f"Copied tree: {self.name}")
        return tree_copy

    def save(self, filepath: str, fmt: str = 'newick') -> None:
        """
        Save tree to file.

        Parameters
        ----------
        filepath : str
            Output file path.
        fmt : str, default='newick'
            Output format (only 'newick' currently supported).

        Raises
        ------
        ValueError
            If format is not supported.
        """
        if fmt.lower() != 'newick':
            self._log(f"Save failed: unsupported format '{fmt}'")
            raise ValueError(f"Unsupported format: {fmt}")

        self.contents.write_tree_newick(filepath)
        self._log(f"Saved tree '{self.name}' to {filepath}")

    def terminal_names(self) -> List[str]:
        """
        Get names of all leaf (terminal) nodes.

        Returns
        -------
        List[str]
            Leaf node labels in traversal order.
        """
        labels = list(self.contents.labels(leaves=True, internal=False))
        self._log(f"Retrieved {len(labels)} terminal names for '{self.name}'")
        return labels

    def distance_matrix(self) -> Tuple[torch.Tensor, List[str]]:
        """
        Compute pairwise patristic distances between all leaves.

        Returns
        -------
        dist_matrix : torch.Tensor
            Shape (n, n) symmetric matrix of pairwise distances.
        labels : List[str]
            Leaf names corresponding to matrix indices.
        """
        labels = self.terminal_names()
        n = len(labels)

        # Build distance lookup from treeswift
        dist_dict = self.contents.distance_matrix(leaf_labels=True)
        label_dists = [dist_dict.get(lbl, {}) for lbl in labels]

        # Vectorized construction
        row_idx = np.repeat(np.arange(n), n)
        col_idx = np.tile(np.arange(n), n)
        distances = np.array([
            label_dists[i].get(labels[j], 0.0)
            for i, j in zip(row_idx, col_idx)
        ])

        dist_matrix = torch.tensor(distances, dtype=torch.float32).reshape(n, n)
        self._log(f"Distance matrix computed for '{self.name}': {n} terminals")
        return dist_matrix, labels

    def diameter(self) -> torch.Tensor:
        """
        Compute tree diameter (maximum pairwise distance).

        Returns
        -------
        torch.Tensor
            Scalar tensor containing the diameter value.
        """
        diam = torch.tensor(self.contents.diameter())
        self._log(f"Tree diameter: {diam.item():.6f}")
        return diam

    def normalize(self) -> None:
        """
        Scale branch lengths so tree diameter equals 1.

        Modifies tree in-place. Does nothing if diameter is zero.
        """
        diam = self.contents.diameter()
        if np.isclose(diam, 0.0):
            self._log("Diameter is zero; skipping normalization.")
            return

        scale = 1.0 / diam
        for node in self.contents.traverse_postorder():
            edge_len = node.get_edge_length()
            if edge_len is not None:
                node.set_edge_length(edge_len * scale)
        self._log(f"Normalized tree with scale factor: {scale:.6f}")

    def embed(
        self,
        dim: int,
        geometry: str = 'hyperbolic',
        **kwargs
    ) -> 'embedding.LoidEmbedding | embedding.EuclideanEmbedding':
        """
        Embed tree into geometric space.

        Parameters
        ----------
        dim : int
            Target embedding dimension.
        geometry : {'hyperbolic', 'euclidean'}
            Target geometry type.
        **kwargs : dict
            Optional parameters:
            - precise_opt : bool - Enable optimization refinement.
            - epochs : int - Optimization epochs.
            - lr_init : float - Initial learning rate.
            - dist_cutoff : float - Distance scaling cutoff.
            - export_video : bool - Generate optimization video.
            - save_mode : bool - Save intermediate states.
            - scale_fn, lr_fn, weight_exp_fn : Callable - Custom schedules.

        Returns
        -------
        embedding.LoidEmbedding or embedding.EuclideanEmbedding
            Geometric embedding with points and labels.

        Raises
        ------
        ValueError
            If dim is None.
        """
        if dim is None:
            raise ValueError("Parameter 'dim' is required.")

        # Parameter defaults
        params = {
            'precise_opt': kwargs.get('precise_opt', conf.ENABLE_ACCURATE_OPTIMIZATION),
            'epochs': kwargs.get('epochs', conf.TOTAL_EPOCHS),
            'lr_init': kwargs.get('lr_init', conf.INITIAL_LEARNING_RATE),
            'dist_cutoff': kwargs.get('dist_cutoff', conf.MAX_RANGE),
            'export_video': kwargs.get('export_video', conf.ENABLE_VIDEO_EXPORT),
            'save_mode': kwargs.get('save_mode', conf.ENABLE_SAVE_MODE),
            'scale_fn': kwargs.get('scale_fn'),
            'lr_fn': kwargs.get('lr_fn'),
            'weight_exp_fn': kwargs.get('weight_exp_fn'),
            'curvature': kwargs.get('curvature'),
        }
        params['save_mode'] |= params['export_video']
        params['export_video'] &= params['precise_opt']


        try:
            dist_matrix = self.distance_matrix()[0]
            curvature = None

            # Hyperbolic: scale distances and compute curvature
            if geometry == 'hyperbolic':
                if params['curvature'] is not None:
                    if params['curvature'] >= 0:
                        self._log(f"Wrong input curvature. It has to be negative.")
                        print(f"Wrong input curvature. It has to be negative.")
                        return None
                    curvature = params['curvature']
                    params['scale_fn'] = lambda x1, x2, x3: False
                    scale = np.sqrt(np.abs(curvature))
                else:
                    scale = params['dist_cutoff'] / self.diameter()
                    curvature = -(scale ** 2)
                dist_matrix = dist_matrix * scale
                
            # Naive embedding initialization
            self._log(f"Computing naive {geometry} embedding...")
            points = utils.naive_embedding(dist_matrix, dim, geometry=geometry)
            self._log(f"Naive {geometry} embedding complete.")

            # Precise optimization refinement
            if params['precise_opt']:
                self._log(f"Refining with precise {geometry} optimization...")
                opt_result = utils.precise_embedding(
                    dist_matrix, dim,
                    geometry=geometry,
                    init_pts=points,
                    log_fn=self._log,
                    time_stamp=self._timestamp,
                    **params
                )
                if geometry == 'hyperbolic':
                    points, opt_scale = opt_result
                    curvature *= opt_scale ** 2
                else:
                    points = opt_result
                self._log(f"Precise {geometry} embedding complete.")

            # Construct embedding object
            labels = self.terminal_names()
            if geometry == 'hyperbolic':
                result = embedding.LoidEmbedding(
                    points=points, labels=labels, curvature=curvature
                )
            else:
                result = embedding.EuclideanEmbedding(
                    points=points, labels=labels
                )

        except Exception as e:
            self._log(f"Embedding error: {e}")
            raise

        # Save result
        self._save_embedding(result, geometry, dim)

        if params['export_video']:
            self._generate_video(fps=params['epochs'] // conf.VIDEO_LENGTH)

        return result

    def _save_embedding(
        self,
        emb: 'embedding.LoidEmbedding | embedding.EuclideanEmbedding',
        geometry: str,
        dim: int
    ) -> None:
        """Save embedding object to timestamped directory."""
        out_dir = os.path.join(
            conf.OUTPUT_DIRECTORY,
            self._timestamp.strftime('%Y-%m-%d_%H-%M-%S')
        )
        os.makedirs(out_dir, exist_ok=True)

        filepath = os.path.join(out_dir, f"{geometry}_embedding_{dim}d.pkl")
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(emb, f, protocol=pickle.HIGHEST_PROTOCOL)
            self._log(f"Embedding saved to {filepath}")
        except (IOError, pickle.PicklingError) as e:
            self._log(f"Save error: {e}")
            raise

    def _generate_video(self, fps: int = 10) -> None:
        """
        Generate MP4 video of optimization evolution.

        Renders relative error heatmaps, distance matrix, and training
        metrics (RMS, learning rate, weight evolution) frame by frame.

        Parameters
        ----------
        fps : int, default=10
            Output video frame rate.
        """
        # Theme configuration
        THEME = {
            'background': '#1a1a2e',
            'panel': '#1e2a4a',
            'grid': '#2a2a4a',
            'text': '#e8e8e8',
            'text_dim': '#a0a0b0',
            'accent': '#00d4ff',
            'accent_alt': '#ff6b6b',
            'highlight': '#ffd93d',
        }

        plt.rcParams.update({
            'figure.facecolor': THEME['background'],
            'figure.edgecolor': THEME['background'],
            'axes.facecolor': THEME['panel'],
            'axes.edgecolor': THEME['grid'],
            'axes.labelcolor': THEME['text'],
            'axes.titlecolor': THEME['text'],
            'axes.grid': True,
            'axes.axisbelow': True,
            'axes.linewidth': 0.8,
            'axes.titleweight': 'bold',
            'axes.titlesize': 11,
            'axes.labelsize': 9,
            'grid.color': THEME['grid'],
            'grid.linewidth': 0.4,
            'grid.alpha': 0.5,
            'xtick.color': THEME['text_dim'],
            'ytick.color': THEME['text_dim'],
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'text.color': THEME['text'],
            'font.family': 'sans-serif',
            'font.size': 9,
            'legend.facecolor': THEME['panel'],
            'legend.edgecolor': THEME['grid'],
            'legend.fontsize': 8,
        })

        base_dir = os.path.join(
            conf.OUTPUT_DIRECTORY,
            self._timestamp.strftime('%Y-%m-%d_%H-%M-%S')
        )

        # Load optimization data
        weights = -np.load(os.path.join(base_dir, "weight_exponents.npy"))
        lrs = np.log10(np.load(os.path.join(base_dir, "learning_rates.npy")) + conf.EPSILON)

        try:
            scales = np.load(os.path.join(base_dir, "scales.npy"))
        except FileNotFoundError:
            scales = None

        re_files = sorted(
            [f for f in os.listdir(base_dir) if f.startswith('RE') and f.endswith('.npy')],
            key=lambda f: int(f.split('_')[1].split('.')[0])
        )[:len(weights)]

        n_frames = len(re_files)

        # Parallel load RE matrices
        re_matrices = Parallel(n_jobs=-1, prefer="threads")(
            delayed(np.load)(os.path.join(base_dir, f)) for f in re_files
        )
        re_stack = np.stack(re_matrices, axis=0)
        del re_matrices

        # Compute statistics
        triu_idx = np.triu_indices(re_stack.shape[1], k=1)
        triu_vals = re_stack[:, triu_idx[0], triu_idx[1]]

        log_re_min = np.log10(np.nanmin(triu_vals) + conf.EPSILON)
        log_re_max = np.log10(np.nanmax(triu_vals) + conf.EPSILON)
        rms_vals = np.sqrt(np.nanmean(triu_vals ** 2, axis=1))
        del triu_vals

        rms_bounds = (rms_vals.min() * 0.9, rms_vals.max() * 1.1)
        lr_bounds = (lrs.min() - 0.1, lrs.max() + 0.1)

        # Prepare distance matrix display
        log_dist = np.log10(self.distance_matrix()[0].numpy() + conf.EPSILON)
        diag_mask = np.eye(log_dist.shape[0], dtype=bool)
        masked_dist = np.where(diag_mask, np.nan, log_dist)

        # Log-transform RE matrices
        log_re_stack = np.log10(re_stack + conf.EPSILON)
        log_re_stack[:, diag_mask] = np.nan
        del re_stack

        epochs = np.arange(1, n_frames + 1)
        is_hyperbolic = scales is not None and not np.all(scales == 1)

        # Precompute scale-learning masks
        if is_hyperbolic:
            scale_active = scales.astype(bool)
            scale_changed = np.concatenate([[True], np.diff(scales) != 0])
            mask_changing = scale_active & scale_changed
            mask_unchanged = scale_active & ~scale_changed

        # Setup output
        out_dir = os.path.join(
            conf.OUTPUT_VIDEO_DIRECTORY,
            self._timestamp.strftime('%Y-%m-%d_%H-%M-%S')
        )
        os.makedirs(out_dir, exist_ok=True)
        video_path = os.path.join(out_dir, 're_evolution.mp4')

        self._log("Generating optimization video...")

        # Create figure
        fig = plt.figure(figsize=(14, 12), dpi=100)
        gs = GridSpec(4, 2, height_ratios=[1, 1, 2, 2], hspace=0.35, wspace=0.25)

        ax_rms = fig.add_subplot(gs[0, :])
        ax_weight = fig.add_subplot(gs[1, 0])
        ax_lr = fig.add_subplot(gs[1, 1])
        ax_re = fig.add_subplot(gs[2:, 0])
        ax_dist = fig.add_subplot(gs[2:, 1])

        # Style axes borders
        for ax in [ax_rms, ax_weight, ax_lr, ax_re, ax_dist]:
            for spine in ax.spines.values():
                spine.set_edgecolor('#4a4a6a')
                spine.set_linewidth(1.5)

        # RMS plot
        line_rms, = ax_rms.plot(
            [], [], color=THEME['accent'], linewidth=2,
            marker='o', markersize=5, markerfacecolor=THEME['accent'],
            markeredgecolor='white', markeredgewidth=0.1
        )
        ax_rms.set_xlim(1, n_frames)
        ax_rms.set_ylim(*rms_bounds)
        ax_rms.set_yscale('log')
        ax_rms.set_xlabel('Epoch')
        ax_rms.set_ylabel('RMS Relative Error')
        ax_rms.set_title('Relative Error Evolution')

        # Weight plot
        line_weight, = ax_weight.plot(
            [], [], color=THEME['accent'], linewidth=2,
            marker='o', markersize=5, markerfacecolor=THEME['highlight'],
            markeredgecolor='white', markeredgewidth=0.1
        )
        line_scale_on = line_scale_off = None
        if is_hyperbolic:
            line_scale_on, = ax_weight.plot(
                [], [], 'o', markersize=7, markerfacecolor='#ff3333',
                markeredgecolor='white', markeredgewidth=0.01, 
                label='Scale Learning On'
            )
            line_scale_off, = ax_weight.plot(
                [], [], 'o', markersize=5, markerfacecolor=THEME['accent'],
                markeredgecolor='white', markeredgewidth=0.01,
                label='Scale Learning Off'
            )
            ax_weight.legend(loc='upper right')
        ax_weight.set_xlim(1, n_frames)
        ax_weight.set_ylim(0, 1)
        ax_weight.set_xlabel('Epoch')
        ax_weight.set_ylabel('−Weight Exponent')
        ax_weight.set_title('Weight Evolution')

        # Learning rate plot
        line_lr, = ax_lr.plot(
            [], [], color='#50fa7b', linewidth=2,
            marker='o', markersize=5, markerfacecolor=THEME['accent'],
            markeredgecolor='white', markeredgewidth=0.1
        )
        ax_lr.set_xlim(1, n_frames)
        ax_lr.set_ylim(*lr_bounds)
        ax_lr.set_xlabel('Epoch')
        ax_lr.set_ylabel('log₁₀(Learning Rate)')
        ax_lr.set_title('Learning Rate Schedule')

        # RE heatmap
        ax_re.set_facecolor('#0d0d1a')
        im_re = ax_re.imshow(
            log_re_stack[0], cmap='magma',
            vmin=log_re_min, vmax=log_re_max,
            interpolation='nearest', aspect='equal'
        )
        title_re = ax_re.set_title('Relative Error Matrix · Epoch 0')
        ax_re.set_xticks([])
        ax_re.set_yticks([])
        cbar_re = fig.colorbar(im_re, ax=ax_re, fraction=0.046, pad=0.04, shrink=0.9)
        cbar_re.set_label('log₁₀(RE)')

        # Distance heatmap
        ax_dist.set_facecolor('#0d0d1a')
        ax_dist.imshow(masked_dist, cmap='viridis', interpolation='nearest', aspect='equal')
        ax_dist.set_title('Distance Matrix')
        ax_dist.set_xticks([])
        ax_dist.set_yticks([])
        cbar_dist = fig.colorbar(
            ax_dist.images[0], ax=ax_dist,
            fraction=0.046, pad=0.04, shrink=0.9
        )
        cbar_dist.set_label('log₁₀(Distance)')

        # Heatmap borders
        for ax in (ax_re, ax_dist):
            for spine in ax.spines.values():
                spine.set_edgecolor(THEME['accent'])
                spine.set_linewidth(1.5)
                spine.set_alpha(0.4)

        fig.text(
            0.99, 0.01, 'RE Matrix Evolution',
            fontsize=8, color=THEME['text_dim'], alpha=0.5,
            ha='right', va='bottom', style='italic'
        )
        fig.subplots_adjust(left=0.06, right=0.94, top=0.95, bottom=0.06)

        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()

        # FFmpeg pipe
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'rgba',
            '-r', str(fps), '-i', '-',
            '-c:v', 'libx264', '-preset', 'ultrafast',
            '-crf', '23', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', video_path
        ]

        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

        try:
            for epoch in range(n_frames):
                x = epochs[:epoch + 1]

                line_rms.set_data(x, rms_vals[:epoch + 1])
                line_weight.set_data(x, weights[:epoch + 1])
                line_lr.set_data(x, lrs[:epoch + 1])

                if is_hyperbolic:
                    m_on = mask_changing[:epoch + 1]
                    m_off = mask_unchanged[:epoch + 1]
                    line_scale_on.set_data(x[m_on], weights[:epoch + 1][m_on])
                    line_scale_off.set_data(x[m_off], weights[:epoch + 1][m_off])

                im_re.set_array(log_re_stack[epoch])
                title_re.set_text(f'Relative Error Matrix · Epoch {epoch}')

                fig.canvas.draw()
                proc.stdin.write(memoryview(fig.canvas.buffer_rgba()))

        finally:
            proc.stdin.close()
            proc.wait()

        plt.close(fig)
        plt.rcdefaults()

        self._log(f"Video saved: {video_path}")


# =============================================================================
# MultiTree: Collection of Phylogenetic Trees
# =============================================================================

class MultiTree:
    """
    Collection of phylogenetic trees with batch operations.

    Supports aggregated distance computation across trees with different
    leaf sets, batch normalization, and parallel multi-tree embedding.

    Attributes
    ----------
    name : str
        Collection identifier.
    trees : List[Tree]
        Contained Tree instances.

    Examples
    --------
    >>> mtree = MultiTree("path/to/trees.newick")
    >>> mtree = MultiTree("collection", [tree1, tree2, tree3])
    >>> avg_dist, confidence, labels = mtree.distance_matrix()
    >>> embeddings = mtree.embed(dim=3, geometry='hyperbolic')
    """

    def __init__(self, *source: Union[str, List[Union['Tree', ts.Tree]]]):
        """
        Initialize MultiTree from file or list of trees.

        Parameters
        ----------
        *source : str or (str, List[Tree | treeswift.Tree])
            Either a file path to multi-tree Newick file, or
            (name, list_of_trees) tuple.

        Raises
        ------
        ValueError
            If input format is invalid.
        FileNotFoundError
            If specified file does not exist.
        """
        self._timestamp = get_time() or datetime.now()
        self.trees: List[Tree] = []

        if len(source) == 1 and isinstance(source[0], str):
            filepath = source[0]
            self.name = os.path.basename(filepath)
            self.trees = self._load_trees(filepath)

        elif len(source) == 2 and isinstance(source[0], str) and isinstance(source[1], list):
            self.name = source[0]
            tree_list = source[1]

            if all(isinstance(t, Tree) for t in tree_list):
                self.trees = tree_list
            elif all(isinstance(t, ts.Tree) for t in tree_list):
                self.trees = [
                    Tree(f"tree_{i}", t) for i, t in enumerate(tree_list)
                ]
            else:
                raise ValueError(
                    "List must contain only Tree or treeswift.Tree instances."
                )
        else:
            raise ValueError("Invalid input format for MultiTree.")

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _log(self, message: str) -> None:
        """Log message if global logging is enabled."""
        if logging_enabled():
            get_logger().info(message)

    def _load_trees(self, filepath: str) -> List[Tree]:
        """Load multiple trees from Newick file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        try:
            return [
                Tree(f'tree_{i + 1}', t)
                for i, t in enumerate(ts.read_tree_newick(filepath))
            ]
        except Exception as e:
            raise ValueError(f"Error loading trees: {e}")

    # -------------------------------------------------------------------------
    # Container Protocol
    # -------------------------------------------------------------------------

    def __getitem__(self, index: Union[int, slice]) -> Union[Tree, 'MultiTree']:
        """Retrieve tree by index or slice."""
        if isinstance(index, slice):
            return MultiTree(self.name, self.trees[index])
        return self.trees[index]

    def __len__(self) -> int:
        """Number of trees in collection."""
        return len(self.trees)

    def __iter__(self) -> Iterator[Tree]:
        """Iterate over contained trees."""
        return iter(self.trees)

    def __contains__(self, item: Tree) -> bool:
        """Check membership."""
        return item in self.trees

    def __repr__(self) -> str:
        return f"MultiTree({self.name}, n={len(self.trees)})"

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def update_time(self) -> None:
        """Update internal timestamp to current time."""
        self._timestamp = datetime.now()
        self._log("Timestamp updated.")

    def copy(self) -> 'MultiTree':
        """
        Create a deep copy of this collection.

        Returns
        -------
        MultiTree
            Independent copy with all trees duplicated.
        """
        self._log(f"Copied MultiTree '{self.name}'")
        return copy.deepcopy(self)

    def save(self, filepath: str, fmt: str = 'newick') -> None:
        """
        Save all trees to file.

        Parameters
        ----------
        filepath : str
            Output file path.
        fmt : str, default='newick'
            Output format (only 'newick' supported).

        Raises
        ------
        ValueError
            If format is not supported.
        """
        if fmt.lower() != 'newick':
            self._log(f"Save failed: unsupported format '{fmt}'")
            raise ValueError(f"Unsupported format: {fmt}")

        try:
            with open(filepath, 'w') as f:
                for tree in self.trees:
                    f.write(tree.contents.newick() + "\n")
            self._log(f"Saved {len(self.trees)} trees to {filepath}")
        except Exception as e:
            self._log(f"Save failed: {e}")
            raise

    def terminal_names(self) -> List[str]:
        """
        Get sorted union of all leaf names across trees.

        Returns
        -------
        List[str]
            Alphabetically sorted unique leaf labels.
        """
        names = sorted({
            name for tree in self.trees
            for name in tree.terminal_names()
        })
        self._log(f"Retrieved {len(names)} terminal names for '{self.name}'")
        return names

    def common_terminals(self) -> List[str]:
        """
        Get sorted intersection of leaf names across all trees.

        Returns
        -------
        List[str]
            Alphabetically sorted labels present in every tree.
        """
        if not self.trees:
            return []

        common = set(self.trees[0].terminal_names())
        for tree in self.trees[1:]:
            common.intersection_update(tree.terminal_names())

        self._log(f"Found {len(common)} common terminals for '{self.name}'")
        return sorted(common)

    def distance_matrix(
        self,
        method: str = "agg",
        func: Callable[[torch.Tensor], torch.Tensor] = torch.nanmean,
        max_iter: int = 1000,
        n_jobs: int = -1,
        tol: float = 1e-10,
        sigma_max: float = 3.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[str]]:
        """
        Compute aggregated distance matrix across all trees.

        Handles trees with different leaf sets by aligning to the global
        label set and treating missing pairs as NaN.

        Parameters
        ----------
        method : {'agg', 'fp'}
            'agg': Direct aggregation with func.
            'fp': Fixed-point iteration with adaptive Gaussian weighting.
        func : Callable
            Aggregation function (e.g., torch.nanmean, torch.nanmedian).
        max_iter : int
            Maximum iterations for fixed-point method.
        n_jobs : int
            Parallel jobs (-1 for all cores).
        tol : float
            Convergence tolerance for fixed-point.
        sigma_max : float
            Maximum sigma for Gaussian similarity kernel.

        Returns
        -------
        avg_matrix : torch.Tensor
            Shape (n, n) aggregated distance matrix.
        confidence : torch.Tensor
            Shape (n, n) fraction of trees contributing to each entry.
        labels : List[str]
            Leaf names corresponding to matrix indices.

        Raises
        ------
        ValueError
            If no trees are available.
        """
        if not self.trees:
            self._log("No trees available for distance computation.")
            raise ValueError("No trees available.")

        labels = self.terminal_names()
        label_idx = {lbl: i for i, lbl in enumerate(labels)}
        n_labels = len(labels)
        n_trees = len(self.trees)

        def align_tree_matrix(tree: Tree) -> torch.Tensor:
            """Align single tree's distance matrix to global label set."""
            tree_labels = tree.terminal_names()
            indices = torch.tensor(
                [label_idx[lbl] for lbl in tree_labels],
                dtype=torch.long
            )
            aligned = torch.full((n_labels, n_labels), float('nan'))
            aligned[indices[:, None], indices] = tree.distance_matrix()[0]
            aligned.fill_diagonal_(0.0)
            return aligned

        def unwrap(result: torch.Tensor | tuple) -> torch.Tensor:
            """Extract tensor from aggregation result (handles nanmedian)."""
            return result[0] if isinstance(result, tuple) else result

        # Parallel matrix computation
        aligned = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(align_tree_matrix)(tree) for tree in self.trees
        )
        dist_stack = torch.stack(aligned)  # (n_trees, n_labels, n_labels)

        valid_mask = ~torch.isnan(dist_stack)
        confidence = valid_mask.float().mean(dim=0)

        if method == "fp":
            return self._fixed_point_aggregate(
                dist_stack, valid_mask, confidence, labels,
                func, unwrap, max_iter, tol, sigma_max
            )

        # Standard aggregation
        avg_matrix = unwrap(func(dist_stack, dim=0))

        # Interpolate remaining NaNs using row/column means
        nan_mask = torch.isnan(avg_matrix)
        if nan_mask.any():
            row_mean = unwrap(func(avg_matrix, dim=1))
            col_mean = unwrap(func(avg_matrix, dim=0))
            fill = (row_mean[:, None] + col_mean[None, :]) / 2
            avg_matrix = torch.where(nan_mask, fill, avg_matrix)

        self._log("Distance matrix computation complete.")
        return avg_matrix, confidence, labels

    def _fixed_point_aggregate(
        self,
        dist_stack: torch.Tensor,
        valid_mask: torch.Tensor,
        confidence: torch.Tensor,
        labels: List[str],
        func: Callable,
        unwrap: Callable,
        max_iter: int,
        tol: float,
        sigma_max: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Fixed-point aggregation with adaptive Gaussian weighting.

        Iteratively refines average matrix by weighting each tree's
        contribution based on its similarity to current estimate.
        """
        n_trees = dist_stack.shape[0]
        device = dist_stack.device

        dist_flat = dist_stack.view(n_trees, -1)
        valid_flat = valid_mask.view(n_trees, -1)

        avg_matrix = unwrap(func(dist_stack, dim=0))
        prev_weights = torch.zeros(n_trees, device=device)

        with tqdm(total=max_iter, desc="Fixed-point iteration", unit="iter") as pbar:
            for i in range(max_iter):
                sigma = min(2 * sigma_max * i / max_iter, sigma_max)
                avg_flat = avg_matrix.view(-1)

                # Squared differences where valid
                diff = torch.where(valid_flat, dist_flat - avg_flat, torch.zeros_like(dist_flat))
                diff_sq = (diff ** 2).sum(dim=1)

                # Reference norm
                ref = torch.where(valid_flat, avg_flat.expand(n_trees, -1), torch.zeros_like(dist_flat))
                ref_sq = (ref ** 2).sum(dim=1).clamp(min=1e-10)

                # Gaussian similarity weights
                sim = torch.exp(-sigma * diff_sq / ref_sq)
                weights = sim / sim.sum().clamp(min=1e-10)

                # Weighted average
                w_exp = weights.view(n_trees, 1, 1)
                weighted = torch.where(valid_mask, w_exp * dist_stack, torch.zeros_like(dist_stack))
                w_sum = torch.where(valid_mask, w_exp.expand_as(dist_stack), torch.zeros_like(dist_stack))
                w_sum = w_sum.sum(dim=0).clamp(min=1e-10)

                avg_matrix = weighted.sum(dim=0) / w_sum

                # Convergence check
                change = torch.sqrt(n_trees * ((weights - prev_weights) ** 2).sum())
                if change < tol:
                    pbar.update(max_iter - i)
                    break

                prev_weights = weights
                pbar.update(1)

        return avg_matrix, confidence, labels

    def normalize(self, batch_mode: bool = False) -> List[float]:
        """
        Normalize branch lengths across all trees.

        Optimizes scale factors so that the weighted average distance
        matrix has minimal variance. Each tree's branch lengths are
        multiplied by its optimal scale factor.

        Parameters
        ----------
        batch_mode : bool, default=False
            If True, use stochastic batch optimization (faster for
            large tree collections).

        Returns
        -------
        List[float]
            Scale factors applied to each tree.
        """
        labels = self.terminal_names()
        n_labels = len(labels)
        n_trees = len(self.trees)
        label_idx = {lbl: i for i, lbl in enumerate(labels)}

        # Adaptive hyperparameters
        sqrt_n = np.sqrt(n_labels)
        sqrt_t = np.sqrt(n_trees)
        lr_log_start = -np.log10(n_labels) + (1 if batch_mode else -1)
        lr_log_end = -np.log10(n_labels)
        max_iter = 10 * int(sqrt_n + 1) if batch_mode else 10 * n_labels
        n_passes = int(n_labels / sqrt_n + 1) if batch_mode else 1
        batch_size = int(sqrt_t + 1) if batch_mode else n_trees

        # Build aligned distance matrices
        dist_matrices = torch.full((n_trees, n_labels, n_labels), float('nan'))
        for t_idx, tree in enumerate(self.trees):
            tree_labels = tree.terminal_names()
            indices = torch.tensor([label_idx[lbl] for lbl in tree_labels], dtype=torch.long)
            dist_matrices[t_idx, indices[:, None], indices] = tree.distance_matrix()[0]
        dist_matrices.diagonal(dim1=-2, dim2=-1).fill_(0)

        valid_mask = ~torch.isnan(dist_matrices)
        valid_count = valid_mask.sum(dim=0).clamp(min=1).float()
        dist_matrices = dist_matrices.nan_to_num_(0.0)

        scales = torch.ones(n_trees, dtype=torch.float32)
        norm_factor = 1.0 / (n_labels ** 2)

        # Progress tracking
        n_batches = (n_trees + batch_size - 1) // batch_size
        total_iter = n_passes * max_iter * n_batches + (max_iter if batch_mode else 0)
        pbar = tqdm(total=total_iter, desc="Normalizing", unit="iter")

        lr_schedule = 10 ** (
            lr_log_start + (lr_log_end - lr_log_start) *
            torch.arange(max_iter) / max_iter
        )

        def optimize_batch(
            batch_idx: List[int],
            other_weighted_sum: torch.Tensor,
            batch_valid: torch.Tensor,
            batch_dist: torch.Tensor
        ) -> torch.Tensor:
            """Optimize scales for a batch via gradient descent."""
            batch_sum = scales[batch_idx].sum()
            params = scales[batch_idx].clone().requires_grad_(True)
            optimizer = Adam([params], lr=lr_schedule[0].item())

            for it in range(max_iter):
                optimizer.param_groups[0]['lr'] = lr_schedule[it].item()

                # Softplus + sum constraint
                norm_params = torch.nn.functional.softplus(params)
                norm_params = norm_params * (batch_sum / norm_params.sum())

                weighted_batch = batch_dist * norm_params[:, None, None]
                avg = (weighted_batch.sum(dim=0) + other_weighted_sum) / valid_count

                residual = (weighted_batch - avg.unsqueeze(0)) * batch_valid
                loss = residual.pow(2).sum() * norm_factor

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(1)
                self._log(f"Iter {it}: loss={loss.item():.6f}, lr={lr_schedule[it]:.2e}")

            scales[batch_idx] = norm_params.detach()
            return avg

        # Main optimization loop
        tree_indices = list(range(n_trees))
        for _ in range(n_passes):
            if batch_mode:
                random.shuffle(tree_indices)

            for start in range(0, n_trees, batch_size):
                batch_idx = tree_indices[start:start + batch_size]
                other_idx = list(set(tree_indices) - set(batch_idx))

                other_sum = (
                    (dist_matrices[other_idx] * scales[other_idx, None, None]).sum(dim=0)
                    if other_idx else torch.zeros(n_labels, n_labels)
                )
                optimize_batch(batch_idx, other_sum, valid_mask[batch_idx], dist_matrices[batch_idx])

        # Final global pass in batch mode
        if batch_mode:
            all_idx = list(range(n_trees))
            optimize_batch(all_idx, torch.zeros(n_labels, n_labels), valid_mask, dist_matrices)

        pbar.close()

        # Apply scales to trees
        final_scales = scales.tolist()
        for t_idx, tree in enumerate(self.trees):
            s = final_scales[t_idx]
            for node in tree.contents.traverse_postorder():
                edge_len = node.get_edge_length()
                if edge_len is not None:
                    node.set_edge_length(edge_len * s)

        return final_scales

    def embed(
        self,
        dim: int,
        geometry: str = 'hyperbolic',
        **kwargs
    ) -> 'embedding.MultiEmbedding':
        """
        Embed all trees into geometric space.

        Parameters
        ----------
        dim : int
            Target embedding dimension.
        geometry : {'hyperbolic', 'euclidean'}
            Target geometry type.
        **kwargs : dict
            Optional parameters (same as Tree.embed):
            - precise_opt, epochs, lr_init, dist_cutoff
            - export_video, save_mode, normalize
            - scale_fn, lr_fn, weight_exp_fn

        Returns
        -------
        embedding.MultiEmbedding
            Collection of embeddings for all trees.

        Raises
        ------
        ValueError
            If dim is None.
        """
        if dim is None:
            raise ValueError("Parameter 'dim' is required.")

        # Parameter defaults
        defaults = [
            ('precise_opt', conf.ENABLE_ACCURATE_OPTIMIZATION),
            ('epochs', conf.TOTAL_EPOCHS),
            ('lr_init', conf.INITIAL_LEARNING_RATE),
            ('dist_cutoff', conf.MAX_RANGE),
            ('save_mode', conf.ENABLE_SAVE_MODE),
            ('export_video', conf.ENABLE_VIDEO_EXPORT),
            ('scale_fn', None),
            ('lr_fn', None),
            ('weight_exp_fn', None),
            ('normalize', False),
        ]
        params = {k: kwargs.get(k, v) for k, v in defaults}

        if params['normalize']:
            self.normalize(batch_mode=params['precise_opt'])

        params['save_mode'] |= params['export_video']
        params['export_video'] &= params['precise_opt']

        n_trees = len(self.trees)
        n_jobs = min(n_trees, os.cpu_count())

        try:
            if geometry == 'hyperbolic':
                self._log("Starting hyperbolic multi-embedding...")
                scale = params['dist_cutoff'] / self.distance_matrix()[0].max()
                curvature = -(scale ** 2)

                def process_hyperbolic(idx_tree: Tuple[int, Tree]):
                    idx, tree = idx_tree
                    dist = tree.distance_matrix()[0]
                    pts = utils.naive_embedding(dist * scale, dim, geometry='hyperbolic')
                    emb = embedding.LoidEmbedding(
                        points=pts, labels=tree.terminal_names(), curvature=curvature
                    )
                    return idx, dist, emb

                results = Parallel(n_jobs=n_jobs, backend='loky', return_as='generator')(
                    delayed(process_hyperbolic)((i, t)) for i, t in enumerate(self.trees)
                )

                dist_mats = [None] * n_trees
                emb_list = [None] * n_trees

                for idx, dist, emb in results:
                    dist_mats[idx] = dist
                    emb_list[idx] = emb
                    self._log(f"Naive hyperbolic embedding {idx + 1}/{n_trees} complete")

                multi_emb = embedding.MultiEmbedding()
                for emb in emb_list:
                    multi_emb.append(emb)
                del emb_list
                gc.collect()

                self._log("Naive hyperbolic embeddings complete.")

                if params['precise_opt']:
                    self._log("Refining with precise optimization...")
                    pts_list, curvature = utils.precise_multiembedding(
                        dist_mats, multi_emb,
                        geometry="hyperbolic",
                        log_fn=self._log,
                        time_stamp=self._timestamp,
                        **params
                    )

                    multi_emb = embedding.MultiEmbedding()
                    tree_labels = [t.terminal_names() for t in self.trees]
                    for pts, labels in zip(pts_list, tree_labels):
                        multi_emb.append(
                            embedding.LoidEmbedding(
                                points=pts, labels=labels, curvature=curvature
                            )
                        )
                    del pts_list, dist_mats
                    gc.collect()
                    self._log("Precise hyperbolic embeddings complete.")
                else:
                    del dist_mats
                    gc.collect()

            else:  # Euclidean
                self._log("Starting Euclidean multi-embedding...")

                def process_euclidean(idx_tree: Tuple[int, Tree]):
                    idx, tree = idx_tree
                    dist = tree.distance_matrix()[0]
                    pts = utils.naive_embedding(dist, dim, geometry='euclidean')
                    emb = embedding.EuclideanEmbedding(
                        points=pts, labels=tree.terminal_names()
                    )
                    return idx, dist, emb

                results = Parallel(n_jobs=n_jobs, backend='loky', return_as='generator')(
                    delayed(process_euclidean)((i, t)) for i, t in enumerate(self.trees)
                )

                dist_mats = [None] * n_trees
                emb_list = [None] * n_trees

                for idx, dist, emb in results:
                    dist_mats[idx] = dist
                    emb_list[idx] = emb
                    self._log(f"Naive Euclidean embedding {idx + 1}/{n_trees} complete")

                multi_emb = embedding.MultiEmbedding()
                for emb in emb_list:
                    multi_emb.append(emb)
                del emb_list
                gc.collect()

                self._log("Naive Euclidean embeddings complete.")

                if params['precise_opt']:
                    self._log("Refining with precise optimization...")
                    pts_list, _ = utils.precise_multiembedding(
                        dist_mats, multi_emb,
                        geometry="euclidean",
                        log_fn=self._log,
                        time_stamp=self._timestamp,
                        **params
                    )

                    multi_emb = embedding.MultiEmbedding()
                    tree_labels = [t.terminal_names() for t in self.trees]
                    for pts, labels in zip(pts_list, tree_labels):
                        multi_emb.append(
                            embedding.EuclideanEmbedding(points=pts, labels=labels)
                        )
                    del pts_list, dist_mats
                    gc.collect()
                    self._log("Precise Euclidean embeddings complete.")
                else:
                    del dist_mats
                    gc.collect()

        except Exception as e:
            self._log(f"Multi-embedding error: {e}")
            raise

        # Save result
        self._save_embedding(multi_emb, geometry, dim)

        if params['export_video']:
            self._generate_video(fps=params['epochs'] // conf.VIDEO_LENGTH)

        return multi_emb

    def _save_embedding(
        self,
        emb: 'embedding.MultiEmbedding',
        geometry: str,
        dim: int
    ) -> None:
        """Save multi-embedding to timestamped directory."""
        out_dir = os.path.join(
            conf.OUTPUT_DIRECTORY,
            self._timestamp.strftime('%Y-%m-%d_%H-%M-%S')
        )
        os.makedirs(out_dir, exist_ok=True)

        filepath = os.path.join(out_dir, f"{geometry}_multiembedding_{dim}d.pkl")
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(emb, f, protocol=pickle.HIGHEST_PROTOCOL)
            self._log(f"Multi-embedding saved to {filepath}")
        except (IOError, pickle.PicklingError) as e:
            self._log(f"Save error: {e}")
            raise

    def _generate_video(self, fps: int = 10) -> None:
        """
        Generate MP4 video of multi-tree optimization evolution.

        Shows per-tree RMS relative error, weight evolution, learning
        rate schedule, and cost function across all trees.

        Parameters
        ----------
        fps : int, default=10
            Output video frame rate.
        """
        # Theme configuration
        THEME = {
            'background': '#1a1a2e',
            'panel': '#1e2a4a',
            'grid': '#2a2a4a',
            'text': '#e8e8e8',
            'text_dim': '#a0a0b0',
            'accent': '#00d4ff',
            'accent_alt': '#ff6b6b',
            'highlight': '#ffd93d',
        }

        plt.rcParams.update({
            'figure.facecolor': THEME['background'],
            'figure.edgecolor': THEME['background'],
            'axes.facecolor': THEME['panel'],
            'axes.edgecolor': THEME['grid'],
            'axes.labelcolor': THEME['text'],
            'axes.titlecolor': THEME['text'],
            'axes.grid': True,
            'axes.axisbelow': True,
            'axes.linewidth': 0.8,
            'axes.titleweight': 'bold',
            'axes.titlesize': 11,
            'axes.labelsize': 9,
            'grid.color': THEME['grid'],
            'grid.linewidth': 0.4,
            'grid.alpha': 0.5,
            'xtick.color': THEME['text_dim'],
            'ytick.color': THEME['text_dim'],
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'text.color': THEME['text'],
            'font.family': 'sans-serif',
            'font.size': 9,
            'legend.facecolor': THEME['panel'],
            'legend.edgecolor': THEME['grid'],
            'legend.fontsize': 8,
        })

        base_dir = os.path.join(
            conf.OUTPUT_DIRECTORY,
            self._timestamp.strftime('%Y-%m-%d_%H-%M-%S')
        )

        # Load metadata
        try:
            metadata = np.load(
                os.path.join(base_dir, "metadata.npy"),
                allow_pickle=True
            ).item()
            n_trees = metadata['num_trees']
        except FileNotFoundError:
            tree_dirs = sorted([
                d for d in os.listdir(base_dir) if d.startswith('tree_')
            ])
            n_trees = len(tree_dirs)

        # Load aggregate data
        weights = -np.load(os.path.join(base_dir, "weight_exponents.npy"))
        lrs = np.log10(np.load(os.path.join(base_dir, "learning_rates.npy")) + conf.EPSILON)
        agg_costs = np.load(os.path.join(base_dir, "costs.npy"))

        try:
            scales = np.load(os.path.join(base_dir, "scales.npy"))
        except FileNotFoundError:
            scales = None

        n_frames = len(weights)
        epochs = np.arange(1, n_frames + 1)
        is_hyperbolic = scales is not None and not np.all(scales == 1)

        if is_hyperbolic:
            scale_active = scales.astype(bool)
            scale_changed = np.concatenate([[True], np.diff(scales) != 0])
            mask_changing = scale_active & scale_changed
            mask_unchanged = scale_active & ~scale_changed

        # Load per-tree data
        self._log(f"Loading data for {n_trees} trees...")

        all_rms = []
        all_costs = []

        for t_idx in range(n_trees):
            tree_dir = os.path.join(base_dir, f"tree_{t_idx}")
            all_costs.append(np.load(os.path.join(tree_dir, "costs.npy")))
            all_rms.append(np.load(os.path.join(tree_dir, "rmse.npy")))

        all_rms = np.array(all_rms)  # (n_trees, n_frames)
        all_costs = np.array(all_costs)

        # Identify min/max RMS trees
        final_rms = all_rms[:, -1]
        min_rms_idx = np.argmin(final_rms)
        max_rms_idx = np.argmax(final_rms)

        # Axis bounds
        rms_bounds = (max(np.nanmin(all_rms) * 0.9, 1e-20), np.nanmax(all_rms) * 1.1)

        lr_bounds = (lrs.min() - 0.1, lrs.max() + 0.1)
        weight_bounds = (weights.min() * 0.95, weights.max() * 1.05)
        cost_bounds = (
            max(min(np.nanmin(agg_costs), np.nanmin(all_costs)) * 0.9, 1e-20),
            max(np.nanmax(agg_costs), np.nanmax(all_costs)) * 1.1
        )
        

        # Setup output
        out_dir = os.path.join(
            conf.OUTPUT_VIDEO_DIRECTORY,
            self._timestamp.strftime('%Y-%m-%d_%H-%M-%S')
        )
        os.makedirs(out_dir, exist_ok=True)
        video_path = os.path.join(out_dir, 're_evolution_multi.mp4')

        self._log(f"Generating video for {n_trees} trees...")

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
        ax_rms, ax_weight = axes[0]
        ax_lr, ax_cost = axes[1]

        # Tree colormap
        cmap = get_cmap('plasma')
        norm = Normalize(vmin=0, vmax=n_trees - 1)
        tree_colors = [cmap(norm(i)) for i in range(n_trees)]

        # RMS plot
        lines_rms = []
        for t_idx in range(n_trees):
            is_extremal = t_idx in [min_rms_idx, max_rms_idx]
            alpha = 1.0 if is_extremal else 0.3
            lw = 2.0 if is_extremal else 0.5
            line, = ax_rms.plot([], [], color=tree_colors[t_idx], linewidth=lw, alpha=alpha)
            lines_rms.append(line)

        ax_rms.set_xlim(1, n_frames)
        ax_rms.set_ylim(*rms_bounds)
        ax_rms.set_yscale('log')
        ax_rms.set_xlabel('Epoch')
        ax_rms.set_ylabel('Median RE (log)')
        ax_rms.set_title(f'Median Relative Error ({n_trees} Trees)')

        annot_min = ax_rms.annotate(
            '', xy=(0, 0), fontsize=7, fontweight='bold', color='#50fa7b',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=THEME['panel'],
                      edgecolor='#50fa7b', linewidth=0.5, alpha=0.9)
        )
        annot_max = ax_rms.annotate(
            '', xy=(0, 0), fontsize=7, fontweight='bold', color='#ff5555',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=THEME['panel'],
                      edgecolor='#ff5555', linewidth=0.5, alpha=0.9)
        )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_rms = fig.colorbar(sm, ax=ax_rms, fraction=0.046, pad=0.04, shrink=0.8)
        cbar_rms.set_label('Tree Index')

        # Weight plot
        line_weight, = ax_weight.plot(
            [], [], color=THEME['accent'], linewidth=2,
            marker='o', markersize=3, markerfacecolor=THEME['highlight'],
            markeredgecolor='white', markeredgewidth=0.01
        )
        line_scale_on = line_scale_off = None
        if is_hyperbolic:
            line_scale_on, = ax_weight.plot(
                [], [], 'o', markersize=5, markerfacecolor='#ff3333',
                markeredgecolor='white', markeredgewidth=0.01,
                label='Scale Learning On'
            )
            line_scale_off, = ax_weight.plot(
                [], [], 'o', markersize=3, markerfacecolor=THEME['accent'],
                markeredgecolor='white', markeredgewidth=0.01,
                label='Scale Learning Off'
            )
            ax_weight.legend(loc='upper right', fontsize=7)
        ax_weight.set_xlim(1, n_frames)
        ax_weight.set_ylim(*weight_bounds)
        ax_weight.set_xlabel('Epoch')
        ax_weight.set_ylabel('−Weight Exponent')
        ax_weight.set_title('Weight Evolution')

        # Learning rate plot
        line_lr, = ax_lr.plot(
            [], [], color='#50fa7b', linewidth=2,
            marker='o', markersize=3, markerfacecolor=THEME['accent'],
            markeredgecolor='white', markeredgewidth=0.01
        )
        ax_lr.set_xlim(1, n_frames)
        ax_lr.set_ylim(*lr_bounds)
        ax_lr.set_xlabel('Epoch')
        ax_lr.set_ylabel('log₁₀(Learning Rate)')
        ax_lr.set_title('Learning Rate Schedule')

        # Cost plot
        lines_cost = []
        for t_idx in range(n_trees):
            line, = ax_cost.plot(
                [], [], color=tree_colors[t_idx],
                linewidth=0.4, alpha=0.2
            )
            lines_cost.append(line)

        line_agg_cost, = ax_cost.plot(
            [], [], color='white', linewidth=2.5, label='Aggregate'
        )
        ax_cost.set_xlim(1, n_frames)
        ax_cost.set_ylim(*cost_bounds)
        ax_cost.set_yscale('log')
        ax_cost.set_xlabel('Epoch')
        ax_cost.set_ylabel('Cost (log)')
        ax_cost.set_title('Cost Evolution')
        ax_cost.legend(loc='upper right', fontsize=8)

        sm_cost = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm_cost.set_array([])
        cbar_cost = fig.colorbar(sm_cost, ax=ax_cost, fraction=0.046, pad=0.04, shrink=0.8)
        cbar_cost.set_label('Tree Index')

        # Borders
        for ax in [ax_rms, ax_weight, ax_lr, ax_cost]:
            for spine in ax.spines.values():
                spine.set_edgecolor('#4a4a6a')
                spine.set_linewidth(1.0)

        fig.tight_layout()
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()

        # FFmpeg pipe
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'rgba',
            '-r', str(fps), '-i', '-',
            '-c:v', 'libx264', '-preset', 'ultrafast',
            '-crf', '23', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', video_path
        ]

        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

        try:
            for epoch in range(n_frames):
                x = epochs[:epoch + 1]

                for t_idx in range(n_trees):
                    lines_rms[t_idx].set_data(x, all_rms[t_idx, :epoch + 1])

                # Update annotations
                min_val = all_rms[min_rms_idx, epoch]
                max_val = all_rms[max_rms_idx, epoch]
                annot_min.set_text(f'Min: T{min_rms_idx}')
                annot_min.xy = (epoch + 1, min_val)
                annot_min.set_position((epoch + 1.5, min_val * 0.85))
                annot_max.set_text(f'Max: T{max_rms_idx}')
                annot_max.xy = (epoch + 1, max_val)
                annot_max.set_position((epoch + 1.5, max_val * 1.15))

                line_weight.set_data(x, weights[:epoch + 1])

                if is_hyperbolic:
                    m_on = mask_changing[:epoch + 1]
                    m_off = mask_unchanged[:epoch + 1]
                    line_scale_on.set_data(x[m_on], weights[:epoch + 1][m_on])
                    line_scale_off.set_data(x[m_off], weights[:epoch + 1][m_off])

                line_lr.set_data(x, lrs[:epoch + 1])

                for t_idx in range(n_trees):
                    lines_cost[t_idx].set_data(x, all_costs[t_idx, :epoch + 1])
                line_agg_cost.set_data(x, agg_costs[:epoch + 1])

                fig.canvas.draw()
                proc.stdin.write(memoryview(fig.canvas.buffer_rgba()))

        finally:
            proc.stdin.close()
            proc.wait()

        plt.close(fig)
        plt.rcdefaults()

        self._log(f"Multi-tree video saved: {video_path}")
