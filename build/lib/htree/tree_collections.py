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
        if n == 0:
            return torch.empty((0, 0), dtype=torch.float32), labels
        dist_dict = self.contents.distance_matrix(leaf_labels=True)
        # O(1) index lookup via hash map
        label_to_idx: dict = {lbl: idx for idx, lbl in enumerate(labels)}
        # Pre-allocate contiguous memory block
        matrix = np.zeros((n, n), dtype=np.float32)
        def fill_row(i: int) -> None:
            """Fill row i via dict iteration (avoids n random lookups per row)."""
            row_dict = dist_dict.get(labels[i])
            if row_dict is not None:
                row = matrix[i]  # Direct view, no copy
                for lbl_j, dist in row_dict.items():
                    j = label_to_idx.get(lbl_j)
                    if j is not None:
                        row[j] = dist
        # Thread backend: shared memory, zero serialization overhead
        Parallel(n_jobs=-1, prefer="threads")(
            delayed(fill_row)(i) for i in range(n)
        )
        # Zero-copy tensor creation
        dist_matrix = torch.from_numpy(matrix)
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
        defaults = {
            'precise_opt': conf.ENABLE_ACCURATE_OPTIMIZATION, 'epochs': conf.TOTAL_EPOCHS,
            'lr_init': conf.INITIAL_LEARNING_RATE, 'dist_cutoff': conf.MAX_RANGE,
            'export_video': conf.ENABLE_VIDEO_EXPORT, 'save_mode': conf.ENABLE_SAVE_MODE,
            'scale_fn': None, 'lr_fn': None, 'weight_exp_fn': None, 'curvature': None,
        }
        params = {k: kwargs.get(k, v) for k, v in defaults.items()}
        params['save_mode'] |= params['export_video']
        params['export_video'] &= params['precise_opt']
        is_hyperbolic = geometry == 'hyperbolic'
        try:
            dist_matrix, curvature = self.distance_matrix()[0], None
            # Hyperbolic: scale distances and compute curvature
            if is_hyperbolic:
                if params['curvature'] is not None and params['curvature'] >= 0:
                    self._log("Wrong input curvature. It has to be negative.")
                    print("Wrong input curvature. It has to be negative.")
                    return None
                if params['curvature'] is not None:
                    curvature, params['scale_fn'] = params['curvature'], lambda x1, x2, x3: False
                    scale = np.sqrt(-curvature)
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
                    dist_matrix, dim, geometry=geometry, init_pts=points,
                    log_fn=self._log, time_stamp=self._timestamp, **params)
                points, opt_scale = (opt_result, 1) if not is_hyperbolic else opt_result
                curvature = curvature * opt_scale ** 2 if is_hyperbolic else None
                self._log(f"Precise {geometry} embedding complete.")
            # Construct embedding object
            labels = self.terminal_names()
            result = (embedding.LoidEmbedding(points=points, labels=labels, curvature=curvature)
                      if is_hyperbolic else embedding.EuclideanEmbedding(points=points, labels=labels))
        except Exception as e:
            self._log(f"Embedding error: {e}")
            raise
        # Save embedding to timestamped directory
        out_dir = os.path.join(conf.OUTPUT_DIRECTORY, self._timestamp.strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(out_dir, exist_ok=True)
        filepath = os.path.join(out_dir, f"{geometry}_embedding_{dim}d.pkl")
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            self._log(f"Embedding saved to {filepath}")
        except (IOError, pickle.PicklingError) as e:
            self._log(f"Save error: {e}")
            raise
        if params['export_video']:
            self._generate_video(fps=params['epochs'] // conf.VIDEO_LENGTH)
        return result

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
            'background': '#1a1a2e', 'panel': '#1e2a4a', 'grid': '#2a2a4a',
            'text': '#e8e8e8', 'text_dim': '#a0a0b0', 'accent': '#00d4ff',
            'accent_alt': '#ff6b6b', 'highlight': '#ffd93d',
        }

        plt.rcParams.update({
            'figure.facecolor': THEME['background'], 'figure.edgecolor': THEME['background'],
            'axes.facecolor': THEME['panel'], 'axes.edgecolor': THEME['grid'],
            'axes.labelcolor': THEME['text'], 'axes.titlecolor': THEME['text'],
            'axes.grid': True, 'axes.axisbelow': True, 'axes.linewidth': 0.8,
            'axes.titleweight': 'bold', 'axes.titlesize': 11, 'axes.labelsize': 9,
            'grid.color': THEME['grid'], 'grid.linewidth': 0.4, 'grid.alpha': 0.5,
            'xtick.color': THEME['text_dim'], 'ytick.color': THEME['text_dim'],
            'xtick.labelsize': 8, 'ytick.labelsize': 8, 'text.color': THEME['text'],
            'font.family': 'sans-serif', 'font.size': 9, 'legend.facecolor': THEME['panel'],
            'legend.edgecolor': THEME['grid'], 'legend.fontsize': 8,
        })

        base_dir = os.path.join(conf.OUTPUT_DIRECTORY, self._timestamp.strftime('%Y-%m-%d_%H-%M-%S'))
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
        re_stack = np.stack(Parallel(n_jobs=-1, prefer="threads")(
            delayed(np.load)(os.path.join(base_dir, f)) for f in re_files), axis=0)

        # Compute statistics
        triu_idx = np.triu_indices(re_stack.shape[1], k=1)
        triu_vals = re_stack[:, triu_idx[0], triu_idx[1]]
        log_re_min, log_re_max = np.log10(np.nanmin(triu_vals) + conf.EPSILON), np.log10(np.nanmax(triu_vals) + conf.EPSILON)
        rms_vals = np.sqrt(np.nanmean(triu_vals ** 2, axis=1))
        del triu_vals

        rms_bounds, lr_bounds = (rms_vals.min() * 0.9, rms_vals.max() * 1.1), (lrs.min() - 0.1, lrs.max() + 0.1)
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
            mask_changing, mask_unchanged = scale_active & scale_changed, scale_active & ~scale_changed

        # Setup output
        out_dir = os.path.join(conf.OUTPUT_VIDEO_DIRECTORY, self._timestamp.strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(out_dir, exist_ok=True)
        video_path = os.path.join(out_dir, 're_evolution.mp4')
        self._log("Generating optimization video...")
        # Create figure
        fig = plt.figure(figsize=(14, 12), dpi=100)
        gs = GridSpec(4, 2, height_ratios=[1, 1, 2, 2], hspace=0.35, wspace=0.25)
        ax_rms, ax_weight, ax_lr = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
        ax_re, ax_dist = fig.add_subplot(gs[2:, 0]), fig.add_subplot(gs[2:, 1])

        # Style axes borders
        for ax in [ax_rms, ax_weight, ax_lr, ax_re, ax_dist]:
            for spine in ax.spines.values():
                spine.set_edgecolor('#4a4a6a')
                spine.set_linewidth(1.5)

        marker_kw = dict(marker='o', markersize=5, markeredgecolor='white', markeredgewidth=0.1)
        # RMS plot
        line_rms, = ax_rms.plot([], [], color=THEME['accent'], linewidth=2, markerfacecolor=THEME['accent'], **marker_kw)
        ax_rms.set(xlim=(1, n_frames), ylim=rms_bounds, yscale='log', xlabel='Epoch',
                   ylabel='RMS Relative Error', title='Relative Error Evolution')

        # Weight plot
        line_weight, = ax_weight.plot([], [], color=THEME['accent'], linewidth=2, markerfacecolor=THEME['highlight'], **marker_kw)
        line_scale_on = line_scale_off = None
        if is_hyperbolic:
            line_scale_on, = ax_weight.plot([], [], 'o', markersize=7, markerfacecolor='#ff3333',
                                            markeredgecolor='white', markeredgewidth=0.01, label='Scale Learning On')
            line_scale_off, = ax_weight.plot([], [], 'o', markersize=5, markerfacecolor=THEME['accent'],
                                             markeredgecolor='white', markeredgewidth=0.01, label='Scale Learning Off')
            ax_weight.legend(loc='upper right')
        ax_weight.set(xlim=(1, n_frames), ylim=(0, 1), xlabel='Epoch', ylabel='−Weight Exponent', title='Weight Evolution')
        # Learning rate plot
        line_lr, = ax_lr.plot([], [], color='#50fa7b', linewidth=2, markerfacecolor=THEME['accent'], **marker_kw)
        ax_lr.set(xlim=(1, n_frames), ylim=lr_bounds, xlabel='Epoch', ylabel='log₁₀(Learning Rate)', title='Learning Rate Schedule')

        cbar_kw = dict(fraction=0.046, pad=0.04, shrink=0.9)
        # RE heatmap
        ax_re.set_facecolor('#0d0d1a')
        im_re = ax_re.imshow(log_re_stack[0], cmap='magma', vmin=log_re_min, vmax=log_re_max, interpolation='nearest', aspect='equal')
        title_re = ax_re.set_title('Relative Error Matrix · Epoch 0')
        ax_re.set(xticks=[], yticks=[])
        fig.colorbar(im_re, ax=ax_re, **cbar_kw).set_label('log₁₀(RE)')
        # Distance heatmap
        ax_dist.set_facecolor('#0d0d1a')
        ax_dist.imshow(masked_dist, cmap='viridis', interpolation='nearest', aspect='equal')
        ax_dist.set(title='Distance Matrix', xticks=[], yticks=[])
        fig.colorbar(ax_dist.images[0], ax=ax_dist, **cbar_kw).set_label('log₁₀(Distance)')
        # Heatmap borders
        for ax in (ax_re, ax_dist):
            for spine in ax.spines.values():
                spine.set_edgecolor(THEME['accent'])
                spine.set_linewidth(1.5)
                spine.set_alpha(0.4)

        fig.text(0.99, 0.01, 'RE Matrix Evolution', fontsize=8, color=THEME['text_dim'],
                 alpha=0.5, ha='right', va='bottom', style='italic')
        fig.subplots_adjust(left=0.06, right=0.94, top=0.95, bottom=0.06)
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        # FFmpeg pipe
        proc = subprocess.Popen([
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'rgba', '-r', str(fps), '-i', '-',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', video_path
        ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

        try:
            for epoch in range(n_frames):
                x = epochs[:epoch + 1]
                line_rms.set_data(x, rms_vals[:epoch + 1])
                line_weight.set_data(x, weights[:epoch + 1])
                line_lr.set_data(x, lrs[:epoch + 1])
                if is_hyperbolic:
                    m_on, m_off = mask_changing[:epoch + 1], mask_unchanged[:epoch + 1]
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
        """
        if not self.trees:
            self._log("No trees available for distance computation.")
            raise ValueError("No trees available.")
        labels = self.terminal_names()
        n_labels, label_idx = len(labels), {lbl: i for i, lbl in enumerate(labels)}
        def align_tree_matrix(tree: Tree) -> torch.Tensor:
            """Align single tree's distance matrix to global label set."""
            indices = torch.tensor([label_idx[lbl] for lbl in tree.terminal_names()], dtype=torch.long)
            aligned = torch.full((n_labels, n_labels), float('nan'))
            aligned[indices[:, None], indices] = tree.distance_matrix()[0]
            aligned.fill_diagonal_(0.0)
            return aligned
        def unwrap(result: torch.Tensor | tuple) -> torch.Tensor:
            """Extract tensor from aggregation result (handles nanmedian)."""
            return result[0] if isinstance(result, tuple) else result
        # Parallel matrix computation
        dist_stack = torch.stack(Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(align_tree_matrix)(tree) for tree in tqdm(self.trees, desc="Aligning trees")
        ))
        valid_mask = ~torch.isnan(dist_stack)
        confidence = valid_mask.float().mean(dim=0)
        tol = 1e-14
        if method == "fp":
            # ============================================================
            # OPTIMIZED FIXED-POINT WITH ANDERSON ACCELERATION (m=2)
            # ============================================================
            # Warm start: median is an excellent robust initial estimate
            D = self.distance_matrix(func = torch.nanmedian)[0]
            D.fill_diagonal_(0.0)
            # Pre-compute constants (avoid repeated allocations/computations)
            dist_clean, valid_float = dist_stack.nan_to_num(0.0), valid_mask.float()
            neg_inv_2sigma2 = -0.5 / (sigma_max ** 2)
            # Anderson acceleration history (m=2 is optimal for this problem class)
            G_prev, D_prev = None, None
            pbar = tqdm(range(max_iter), desc="Fixed-point iteration")
            for _ in pbar:
                # Compute G(D) = Gaussian-weighted mean (fused operations)
                residuals = dist_clean - D.unsqueeze(0)
                weights = torch.exp(residuals.square().mul_(neg_inv_2sigma2)).mul_(valid_float)
                G = weights.mul(dist_clean).sum(dim=0).div_(weights.sum(dim=0).clamp_(min=tol))
                G.fill_diagonal_(0.0)
                # Convergence check (before acceleration to measure true residual)
                F = G - D
                pbar.set_postfix({"residual": f"{(max_residual := F.abs().max().item()):.2e}"})
                if max_residual < tol:
                    D = G
                    break
                # Anderson acceleration: extrapolate using secant-like update
                D_new = G
                if G_prev is not None:
                    # dG = G - G_prev, dD = D - D_prev
                    # Optimal mixing: theta = -<F, dG-dD> / ||dG-dD||^2
                    dG, dD = G - G_prev, D - D_prev
                    dF = dG - dD  # change in residual
                    dF_flat, F_flat = dF.view(-1), F.view(-1)
                    if (denom := dF_flat.dot(dF_flat)) > tol:
                        # Clamp for stability, allow slight extrapolation
                        D_new = G + max(-0.5, min(-F_flat.dot(dF_flat) / denom, 2.0)) * dG
                # Store history for next iteration
                G_prev, D_prev, D = G, D, D_new
                D.fill_diagonal_(0.0)
            pbar.close()
            self._log("Distance matrix computation complete.")
            return D, confidence, labels
        # Standard aggregation
        avg_matrix = unwrap(func(dist_stack, dim=0))
        # Interpolate remaining NaNs using row/column means
        if (nan_mask := torch.isnan(avg_matrix)).any():
            row_mean, col_mean = unwrap(func(avg_matrix, dim=1)), unwrap(func(avg_matrix, dim=0))
            avg_matrix = torch.where(nan_mask, (row_mean[:, None] + col_mean[None, :]) / 2, avg_matrix)
        self._log("Distance matrix computation complete.")
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
            Unused, kept for API compatibility.
        Returns
        -------
        List[float]
            Scale factors applied to each tree.
        """
        labels = self.terminal_names()
        n_labels, n_trees = len(labels), len(self.trees)
        label_idx = {lbl: i for i, lbl in enumerate(labels)}
        # Parallel computation of distance matrices and indices
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(lambda t: ([label_idx[lbl] for lbl in t.terminal_names()], t.distance_matrix()[0]))(tree)
            for tree in self.trees
        )
        # Build aligned distance matrices (n_trees, n_labels, n_labels)
        dist_matrices = torch.zeros((n_trees, n_labels, n_labels))
        valid_mask = torch.zeros((n_trees, n_labels, n_labels), dtype=torch.bool)
        for t_idx, (indices, dist_mat) in enumerate(results):
            idx_tensor = torch.tensor(indices, dtype=torch.long)
            ix_grid = (idx_tensor[:, None], idx_tensor)
            dist_matrices[t_idx, ix_grid[0], ix_grid[1]] = dist_mat
            valid_mask[t_idx, ix_grid[0], ix_grid[1]] = True
        # Precompute quadratic form: loss = scales^T @ A @ scales
        N_inv = 1.0 / (n_trees * n_labels * n_labels)
        dist_over_count = dist_matrices / valid_mask.sum(dim=0).clamp(min=1).float()
        A = (torch.diag((dist_matrices ** 2).sum(dim=(1, 2))) - torch.einsum('tij,sij->ts', dist_matrices, dist_over_count)) * N_inv
        A = (A + A.T) * 0.5  # Symmetrize for numerical stability
        A2 = A * 2.0  # Pre-compute for gradient
        # Optimization with explicit gradients and manual Adam
        params, m, v = torch.zeros(n_trees), torch.zeros(n_trees), torch.zeros(n_trees)
        lr, beta1, beta2, eps, tol = 0.1, 0.9, 0.999, 1e-8, 1e-14
        prev_loss, inv_n_trees = float('inf'), 1.0 / n_trees
        with torch.no_grad():
            for step in tqdm(range(1000), desc="Normalizing"):
                # Forward: params -> softplus -> normalize to sum=n_trees
                raw_scales = torch.nn.functional.softplus(params)
                scales = raw_scales * (n_trees / raw_scales.sum())
                # Loss and early stopping
                As = A @ scales
                if abs(prev_loss - (loss_val := scales.dot(As).item())) < tol:
                    break
                prev_loss = loss_val
                # Gradient through normalization and softplus
                grad_scales = A2 @ scales
                grad_params = (n_trees / raw_scales.sum()) * (grad_scales - scales.dot(grad_scales) * inv_n_trees) * torch.sigmoid(params)
                # Adam update
                m.mul_(beta1).add_(grad_params, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad_params, grad_params, value=1.0 - beta2)
                bias_correction1, bias_correction2 = 1.0 - beta1 ** (step + 1), 1.0 - beta2 ** (step + 1)
                params.addcdiv_(m, v.div(bias_correction2).sqrt_().add_(eps), value=-lr / bias_correction1)
        # Compute final scales and apply to trees
        raw_scales = torch.nn.functional.softplus(params)
        final_scales = (raw_scales * (n_trees / raw_scales.sum())).tolist()
        Parallel(n_jobs=-1, prefer="threads")(
            delayed(lambda t, s: [node.set_edge_length(node.get_edge_length() * s) 
                                  for node in t.contents.traverse_postorder() 
                                  if node.get_edge_length() is not None])(tree, scale)
            for tree, scale in zip(self.trees, final_scales)
        )
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
            ('scale_fn', None), ('lr_fn', None), ('weight_exp_fn', None),
            ('normalize', False),
        ]
        params = {k: kwargs.get(k, v) for k, v in defaults}
        if params['normalize']:
            self.normalize(batch_mode=params['precise_opt'])

        params['save_mode'] |= params['export_video']
        params['export_video'] &= params['precise_opt']
        n_trees = len(self.trees)
        n_jobs = min(n_trees, os.cpu_count())
        is_hyperbolic = geometry == 'hyperbolic'
        EmbClass = embedding.LoidEmbedding if is_hyperbolic else embedding.EuclideanEmbedding

        try:
            self._log(f"Starting {geometry} multi-embedding...")
            scale = params['dist_cutoff'] / self.distance_matrix()[0].max() if is_hyperbolic else 1
            curvature = -(scale ** 2) if is_hyperbolic else None

            def process_tree(idx_tree: Tuple[int, Tree]):
                idx, tree = idx_tree
                dist = tree.distance_matrix()[0]
                pts = utils.naive_embedding(dist * scale, dim, geometry=geometry)
                emb = EmbClass(points=pts, labels=tree.terminal_names(), curvature=curvature) if is_hyperbolic \
                    else EmbClass(points=pts, labels=tree.terminal_names())
                return idx, dist, emb

            results = Parallel(n_jobs=n_jobs, backend='loky', return_as='generator')(
                delayed(process_tree)((i, t)) for i, t in enumerate(self.trees)
            )
            dist_mats, emb_list = [None] * n_trees, [None] * n_trees
            for idx, dist, emb in results:
                dist_mats[idx], emb_list[idx] = dist, emb
                self._log(f"Naive {geometry} embedding {idx + 1}/{n_trees} complete")

            multi_emb = embedding.MultiEmbedding()
            for emb in emb_list:
                multi_emb.append(emb)
            del emb_list
            gc.collect()
            self._log(f"Naive {geometry} embeddings complete.")

            if params['precise_opt']:
                self._log("Refining with precise optimization...")
                pts_list, curvature = utils.precise_multiembedding(
                    dist_mats, multi_emb, geometry=geometry,
                    log_fn=self._log, time_stamp=self._timestamp, **params
                )
                multi_emb = embedding.MultiEmbedding()
                for pts, labels in zip(pts_list, [t.terminal_names() for t in self.trees]):
                    multi_emb.append(EmbClass(points=pts, labels=labels, curvature=curvature) if is_hyperbolic
                                     else EmbClass(points=pts, labels=labels))
                del pts_list
                self._log(f"Precise {geometry} embeddings complete.")
            del dist_mats
            gc.collect()

        except Exception as e:
            self._log(f"Multi-embedding error: {e}")
            raise

        # Save result
        out_dir = os.path.join(conf.OUTPUT_DIRECTORY, self._timestamp.strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(out_dir, exist_ok=True)
        filepath = os.path.join(out_dir, f"{geometry}_multiembedding_{dim}d.pkl")
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(multi_emb, f, protocol=pickle.HIGHEST_PROTOCOL)
            self._log(f"Multi-embedding saved to {filepath}")
        except (IOError, pickle.PicklingError) as e:
            self._log(f"Save error: {e}")
            raise

        if params['export_video']:
            self._generate_video(fps=params['epochs'] // conf.VIDEO_LENGTH)
        return multi_emb

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
            'background': '#1a1a2e', 'panel': '#1e2a4a', 'grid': '#2a2a4a',
            'text': '#e8e8e8', 'text_dim': '#a0a0b0', 'accent': '#00d4ff',
            'accent_alt': '#ff6b6b', 'highlight': '#ffd93d',
        }
        plt.rcParams.update({
            'figure.facecolor': THEME['background'], 'figure.edgecolor': THEME['background'],
            'axes.facecolor': THEME['panel'], 'axes.edgecolor': THEME['grid'],
            'axes.labelcolor': THEME['text'], 'axes.titlecolor': THEME['text'],
            'axes.grid': True, 'axes.axisbelow': True, 'axes.linewidth': 0.8,
            'axes.titleweight': 'bold', 'axes.titlesize': 11, 'axes.labelsize': 9,
            'grid.color': THEME['grid'], 'grid.linewidth': 0.4, 'grid.alpha': 0.5,
            'xtick.color': THEME['text_dim'], 'ytick.color': THEME['text_dim'],
            'xtick.labelsize': 8, 'ytick.labelsize': 8, 'text.color': THEME['text'],
            'font.family': 'sans-serif', 'font.size': 9,
            'legend.facecolor': THEME['panel'], 'legend.edgecolor': THEME['grid'], 'legend.fontsize': 8,
        })
        base_dir = os.path.join(conf.OUTPUT_DIRECTORY, self._timestamp.strftime('%Y-%m-%d_%H-%M-%S'))
        # Load metadata
        try:
            n_trees = np.load(os.path.join(base_dir, "metadata.npy"), allow_pickle=True).item()['num_trees']
        except FileNotFoundError:
            n_trees = len([d for d in os.listdir(base_dir) if d.startswith('tree_')])
        # Load aggregate data
        weights = -np.load(os.path.join(base_dir, "weight_exponents.npy"))
        lrs = np.log10(np.load(os.path.join(base_dir, "learning_rates.npy")) + conf.EPSILON)
        agg_costs = np.load(os.path.join(base_dir, "costs.npy"))
        try:
            scales = np.load(os.path.join(base_dir, "scales.npy"))
        except FileNotFoundError:
            scales = None
        n_frames, epochs = len(weights), np.arange(1, len(weights) + 1)
        is_hyperbolic = scales is not None and not np.all(scales == 1)
        if is_hyperbolic:
            scale_active, scale_changed = scales.astype(bool), np.concatenate([[True], np.diff(scales) != 0])
            mask_changing, mask_unchanged = scale_active & scale_changed, scale_active & ~scale_changed
        # Load per-tree data
        self._log(f"Loading data for {n_trees} trees...")
        all_rms = np.array([np.load(os.path.join(base_dir, f"tree_{t}", "rmse.npy")) for t in range(n_trees)])
        all_costs = np.array([np.load(os.path.join(base_dir, f"tree_{t}", "costs.npy")) for t in range(n_trees)])
        # Identify min/max RMS trees
        min_rms_idx, max_rms_idx = np.argmin(all_rms[:, -1]), np.argmax(all_rms[:, -1])
        extremal = {min_rms_idx, max_rms_idx}
        # Axis bounds
        rms_bounds = (max(np.nanmin(all_rms) * 0.9, 1e-20), np.nanmax(all_rms) * 1.1)
        weight_bounds = (weights.min() * 0.95, weights.max() * 1.05)
        cost_bounds = (max(min(np.nanmin(agg_costs), np.nanmin(all_costs)) * 0.9, 1e-20),
                       max(np.nanmax(agg_costs), np.nanmax(all_costs)) * 1.1)
        # Setup output
        out_dir = os.path.join(conf.OUTPUT_VIDEO_DIRECTORY, self._timestamp.strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(out_dir, exist_ok=True)
        video_path = os.path.join(out_dir, 're_evolution_multi.mp4')
        self._log(f"Generating video for {n_trees} trees...")
        # Create figure
        fig, ((ax_rms, ax_weight), (ax_lr, ax_cost)) = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
        cmap, norm = get_cmap('plasma'), Normalize(vmin=0, vmax=n_trees - 1)
        tree_colors = [cmap(norm(i)) for i in range(n_trees)]
        marker_style = dict(marker='o', markersize=3, markeredgecolor='white', markeredgewidth=0.01)
        bbox_base = dict(boxstyle='round,pad=0.2', facecolor=THEME['panel'], linewidth=0.5, alpha=0.9)
        # RMS plot
        lines_rms = [ax_rms.plot([], [], color=tree_colors[t], linewidth=2.0 if t in extremal else 0.5,
                                 alpha=1.0 if t in extremal else 0.3)[0] for t in range(n_trees)]
        ax_rms.set(xlim=(1, n_frames), ylim=rms_bounds, yscale='log', xlabel='Epoch',
                   ylabel='Median RE (log)', title=f'Median Relative Error ({n_trees} Trees)')
        annot_min = ax_rms.annotate('', xy=(0, 0), fontsize=7, fontweight='bold', color='#50fa7b',
                                    bbox={**bbox_base, 'edgecolor': '#50fa7b'})
        annot_max = ax_rms.annotate('', xy=(0, 0), fontsize=7, fontweight='bold', color='#ff5555',
                                    bbox={**bbox_base, 'edgecolor': '#ff5555'})
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax_rms, fraction=0.046, pad=0.04, shrink=0.8).set_label('Tree Index')
        # Weight plot
        line_weight, = ax_weight.plot([], [], color=THEME['accent'], linewidth=2,
                                      markerfacecolor=THEME['highlight'], **marker_style)
        line_scale_on = line_scale_off = None
        if is_hyperbolic:
            line_scale_on, = ax_weight.plot([], [], 'o', markersize=5, markerfacecolor='#ff3333',
                                            markeredgecolor='white', markeredgewidth=0.01, label='Scale Learning On')
            line_scale_off, = ax_weight.plot([], [], 'o', markersize=3, markerfacecolor=THEME['accent'],
                                             markeredgecolor='white', markeredgewidth=0.01, label='Scale Learning Off')
            ax_weight.legend(loc='upper right', fontsize=7)
        ax_weight.set(xlim=(1, n_frames), ylim=weight_bounds, xlabel='Epoch',
                      ylabel='−Weight Exponent', title='Weight Evolution')
        # Learning rate plot
        line_lr, = ax_lr.plot([], [], color='#50fa7b', linewidth=2, markerfacecolor=THEME['accent'], **marker_style)
        ax_lr.set(xlim=(1, n_frames), ylim=(lrs.min() - 0.1, lrs.max() + 0.1), xlabel='Epoch',
                  ylabel='log₁₀(Learning Rate)', title='Learning Rate Schedule')
        # Cost plot
        lines_cost = [ax_cost.plot([], [], color=tree_colors[t], linewidth=0.4, alpha=0.2)[0] for t in range(n_trees)]
        line_agg_cost, = ax_cost.plot([], [], color='white', linewidth=2.5, label='Aggregate')
        ax_cost.set(xlim=(1, n_frames), ylim=cost_bounds, yscale='log', xlabel='Epoch',
                    ylabel='Cost (log)', title='Cost Evolution')
        ax_cost.legend(loc='upper right', fontsize=8)
        fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax_cost,
                     fraction=0.046, pad=0.04, shrink=0.8).set_label('Tree Index')
        # Borders
        for ax in [ax_rms, ax_weight, ax_lr, ax_cost]:
            for spine in ax.spines.values():
                spine.set(edgecolor='#4a4a6a', linewidth=1.0)
        fig.tight_layout()
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        # FFmpeg pipe
        proc = subprocess.Popen([
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'rgba', '-r', str(fps), '-i', '-',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', video_path
        ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            for epoch in range(n_frames):
                x = epochs[:epoch + 1]
                for t_idx in range(n_trees):
                    lines_rms[t_idx].set_data(x, all_rms[t_idx, :epoch + 1])
                    lines_cost[t_idx].set_data(x, all_costs[t_idx, :epoch + 1])
                # Update annotations
                min_val, max_val = all_rms[min_rms_idx, epoch], all_rms[max_rms_idx, epoch]
                annot_min.set_text(f'Min: T{min_rms_idx}')
                annot_min.xy = (epoch + 1, min_val)
                annot_min.set_position((epoch + 1.5, min_val * 0.85))
                annot_max.set_text(f'Max: T{max_rms_idx}')
                annot_max.xy = (epoch + 1, max_val)
                annot_max.set_position((epoch + 1.5, max_val * 1.15))
                line_weight.set_data(x, weights[:epoch + 1])
                if is_hyperbolic:
                    m_on, m_off = mask_changing[:epoch + 1], mask_unchanged[:epoch + 1]
                    line_scale_on.set_data(x[m_on], weights[:epoch + 1][m_on])
                    line_scale_off.set_data(x[m_off], weights[:epoch + 1][m_off])
                line_lr.set_data(x, lrs[:epoch + 1])
                line_agg_cost.set_data(x, agg_costs[:epoch + 1])
                fig.canvas.draw()
                proc.stdin.write(memoryview(fig.canvas.buffer_rgba()))
        finally:
            proc.stdin.close()
            proc.wait()
        plt.close(fig)
        plt.rcdefaults()
        self._log(f"Multi-tree video saved: {video_path}")