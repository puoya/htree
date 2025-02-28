import os
import gc
import copy
import torch
import pickle
import logging
import imageio
import numpy as np
import treeswift as ts
import matplotlib.patches as patches
from datetime import datetime
import matplotlib.pyplot as plt
from . import conf, utils, embedding

# import searborn as sns
# import conf
# import utils
# import embedding

from collections.abc import Collection
from typing import Union, Set, Optional, List, Callable, Tuple, Dict, Iterator

#############################################################################################
# Class for handling tree operations using treeswift and additional utilities.
#############################################################################################
class Tree:
    def __init__(self, *args: Tuple[str, Optional[ts.Tree]], enable_logging: bool = False):
        """
        Initialize a Tree object.
        
        Parameters:
            args (Tuple[str, Optional[ts.Tree]]): 
                Variable length arguments:
                - Single argument: File path (str) to load the tree.
                - Two arguments: Name (str) and treeswift.Tree object.
            enable_logging (bool): Enable logging if True.
        """
        self._logger = None
        self._current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if enable_logging:
            self._setup_logging()

        if len(args) == 1 and isinstance(args[0], str):
            self.name = os.path.basename(args[0])
            self.contents = self._load_tree(args[0])
            self._log_info(f"Initialized tree from file: {args[0]}")
        elif len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], ts.Tree):
            self.name = args[0]
            self.contents = args[1]
            self._log_info(f"Initialized tree with name: {self.name}")
        else:
            raise ValueError("Expected a single file path or a name and treeswift.Tree object.")

    def _setup_logging(self, log_dir: str = conf.LOG_DIRECTORY, log_level: int = logging.INFO, 
                       log_format: str = '%(asctime)s - %(levelname)s - %(message)s') -> None:
        """
        Configure logging.

        Parameters:
            log_dir (str): Directory for log files.
            log_level (int): Logging level.
            log_format (str): Format for log messages.
        """
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"Tree_{self._current_time}.log")
        logging.basicConfig(filename=log_file, level=log_level, format=log_format)
        self._logger = logging.getLogger(__name__)
        self._log_info("Logging initialized.")

    def _log_info(self, message: str) -> None:
        """
        Log an informational message if logging is enabled.

        Parameters:
            message (str): The message to log.
        """
        if self._logger:
            self._logger.info(message)

    @classmethod
    def _from_contents(cls, name: str, contents: ts.Tree, enable_logging: bool = False) -> 'Tree':
        """
        Create a Tree object from a treeswift.Tree object.

        Parameters:
            name (str): Name of the tree.
            contents (ts.Tree): treeswift.Tree object.
            enable_logging (bool): Enable logging if True.

        Returns:
            Tree: A new Tree instance.
        """
        instance = cls(name, contents, enable_logging=enable_logging)
        instance._log_info(f"Tree created from contents with name: {name}")
        return instance

    def copy(self) -> 'Tree':
        """
        Create a deep copy of the Tree object.

        Returns:
            Tree: A deep copy of the current Tree instance.
        """
        tree_copy = copy.deepcopy(self)
        self._log_info(f"Copied tree: {self.name}")
        return tree_copy

    def _load_tree(self, file_path: str) -> ts.Tree:
        """
        Load a tree from a Newick file.

        Parameters:
            file_path (str): Path to the Newick file.

        Returns:
            ts.Tree: The loaded treeswift.Tree object.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        self._log_info(f"Loading tree from: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        return ts.read_tree_newick(file_path)

    def save(self, file_path: str, format: str = 'newick') -> None:
        """
        Save the tree to a file in the specified format.

        Parameters:
            file_path (str): Destination file path.
            format (str): File format ('newick' supported).

        Raises:
            ValueError: If the specified format is unsupported.
        """
        try:
            if format.lower() == 'newick':
                self.contents.write_tree_newick(file_path)
                self._log_info(f"Tree saved: {self.name}")
            else:
                raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            self._log_info(f"Failed to save tree: {self.name}. Error: {e}")
            raise

    def terminal_names(self) -> List[str]:
        """
        Retrieve terminal (leaf) names in the tree.

        Returns:
            List[str]: List of terminal names.
        """
        leaf_names = list(self.contents.labels(leaves=True, internal=False))
        self._log_info(f"Retrieved terminal names for tree: {self.name}")
        return leaf_names

    def distance_matrix(self) -> torch.Tensor:
        """
        Compute the distance matrix for the tree.

        Returns:
            torch.Tensor: Distance matrix with terminal names as labels.
        """
        labels = self.terminal_names()
        n = len(labels)
        dist_matrix = torch.zeros((n, n))
        dist_dict = self.contents.distance_matrix(leaf_labels=True)
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        for label1, row in dist_dict.items():
            i = label_to_idx[label1]
            for label2, dist in row.items():
                j = label_to_idx[label2]
                dist_matrix[i, j] = torch.tensor(dist)
        self._log_info(f"Computed distance matrix for tree: {self.name}")
        return dist_matrix

    def diameter(self) -> torch.Tensor:
        """
        Calculate the diameter of the tree.

        Returns:
            torch.Tensor: Diameter of the tree.
        """
        tree_diameter = torch.tensor(self.contents.diameter())
        self._log_info(f"Tree diameter: {tree_diameter.item()}")
        return tree_diameter

    def normalize(self) -> None:
        """
        Normalize the tree branch lengths so that the tree's diameter is 1.
        """
        self._log_info(f"Normalizing tree: {self.name}")
        tree_diameter = self.diameter().item()
        if not np.isclose(tree_diameter, 0.0):
            scale_factor = 1.0 / tree_diameter
            for node in self.contents.traverse_postorder():
                if node.get_edge_length() is not None:
                    node.set_edge_length(node.get_edge_length() * scale_factor)
            self._log_info(f"Tree normalized with scale factor: {scale_factor}")
        else:
            self._log_info(f"Tree diameter is zero and cannot be normalized.")
    
    def embed(self, dimension: int, geometry: str = 'hyperbolic', **kwargs) -> 'Embedding':
        """
        Embed the tree into a specified geometric space (hyperbolic or Euclidean).

        Parameters:
        - dimension (int): The number of dimensions for the embedding (required).
        - geometry (str): Type of geometric space to use: 'hyperbolic' or 'euclidean' (default is 'hyperbolic').
        - accurate (bool, optional): Accurate embedding. 
          Choices are False for a basic embedding or True for an advanced, optimized embedding (default is False).
        - total_epochs (int, optional): Number of epochs for optimization in accurate mode (default is 2000).
        - initial_lr (float, optional): Initial learning rate for the optimization process (default is 0.1).
        - max_diameter (float, optional): Maximum diameter for scaling the distance matrix (default is 10.0).
        - enable_save (bool, optional): Whether to save intermediate states during optimization (default is True).
        - enable_movie (bool, optional): Whether to generate a visualization of the embedding process (default is True).
        - learning_rate (float, optional): Specific learning rate for the optimization process (default is None, which uses adaptive learning).
        - scale_learning (callable, optional): Function to dynamically adjust the scale during optimization (default is None).
        - weight_exponent (float, optional): Exponent for weights in optimization, affecting how distances are weighted (default is None).
        
        Returns:
        - Embedding: The resulting embedding of the tree in the specified geometric space.

        Raises:
        - ValueError: If 'dimension' is not provided.
        - RuntimeError: For any errors encountered during the embedding process.
        """

        if dimension is None:
            raise ValueError("The 'dimension' parameter is required.")
        accurate        = kwargs.get('accurate', conf.ENABLE_ACCURATE_OPTIMIZATION)
        total_epochs    = kwargs.get('total_epochs', conf.TOTAL_EPOCHS)
        initial_lr      = kwargs.get('initial_lr', conf.INITIAL_LEARNING_RATE)
        max_diameter    = kwargs.get('max_diameter', conf.MAX_RANGE)
        enable_save     = kwargs.get('enable_save', conf.ENABLE_SAVE_MODE)
        enable_movie    = kwargs.get('enable_movie', conf.ENABLE_VIDEO_EXPORT)
        learning_rate   = kwargs.get('learning_rate', None)
        scale_learning  = kwargs.get('scale_learning', None)
        weight_exponent = kwargs.get('weight_exponent', None)
        enable_save     = enable_movie or enable_save
        
        if geometry == 'hyperbolic':
            try:
                scale_factor = max_diameter / self.diameter()
                initial_curvature = -(scale_factor ** 2)
                distance_matrix = self.distance_matrix() * scale_factor

                # Naive hyperbolic embedding
                self._log_info("Initiating naive embedding of the tree in hyperbolic space.")
                gramian = -torch.cosh(distance_matrix)
                points = utils.lgram_to_points(gramian, dimension)
                for n in range(points.size(1)):
                    points[:,n] = utils.project_to_hyperbolic_space(points[:,n])
                    points[0,n] = torch.sqrt(1+torch.sum(points[1:,n]**2))
                embeddings = embedding.LoidEmbedding(points=points, labels = self.terminal_names(), curvature=initial_curvature)
                self._log_info("Naive hyperbolic embedding completed.")

                if accurate:
                    self._log_info("Initiating precise hyperbolic embedding.")
                    initial_tangents = utils.hyperbolic_log(points)
                    tangents, scale = utils.hyperbolic_embedding(   distance_matrix, 
                                                                    dimension,
                                                                    initial_tangents    = initial_tangents,
                                                                    total_epochs        = total_epochs,
                                                                    log_function        = self._log_info,
                                                                    learning_rate       = learning_rate,
                                                                    scale_learning      = scale_learning,
                                                                    weight_exponent     = weight_exponent,
                                                                    initial_lr          = initial_lr,
                                                                    enable_save         = enable_save,
                                                                    time                = self._current_time
                                                                )
                    embeddings.curvature *= np.abs(scale) ** 2
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
        else:
            try:
                distance_matrix = self.distance_matrix() 

                # Naive hyperbolic embedding
                self._log_info("Initiating naive embedding of the tree in euclidean space.")
                n = distance_matrix.size(0)
                J = torch.eye(n) - torch.ones((n, n)) / n
                gramian = -0.5*J@distance_matrix@J
                eigenvalues, eigenvectors = torch.linalg.eigh(gramian)
                sorted_indices = torch.argsort(eigenvalues, descending=True)
                eigenvalues = eigenvalues[sorted_indices]
                eigenvectors = eigenvectors[:, sorted_indices]
                top_eigenvalues = eigenvalues[:dimension].clamp(min=0)  # Clamp to ensure non-negative
                top_eigenvectors = eigenvectors[:, :dimension]
                points = top_eigenvectors * torch.sqrt(top_eigenvalues).unsqueeze(0)
                points = points.t()
                embeddings = embedding.EuclideanEmbedding(points=points, labels = self.terminal_names())
                self._log_info("Naive euclidean embedding completed.")

                # Precise hyperbolic embedding
                if accurate:
                    self._log_info("Initiating precise euclidean embedding.")
                    points = utils.euclidean_embedding( distance_matrix,
                                                        dimension,
                                                        initial_points  = points,
                                                        epochs          = total_epochs,
                                                        log_function    = self._log_info,
                                                        learning_rate   = learning_rate,
                                                        weight_exponent = weight_exponent,
                                                        initial_lr      = initial_lr,
                                                        enable_save     = enable_save,
                                                        time            = self._current_time
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

        if enable_movie and accurate:
            self._create_figures()
            fps = total_epochs // conf.VIDEO_LENGTH 
            self._create_movie(fps = fps)
        directory = f'{conf.OUTPUT_DIRECTORY}/{self._current_time}'
        os.makedirs(directory, exist_ok=True)
        filename = f'{geometry}_space.pkl'
        filepath = f'{directory}/{filename}'
        try:
            with open(filepath, 'wb') as file:
                pickle.dump(embeddings, file)
            self._log_info(f"Object successfully saved to {filepath}")
        except IOError as ioe:
            self._log_info(f"IO error while saving the object: {ioe}")
            raise
        except pickle.PicklingError as pe:
            self._log_info(f"Pickling error while saving the object: {pe}")
            raise
        except Exception as e:
            self._log_info(f"Unexpected error while saving the object: {e}")
            raise
        return embeddings

    def _create_figures(self):
        """
        Create and save figures based on RE matrices and distance matrices.
        """
        timestamp = self._current_time
        output_dir = f'{conf.OUTPUT_FIGURES_DIRECTORY}/{timestamp}'
        os.makedirs(output_dir, exist_ok=True)
        path = f'{conf.OUTPUT_DIRECTORY}/{timestamp}'
        weight_history = -np.load(os.path.join(path, "weight_history.npy"))
        lr_history = np.log10(np.abs(np.load(os.path.join(path, "lr_history.npy"))))        
        npy_files = sorted([f for f in os.listdir(path) if f.startswith('RE') and f.endswith('.npy')],
                           key=lambda f: int(f.split('_')[1].split('.')[0]))
        total_epochs = len(npy_files)
        scale_history_path = os.path.join(path, "scale_history.npy")
        if os.path.exists(scale_history_path):
            scale_history = np.load(scale_history_path)
            flag = True
        else:
            flag = False

        log10_distance_matrix = np.log10(self.distance_matrix() + conf.EPSILON)
        mask = np.eye(log10_distance_matrix.shape[0], dtype=bool)

        rms_values = []
        upper_diag_indices = None
        min_heatplot = float('inf')
        max_heatplot = float('-inf')
        for file_name in npy_files:
            re_matrix = np.load(os.path.join(path, file_name))
            log10_re_matrix = np.log10(re_matrix + conf.EPSILON)
            np.fill_diagonal(log10_re_matrix, np.nan)
            current_min = np.nanmin(log10_re_matrix)
            current_max = np.nanmax(log10_re_matrix)
            min_heatplot = min(min_heatplot, current_min)
            max_heatplot = max(max_heatplot, current_max)
            if upper_diag_indices is None:
                upper_diag_indices = np.triu_indices_from(re_matrix, k=1)
            upper_diag_elements = re_matrix[upper_diag_indices]
            rms = np.sqrt(np.mean(upper_diag_elements))
            rms_values.append(rms)
        del log10_re_matrix, re_matrix, upper_diag_indices
        gc.collect()

        if rms_values:
            max_rms = max(rms_values) * 1.1
            min_rms = min(rms_values) * 0.9
        else:
            max_rms = 1
            min_rms = 0
        max_lr = max(lr_history)+0.1
        min_lr = min(lr_history)-0.1

        for i, file_name in enumerate(npy_files):
            epoch = int(file_name.split('_')[1].split('.')[0])
            save_path = os.path.join(output_dir, f'heatmap_{epoch}.png')

            log10_re_matrix = np.log10(np.load(os.path.join(path, file_name)) + conf.EPSILON)

            fig = plt.figure(figsize=(12, 12), tight_layout=True)
            gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 2, 2], width_ratios=[1, 1])

            ax_rms = fig.add_subplot(gs[0, :])
            ax_rms.plot(range(1, len(rms_values[:epoch]) + 1), rms_values[:epoch], marker='o')
            ax_rms.set_xlim(1, total_epochs)
            ax_rms.set_ylim(min_rms, max_rms)
            ax_rms.set_xlabel('Epoch')
            ax_rms.set_ylabel('RMS of RE')
            ax_rms.set_title('Evolution of Relative Errors')

            # Additional plots for weight and learning rate
            ax_weight = fig.add_subplot(gs[1, 0])
            epochs_range = range(1, len(weight_history[:epoch]) + 1)
            weight_color = 'blue'
            highlight_color = 'red'
            if flag:
                highlight_mask = scale_history[:epoch]

                ax_weight.plot(epochs_range, weight_history[:epoch], marker='o', linestyle='', color=weight_color, label='Scale Learning Disabled')
                if highlight_mask is not None:
                    highlighted_weights = [weight_history[i] if highlight_mask[i] else np.nan for i in range(len(weight_history[:epoch]))]
                    ax_weight.plot(epochs_range, highlighted_weights, marker='o', linestyle='', color=highlight_color, label='Scale Learning Enabled')

                ax_weight.set_xlim(1, total_epochs)
                ax_weight.set_ylim(0, 1)
                ax_weight.set_xlabel('Epoch')
                ax_weight.set_ylabel('-Weight Exponent')
                ax_weight.set_title('Evolution of Weights')
                ax_weight.legend()
            else:
                ax_weight.plot(epochs_range, weight_history[:epoch], marker='o', linestyle='', color=weight_color)
                ax_weight.set_xlim(1, total_epochs)
                ax_weight.set_ylim(0, 1)
                ax_weight.set_xlabel('Epoch')
                ax_weight.set_ylabel('-Weight Exponent')
                ax_weight.set_title('Evolution of Weights')

            ax_lr = fig.add_subplot(gs[1, 1])
            ax_lr.plot(range(1, len(lr_history[:epoch]) + 1), lr_history[:epoch], marker='o')
            ax_lr.set_xlim(1, total_epochs)
            ax_lr.set_ylim(min_lr, max_lr)
            ax_lr.set_xlabel('Epoch')
            ax_lr.set_ylabel('log10(Learning Rate)')
            ax_lr.set_title('Evolution of Learning Rates')

            ax1 = fig.add_subplot(gs[2:, 0])
            ax2 = fig.add_subplot(gs[2:, 1])

            # sns.heatmap(log10_re_matrix, mask=mask, ax=ax1, cmap='viridis', cbar_kws={'label': 'log10(RE)'}, vmin=min_heatplot, vmax=max_heatplot, square=True, xticklabels=False, yticklabels=False)
            # ax1.set_title(f'Relative Error (RE) Matrix (Epoch {epoch})')
            # cbar = ax1.collections[0].colorbar
            # cbar.set_ticks(cbar.get_ticks())
            # cbar.set_ticklabels([f'{tick:.2f}' for tick in cbar.get_ticks()])


            log10_re_matrix_masked = np.where(mask, np.nan, log10_re_matrix)
            log10_distance_matrix_masked = np.where(mask, np.nan, log10_distance_matrix)

            im1 = ax1.imshow(log10_re_matrix_masked, cmap='viridis', vmin=min_heatplot, vmax=max_heatplot)
            ax1.set_title(f'Relative Error (RE) Matrix (Epoch {epoch})')
            # Add colorbar
            cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label('log10(RE)')
            cbar1.set_ticks(cbar1.get_ticks())
            cbar1.set_ticklabels([f'{tick:.2f}' for tick in cbar1.get_ticks()])

            # Plot second heatmap (Distance Matrix)
            im2 = ax2.imshow(log10_distance_matrix_masked, cmap='viridis')
            ax2.set_title('Distance Matrix')

            # Add colorbar
            cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label('log10(Distance)')
            cbar2.set_ticks(cbar2.get_ticks())
            cbar2.set_ticklabels([f'{tick:.2f}' for tick in cbar2.get_ticks()])

            # sns.heatmap(log10_distance_matrix, mask=mask, ax=ax2, cmap='viridis', cbar_kws={'label': 'log10(Distance)'}, square=True, xticklabels=False, yticklabels=False)
            # ax2.set_title('Distance Matrix')
            # cbar = ax2.collections[0].colorbar
            # cbar.set_ticks(cbar.get_ticks())
            # cbar.set_ticklabels([f'{tick:.2f}' for tick in cbar.get_ticks()])
            # Hide x and y ticks for heatmaps
            for ax in (ax1, ax2):
                ax.set_xticks([])
                ax.set_yticks([])

            # Add thin black frame around heatmaps
            for ax in [ax1, ax2]:
                rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color='black', fill=False, linewidth=2)
                ax.add_patch(rect)

            plt.tight_layout()
            plt.savefig(save_path, dpi=200)
            plt.close('all')

            # Free memory explicitly
            del log10_re_matrix, fig, gs, ax_rms, ax_weight, ax_lr, ax1, ax2
            gc.collect()
            self._log_info(f"Figure saved: {save_path}")
        gc.collect()

    def _create_movie(self, fps: int = 10):
        """
        Create a movie from the saved heatmaps.

        Parameters:
        - fps (int): Frames per second for the movie.
        """
        
        timestamp = self._current_time    
        output_dir = f'{conf.OUTPUT_VIDEO_DIRECTORY}/{timestamp}'
        os.makedirs(output_dir, exist_ok=True)

        image_dir = f'{conf.OUTPUT_FIGURES_DIRECTORY}/{timestamp}'
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

        # Custom sorting function to sort filenames numerically based on the heatmap number
        def extract_number(filename):
            base_name = filename.split('_')[-1].replace('.png', '')
            return int(base_name)

        image_files.sort(key=extract_number)  # Ensure files are in the correct numerical order
        
        total_epoch = len(image_files)
        # Construct full file paths for images
        image_files = [os.path.join(image_dir, f) for f in image_files]
        movie_path = os.path.join(output_dir, 're_vs_distance_evolution.mp4')
        try:
            with imageio.get_writer(movie_path, fps=fps) as writer:
                for image_file in image_files:
                    image = imageio.imread(image_file)
                    writer.append_data(image)

            self._log_info(f"Movie created: {movie_path}")
        except Exception as e:
            self._log_info(f"Error creating movie: {e}")
            raise

    def __repr__(self) -> str:
        """
        Return a string representation of the Tree object.

        Returns:
            str: String representation of the tree.
        """
        repr_str = f"Tree({self.name})"
        self._log_info(f"Tree representation: {repr_str}")
        return repr_str
#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
class MultiTree:
    def __init__(self, *source: Union[str, List[Union['Tree', 'ts.Tree']]], enable_logging: bool = False):
        """
        Initialize a MultiTree object. Load trees from a source, which can be:
         -- a string (file path); trees are loaded from the file
         -- a list of Tree objects
         -- a list of treeswift.Tree objects (automatically wrapped in Tree instances)

        Parameters:
        enable_logging (bool): Flag to enable or disable logging.
        """
        self._logger = None
        self._current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if enable_logging:
            self._setup_logging()

        self.trees = []  # Store trees as a list instead of a dictionary
        if len(source) == 1 and isinstance(source[0], str):
            # If the input is a file path, load trees from file
            file_path = source[0]
            self.name = os.path.basename(file_path)
            self.trees = self._load_trees(file_path)
            self._log_info(f"Initialized MultiTree with trees from file: {file_path}")
        elif len(source) == 2 and isinstance(source[0], str) and isinstance(source[1], list):
            # If the input is a list of Tree objects or treeswift.Tree objects
            self.name = source[0]
            if all(isinstance(t, Tree) for t in source[1]):
                self.trees = source[1]  # Use as-is if they are Tree instances
            elif all(isinstance(t, ts.Tree) for t in source[1]):
                self.trees = [Tree(f"Tree_{i}", ts_tree) for i, ts_tree in enumerate(source[1])]
            else:
                raise ValueError("All elements in the list must be either Tree or treeswift.Tree instances.")
            self._log_info(f"Initialized MultiTree with a list of trees. Name: {self.name}")
        else:
            raise ValueError("Provide a valid source: a file path as a string or a name (string) and a list of Tree or treeswift.Tree objects.")

    def __getitem__(self, index: Union[int, slice]) -> Union['Tree', 'MultiTree']:
        """
        Make MultiTree subscriptable. Allows retrieving individual trees or a sub-MultiTree.
        """
        if isinstance(index, slice):
            return MultiTree(self.name, self.trees[index])
        return self.trees[index]

    def __len__(self) -> int:
        """Return the number of trees in the MultiTree."""
        length = len(self.trees)
        self._log_info(f"Number of trees: {length}")
        return len(self.trees)

    def __iter__(self) -> Iterator['Tree']:
        """Allow iteration over the trees in MultiTree."""
        self._log_info("Returning an iterator over trees.")
        return iter(self.trees)

    def __contains__(self, item) -> bool:
        """
        Check if an item is in the collection.

        Parameters:
        item: The item to check for.

        Returns:
        bool: True if the item is in the collection, False otherwise.
        """
        contains = item in self.trees
        self._log_info(f"Item {'is' if contains else 'is not'} in MultiTree.")
        return contains

    def __repr__(self) -> str:
        """
        Return a string representation of the MultiTree object.
        """
        repr_str = f"MultiTree({self.name}, {len(self.trees)} trees)"
        self._log_info(f"Representation: {repr_str}")
        return repr_str


    def _load_trees(self, file_path: str) -> List['Tree']:
        """
        Load trees from a Newick file and return a list of Tree objects.
        """
        if not os.path.exists(file_path):
            self._log_info(f"The file {file_path} does not exist")
            raise FileNotFoundError(f"The file {file_path} does not exist")
        
        self._log_info(f"Loading multitree from file: {file_path}")
        tree_list = []
        try:
            for idx, tree in enumerate(ts.read_tree_newick(file_path)):
                tree_list.append(Tree(f'tree_{idx+1}', tree))
            self._log_info(f"Loaded {len(tree_list)} trees from {file_path}.")
        except Exception as e:
            self._log_info(f"Failed to load trees from {file_path}: {e}")
            raise ValueError(f"Failed to load trees from {file_path}: {e}")
        return tree_list

    def _setup_logging(self, log_dir: str = conf.LOG_DIRECTORY, log_level: int = logging.INFO, log_format: str = '%(asctime)s - %(levelname)s - %(message)s'):
        """
        Set up logging configuration.
        """
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"MultiTree_{self._current_time}.log")
        logging.basicConfig(filename=log_file, level=log_level, format=log_format)
        self._logger = logging.getLogger(__name__)

    def _log_info(self, message: str):
        """
        Log an informational message.
        """
        if self._logger:
            self._logger.info(message)

    def copy(self) -> 'MultiTree':
        """
        Create a deep copy of the MultiTree object.
        """
        multitree_copy = copy.deepcopy(self)
        self._log_info(f"MultiTree '{self.name}' copied.")
        return multitree_copy

    def save(self, file_path: str, format: str = 'newick') -> None:
        """
        Save all trees to a file in the specified format.
        """
        if format == 'newick':
            try:
                with open(file_path, 'w') as f:
                    for tree in self.trees:
                        f.write(tree.contents.newick() + "\n")
                self._log_info(f"Trees saved to {file_path} in {format} format.")
            except Exception as e:
                self._log_info(f"Failed to save trees to {file_path}: {e}")
                raise
        else:
            self._log_info(f"Unsupported format: {format}")
            raise ValueError(f"Unsupported format: {format}")

    def distance_matrix(self, 
                        func: Callable[[torch.Tensor], torch.Tensor] = torch.nanmean, 
                        confidence: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute the distance matrix of each individual tree and compute the element-wise aggregate
        (mean, median, max, min) of them using PyTorch tensors. The aggregate function can be specified as an input.
        
        Additionally, return a confidence matrix indicating the ratio of non-NaN values at each element.

        Parameters:
        func (Callable[[torch.Tensor], torch.Tensor]): Function to compute the aggregate. Default is torch.nanmean.
        confidence (bool): If True, also return the confidence matrix. Default is False.

        Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]:
            - The aggregated distance matrix as a torch.Tensor.
            - (Optional) Confidence matrix indicating the ratio of non-NaN values.
        """

        if not self.trees:
            self._log_info("No trees available to compute distance matrices.")
            raise ValueError("No trees available to compute distance matrices")

        all_labels = self.terminal_names()
        n = len(all_labels)
        
        stacked_matrices = torch.zeros((len(self.trees),n, n))
        stacked_counts = torch.zeros((len(self.trees),n, n))
        cnt = 0

        for tree in self.trees:
            labels = tree.terminal_names()
            idx = [all_labels.index(label) for label in labels]
            idx = torch.tensor(idx)
            
            # Populate the distance matrix
            distance_matrix = torch.full((n, n), float('nan'))
            distance_matrix[idx[:, None], idx] = tree.distance_matrix()
            distance_matrix.fill_diagonal_(0)
            stacked_matrices[cnt,:,:] = distance_matrix

            if confidence:
                count_matrix = torch.zeros((n, n))  # Matrix to count non-NaN entries
                count_matrix[idx[:, None], idx] = 1
                count_matrix.fill_diagonal_(0)
                stacked_counts[cnt,:,:] = count_matrix
            cnt = cnt +1

        # Apply the aggregation function element-wise, excluding NaNs
        aggregated_matrix = func(stacked_matrices, dim=0)
        if isinstance(aggregated_matrix, tuple):
            aggregated_matrix = aggregated_matrix[0]


        # Identify indices where NaN values exist
        nan_indices = torch.nonzero(torch.isnan(aggregated_matrix), as_tuple=False)
        # Iterate over all NaN indices and compute the replacement values
        aggregated_matrix_ = aggregated_matrix.clone()
        for idx in nan_indices:
            i, j = idx[0], idx[1]
            
            # Extract non-NaN values in the i-th row and j-th column
            row_non_nan = aggregated_matrix_[i, ~torch.isnan(aggregated_matrix_[i, :])]
            col_non_nan = aggregated_matrix_[~torch.isnan(aggregated_matrix_[:, j]), j]

            if row_non_nan.numel() == 0 and col_non_nan.numel() == 0:
                continue  # No non-NaN values available to estimate
            combined_non_nan = torch.cat((row_non_nan, col_non_nan))
            # Replace the NaN value with the estimated value
            aggregated_matrix[i, j] = func(combined_non_nan)

        self._log_info("Distance matrix computation complete.")

        if confidence:
            # Compute confidence matrix
            confidence_matrix = torch.sum(stacked_counts > 0, dim=0) / len(self.trees)
            self._log_info("Confidence matrix computation complete.")
            return aggregated_matrix, confidence_matrix
        else:
            return aggregated_matrix

    def terminal_names(self) -> List[str]:
        """
        Retrieve terminal (leaf) names in the MultiTree.

        Returns:
            List[str]: List of terminal names.
        """
        leaf_names = set()
        for tree in self.trees:
            leaf_names.update(tree.terminal_names())
        leaf_names = sorted(leaf_names)
        self._log_info(f"Retrieved terminal names for MultiTree: {self.name}")
        return leaf_names

    def embed(self, dimension: int, geometry: str = 'hyperbolic', **kwargs) -> 'Embedding':
        """
        Embed the MultiTree into a specified geometric space (hyperbolic or Euclidean).

        Parameters:
        - dimension (int): The number of dimensions for the embedding (required).
        - geometry (str): Type of geometric space to use: 'hyperbolic' or 'euclidean' (default is 'hyperbolic').
        - accurate (bool, optional): Accurate embedding. 
          Choices are 'naive' for a basic embedding or 'precise' for an advanced, optimized embedding (default is 'naive').
        - total_epochs (int, optional): Number of epochs for optimization in accurate mode (default is 2000).
        - initial_lr (float, optional): Initial learning rate for the optimization process (default is 0.1).
        - max_diameter (float, optional): Maximum diameter for scaling the distance matrix (default is 10.0).
        - enable_save (bool, optional): Whether to save intermediate states during optimization (default is True).
        - learning_rate (float, optional): Specific learning rate for the optimization process (default is None, which uses adaptive learning).
        - scale_learning (callable, optional): Function to dynamically adjust the scale during optimization (default is None).
        - weight_exponent (float, optional): Exponent for weights in optimization, affecting how distances are weighted (default is None).

        Returns:
        - Embedding: The resulting embedding of the tree in the specified geometric space.

        Raises:
        - ValueError: If 'dimension' is not provided.
        - RuntimeError: For any errors encountered during the embedding process.
        """

        if dimension is None:
            raise ValueError("The 'dimension' parameter is required.")
        
        # Extract and set parameters for embedding
        accurate = kwargs.get('accurate', conf.ENABLE_ACCURATE_OPTIMIZATION)
        total_epochs = kwargs.get('total_epochs', conf.TOTAL_EPOCHS)
        initial_lr = kwargs.get('initial_lr', conf.INITIAL_LEARNING_RATE)
        max_diameter = kwargs.get('max_diameter', conf.MAX_RANGE)
        enable_save = kwargs.get('enable_save', conf.ENABLE_SAVE_MODE)
        learning_rate = kwargs.get('learning_rate', None)
        scale_learning = kwargs.get('scale_learning', None)
        weight_exponent = kwargs.get('weight_exponent', None)

        if geometry == 'hyperbolic':
            try:
                # Constants and initial setup
                diameters = []
                distance_matrices = []
                for tree in self.trees:
                    diameters.append(tree.diameter())
                diameters = torch.tensor(diameters)

                scale_factor = max_diameter / torch.max(diameters)
                initial_curvature = -(scale_factor ** 2)
                distance_matrices = {}

                # Naive hyperbolic embedding
                multi_embeddings = embedding.MultiEmbedding()
                for name, tree in enumerate(self.trees):
                    self._log_info(f"Initiating naive embedding of {name} in hyperbolic space.")
                    # print(tree.distance_matrix(),scale_factor)
                    distance_matrix = tree.distance_matrix() * scale_factor
                    distance_matrices[name] = distance_matrix
                    n = distance_matrix.size(0)
                    gramian = -torch.cosh(distance_matrix)
                    points = utils.lgram_to_points(gramian, dimension).detach()
                    points= points.double()
                    for n in range(points.size(1)):
                        points[:,n] = utils.project_to_hyperbolic_space(points[:,n])
                        points[0,n] = torch.sqrt(1+torch.sum(points[1:,n]**2))

                    # Initialize Loid embedding
                    multi_embeddings.add_embedding(embedding.LoidEmbedding( points      = points, 
                                                                            labels      = tree.terminal_names(), 
                                                                            curvature   = initial_curvature))
                    self._log_info(f"Naive hyperbolic embedding of {name} is completed.")
                # Precise hyperbolic embedding
                if accurate:
                    initial_tangents = {}
                    self._log_info("Initiating precise hyperbolic embedding.")
                    for name, tree in enumerate(self.trees):
                        initial_tangents[name] =  utils.hyperbolic_log(multi_embeddings.embeddings[name].points)

                    tangents_dic, consensus_scale = utils.hyperbolic_embedding_consensus(distance_matrices,
                                                                                         dimension,
                                                                                         initial_tangents   = initial_tangents,
                                                                                         total_epochs       = total_epochs,
                                                                                         log_function       = self._log_info,
                                                                                         learning_rate      = learning_rate,
                                                                                         scale_learning     = scale_learning,
                                                                                         weight_exponent    = weight_exponent,
                                                                                         initial_lr         = initial_lr,
                                                                                         enable_save        = enable_save,
                                                                                         time               = self._current_time
                                                                                         )
                    for name, tree in enumerate(self.trees):
                        multi_embeddings.embeddings[name].points    = utils.hyperbolic_exponential(tangents_dic[name])
                        multi_embeddings.embeddings[name].curvature = (multi_embeddings.embeddings[name].curvature)*consensus_scale
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
        else:
            try:
                distance_matrices = {}
                # Naive hyperbolic embedding
                
                multi_embeddings = embedding.MultiEmbedding()
                for name, tree in enumerate(self.trees):
                    self._log_info(f"Initiating naive embedding of {name} in hyperbolic space.")
                    distance_matrix = tree.distance_matrix()
                    distance_matrices[name] = distance_matrix
                    n = distance_matrix.size(0)

                    # Step 1: Convert the distance matrix to a Gram matrix using double centering
                    J = torch.eye(n) - torch.ones((n, n)) / n
                    gramian = -0.5 * J @ distance_matrix @ J
                    # Step 2: Perform eigen decomposition on the Gram matrix
                    eigenvalues, eigenvectors = torch.linalg.eigh(gramian)
                    # Step 3: Sort eigenvalues and eigenvectors in descending order
                    sorted_indices = torch.argsort(eigenvalues, descending=True)
                    eigenvalues = eigenvalues[sorted_indices]
                    eigenvectors = eigenvectors[:, sorted_indices]
                    # Step 4: Select the top "dimension" eigenvalues and corresponding eigenvectors
                    top_eigenvalues = eigenvalues[:min(dimension, n)].clamp(min=0)  # Clamp to ensure non-negative
                    top_eigenvectors = eigenvectors[:, :min(dimension, n)]
                    # Step 5: Compute the coordinates in d-dimensional space
                    points = top_eigenvectors * torch.sqrt(top_eigenvalues).unsqueeze(0)
                    points = points.t()
                    points = points.double()
                    if dimension > n:
                        zeros_to_append = torch.zeros((dimension - n, points.size(1)), dtype=points.dtype)
                        points = torch.cat((points, zeros_to_append), dim=0)
                    self._log_info(f"Naive hyperbolic embedding of {name} is completed.")
                    
                    # Precise hyperbolic embedding
                    if accurate:    
                        self._log_info("Initiating precise euclidean embedding.")
                        points = utils.euclidean_embedding(distance_matrix,
                                                           dimension,
                                                           initial_points   = points,
                                                           total_epochs     = total_epochs,
                                                           log_function     = self._log_info,
                                                           learning_rate    = learning_rate,
                                                           weight_exponent  = weight_exponent,
                                                           initial_lr       = initial_lr,
                                                           enable_save      = enable_save,
                                                           time             = self._current_time
                                                           )
                        self._log_info(f"Precise euclidean embedding of {name} is completed.")
                    multi_embeddings.add_embedding(embedding.EuclideanEmbedding(points=points,labels=tree.terminal_names()))
            except ValueError as ve:
                self._log_info(f"Value error during euclidean embedding: {ve}")
                raise
            except RuntimeError as re:
                self._log_info(f"Runtime error during euclidean embedding: {re}")
                raise
            except Exception as e:
                self._log_info(f"Unexpected error during euclidean embedding: {e}")
                raise
                
        directory = f'{conf.OUTPUT_DIRECTORY}/{self._current_time}'
        os.makedirs(directory, exist_ok=True)
        filename = f'{geometry}_spaces.pkl'
        filepath = f'{directory}/{filename}'
        try:
            with open(filepath, 'wb') as file:
                pickle.dump(multi_embeddings, file)
            self._log_info(f"Object successfully saved to {filepath}")
        except IOError as ioe:
            self._log_info(f"IO error while saving the object: {ioe}")
            raise
        except pickle.PicklingError as pe:
            self._log_info(f"Pickling error while saving the object: {pe}")
            raise
        except Exception as e:
            self._log_info(f"Unexpected error while saving the object: {e}")
            raise
        
        return multi_embeddings
