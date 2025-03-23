from collections.abc import Collection
from typing import Union, Set, Optional, List, Callable, Tuple, Dict, Iterator

from tree_collections import Tree
# # import htree.logger

# Enable logging for the session
import logger
import torch
import treeswift as ts

import matplotlib
matplotlib.use('Qt5Agg')  # Or use 'TkAgg' for Tkinter



def custom_learning_rate(epoch: int, total_epochs: int, loss_list: List[float]) -> float:
    """ 
    Calculate a dynamic learning rate based on the current epoch and total number of epochs.
    Parameters:
    - epoch (int): The current epoch in the training process.
    - total_epochs (int): The total number of epochs in the training process.
    - loss_list (list): A list of recorded loss values (can be used for further custom logic).

    Returns:
    - float: The dynamic learning rate for the current epoch.

    Raises:
    - ValueError: If `total_epochs` is less than or equal to 1.
    """

    if total_epochs <= 1:
        raise ValueError("Total epochs must be greater than 1.")

    # Example: Reduce learning rate as training progresses
    decay_factor = 0.5  # Factor by which to decay the learning rate
    loss_threshold = 0.01  # Loss threshold for further reduction
    decay_start_epoch = int(0.7 * total_epochs)  # When to start decaying

    # Reduce learning rate if the epoch is beyond a certain point
    if epoch > decay_start_epoch:
        # Learning rate decays based on the remaining epochs
        decay_rate = 1 - (epoch - decay_start_epoch) / (total_epochs - decay_start_epoch)
    else:
        decay_rate = 1.0  # No decay before the threshold
    # Further adjust learning rate if recent loss has not improved sufficiently
    if len(loss_list) > 1 and loss_list[-1] > loss_threshold:
        decay_rate *= decay_factor
    return  decay_rate

def custom_scale(epoch: int, total_epochs: int, loss_list: List[float]) -> bool:
    """
    Determine whether scale learning should occur based on the current epoch and total number of epochs.

    Parameters:
    - epoch (int): The current epoch in the training process.
    - total_epochs (int): The total number of epochs in the training process.
    - loss_list (list): A list of recorded loss values (can be used for further custom logic).

    Returns:
    - bool: `True` if scale learning should occur, `False` otherwise.
    
    Raises:
    - ValueError: If `total_epochs` is less than or equal to 1.
    """
    if total_epochs <= 1:
        raise ValueError("Total epochs must be greater than 1.")

    # Define the ratio of epochs during which scale learning should be applied
    curv_ratio = 0.3  # For example, learning happens during the first 30% of epochs
    
    return epoch < int(0.6 * total_epochs)

def custom_weight_exponent(epoch: int, total_epochs: int,loss_list: List[float]) -> float:
  """
  Calculate the weight exponent based on the current epoch and total number of epochs.

  Parameters:
  - epoch (int): The current epoch in the training process.
  - total_epochs (int): The total number of epochs in the training process.
- loss_list (list): A list of recorded loss values (can be used for further custom logic).

  Returns:
  - float: The calculated weight exponent for the current epoch.
  
  Raises:
  - ValueError: If `total_epochs` is less than or equal to 1.
  """
  if total_epochs <= 1:
      raise ValueError("Total epochs must be greater than 1.")

  # Define a ratio that determines how long to apply no weights
  no_weight_ratio = 0.3  # Example ratio: first 30% of epochs without weighting
  no_weight_epochs = int(no_weight_ratio * total_epochs)
  # No weighting for the first part of the training
  if epoch < no_weight_epochs:
      return 0.0  # No weighting initially
  
  # Gradually increase the negative weight exponent after the no-weight phase
  return -(epoch - no_weight_epochs) / (total_epochs - 1 - no_weight_epochs)




logger.set_logger(True)
tree = Tree("path/to/treefile.tre")
# terminals = tree.terminal_names()[:4]
# print(terminals[:4])
# dist_matrix = tree.distance_matrix()[0]
# print(dist_matrix[:4,:4])
# # print(tree.distance_matrix()[1])
# print(tree.distance_matrix())
# diameter = tree.diameter()
# print(diameter)
# t = ts.read_tree_newick("path/to/treefile.tre")
# tree = Tree("Name", t)
# print(tree)
# tree = Tree("path/to/treefile.tre")
# print(tree)
# tree.normalize()
# diameter = tree.diameter()
# print(diameter)
# embedding = tree.embed(dim=4, geometry = 'euclidean')
# print(embedding)
# print(embedding.points)
# embedding = tree.embed(dim=3,save_mode=True,geometry = 'euclidean')
# print(embedding)
# print(embedding.points)
# embedding = tree.embed(dim=3,export_video=True, epochs=1000,geometry = 'euclidean')
# print(embedding)
# print(embedding.points)
# embedding = tree.embed(dim=10,precise_opt=False,export_video=True, epochs=100,geometry = 'euclidean')
# print(embedding)
# print(embedding.points)
# embedding = tree.embed(dim=10,precise_opt=True,geometry = 'euclidean')
# print(embedding)
# print(embedding.points)
# embedding = tree.embed(dim=3,precise_opt=True,export_video=False, epochs=1000,geometry = 'euclidean')
# print(embedding)
# print(embedding.points)
# embedding = tree.embed(dim=4, precise_opt=True,epochs=2000,geometry = 'euclidean')
# print(embedding)
# print(embedding.points)
# logger.set_logger(False)
# embedding = tree.embed(dim=3, save_mode=True,precise_opt=True,geometry = 'euclidean')
# print(embedding)
# print(embedding.points)
# embedding = tree.embed(dim=3, export_video=True, epochs=200,precise_opt=True,geometry = 'euclidean')
# print(embedding)
# print(embedding.points)
# embedding = tree.embed(dim=3, epochs=100,precise_opt=True,geometry = 'euclidean')
# print(embedding)
# print(embedding.points)
# logger.set_logger(True)
# embedding = tree.embed(dim=3, epochs=200,precise_opt=True,geometry = 'euclidean')
# print(embedding)
# print(embedding.points)
# embedding = tree.embed(dim=3, epochs=200,save_mode=True,geometry = 'euclidean')
# print(embedding)
# print(embedding.points)
# embedding = tree.embed(dim=3, epochs=200,save_mode=True,lr_init = 0.1,geometry = 'euclidean')
# print(embedding)
# print(embedding.points)
# embedding = tree.embed(dim=3, epochs=200,precise_opt=True,save_mode=True,lr_init = 0.02,geometry = 'euclidean')
# print(embedding)
# print(embedding.points)
# embedding = tree.embed(dim=10, precise_opt=True, epochs=1000,geometry = 'euclidean')
# print(embedding)
# print(embedding.points)
# embedding = tree.embed(dim=2, precise_opt=True,scale_fn=custom_scale,geometry = 'euclidean')
# print(embedding.points)
# embedding = tree.embed(dim=2, precise_opt=True, lr_fn =custom_learning_rate, lr_init = 0.01,geometry = 'euclidean')
# print(embedding.points)
# embedding = tree.embed(dim=2, precise_opt=True, weight_exp_fn=custom_weight_exponent,geometry = 'euclidean')
# print(embedding.points)
# embedding = tree.embed(dim=2, precise_opt=True,weight_exp_fn=custom_weight_exponent, export_video=True, epochs = 1000,geometry = 'euclidean')
# print(embedding.points)




# logger.set_logger(True)
# embedding = tree.embed(dim=4)
# print(embedding)
# print(embedding.points)
# print(embedding.distance_matrix())
# print(tree.distance_matrix()[0])
# embedding = tree.embed(dim=3,save_mode=True)
# print(embedding)
# print(embedding.points)
# embedding = tree.embed(dim=3,export_video=True, epochs=1000)
# print(embedding)
# print(embedding.points)
# embedding = tree.embed(dim=10,export_video=False, precise_opt=True, epochs=100,save_mode=True)
# print(embedding)
# print(embedding._norm2())
# print(embedding.points)
# print(tree._current_time)
# tree.update_time()
# print(tree._current_time)
# embedding = tree.embed(dim=10,precise_opt=False, save_mode=True)
# print(embedding)
# print(embedding.points)
# logger.set_logger(True)
# embedding = tree.embed(dim=3,precise_opt=True,export_video=True, epochs=1000)
# print(embedding)
# print(embedding.points)
# print(embedding._norm2())
# embedding = tree.embed(dim=4, precise_opt=True,epochs=2000)
# print(embedding)
# print(embedding.points)
# logger.set_logger(False)
# embedding = tree.embed(dim=3, save_mode=True,precise_opt=True)
# print(embedding)
# print(embedding.points)
# embedding = tree.embed(dim=3, export_video=True, epochs=200,precise_opt=True)
# print(embedding)
# print(embedding.points)
# embedding = tree.embed(dim=3, epochs=1000,precise_opt=True)
# print(embedding)
# print(embedding.points)
# logger.set_logger(True)
# print(embedding.points)
# embedding = tree.embed(dim=3, epochs=200,precise_opt=True)
# print(embedding)
# print(embedding.points)
# embedding = tree.embed(dim=3, epochs=200,save_mode=True)
# print(embedding)
# print(embedding.points)
# embedding = tree.embed(dim=3, epochs=200,save_mode=True,lr_init = 0.1)
# print(embedding)
# print(embedding.points)
# embedding = tree.embed(dim=3, epochs=200,precise_opt=True,save_mode=True,lr_init = 0.02)
# print(embedding)
# print(embedding.points)
# tree.update_time()
# embedding = tree.embed(dim=10, precise_opt=True, epochs=1000)
# print(embedding)
# print(embedding.points)
# embedding = tree.embed(dim=2, precise_opt=True,scale_fn=custom_scale)
# print(embedding.points)
# embedding = tree.embed(dim=2, precise_opt=True, lr_fn =custom_learning_rate, lr_init = 0.01)
# print(embedding.points)
# embedding = tree.embed(dim=2, precise_opt=True, weight_exp_fn=custom_weight_exponent)
# print(embedding.points)
# embedding = tree.embed(dim=2, precise_opt=True,weight_exp_fn=custom_weight_exponent, export_video=True, epochs = 500)
# print(embedding.points)




# from tree_collections import MultiTree
# logger.set_logger(True)
# # Initialize from a Newick file
# multitree = MultiTree("path/to/trees.tre")
# multitree = multitree[:10]
# print(multitree)
# terminals = multitree.terminal_names()  
# print(terminals)
# terminals = multitree.common_terminals()
# print(terminals)
# D,C,L = multitree.distance_matrix(method = 'fp', func= torch.nanmedian)
# print(D,C,L)
# D,C,L = multitree.distance_matrix(method = 'fp', func= torch.nanmean)
# print(D,C,L)
# D,C,L = multitree.distance_matrix(method = 'fp')
# print(D,C,L)
# D,C,L = multitree.distance_matrix(func= torch.nanmedian)
# print(D,C,L)
# D,C,L = multitree.distance_matrix(func= torch.nanmean)
# print(D,C,L)
# D,C,L = multitree.distance_matrix()
# print(D,C,L)
# D,C,L = multitree.distance_matrix()
# print(D.max())
# x = multitree.normalize(batch_mode=False)
# print(x)
# x = multitree.normalize(batch_mode=True)
# print(x)
# x = multitree.normalize(batch_mode=False)
# print(x)
# D,C,L = multitree.distance_matrix()
# print(C)
# print(D.max())
# multi = multitree.embed(dim=3, geometry='euclidean')
# print(multi)
# print(multi.embeddings)
# embedding = multitree.embed(dim=3, precise_opt=True,scale_fn=custom_scale)
# print(embedding.embeddings)
# multi = multitree.embed(dim=2, geometry='hyperbolic')
# print(multi)
# print(multi.embeddings)
# print(multi.embeddings[3].points)
# multi = multitree.embed(dim=2, geometry='hyperbolic',precise_opt=True , epochs = 1000, save_mode = True)
# print(multi)




from embedding import Embedding
import numpy as np
import torch
logger.set_logger(True)
# n_points = 10
# dimension = 2
# labels = [str(i) for i in range(n_points)]
# points = np.random.randn(dimension,n_points)
# embedding = Embedding(geometry='euclidean')
# embedding.save('embedding.pkl')
# loaded_embedding = Embedding.load('embedding.pkl')
# print(loaded_embedding)
# embd = Embedding(geometry='hyperbolic', points = points)
# print(embd)
# # print(embd.distance_matrix())
# embd.labels = labels
# # embd.points = np.random.randn(dimension,150)
# print(embd.labels)
# print(embd.geometry)
# print(embd)
# points = torch.randn(dimension,n_points)
# print(Embedding(geometry='euclidean', points = points))


# from embedding import EuclideanEmbedding
# import numpy as np
# n_points = 10
# dimension = 2
# labels = [str(i) for i in range(n_points)]
# points = np.random.randn(dimension,n_points)/10
# embedding = EuclideanEmbedding(points = points, labels = labels)
# print(embedding.points)
# translatio_vector = np.random.randn(dimension)
# print(translatio_vector)
# embedding.translate(translatio_vector)
# print(embedding.points)
# # Convert degrees to radians
# theta = np.radians(30)

# # Define the 2x2 rotation matrix
# rotation_matrix = np.array([
#     [np.cos(theta), -np.sin(theta)],
#     [np.sin(theta), np.cos(theta)]
# ])
# embedding.rotate(rotation_matrix)
# print(embedding.points)
# new_points = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# embedding.points = new_points  # This will automatically update dimensions and other attributes.
# print(embedding)
# print(embedding.points)
# centroid = embedding.centroid()
# print(embedding.distance_matrix())
# print(centroid)
# embedding.center()
# print(embedding.points)
# centroid = embedding.centroid()
# print(centroid)
# print(embedding.distance_matrix())

from embedding import PoincareEmbedding
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def plot_embedding_comparison(points_before, points_after, center_before, center_after, title, ax):
    # Plot Poincaré unit circle (domain) in black
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = np.sin(theta)
    circle_y = np.cos(theta)
    ax.plot(circle_x, circle_y, 'k-')
    ax.scatter(points_before[0, :], points_before[1, :], color='blue', label='Before', edgecolor='k')
    ax.scatter(points_after[0, :], points_after[1, :], color='red', label='After', edgecolor='k')
    ax.scatter(center_before[0], center_before[1], color='blue', marker='x', s=100, label='Center')
    ax.scatter(center_after[0], center_after[1], color='red', marker='x', s=100, label='Center')
    ax.set_aspect('equal')
    ax.legend()
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.set_title(title)

# n_points = 10
# dimension = 3
# labels = [str(i) for i in range(n_points+1)]
# points = np.random.randn(dimension,n_points)/10
# # print(points)

# embedding = PoincareEmbedding(points = points, labels = labels)
# print(embedding)
# embedding.curvature = -.5
# print(embedding)
# print(embedding.dimension)
# print(embedding.n_points)
# import torch
# new_points = torch.tensor([[-0.5, 0.5, 0.5, -0.5], [-0.5, 0.5, -0.5, 0.5]])
# embedding.points = new_points
# print(embedding.dimension)
# print(embedding.n_points)

# # # Translate the points
# print(embedding.points)
# print(embedding.distance_matrix())

# # plt.figure(figsize=(6,12))
# fig, axs = plt.subplots(1, 2, figsize=(14, 7))
# points_before = embedding.points

# translation_vector = np.array([0.25, -0.75])

# theta = np.radians(30)
# rotation_matrix = np.array([ [np.cos(theta), -np.sin(theta)],
#     [np.sin(theta), np.cos(theta)]])
# # Rotate the points
# embedding.rotate(rotation_matrix)

# center = embedding.centroid()





# # Plot for translation
# plot_embedding_comparison(points_before, embedding.points, center, embedding.centroid(), 'Effect of Translation', axs[0])
# # plot_embedding_comparison(points_before, embedding.points, center, embedding.centroid(), 'Effect of Rotation', axs[0])

# points_before = embedding.points
# center = embedding.centroid()

# theta = np.radians(-30)
# rotation_matrix = np.array([ [np.cos(theta), -np.sin(theta)],
#     [np.sin(theta), np.cos(theta)]])
# # Rotate the points
# # embedding.rotate(rotation_matrix)
# embedding.translate(translation_vector)

# print(embedding)
# print(embedding.points)
# print(embedding.distance_matrix())
# print(embedding.centroid())
# print(embedding.centroid(mode = 'Frechet'))


# print(center)
# embedding.center()

# center = embedding.centroid()

# # Plot for centering operations
# plot_embedding_comparison(points_before, embedding.points, center, embedding.centroid(), 'Effect of Centering', axs[1])
# # plot_embedding_comparison(points_before, embedding.points, center, embedding.centroid(), 'Effect of Rotation', axs[1])

# # print(embedding.points)

# plt.tight_layout()
# plt.show()

# embedding.curvature = -0.5
# print(embedding.distance_matrix())

# print(embedding.centroid())
# print(embedding.centroid(mode = 'Frechet'))

# print(embedding.points)
# loid_embedding = embedding.switch_model()
# print(loid_embedding.points)
# print(loid_embedding.curvature)


# import matplotlib.pyplot as plt
# from embedding import EuclideanEmbedding
# import numpy as np

# # Function to generate points at the vertices of a regular pentagon
# def generate_pentagon_vertices():
#     angles = np.linspace(0, 2 * np.pi, 6)[:-1]  # 5 angles for 5 vertices, excluding the last one to avoid duplication
#     x = np.cos(angles)
#     y = np.sin(angles)
#     return np.vstack((x, y))

# # Function to plot points with  edges, and axes
# def plot_pentagon(ax, points, title, color, marker, label_text):
#     ax.scatter(points[0, :], points[1, :], c=color, s=100, edgecolors='black', marker=marker, label=label_text)
#     # Draw edges between consecutive points
#     for i in range(points.shape[1]):
#         start = i
#         end = (i + 1) % points.shape[1]
#         ax.plot([points[0, start], points[0, end]], [points[1, start], points[1, end]], color=color)
    
#     # Draw the axes
#     ax.axhline(0, color='black', linewidth=0.5)
#     ax.axvline(0, color='black', linewidth=0.5)
    
#     ax.set_title(title)
#     ax.set_xlabel('X-axis')
#     ax.set_ylabel('Y-axis')
#     ax.set_aspect('equal', adjustable='box')
#     ax.legend()

# # Define pentagon vertices
# points = generate_pentagon_vertices()
# n_points = points.shape[1]
# labels = [str(i) for i in range(n_points)]

# # Initialize embedding
# embedding = EuclideanEmbedding(points=points, labels=labels)

# ### Figure 1: Translation ###
# # Create a figure for the translation
# fig1, ax1 = plt.subplots(figsize=(8, 8))

# # Plot the original pentagon vertices with edges and labels
# plot_pentagon(ax1, embedding.points, "Translation of Pentagon Vertices", color='black', marker='o', label_text='Original Points')


# # Translate the points
# translation_vector = np.random.randn(2)/2  # 2D translation vector
# embedding.translate(translation_vector)

# # Plot the translated pentagon vertices with edges and labels
# plot_pentagon(ax1, embedding.points, "Translation of Pentagon Vertices", color='green', marker='^', label_text='Translated Points')

# # Show the plot for translation
# plt.show()

# ### Figure 2: Rotation ###
# # Create a figure for the rotation
# fig2, ax2 = plt.subplots(figsize=(8, 8))

# # Plot the translated pentagon vertices with edges and labels before rotation
# plot_pentagon(ax2, embedding.points, labels, "Rotation of Pentagon Vertices", color='black', marker='o', label_text='Before Rotation')

# # Convert degrees to radians and define the 2x2 rotation matrix
# theta = np.radians(30)
# rotation_matrix = np.array([
#     [np.cos(theta), -np.sin(theta)],
#     [np.sin(theta), np.cos(theta)]
# ])

# # Rotate the points
# embedding.rotate(rotation_matrix)

# # Plot the pentagon vertices with edges and labels after rotation
# plot_pentagon(ax2, embedding.points, labels, "Rotation of Pentagon Vertices", color='red', marker='^', label_text='After Rotation')

# # Show the plot for rotation
# plt.show()





# from embedding import LoidEmbedding
# import numpy as np
# n_points = 10
# dimension = 3
# labels = [str(i) for i in range(n_points+1)]
# points = np.random.randn(dimension, n_points)
# norm_points = np.linalg.norm(points, axis=0)
# points = np.vstack([np.sqrt(1 + norm_points**2), points])
# embedding = LoidEmbedding(points=points, labels=labels)
# print(embedding)
# embedding.curvature = -2
# print(embedding)
# print(embedding._norm2())
# print(embedding.dimension)
# print(embedding.n_points)

# new_points = np.array([[-0.5, 0.5, 0.5, -0.5], [-0.5, 0.5, -0.5, 0.5]])
# norm_points = np.linalg.norm(new_points, axis=0)
# new_points = np.vstack([np.sqrt(1 + norm_points**2), new_points])
# # print(new_points)
# embedding.points = new_points

# print(embedding.dimension)
# print(embedding.n_points)

# print(embedding.distance_matrix())

# embedding.curvature = -.5
# print(embedding.distance_matrix())

# print(embedding.centroid())
# print(embedding.centroid(mode = 'Frechet'))
# poincare_embedding = embedding.switch_model()
# print(poincare_embedding)
# print(poincare_embedding.distance_matrix())



# import numpy as np
# import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D

# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')

# def plot_hyperbolic_sheet_with_points(points, color='red', label='Scatter Points', max_radius=2):
#     if isinstance(points, torch.Tensor):
#         points = points.numpy()
#     # Extract x, y, z from points where z = sqrt(1 + x^2 + y^2)
#     x_points = points[2,:]  # z from original points as x
#     y_points = points[1,:]  # y remains as y
#     z_points = np.sqrt(1 + x_points**2 + y_points**2)  # Compute z based on x and y
#     def hyperbolic_z(x, y):
#         return np.sqrt(1 + x**2 + y**2)
#     # Generate circular grid in polar coordinates (r, theta) with controllable radius
#     theta_vals = np.linspace(0, 2 * np.pi, 100)  # Angular coordinate
#     r_vals = np.linspace(0, max_radius, 50)  # Radial coordinate
#     # Convert polar to cartesian coordinates for a circular grid
#     theta_grid, r_grid = np.meshgrid(theta_vals, r_vals)
#     x_grid = r_grid * np.cos(theta_grid)
#     y_grid = r_grid * np.sin(theta_grid)
#     z_grid = hyperbolic_z(x_grid, y_grid)
    
#     ax.plot_wireframe(x_grid, y_grid, z_grid, color='lightblue', alpha=0.5, linewidth=0.5)
#     ax.scatter(x_points, y_points, z_points, color=color, s=100, label=label)
#     ax.set_xticks([])  # Remove x-axis ticks
#     ax.set_yticks([])  # Remove y-axis ticks
#     ax.set_zticks([])  # Remove z-axis ticks
#     ax.xaxis.pane.fill = False
#     ax.yaxis.pane.fill = False
#     ax.zaxis.pane.fill = False
#     ax.xaxis.line.set_color((0.5, 0.5, 0.5))  # Light gray axis lines
#     ax.yaxis.line.set_color((0.5, 0.5, 0.5))  
#     ax.zaxis.line.set_color((0.5, 0.5, 0.5))
#     ax.set_xlabel('X axis')
#     ax.set_ylabel('Y axis')
#     ax.set_zlabel('Z axis')
#     ax.set_title('Hyperbolic Sheet with Scatter Points')
#     ax.legend()
    

# # Example usage:
# import torch

# print(embedding.centroid())
# # plot_hyperbolic_sheet_with_points(embedding.points, color='red', label='Points', max_radius=2.5)
# translation_vector = np.random.randn(2,1)/2  # 2D translation vector
# norm_point = np.linalg.norm(translation_vector, axis=0)
# translation_vector = np.vstack([np.sqrt(1 + norm_point**2), translation_vector])
# # embedding.translate(translation_vector)
# print(translation_vector)
# embedding.translate(translation_vector)
# print(embedding.centroid())
# # plot_hyperbolic_sheet_with_points(embedding.points, color='blue', label='After Tranlation', max_radius=2.5)
# embedding.center()
# print(embedding.centroid())
# plt.show()




# import torch

# theta = np.radians(30)
# rotation_matrix = np.array([ [np.cos(theta), -np.sin(theta)],
#     [np.sin(theta), np.cos(theta)]])
# # Rotate the points

# R = np.zeros((3,3))
# R[0,0] = 1
# R[1:,1:] = rotation_matrix

# plot_hyperbolic_sheet_with_points(embedding.points, color='red', label='Points', max_radius=1.5)
# embedding.rotate(R)
# plot_hyperbolic_sheet_with_points(embedding.points, color='blue', label='After Rotation', max_radius=1.5)


# plt.show()


# from procrustes import EuclideanProcrustes
# from embedding import EuclideanEmbedding
# import numpy as np
# import matplotlib.pyplot as plt

# n_points = 10
# dimension = 2

# points = np.random.randn(dimension, n_points)
# embedding = EuclideanEmbedding(points=points)
# source_embedding = embedding.copy()
# target_embedding = embedding.copy()

# # Apply transformation: translation + rotation
# translation_vector = np.random.randn(dimension)
# target_embedding.translate(translation_vector)

# # Convert degrees to radians and define the 2x2 rotation matrix
# theta = np.radians(30)
# rotation_matrix = np.array([
#     [np.cos(theta), -np.sin(theta)],
#     [np.sin(theta), np.cos(theta)]
# ])
# target_embedding.rotate(rotation_matrix)

# # Define a noise variance
# noise_variance = 0.1  # Example variance
# noisy_target_embedding = target_embedding.copy()

# # # Add Gaussian noise to the points with specified variance
# noise = np.random.normal(0, np.sqrt(noise_variance), size=noisy_target_embedding.points.shape)
# # noisy_target_embedding.points += noise

# # Perform Procrustes alignment using the default mode
# model = EuclideanProcrustes(source_embedding, noisy_target_embedding)
# model = EuclideanProcrustes(source_embedding, target_embedding)
# source_aligned = model.map(source_embedding)
# print(source_embedding.points)
# print(target_embedding.points)
# # Perform Procrustes alignment using mode='accurate'
# print(source_aligned.points)
# print(target_embedding.points)


# # Define a function to compute the alignment cost (mean squared error between aligned and noisy target points)
# def compute_alignment_cost(aligned_points, target_points):
#     return np.mean(np.linalg.norm(aligned_points - target_points)**2)

# # Compute the alignment costs for both modes
# cost = compute_alignment_cost(source_aligned.points, target_embedding.points)
# # print('asd')
# # print(cost)


# # # Define noise variances
# noise_variances = np.linspace(0, .01, 1000)  # Variances from 0 to 1
# alignment_errors = []

# # Loop through different noise variances
# for noise_variance in noise_variances:
#     noisy_target_embedding = target_embedding.copy()
    
#     # Add Gaussian noise to the points with specified variance
#     noise = np.random.normal(0, np.sqrt(noise_variance), size=noisy_target_embedding.points.shape)
#     # print(type(noisy_target_embedding.points))
#     noisy_target_embedding.points += noise

#     # Apply Procrustes alignment
#     model = EuclideanProcrustes(source_embedding, noisy_target_embedding)
#     source_aligned = model.map(source_embedding)

#     # Compute alignment error (e.g., mean squared error between aligned and target points)
#     alignment_error = np.mean(np.linalg.norm(source_aligned.points - noisy_target_embedding.points, axis=0))
#     alignment_errors.append(alignment_error)

# # alignment_errors = np.array(alignment_errors, dtype=np.float64)
# # noise_variances = np.array(noise_variances, dtype=np.float64)
# # print(noise_variances)
# # print(alignment_errors.to(np))
# alignment_errors = np.array([float(x) for x in alignment_errors])
# # print(type(alignment_errors))
# # print(type(noise_variances))
# print(len(alignment_errors))
# print(len(noise_variances))

# # Plot quality of embedding vs. noise variance
# # print(noise_variances,alignment_errors)

# # valid_indices = ~np.isnan(alignment_errors) & ~np.isinf(alignment_errors)
# # print(type(noise_variances), type(alignment_errors))
# # plt.scatter(np.arange(1,10), np.arange(1,10))
# plt.scatter(noise_variances, alignment_errors)

# plt.xlabel('Noise Variance')
# plt.ylabel('Alignment Error')
# plt.title('Quality of Embedding vs. Noise Variance')
# plt.grid(True)
# plt.show()



# from procrustes import HyperbolicProcrustes
# from embedding import PoincareEmbedding
# import numpy as np
# import matplotlib.pyplot as plt

# n_points = 10
# dimension = 2

# points = np.random.randn(dimension, n_points)/10
# embedding = PoincareEmbedding(points=points)
# source_embedding = embedding.copy()
# target_embedding = embedding.copy()

# # Apply transformation: translation + rotation
# translation_vector = np.random.randn(dimension)/4
# target_embedding.translate(translation_vector)

# # Convert degrees to radians and define the 2x2 rotation matrix
# theta = np.radians(27.5)
# rotation_matrix = np.array([
#     [np.cos(theta), -np.sin(theta)],
#     [np.sin(theta), np.cos(theta)]
# ])
# target_embedding.rotate(rotation_matrix)

# # # Define a noise variance
# noise_variance = 0.1  # Example variance
# noisy_target_embedding = target_embedding.copy()

# # # Add Gaussian noise to the points with specified variance
# noise = np.random.normal(0, np.sqrt(noise_variance), size=noisy_target_embedding.points.shape)
# noisy_target_embedding.points += noise

# # # Perform Procrustes alignment using the default mode
# # model = EuclideanProcrustes(source_embedding, noisy_target_embedding)
# model = HyperbolicProcrustes(source_embedding, target_embedding)
# source_aligned = model.map(source_embedding)
# print(source_embedding.points)
# print(target_embedding.points)
# # Perform Procrustes alignment using mode='accurate'
# print(source_aligned.points)

# source_embedding  = source_embedding.switch_model()
# print(source_embedding)
# print(source_embedding.points)
# source_aligned = model.map(source_embedding) # loid to poincare

# print(target_embedding.points)
# print(source_aligned.points)


# import torch


# Define a function to compute the alignment cost (mean squared error between aligned and noisy target points)
# def compute_alignment_cost(src_embedding, trg_embedding):
#     cost = sum((torch.norm(src_embedding.points[:, n]- trg_embedding.points[:, n]))**2 for n in range(src_embedding.n_points))
#     return cost

# # # Compute the alignment costs for both modes
# cost = compute_alignment_cost(source_aligned, target_embedding)
# print(cost)


# # # Define noise variances
# noise_variances = np.linspace(0, .1, 10000)  # Variances from 0 to 1
# alignment_errors = []

# # Loop through different noise variances
# for noise_variance in noise_variances:
#     noisy_target_embedding = target_embedding.copy()
#     noisy_target_embedding.switch_model()
    
#     # Add Gaussian noise to the points with specified variance
#     noise = np.random.normal(0, noise_variance, size=noisy_target_embedding.points.shape)
#     noisy_target_embedding.points += noise

#     noisy_target_embedding.switch_model()

#     # Apply Procrustes alignment
#     model = HyperbolicProcrustes(source_embedding, noisy_target_embedding, mode = 'default')
#     source_aligned = model.map(source_embedding)

#     # Compute alignment error (e.g., mean squared error between aligned and target points)
#     alignment_error = compute_alignment_cost(source_aligned, noisy_target_embedding) 
#     alignment_errors.append(alignment_error)

# # Plot quality of embedding vs. noise variance
# plt.scatter(noise_variances, alignment_errors, marker='o')
# plt.xlabel('Noise Variance')
# plt.ylabel('Alignment Error')
# plt.title('Quality of Embedding vs. Noise Variance')
# plt.grid(True)
# plt.show()










# from embedding import MultiEmbedding
# from embedding import EuclideanEmbedding
# import numpy as np
# import torch

# multi_embedding = MultiEmbedding()

# n_points = 10
# dimension = 2
# labels = [str(i) for i in range(n_points)]
# points = np.random.randn(dimension,n_points)/10
# embedding1 = EuclideanEmbedding(points = points, labels = labels)
# print(embedding1)
# multi_embedding.append(embedding1)

# labels = [str(i) for i in range(14)]
# points = np.random.randn(2,14)/10
# embedding2 = EuclideanEmbedding(points = points, labels = labels)
# multi_embedding.append(embedding2)
# print(embedding2)

# labels = [str(i) for i in range(5)]
# points = np.random.randn(2,5)/10
# embedding3 = EuclideanEmbedding(points = points, labels = labels)
# multi_embedding.append(embedding3)
# print(embedding3)

# labels = [str(i) for i in range(7)]
# points = np.random.randn(2,7)/10
# embedding4 = EuclideanEmbedding(points = points, labels = labels)
# multi_embedding.append(embedding3)
# print(embedding4)
# print(multi_embedding)

# print(multi_embedding)
# print(multi_embedding.embeddings)
# multi_embedding = MultiEmbedding()
# embedding_dic = {'a': embedding1, 'b':embedding2, 'c':embedding3, 'd':embedding4}
# multi_embedding.embeddings = embedding_dic
# print(multi_embedding)
# print(multi_embedding.embeddings)
# # print(multi_embedding.distance_matrix()[0])
# print(multi_embedding.distance_matrix()[1])
# print(multi_embedding.distance_matrix(func = torch.nanmedian))




from tree_collections import MultiTree
import treeswift as ts
import numpy as np
import torch

tree1 = ts.read_tree_newick('path/to/treefile1.tre')
tree2 = ts.read_tree_newick('path/to/treefile2.tre')
tree3 = ts.read_tree_newick('path/to/treefile2.tre')
multitree = MultiTree('name', [tree1, tree2,tree3])
print(multitree)
# print(multitree.distance_matrix()[0][:4,:4])
# multiembedding = multitree.embed(dim = 2, geometry = 'euclidean', precise_opt = True)
# multiembedding = multitree.embed(dim = 2, geometry = 'hyperbolic', precise_opt = False)
# print(multiembedding.embeddings)
# print( multiembedding.distance_matrix()[0][:4,:4])
# print( (multiembedding.distance_matrix()[0][:4,:4])**2)
# print(multiembedding.distance_matrix()[0][:4,:4])
# print(multiembedding.embeddings)
# print(multiembedding.reference_embedding())
# x = multiembedding.reference_embedding(accurate = True)
# print(x)
# print(multiembedding.embeddings[0]._points)
# reference = multiembedding.reference_embedding(dim = 2, precise_opt = True)
# print( (reference.distance_matrix()[0][:4,:4])**2)
# print(reference)
# print( reference.distance_matrix()[0][:4,:4])



multiembedding = multitree.embed(dim = 2,precise_opt = True)
# print(multiembedding.embeddings[0]._points)
# print(multiembedding.embeddings[0].points[:,:4])
# print(multiembedding.embeddings[1].points[:,:4])
# print(multiembedding.distance_matrix()[0][:4,:4])
# print(multiembedding.embeddings[0].points)
# print(multiembedding.embeddings[1].points)
multiembedding.align(func = torch.nanmean,precise_opt = True)
# multiembedding.align(func = torch.nanmean)
print(multiembedding.embeddings[0].points)
print(multiembedding.embeddings[1].points)
# print(multiembedding.embeddings[0].points[:,:4])
# print(multiembedding.embeddings[1].points[:,:4])
# print(multiembedding.distance_matrix()[0][:4,:4])






# import numpy as np
# from embedding import EuclideanEmbedding
# from PCA import PCA

# n_points = 10
# dimension = 2

# points = np.random.randn(dimension, n_points)
# embedding = EuclideanEmbedding(points=points)
# print(embedding.distance_matrix())
# pca = PCA(embedding, enable_logging=True)

# # Map to a lower-dimensional space.
# reduced_embedding = pca.map_to_dimension(target_dimension=1)
# print(reduced_embedding.distance_matrix())

# # Access the mean and subspace.
# mean = pca.get_mean()
# subspace = pca.get_subspace()
# print(mean,subspace)









# from tree_collections import Tree
# from PCA import PCA

# tree = Tree("path/to/treefile.tre", enable_logging=True)
# embedding = tree.embed(dimension=3, geometry='hyperbolic')
# embedding = embedding.switch_model()
# pca = PCA(embedding, enable_logging=True)
# reduced_embedding = pca.map_to_dimension(target_dimension=3)
# print(reduced_embedding.points)
# Map to a lower-dimensional space.
# reduced_embedding = pca.map_to_dimension(target_dimension=2)

# Access the mean and subspace.
# mean = pca.get_mean()
# subspace = pca.get_subspace()









# # Initialize from a list of trees
# # import treeswift as ts
# # tree1 = ts.read_tree_newick("path/to/treefile1.tre")
# # tree2 = ts.read_tree_newick("path/to/treefile2.tre")
# # tree3 = ts.read_tree_newick("path/to/treefile3.tre")
# # tree_list = [tree1, tree2, tree3]  # List of trees
# # multitree = MultiTree('Name', tree_list)
# print(multitree.trees[0].contents)
# print(multitree.trees[1].contents)
# # multitree.save("path/to/output_unnormalized.tre")
# # x = multitree.normalize(batch_mode=False)
# # print(x)
# print(multitree.distance_matrix())
# print(multitree.trees[0].contents)
# print(multitree.trees[1].contents)
# print(x)
# multitree.save("path/to/output_normalized.tre")
# print(multitree)
# print(multitree.trees)
# multitree.save("path/to/outputssss.tre")
# logger.set_logger(False)
# distance_matrix = multitree.distance_matrix(func=torch.nanmedian)[:4,:4]
# distance_matrix, confidence_matrix = multitree.distance_matrix(func=torch.nanmedian,confidence= True)
# print(confidence_matrix)
# print(distance_matrix)
# print(embedding)
# print(embedding.embeddings)
# embedding = multitree.embed(dimension=3, geometry='euclidean',accurate = True)
# print(embedding)
# print(embedding.embeddings)
# embedding = multitree.embed(dimension=3, geometry='hyperbolic')
# print(embedding)
# print(embedding.embeddings)
# embedding = multitree.embed(dimension=2, geometry='hyperbolic')
# print(embedding)
# print(embedding.embeddings)
# embedding = multitree.embed(dimension=2, geometry='hyperbolic', accurate = True)
# print(embedding)
# print(embedding.embeddings)
# multitree.save("path/to/output.tre")
# import logger
# import torch
# import treeswift as ts