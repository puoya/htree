from collections.abc import Collection
from typing import Union, Set, Optional, List, Callable, Tuple, Dict, Iterator

# from tree_collections import Tree
# # Initialize from file
# tree = Tree("path/to/treefile.tre")
# print(tree)
# # Initialize from a treeswift Tree object
# import treeswift as ts
# t = ts.read_tree_newick("path/to/treefile.tre")
# tree = Tree("treefile.tre", t)
# print(tree)




# import logger as logger
# logger.set_logger(True)
# # Initialize with logging enabled
# tree = Tree("path/to/treefile.tre")
# print(tree)
# Tree(treefile.tre)


# Get top four terminal names
# terminals = tree.terminal_names()[:4]
# print(terminals)
# ['Diospyros_malabarica', 'Huperzia_squarrosa', 'Selaginella_moellendorffii_genome', 'Colchicum_autumnale']
# Compute their distance matrix
# dist_matrix, names = tree.distance_matrix()
# print(dist_matrix[:4,:4])
# tensor([[0.0000, 2.0274, 2.1069, 0.9733],
#         [2.0274, 0.0000, 1.7451, 1.3855],
#         [2.1069, 1.7451, 0.0000, 1.4650],
# #         [0.9733, 1.3855, 1.4650, 0.0000]])

# print(names[:4])
# # Get tree diameter
# diameter = tree.diameter()
# print(diameter)
# tensor(2.5342)





# Normalize tree to have a diameter of 1
# tree.normalize()
# diameter = tree.diameter()
# print(diameter)






# Embed tree in 2D hyperbolic space
# embedding = tree.embed(dim=2, geometry='hyperbolic')
# print(embedding)
# # HyperbolicEmbedding(curvature=-15.57, model=loid, points_shape=[3, 14])
# # Embed tree in 3D Euclidean space
# embedding = tree.embed(dim=3, geometry='euclidean')
# print(embedding)
# # EuclideanEmbedding(points_shape=[3, 14])
# print(tree.embed(dim=2, geometry='euclidean'))

# print(tree.embed(dim=2, precise_opt=True))
# print(tree.embed(dim=2, precise_opt=True, epochs=2000))
# print(tree.embed(dim=2, precise_opt=True, lr_init=0.1))

# print(tree.embed(dim=2, dist_cutoff=5.0))
# print(tree.embed(dim=2, precise_opt=True, save_mode=True))
# tree.embed(dim=2, precise_opt=True, export_video=True)


# def custom_learning_rate(epoch: int, total_epochs: int, loss_list: List[float]) -> float:
#     """ 
#     Calculate a dynamic learning rate based on the current epoch and total number of epochs.
#     Parameters:
#     - epoch (int): The current epoch in the training process.
#     - total_epochs (int): The total number of epochs in the training process.
#     - loss_list (list): A list of recorded loss values (can be used for further custom logic).

#     Returns:
#     - float: The dynamic learning rate for the current epoch.

#     Raises:
#     - ValueError: If `total_epochs` is less than or equal to 1.
#     """

#     if total_epochs <= 1:
#         raise ValueError("Total epochs must be greater than 1.")

#     # Example: Reduce learning rate as training progresses
#     decay_factor = 0.5  # Factor by which to decay the learning rate
#     loss_threshold = 0.01  # Loss threshold for further reduction
#     decay_start_epoch = int(0.7 * total_epochs)  # When to start decaying

#     # Reduce learning rate if the epoch is beyond a certain point
#     if epoch > decay_start_epoch:
#         # Learning rate decays based on the remaining epochs
#         decay_rate = 1 - (epoch - decay_start_epoch) / (total_epochs - decay_start_epoch)
#     else:
#         decay_rate = 1.0  # No decay before the threshold
#     # Further adjust learning rate if recent loss has not improved sufficiently
#     if len(loss_list) > 1 and loss_list[-1] > loss_threshold:
#         decay_rate *= decay_factor
#     return  decay_rate


# tree.embed(dim=2, precise_opt=True, lr_fn=custom_learning_rate, lr_init = 0.01)











# def custom_scale(epoch: int, total_epochs: int, loss_list: List[float]) -> bool:
#     """
#     Determine whether scale learning should occur based on the current epoch and total number of epochs.

#     Parameters:
#     - epoch (int): The current epoch in the training process.
#     - total_epochs (int): The total number of epochs in the training process.
#     - loss_list (list): A list of recorded loss values (can be used for further custom logic).

#     Returns:
#     - bool: `True` if scale learning should occur, `False` otherwise.
    
#     Raises:
#     - ValueError: If `total_epochs` is less than or equal to 1.
#     """
#     if total_epochs <= 1:
#         raise ValueError("Total epochs must be greater than 1.")

#     # Define the ratio of epochs during which scale learning should be applied
#     curv_ratio = 0.3  # For example, learning happens during the first 30% of epochs
    
#     return epoch < int(0.6 * total_epochs)

# tree.embed(dim=2, precise_opt=True, scale_fn=custom_scale)








# def custom_weight_exponent(epoch: int, total_epochs: int,loss_list: List[float]) -> float:
#   """
#   Calculate the weight exponent based on the current epoch and total number of epochs.

#   Parameters:
#   - epoch (int): The current epoch in the training process.
#   - total_epochs (int): The total number of epochs in the training process.
#   - loss_list (list): A list of recorded loss values (can be used for further custom logic).

#   Returns:
#   - float: The calculated weight exponent for the current epoch.
  
#   Raises:
#   - ValueError: If `total_epochs` is less than or equal to 1.
#   """
#   if total_epochs <= 1:
#       raise ValueError("Total epochs must be greater than 1.")

#   # Define a ratio that determines how long to apply no weights
#   no_weight_ratio = 0.3  # Example ratio: first 30% of epochs without weighting
#   no_weight_epochs = int(no_weight_ratio * total_epochs)
#   # No weighting for the first part of the training
#   if epoch < no_weight_epochs:
#       return 0.0  # No weighting initially
  
#   # Gradually increase the negative weight exponent after the no-weight phase
#   return -(epoch - no_weight_epochs) / (total_epochs - 1 - no_weight_epochs)
# tree.embed(dim=2, precise_opt=True, weight_exp_fn=custom_weight_exponent)





# # Save tree to a file
# tree.save("path/to/treefile2.tre") 
# # Copy the tree object
# tree_copy = tree.copy()













# from htree.tree_collections import MultiTree
# Initialize from a Newick file
# multitree = MultiTree("path/to/trees.tre")
# print(multitree)
# MultiTree(trees.tre, 844 trees)
# Initialize from a list of trees
# import treeswift as ts
# tree1 = ts.read_tree_newick("path/to/treefile1.tre")
# tree2 = ts.read_tree_newick("path/to/treefile2.tre")
# tree_list = [tree1, tree2]  # List of trees
# multitree = MultiTree('mytrees', tree_list)
# print(multitree)
# # MultiTree(mytrees, 2 trees)
# print(multitree.trees)
# # [Tree(Tree_0), Tree(Tree_1)]
# # Initialize from a list of named trees
# from htree.tree_collections import Tree
# named_trees = [Tree('a', tree1), Tree('b', tree2)]
# multitree = MultiTree('mTree', named_trees)
# print(multitree)
# # MultiTree(mTree, 2 trees)
# print(multitree.trees)
# # [Tree(a), Tree(b)]








# Initialize with logging enabled
# import htree.logger as logger
# logger.set_logger(True)

# multitree = MultiTree("path/to/trees.tre")
# print(multitree)
# MultiTree(trees.tre, 844 trees)



# multitree = MultiTree("path/to/trees.tre")[:10]
# print(multitree)
# MultiTree(mTree, 10 trees)
# print(multitree.trees)
# [Tree(tree_1), Tree(tree_2), Tree(tree_3), Tree(tree_4), Tree(tree_5), Tree(tree_6), Tree(tree_7), Tree(tree_8), Tree(tree_9), Tree(tree_10)]
# Compute the distance matrix with default aggregation (mean)
# avg_mat, conf, labels = multitree.distance_matrix()
# print(avg_mat[:4,:4])
# tensor([[0.0000, 0.7049, 1.2343, 0.5929],
#         [0.7049, 0.0000, 1.3234, 0.6870],
#         [1.2343, 1.3234, 0.0000, 1.0143],
#         [0.5929, 0.6870, 1.0143, 0.0000]])
# # Compute the distance matrix with custom aggregation
# import torch
# avg_mat, conf, labels = multitree.distance_matrix(func=torch.nanmedian)
# print(distance_matrix)
# tensor([[0.0000, 0.5538, 0.9043, 0.5240],
#         [0.5538, 0.0000, 1.1598, 0.5902],
#         [0.9043, 1.1598, 0.0000, 0.8635],
#         [0.5240, 0.5902, 0.8635, 0.0000]])
# avg_mat, conf, labels = multitree.distance_matrix(method='fp')
# print(avg_mat[:4,:4])
# tensor([[0.0000, 0.6000, 0.4000, 0.7000],
#         [0.6000, 0.0000, 0.5000, 0.8000],
#         [0.4000, 0.5000, 0.0000, 0.6000],
#         [0.7000, 0.8000, 0.6000, 0.0000]])
# # Compute the union of all terminal names (removes duplicates)
# print(multitree.terminal_names()[:4])
# ['Allamanda_cathartica', 'Alsophila_spinulosa', 'Amborella_trichopoda', 'Aquilegia_formosa']





# Embed trees in a 2D hyperbolic space
# multiemb_hyperbolic = multitree.embed(dim=2, geometry='hyperbolic')
# print(multiemb_hyperbolic)
# print(multiemb_hyperbolic.embeddings)
# # Embed trees in a 3D Euclidean space
# multiemb_euclidean = multitree.embed(dim=3, geometry='euclidean')
# print(multiemb_euclidean)
# print(multiemb_euclidean.embeddings)












# from htree.embedding import Embedding
# import numpy as np
# # Create an embedding with hyperbolic geometry (norm requirements are not applied)
# n_points = 10
# dimension = 2
# embedding = Embedding(geometry='hyperbolic', points=np.random.randn(dimension,n_points))
# print(embedding)
# # Embedding(geometry=hyperbolic, points_shape=[2, 10])
# # Update points
# # embedding.points = np.random.randn(2,150)
# # raise NotImplementedError("update_dimensions must be implemented by a subclass")
# # Set labels
# labels = [str(i) for i in range(n_points)]
# embedding.labels = list(range(n_points))
# # Save the embedding
# embedding.save('embedding.pkl')
# # Load the embedding
# loaded_embedding = Embedding.load('embedding.pkl')
# print(loaded_embedding)
# # Embedding(geometry=hyperbolic, points_shape=[2, 10])
# import torch
# embedding = Embedding(geometry='euclidean', points = np.random.randn(dimension,n_points) )
# print(embedding)
# Embedding(geometry=euclidean, points_shape=[2, 10])
















# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg') 
# from htree.embedding import EuclideanEmbedding
# import numpy as np
# # # Function to generate points at the vertices of a regular pentagon
# def generate_pentagon_vertices():
#      angles = np.linspace(0, 2 * np.pi, 6)[:-1]  # 5 angles for 5 vertices, excluding the last one to avoid duplication
#      x = np.cos(angles)
#      y = np.sin(angles)
#      return np.vstack((x, y))
# # # Function to plot points with edges, and axes
# def plot_pentagon(ax, points, title, color, marker, label_text):
#      ax.scatter(points[0, :], points[1, :], c=color, s=100, edgecolors='black', marker=marker, label=label_text)    
#      # Draw edges between consecutive points
#      for i in range(points.shape[1]):
#          start = i
#          end = (i + 1) % points.shape[1]
#          ax.plot([points[0, start], points[0, end]], [points[1, start], points[1, end]], color=color)
#      # Draw the axes
#      ax.axhline(0, color='black', linewidth=0.5)
#      ax.axvline(0, color='black', linewidth=0.5)
#      ax.set_title(title)
#      ax.set_xlabel('X-axis')
#      ax.set_ylabel('Y-axis')
#      ax.set_aspect('equal', adjustable='box')
#      ax.legend()
# # # Define pentagon vertices
# points = generate_pentagon_vertices()
# dimension = points.shape[0]
# n_points = points.shape[1]
# # # Initialize embedding
# embedding = EuclideanEmbedding(points=points)
# # # Create a figure for the translation
# # fig1, ax1 = plt.subplots(figsize=(8, 8))
# # # Plot the original pentagon vertices with edges
# # plot_pentagon(ax1, embedding.points, "Translation of Pentagon Vertices", color='black', marker='o', label_text='Original Points')
# # # Translate the points
# # translation_vector = np.random.randn(dimension)/2  # 2D translation vector
# # embedding.translate(translation_vector)
# # # Plot the translated pentagon vertices with edges
# # plot_pentagon(ax1, embedding.points, "Translation of Pentagon Vertices", color='green', marker='^', label_text='Translated Points')
# # # Show the plot for translation
# # plt.show()




# # embedding = EuclideanEmbedding(points=points, labels=labels)
# # Create a figure for the rotation
# fig2, ax2 = plt.subplots(figsize=(8, 8))
# # Plot the translated pentagon vertices with edges and labels before rotation
# plot_pentagon(ax2, embedding.points, "Rotation of Pentagon Vertices", color='black', marker='o', label_text='Before Rotation')
# # Convert degrees to radians and define the 2x2 rotation matrix
# theta = np.radians(30)
# rotation_matrix = np.array([
#     [np.cos(theta), -np.sin(theta)],
#     [np.sin(theta), np.cos(theta)]
# ])
# # Rotate the points
# embedding.rotate(rotation_matrix)
# # Plot the pentagon vertices with edges after rotation
# plot_pentagon(ax2, embedding.points, "Rotation of Pentagon Vertices", color='red', marker='^', label_text='After Rotation')
# # Show the plot for rotation
# plt.show()










# from htree.procrustes import EuclideanProcrustes
# from htree.embedding import EuclideanEmbedding
# import numpy as np
# import htree.logger as logger
# logger.set_logger(True)

# n_points = 10
# dimension = 2
# points = np.random.randn(dimension, n_points)
# embedding = EuclideanEmbedding(points=points)
# Make a copy of the embedding to serve as the source embedding
# source_embedding = embedding.copy()
# print(source_embedding.points)
# tensor([[-1.6209, -0.2308, -0.6212,  0.6591, -0.9084, -1.0451, -0.5635, -1.5712,
#          -1.3372,  0.3394],
#         [ 1.7770, -0.1018,  1.2887,  0.5590, -0.4241,  0.8596, -0.3132,  1.9741,
#          -0.7349, -0.1978]], dtype=torch.float64)
# Create the target embedding by transforming the source embedding
# target_embedding = embedding.copy()
# Apply transformation: translation + rotation
# translation_vector = np.random.randn(dimension)
# target_embedding.translate(translation_vector)
# Convert degrees to radians and define the 2x2 rotation matrix
# theta = np.radians(30)
# rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
# target_embedding.rotate(rotation_matrix)
# print(target_embedding.points)
# tensor([[-2.1462, -0.0029, -1.0362,  0.4374, -0.4286, -1.1888, -0.1853, -2.2017,
#          -0.6445,  0.5389],
#         [ 1.8580,  0.9259,  1.9349,  1.9431,  0.3079,  1.3513,  0.5765,  2.0535,
#          -0.1756,  1.1279]], dtype=torch.float64)
# Initialize the Procrustes model with the source and target embeddings
# model = EuclideanProcrustes(source_embedding, target_embedding)
# Use the model to align the source embedding to the target
# source_aligned = model.map(source_embedding)
# The aligned source embedding should now closely match the target
# print(source_aligned.points)
# tensor([[-2.1462, -0.0029, -1.0362,  0.4374, -0.4286, -1.1888, -0.1853, -2.2017,
#          -0.6445,  0.5389],
#         [ 1.8580,  0.9259,  1.9349,  1.9431,  0.3079,  1.3513,  0.5765,  2.0535,
#          -0.1756,  1.1279]], dtype=torch.float64)

















# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg') 
# # Define noise variances
# noise_variances = np.linspace(0, .1, 10000)  # Variances from 0 to 1
# alignment_errors = []
# # Loop through different noise variances
# for noise_variance in noise_variances:
#     noisy_target_embedding = target_embedding.copy()
#     # Add Gaussian noise to the points with specified variance
#     noise = np.random.normal(0, np.sqrt(noise_variance), size=noisy_target_embedding.points.shape)
#     noisy_target_embedding.points += noise
#     # Apply Procrustes alignment
#     model = EuclideanProcrustes(source_embedding, noisy_target_embedding)
#     source_aligned = model.map(source_embedding)
#     # Compute alignment error (e.g., mean squared error between aligned and target points)
#     alignment_error = np.mean(np.linalg.norm(source_aligned.points - noisy_target_embedding.points, axis=0))
#     alignment_errors.append(alignment_error)
# # Plot quality of embedding vs. noise variance
# plt.scatter(noise_variances, alignment_errors, marker='o')
# plt.xlabel('Noise Variance')
# plt.ylabel('Alignment Error')
# plt.title('Quality of Embedding vs. Noise Variance')
# plt.grid(True)
# plt.show()











# from htree.procrustes import HyperbolicProcrustes
# from htree.embedding import PoincareEmbedding
# import numpy as np
# n_points = 10
# dimension = 2
# # Generate random points for the source embedding
# points = np.random.randn(dimension, n_points)/10
# # Initialize the source embedding
# source_embedding = PoincareEmbedding(points=points)
# print(source_embedding.points)
# # tensor([[ 0.2135, -0.0119,  0.0454, -0.1136,  0.0334, -0.0786, -0.0686, -0.1149,
#          # -0.0842,  0.1199],
#         # [-0.0798,  0.0566,  0.1365,  0.0822, -0.0354, -0.1948, -0.1846,  0.0131,
#          # -0.0577,  0.0051]], dtype=torch.float64)
# # Create a target embedding by copying and transforming the source
# target_embedding = source_embedding.copy()
# translation_vector = np.random.randn(dimension)/4
# target_embedding.translate(translation_vector)
# theta = np.radians(27.5)
# rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
# target_embedding.rotate(rotation_matrix)
# print(target_embedding.points)
# # tensor([[-0.4023, -0.5559, -0.5553, -0.6032, -0.5126, -0.5377, -0.5346, -0.5871,
#          # -0.5603, -0.4777],
#         # [-0.4450, -0.4259, -0.3731, -0.4428, -0.4570, -0.5466, -0.5412, -0.4689,
#          # -0.4906, -0.4155]], dtype=torch.float64)
# # Initialize the Procrustes model with the source and target embeddings
# model = HyperbolicProcrustes(source_embedding, target_embedding,precise_opt = True) # mode = 'default'
# # Map the source embedding to the target space
# source_aligned = model.map(source_embedding)
# print(source_aligned.points)
# # tensor([[-0.4023, -0.5559, -0.5553, -0.6032, -0.5126, -0.5377, -0.5346, -0.5871,
#          # -0.5603, -0.4777],
#         # [-0.4450, -0.4259, -0.3731, -0.4428, -0.4570, -0.5466, -0.5412, -0.4689,
#          # -0.4906, -0.4155]], dtype=torch.float64)
# # Switch the model of the source embedding
# source_embedding  = source_embedding.switch_model()
# print(source_embedding)
# # HyperbolicEmbedding(curvature=-1.00, model=loid, points_shape=[3, 10])
# print(source_embedding.points)
# # tensor([[ 1.1096,  1.0067,  1.0423,  1.0401,  1.0047,  1.0923,  1.0807,  1.0271,
#           # 1.0211,  1.0292],
#         # [ 0.4503, -0.0240,  0.0926, -0.2318,  0.0669, -0.1645, -0.1427, -0.2330,
#          # -0.1702,  0.2433],
#         # [-0.1684,  0.1135,  0.2788,  0.1676, -0.0709, -0.4075, -0.3840,  0.0266,
#          # -0.1167,  0.0103]], dtype=torch.float64)
# source_aligned = model.map(source_embedding)
# print(source_aligned.points)
# # tensor([[-0.4023, -0.5559, -0.5553, -0.6032, -0.5126, -0.5377, -0.5346, -0.5871,
#          # -0.5603, -0.4777],
#         # [-0.4450, -0.4259, -0.3731, -0.4428, -0.4570, -0.5466, -0.5412, -0.4689,
#          # -0.4906, -0.4155]], dtype=torch.float64)









# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg') 
# # Define a trivial function to compute the alignment cost
# def compute_alignment_cost(src_embedding, trg_embedding):
#     cost = sum((torch.norm(src_embedding.points[:, n]- trg_embedding.points[:, n]))**2 for n in range(src_embedding.n_points))
#     return cost
# # Define noise variances
# noise_variances = np.linspace(0, .1, 1000)  # Variances from 0 to 1
# alignment_errors = []
# # Loop through different noise variances
# cnt = 1
# for noise_variance in noise_variances:
#     print(cnt)
#     cnt +=1
#     noisy_target_embedding = target_embedding.copy()
#     noisy_target_embedding.switch_model()
#     # Add Gaussian noise to the points with specified variance
#     noise = np.random.normal(0, np.sqrt(noise_variance)/2, size=noisy_target_embedding.points.shape)
#     noisy_target_embedding.points += noise
#     noisy_target_embedding.switch_model()
#     # Apply Procrustes alignment
#     model = HyperbolicProcrustes(source_embedding, noisy_target_embedding, precise_opt=False)
#     source_aligned = model.map(source_embedding)
#     alignment_error = np.mean(np.linalg.norm(source_aligned.points - noisy_target_embedding.points, axis=0))
#     alignment_errors.append(alignment_error)
# # Plot quality of embedding vs. noise variance
# plt.scatter(noise_variances, alignment_errors, marker='o')
# plt.xlabel('Noise Variance')
# plt.ylabel('Alignment Error')
# plt.title('Quality of Embedding vs. Noise Variance')
# plt.grid(True)
# plt.show()











# from htree.embedding import MultiEmbedding
# from htree.embedding import EuclideanEmbedding
# import numpy as np
# import torch
# # Initialize a MultiEmbedding object
# multi_embedding = MultiEmbedding()
# # Create and add multiple embeddings
# n_points = 10
# dimension = 2
# labels = [str(i) for i in range(n_points)]
# points = np.random.randn(dimension,n_points)/10
# embedding1 = EuclideanEmbedding(points = points, labels = labels)
# print(embedding1)
# # EuclideanEmbedding(points_shape=[2, 10])
# multi_embedding.append(embedding1)
# # Add another embedding with different dimensions
# labels = [str(i) for i in range(14)]
# points = np.random.randn(4,14)/10
# embedding2 = EuclideanEmbedding(points = points, labels = labels)
# print(embedding2)
# # EuclideanEmbedding(points_shape=[4, 14])
# multi_embedding.append(embedding2)
# labels = [str(i) for i in range(5)]
# points = np.random.randn(3,5)/10
# embedding3 = EuclideanEmbedding(points = points, labels = labels)
# print(embedding3)
# # EuclideanEmbedding(points_shape=[3, 5])
# multi_embedding.append(embedding3)
# labels = [str(i) for i in range(7)]
# points = np.random.randn(4,7)/10
# embedding4 = EuclideanEmbedding(points = points, labels = labels)
# print(embedding4)
# # EuclideanEmbedding(points_shape=[4, 7])
# multi_embedding.append(embedding4)
# print(multi_embedding)
# # MultiEmbedding(4 embeddings)
# # Check the embeddings dictionary
# print(multi_embedding.embeddings)
# # [EuclideanEmbedding(points_shape=[2, 10]),EuclideanEmbedding(points_shape=[4, 14]), EuclideanEmbedding(points_shape=[3, 5]), EuclideanEmbedding(points_shape=[4, 7])]
# # another way of initialization the multiembedding
# multi_embedding = MultiEmbedding()
# multi_embedding.embeddings = [embedding1, embedding2, embedding3, embedding4]
# print(multi_embedding.embeddings)
# # [EuclideanEmbedding(points_shape=[2, 10]), EuclideanEmbedding(points_shape=[4, 14]), EuclideanEmbedding(points_shape=[3, 5]), EuclideanEmbedding(points_shape=[4, 7])]
# # computing the aggregate distance matrix over the union of labels
# print(multi_embedding.distance_matrix()[0].shape)
# # torch.Size([14, 14])
# # Compute the aggregated distance matrix
# print(multi_embedding.distance_matrix()[0][:4,:4]) # defualt aggregation method is mean
# # tensor([[0.0000, 0.2846, 0.3471, 0.2391],
#         # [0.2846, 0.0000, 0.4289, 0.2696],
#         # [0.3471, 0.4289, 0.0000, 0.2342],
#         # [0.2391, 0.2696, 0.2342, 0.0000]], dtype=torch.float64)
# # Using a different aggregation function (e.g., torch.nanmedian)
# print(multi_embedding.distance_matrix(func = torch.nanmedian)[0][:4,:4])
# # tensor([[0.0000, 0.2747, 0.3471, 0.2391],
#         # [0.2747, 0.0000, 0.4289, 0.2696],
#         # [0.3471, 0.4289, 0.0000, 0.2342],
#         # [0.2391, 0.2696, 0.2342, 0.0000]], dtype=torch.float64)










# Load trees from Newick files
from tree_collections import MultiTree
import treeswift as ts
import logger as logger
import torch

logger.set_logger(True)

tree1 = ts.read_tree_newick('path/to/treefile1.tre')
tree2 = ts.read_tree_newick('path/to/treefile2.tre')
# Create a MultiTree object
multitree = MultiTree('name', [tree1, tree2, tree2])
print(multitree)
# MultiTree(name, 2 trees)
# Print the first 4x4 section of the average distance matrix from the trees
# print(multitree.distance_matrix()[0][:4,:4])
# tensor([[0.0000, 2.2280, 1.6060, 0.9966],
#         [2.2280, 0.0000, 2.5171, 1.8496],
#         [1.6060, 2.5171, 0.0000, 2.0671],
#         [0.9966, 1.8496, 2.0671, 0.0000]])
# Create joint embeddings for the trees in 2-dimensional hyperbolic space
# multiembedding = multitree.embed(dim = 2,geometry = 'euclidean')
multiembedding = multitree.embed(dim = 2,geometry = 'hyperbolic')
print(multiembedding)
# MultiEmbedding(2 embeddings)
# View the embeddings for each tree
print(multiembedding.embeddings)
# [HyperbolicEmbedding(curvature=-8.92, model=loid, points_shape=[3, 30]), HyperbolicEmbedding(curvature=-8.92, model=loid, points_shape=[3, 14])]
# Show the first 4x4 section of the distance matrix from the embeddings
# print(multiembedding.distance_matrix()[0][:4,:4]) # hyperbolic distances are used to approxiate tree distances
# tensor([[5.9791e-08, 2.1971e+00, 2.9019e-01, 6.6652e-02],
#         [2.1971e+00, 1.3717e-08, 2.4854e+00, 1.2870e+00],
#         [2.9019e-01, 2.4854e+00, 0.0000e+00, 1.4410e+00],
#         [6.6652e-02, 1.2870e+00, 1.4410e+00, 0.0000e+00]], dtype=torch.float64)
# reference = multiembedding.reference_embedding(func = torch.nanmedian, precise_opt= True) # compute the reference point set by embedding the average distance matrix
# print(reference.points)
# HyperbolicEmbedding(curvature=-8.92, model=loid, points_shape=[3, 40])
# print( reference.distance_matrix()[0][:4,:4]) # Show the distance matrix of the reference embedding
# tensor([[0.0000, 2.0969, 3.1035, 1.4822],
#         [2.0969, 0.0000, 2.3563, 1.9331],
#         [3.1035, 2.3563, 0.0000, 3.1467],
#         [1.4822, 1.9331, 3.1467, 0.0000]], dtype=torch.float64)
# reference = multiembedding.reference_embedding(precise_opt=True) # Compute a more accurate reference embedding
# print(reference)
# HyperbolicEmbedding(curvature=-8.92, model=loid, points_shape=[3, 40])
# print( reference.distance_matrix()[0][:4,:4]) # Show the more accurate distance matrix of the reference embedding
# tensor([[0.0000e+00, 1.5942e+00, 2.6063e+00, 4.3736e-01],
#         [1.5942e+00, 0.0000e+00, 2.1983e+00, 1.5878e+00],
#         [2.6063e+00, 2.1983e+00, 0.0000e+00, 2.4947e+00],
#         [4.3736e-01, 1.5878e+00, 2.4947e+00, 7.9983e-08]], dtype=torch.float64)
# multiembedding = multitree.embed(dim = 2, precise_opt=True) # Recompute embeddings in hyperbolic space with more accuracy
# print(multiembedding)
# MultiEmbedding(2 embeddings)
# print(multiembedding.embeddings) # View the embeddings with updated curvature
# [HyperbolicEmbedding(curvature=-8.30, model=loid, points_shape=[3, 30]), HyperbolicEmbedding(curvature=-8.30, model=loid, points_shape=[3, 14])]
# embedding trees in euclidean space
# multiembedding = multitree.embed(dim = 2, geometry = 'euclidean')
# print(multiembedding.embeddings)
# [EuclideanEmbedding(points_shape=[2, 30]), EuclideanEmbedding(points_shape=[2, 14])]
# print((multiembedding.distance_matrix()[0][:4,:4])**2) # Compute the squared Euclidean distance matrix (used to approximate tree distances)
# tensor([[0.0000, 0.9731, 0.0194, 0.0064],
#         [0.9731, 0.0000, 0.7350, 0.3536],
#         [0.0194, 0.7350, 0.0000, 0.4402],
#         [0.0064, 0.3536, 0.4402, 0.0000]], dtype=torch.float64)
# reference = multiembedding.reference_embedding(precise_opt=True)
# print(reference.points)
# EuclideanEmbedding(points_shape=[2, 40])
# print((reference.distance_matrix()[0][:4,:4])**2) # Show the squared distance matrix of the reference embedding
# tensor([[0.0000, 0.8788, 0.3246, 0.2431],
#         [0.8788, 0.0000, 0.5642, 0.3351],
#         [0.3246, 0.5642, 0.0000, 0.5179],
#         [0.2431, 0.3351, 0.5179, 0.0000]], dtype=torch.float64)



import torch

multiembedding.align(func = torch.nanmedian, precise_opt=True,p = 1)
print(multiembedding[0].points)





