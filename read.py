import numpy as np
import torch
from htree.tree_collections import Tree
from htree.tree_collections import MultiTree
# Initialize from file
tree = Tree("path/to/treefile.tre")
# print(tree)
# # Initialize from a treeswift Tree object
# import treeswift as ts
# t = ts.read_tree_newick("path/to/treefile.tre")
print(tree)


import htree.logger as logger
logger.set_logger(True)
print(tree)


# terminals = tree.terminal_names()[:4]
# print(terminals)
# dmT, names = tree.distance_matrix()
# print(dmT[:4,:4])
# print(names[:4])
# Get tree diameter
# diameter = tree.diameter()
# print(diameter)


# tree.normalize()
# diameter = tree.diameter()
# print(diameter)


embedding = tree.embed(dim=2, geometry='euclidean')
print(embedding)
print(embedding.points)
x = embedding.embed(dim=4, geometry='hyperbolic', curvature = -10, precise_opt = True)
print(x)
print(x.points)

# from htree.PCA import PCA

# pca = PCA(embedding)
# emb2 = pca.map_to(dim=2)
# print(emb2)
# # print(embedding.points)
# embedding = tree.embed(dim=2, geometry='hyperbolic', curvature = -100)
# # print(embedding.points)
# # Embed tree in 3D Euclidean space
# embedding = tree.embed(dim=3, geometry='euclidean')
# print(embedding)

# dmT, names = tree.distance_matrix()
# embedding = tree.embed(dim=101, geometry='euclidean',  precise_opt=False)
# print(embedding)
# dmH,_ = embedding.distance_matrix()
# # print(dmH)

# # print( np.linalg.norm(dmT- dmH**2)/np.linalg.norm(dmT))


# dmT, names = tree.distance_matrix()
# embedding = tree.embed(dim=3, geometry='hyperbolic', precise_opt=True)
# print(embedding)
# dmH,_ = embedding.distance_matrix()
# print(dmH)

# print( np.linalg.norm(dmT- dmH)/np.linalg.norm(dmT))

# print(tree.embed(dim=2, precise_opt=True, lr_init=0.1))

# print(tree.embed(dim=2, precise_opt=True, epochs=2000))
# print(tree.embed(dim=2, dist_cutoff=5.0))

# print(tree.embed(dim=2, precise_opt=True, save_mode=True))

# tree.embed(dim=2, precise_opt=True, export_video=True)

# mtree = MultiTree("path/to/trees.tre")
# tree = mtree[190]
# # # print(mtree)

# embedding = tree.embed(dim=12, geometry='euclidean', precise_opt=True, export_video=True)
# print(embedding)
# dmH,_ = embedding.distance_matrix()

# print( np.linalg.norm(dmT- dmH**2)/np.linalg.norm(dmT))

# embedding = tree.embed(dim=76, geometry='hyperbolic', precise_opt=False)
# print(embedding)
# dmH,_ = embedding.distance_matrix()
# print( np.linalg.norm(dmT- dmH)/np.linalg.norm(dmT))

# def custom_weight_exponent(epoch: int, total_epochs: int,loss_list: List[float]) -> float:
#    """
#    Calculate the weight exponent based on the current epoch and total number of epochs.
#    Parameters:
#    - epoch (int): The current epoch in the training process.
#    - total_epochs (int): The total number of epochs in the training process.
#    - loss_list (list): A list of recorded loss values (can be used for further custom logic).

#    Returns:
#    - float: The calculated weight exponent for the current epoch.

#    Raises:
#    - ValueError: If `total_epochs` is less than or equal to 1.
#    """
#    if total_epochs <= 1:
#        raise ValueError("Total epochs must be greater than 1.")

#    # Define a ratio that determines how long to apply no weights
#    no_weight_ratio = 0.3  # Example ratio: first 30% of epochs without weighting
#    no_weight_epochs = int(no_weight_ratio * total_epochs)
#    # No weighting for the first part of the training
#    if epoch < no_weight_epochs:
#        return 0.0  # No weighting initially
#    # Gradually increase the negative weight exponent after the no-weight phase
#    return -(epoch - no_weight_epochs) / (total_epochs - 1 - no_weight_epochs)
# print(tree.embed(dim=2, precise_opt=True))
# print(tree.embed(dim=2, precise_opt=True, weight_exp_fn=custom_weight_exponent))

# tree.save("path/to/treefile11.tre") 

# print( np.linalg.norm(dmT- dmH)/np.linalg.norm(dmT))
# tree.update_time()

# embedding = tree.embed(dim=15, geometry='hyperbolic', precise_opt=True,export_video=True)

# print(embedding)
# dmH,_ = embedding.distance_matrix()

# print( np.linalg.norm(dmT- dmH)/np.linalg.norm(dmT))

# tree.embed(dim=2, precise_opt=True, export_video=True)


# mtree = MultiTree("path/to/trees.tre")
# # print(mtree)

# # print(mtree.terminal_names())

# avg_mat, conf, labels = mtree.distance_matrix(method='fp')
# # # func=torch.nanmedian

# print(avg_mat)



# mtree_normal = mtree.normalize()
# mtree = mtree[:10]
# # # print(mtree_normal)
# # avg_mat, conf, labels = mtree.distance_matrix(method='fp')
# # print(avg_mat)
# # avg_mat, conf, labels = mtree.distance_matrix()
# # print(avg_mat)
# # mtree = mtree[20:60]
# multiemb_hyperbolic = mtree.embed(dim=10, geometry='hyperbolic', precise_opt=True)
# print(multiemb_hyperbolic[0])
# dist= multiemb_hyperbolic.distance_matrix()
# print(dist[0])
# x = multiemb_hyperbolic.reference_embedding(precise_opt=True)
# print(x)
# print(multiemb_hyperbolic[0])




# # Initialize from a Newick file
# multitree = MultiTree("path/to/trees.tre")
# print(multitree)
# # MultiTree(trees.tre, 844 trees)
# # Initialize from a list of trees
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

# import htree.logger as logger
# logger.set_logger(True)
# multitree = MultiTree("path/to/trees.tre")
# print(multitree)
# # MultiTree(trees.tre, 844 trees)


# multitree = MultiTree("path/to/trees.tre")[:10]
# print(multitree)
# # MultiTree(mTree, 10 trees)
# print(multitree.trees)
# # [Tree(tree_1), Tree(tree_2), Tree(tree_3), Tree(tree_4), Tree(tree_5), Tree(tree_6), Tree(tree_7), Tree(tree_8), Tree(tree_9), Tree(tree_10)]
# # Compute the distance matrix with default aggregation (mean)
# avg_mat, conf, labels = multitree.distance_matrix()
# print(avg_mat[:4,:4])
# # tensor([[0.0000, 0.7049, 1.2343, 0.5929],
# #         [0.7049, 0.0000, 1.3234, 0.6870],
# #         [1.2343, 1.3234, 0.0000, 1.0143],
# #         [0.5929, 0.6870, 1.0143, 0.0000]])
# # Compute the distance matrix with custom aggregation
# import torch
# avg_mat, conf, labels = multitree.distance_matrix(func=torch.nanmedian)
# print(avg_mat[:4,:4])
# # tensor([[0.0000, 0.5538, 0.9043, 0.5240],
# #         [0.5538, 0.0000, 1.1598, 0.5902],
# #         [0.9043, 1.1598, 0.0000, 0.8635],
# #         [0.5240, 0.5902, 0.8635, 0.0000]])
# avg_mat, conf, labels = multitree.distance_matrix(method='fp')
# print(avg_mat[:4,:4])
# # tensor([[0.0000, 0.6760, 1.1487, 0.5696],
# #         [0.6760, 0.0000, 1.2801, 0.6613],
# #         [1.1487, 1.2801, 0.0000, 0.9627],
# #         [0.5696, 0.6613, 0.9627, 0.0000]])
# # Compute the union of all terminal names (removes duplicates)
# print(multitree.terminal_names()[:4])
# # ['Allamanda_cathartica', 'Alsophila_spinulosa', 'Amborella_trichopoda', 'Aquilegia_formosa']







# # Embed trees in a 2D hyperbolic space
# multiemb_hyperbolic = multitree.embed(dim=2, geometry='hyperbolic')
# print(multiemb_hyperbolic)
# # MultiEmbedding(10 embeddings)
# print(multiemb_hyperbolic.embeddings)
# # [HyperbolicEmbedding(curvature=-0.84, model=loid, points_shape=[3, 24]), HyperbolicEmbedding(curvature=-0.84, model=loid, points_shape=[3, 76]), HyperbolicEmbedding(curvature=-0.84, model=loid, points_shape=[3, 71]), HyperbolicEmbedding(curvature=-0.84, model=loid, points_shape=[3, 70]), HyperbolicEmbedding(curvature=-0.84, model=loid, points_shape=[3, 63]), HyperbolicEmbedding(curvature=-0.84, model=loid, points_shape=[3, 80]), HyperbolicEmbedding(curvature=-0.84, model=loid, points_shape=[3, 60]), HyperbolicEmbedding(curvature=-0.84, model=loid, points_shape=[3, 76]), HyperbolicEmbedding(curvature=-0.84, model=loid, points_shape=[3, 82]), HyperbolicEmbedding(curvature=-0.84, model=loid, points_shape=[3, 51])]
# # Embed trees in a 3D Euclidean space
# multiemb_euclidean = multitree.embed(dim=3, geometry='euclidean')
# print(multiemb_euclidean)
# # MultiEmbedding(10 embeddings)
# print(multiemb_euclidean.embeddings)
# # [EuclideanEmbedding(points_shape=[3, 24]), EuclideanEmbedding(points_shape=[3, 76]), EuclideanEmbedding(points_shape=[3, 71]), EuclideanEmbedding(points_shape=[3, 70]), EuclideanEmbedding(points_shape=[3, 63]), EuclideanEmbedding(points_shape=[3, 80]), EuclideanEmbedding(points_shape=[3, 60]), EuclideanEmbedding(points_shape=[3, 76]), EuclideanEmbedding(points_shape=[3, 82]), EuclideanEmbedding(points_shape=[3, 51])]


# # Save trees to a Newick file
# multitree.save("path/to/output12.tre")