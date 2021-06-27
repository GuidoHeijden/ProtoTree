################  EXAMPLE EXPERIMENT FOR PROTOTYPE REPLACEMENT  ##################
##  The code in this file functions as an example for how to replace the        ##
##  prototypes of a ProtoTree to correct Clever Hans behaviour. Note that it    ##
##  requires you to already have a trained ProtoTree available as well as its   ##
##  project info.                                                               ##
##################################################################################

import gc
import torch
import pickle
from pixellib.semantic import semantic_segmentation

from util.args import load_args
from util.data import get_dataloaders
from util.log import Log

# The imports below are especially relevant to the correction of Clever Hans behaviour
from prototree.replace_prototype import replace_prototype, replace_prototypes
from prototree.identify_clever_hans import identify_clever_hans
from prototree.find_high_similarity_replacement import find_high_similarity_restriction_images
from prototree.save_after_replacement import save_after_replacement


# Loading a trained ProtoTree (please change the path to the correct locations/files)
with open('./runs/tree_0/project_info.pkl', 'rb') as handle:
    project_info = pickle.load(handle)
with open("./runs/tree_0/checkpoints/tree.pkl", 'rb') as handle:
    tree = pickle.load(handle)


# It might be desired to free up GPU memory by moving the ProtoTree to CPU, for example to fit an
# image segmentation model into the GPU. This can be done as follows:
print("Allocated memory:", torch.cuda.memory_allocated())  # should be 0
print("Reserved memory:", torch.cuda.memory_reserved())  # should be 0
device = torch.device('cpu')
tree = tree.to(device)
for i in project_info.keys():
    project_info[i]['nearest_input'] = project_info[i]['nearest_input'].to(device)
print("Allocated memory:", torch.cuda.memory_allocated())  # in the range of 10^5
print("Reserved memory:", torch.cuda.memory_reserved())  # in the range of 10^8
gc.collect()
torch.cuda.empty_cache()
print("Allocated memory:", torch.cuda.memory_allocated())  # in the range of 10^5
print("Reserved memory:", torch.cuda.memory_reserved())  # in the range of 10^6


# Loading additional data
args = load_args("./runs/tree_0/metadata/")
_, projectloader, _, _, _ = get_dataloaders(args=args)
log = Log("./replacements/logs/all_prototype-similarity_based")  # Note that different logs are used when saving a tree

# Load object for image segmentation, which is used for identifying clever hans prototypes
segment_image = semantic_segmentation()
segment_image.load_pascalvoc_model("./deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")


# Identify the Clever Hans prototypes in the original ProtoTree
project_info = identify_clever_hans(tree=tree, project_info=project_info, project_loader=projectloader,
                                    segment_image=segment_image, threshold=0.2, args=args, log=log)
all_node_indices = [node.index for node in tree.nodes]
save_after_replacement(tree=tree, project_info=project_info, args=args, which_nodes=all_node_indices,
                       folder_name="./replacements/clever_hans_identified_0")

# If initial Clever Hans identification has been done already, load that model below and comment out above
with open('./replacements/clever_hans_identified_0/project_info.pickle', 'rb') as handle:
    project_info = pickle.load(handle)
with open("./replacements/clever_hans_identified_0/checkpoints/tree.pkl", 'rb') as handle:
    tree = pickle.load(handle)

# Retrieve which prototypes are Clever Hans based on the project_info
clever_hans_node_ixs = []
for prototype_info in project_info.values():
    if 'is_clever_hans' in prototype_info.keys() and prototype_info['is_clever_hans']:
        clever_hans_node_ixs.append(prototype_info['node_ix'])
log.log_message("\nClever Hans prototype nodes: "+str(clever_hans_node_ixs))

# Replace Clever Hans prototypes with those that are not Clever Hans and most similar iteratively per layer
for cur_depth in range(1, tree.depth):
    clever_hans_nodes_at_depth_i = [clever_hans_node_ix for clever_hans_node_ix in clever_hans_node_ixs
                                    if tree.nodes_by_index[clever_hans_node_ix].depth == cur_depth]
    project_info, tree = find_high_similarity_restriction_images(tree=tree, project_loader=projectloader, device=device,
                                                                 log=log, segment_image=segment_image, threshold=0.2,
                                                                 which_nodes=clever_hans_nodes_at_depth_i,
                                                                 cur_project_info=project_info, args=args)
    save_after_replacement(tree=tree, project_info=project_info, args=args, which_nodes=clever_hans_nodes_at_depth_i,
                           folder_name="./replacements/all_prototype-similarity_based_depth_"+str(cur_depth)+"_0")

    # Note that the replacement of the previous layer is maintained when replacing the next layer
