import torch
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import DataLoader
import argparse

from prototree.prototree import ProtoTree
from util.log import Log

from pixellib.semantic import semantic_segmentation


def identify_clever_hans(tree: ProtoTree, project_info: dict, project_loader: DataLoader,
                         segment_image: semantic_segmentation, threshold: float, args: argparse.Namespace, log: Log):
    """
    This is modified code from upsample.py where the visualization functionality has been removed. Instead, only the
    overlap of each patch with a mask of the bird is calculated. The function returns the project_info that now includes
    overlap of each patch with the mask, and whether the prototype patch indicates a clever hans prototype given an
    overlap threshold.
    """
    with torch.no_grad():
        imgs = project_loader.dataset.imgs
        log.log_message("\n-- Identification of Clever Hans Prototypes.")
        for node, j in tree._out_map.items():
            torch.cuda.empty_cache()
            if node in tree.branches:  # do not identify clever hans when node is pruned
                log.log_message("\nIdentifying whether node "+str(node.index)+" is a Clever Hans prototype.")
                # Get prototype objects and attributes
                prototype_info = project_info[j]
                W, H = prototype_info['W'], prototype_info['H']
                assert W == H
                decision_node_idx = prototype_info['node_ix']
                x = Image.open(imgs[prototype_info['input_image_ix']][0])
                x_np = np.asarray(x)
                x_np = np.float32(x_np) / 255
                if x_np.ndim == 2:  # convert grayscale to RGB
                    x_np = np.stack((x_np,) * 3, axis=-1)
                img_size = x_np.shape[:2]

                # Get indices for all patches given the upsample threshold
                patches_ix = []
                for i in range(W*H):
                    masked_similarity_map = np.zeros((W, H))
                    masked_similarity_map[i // W, i % W] = 1
                    upsampled_prototype_pattern = cv2.resize(masked_similarity_map,
                                                             dsize=(img_size[1], img_size[0]),
                                                             interpolation=cv2.INTER_CUBIC)
                    patch_indices = find_high_activation_crop(upsampled_prototype_pattern, args.upsample_threshold)
                    patches_ix.append(patch_indices)

                # Image segmentation
                segvalues, _ = segment_image.segmentAsPascalvoc(imgs[prototype_info['input_image_ix']][0], overlay=True)

                # Calculate overlap of patch with segmentation mask for each patch
                patches_segvalues =[segvalues["masks"][patch_ix[0]:patch_ix[1], patch_ix[2]:patch_ix[3]]
                                    for patch_ix in patches_ix]
                overlap_percentages = [round(np.sum(patch_segvalue) / np.size(patch_segvalue), 3)
                                       for patch_segvalue in patches_segvalues]

                # Result of the function
                log.log_message("Bounding box coordinates: "+str(patches_ix[prototype_info["patch_ix"]]))
                log.log_message("Overlap percentage node "+str(decision_node_idx)+": "+ str(overlap_percentages[prototype_info["patch_ix"]]))
                project_info[j]["overlap_percentages"] = overlap_percentages
                project_info[j]["is_clever_hans"] = (overlap_percentages[prototype_info["patch_ix"]] < threshold)
    return project_info


# copied from protopnet (and from upsample.py)
def find_high_activation_crop(mask, threshold):
    threshold = 1. - threshold
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > threshold:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > threshold:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > threshold:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > threshold:
            upper_x = j
            break
    return lower_y, upper_y + 1, lower_x, upper_x + 1


