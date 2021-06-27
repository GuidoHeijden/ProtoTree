import argparse
import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from prototree.prototree import ProtoTree
from util.log import Log

from PIL import Image
from pixellib.semantic import semantic_segmentation


def find_high_similarity_same_image(tree: ProtoTree,
                                    project_info: dict,
                                    node_index: int,
                                    threshold: float) -> dict:
    """
    Replace the patch of a prototype with the closest path that overlaps more than the threshold with the segmentation
    mask of the image.

    Note that as a prerequisite, the cur_project_info should include dict keys "overlap_percentages" and
    "is_clever_hans", which are added after running identify_clever_hans.py
    """
    # Calculate similarity maps for the prototype that needs replacement
    prototype_info = project_info[tree._out_map[tree.nodes_by_index[node_index]]]
    nearest_x = prototype_info['nearest_input']
    with torch.no_grad():
        _, distances_batch, _ = tree.forward_partial(nearest_x)
        sim_map = torch.flatten(torch.exp(-distances_batch[0, tree._out_map[tree.nodes_by_index[node_index]],
                                           :, :])).cpu().numpy()
    del nearest_x

    # Find the index of the patch that overlaps more than the threshold and has highest similarity
    for patch_ix in sim_map.argsort()[::-1]:
        if prototype_info["overlap_percentages"][patch_ix] > threshold:
            replacement_patch_index = patch_ix
            break

    # Retrieve the latent image for the prototype of the node at node_index
    W, H = prototype_info['W'], prototype_info['H']
    xs = prototype_info['nearest_input']
    latent_img_batch, _, _ = tree.forward_partial(xs)
    latent_img = latent_img_batch[0]

    # Replace the necessary info regarding the prototype: prototype_vector, patch_ix
    new_prototype_vector = torch.unsqueeze(torch.unsqueeze(
        latent_img[:, replacement_patch_index // H, replacement_patch_index % W],
        1), 1)
    tree.prototype_layer.prototype_vectors[tree._out_map[tree.nodes_by_index[node_index]]] = new_prototype_vector
    project_info[tree._out_map[tree.nodes_by_index[node_index]]]['patch_ix'] = replacement_patch_index

    return tree, project_info


def find_high_similarity_restriction_images(tree: ProtoTree,
                                            project_loader: DataLoader,
                                            device,
                                            log: Log,
                                            segment_image: semantic_segmentation,
                                            threshold: float,
                                            which_nodes: list,
                                            cur_project_info,
                                            args: argparse.Namespace,
                                            progress_prefix: str = 'Projection'
                                            ) -> dict:
    """
    This function is essentially the same as projecting a ProtoTree with restrictions, but now another restriction is
    added: the patch closest to the prototype_vector is selected when it also overlaps a certain percentage with with
    the segmentation mask of the image. This means that the patch is not allowed to focus on background information.

    Note that as a prerequisite, the cur_project_info should include dict keys "overlap_percentages" and
    "is_clever_hans", which are added after running identify_clever_hans.py
    """
    log.log_message("\nProjecting prototypes to nearest training patch (with class restrictions)...")
    # Set the model to evaluation mode
    tree.eval()
    torch.cuda.empty_cache()
    # The goal is to find the latent patch that minimizes the L2 distance to each prototype
    # To do this we iterate through the train dataset and store for each prototype the closest latent patch seen so far
    # Also store info about the image that was used for projection
    global_min_proto_dist = {j: np.inf for j in range(tree.num_prototypes)}
    global_min_patches = {j: None for j in range(tree.num_prototypes)}
    global_min_info = {j: None for j in range(tree.num_prototypes)}

    # Get the shape of the prototypes
    W1, H1, D = tree.prototype_shape

    # Build a progress bar for showing the status
    projection_iter = tqdm(enumerate(project_loader),
                           total=len(project_loader),
                           desc=progress_prefix,
                           ncols=0
                           )

    with torch.no_grad():
        # Get all images for segmentation later
        imgs = project_loader.dataset.imgs
        # Get a batch of data
        xs, ys = next(iter(project_loader))
        batch_size = xs.shape[0]
        # For each internal node, collect the leaf labels in the subtree with this node as root.
        # Only images from these classes can be used for projection.
        leaf_labels_subtree = dict()

        for branch, j in tree._out_map.items():
            leaf_labels_subtree[branch.index] = set()
            for leaf in branch.leaves:
                leaf_labels_subtree[branch.index].add(torch.argmax(leaf.distribution()).item())

        for i, (xs, ys) in projection_iter:
            xs, ys = xs.to(device), ys.to(device)
            # Get the features and distances
            # - features_batch: features tensor (shared by all prototypes)
            #   shape: (batch_size, D, W, H)
            # - distances_batch: distances tensor (for all prototypes)
            #   shape: (batch_size, num_prototypes, W, H)
            # - out_map: a dict mapping decision nodes to distances (indices)
            features_batch, distances_batch, out_map = tree.forward_partial(xs)

            # Get the features dimensions
            bs, D, W, H = features_batch.shape
            assert W == H

            # Get a tensor containing the individual latent patches
            # Create the patches by unfolding over both the W and H dimensions
            # TODO -- support for strides in the prototype layer? (corresponds to step size here)
            patches_batch = features_batch.unfold(2, W1, 1).unfold(3, H1, 1)  # Shape: (batch_size, D, W, H, W1, H1)

            # Iterate over all decision nodes/prototypes
            for node, j in out_map.items():
                # Only find replacements for the nodes that are clever hans prototypes and not pruned
                if node not in tree.branches or not cur_project_info[j]["is_clever_hans"] \
                        or node.index not in which_nodes:
                    if i == 0:
                        global_min_info[j] = cur_project_info[j]
                        global_min_patches[j] = tree.prototype_layer.prototype_vectors[j].detach().clone()
                    continue

                log.log_message("\nFinding replacement for node " + str(node.index))

                leaf_labels = leaf_labels_subtree[node.index]
                # Iterate over all items in the batch
                # Select the features/distances that are relevant to this prototype
                # - distances: distances of the prototype to the latent patches
                #   shape: (W, H)
                # - patches: latent patches
                #   shape: (D, W, H, W1, H1)
                for batch_i, (distances, patches) in enumerate(zip(distances_batch[:, j, :, :], patches_batch)):
                    # Check if label of this image is in one of the leaves of the subtree
                    if ys[batch_i].item() in leaf_labels:
                        # Sort the patches based on distance to the prototype vector
                        min_distances, min_distance_ixs = torch.sort(torch.flatten(distances), descending=True)
                        # Discard the ones that are not closer than the global_min_proto_dist for this node
                        include_mask = min_distances < global_min_proto_dist[j]
                        min_distance_ixs = min_distance_ixs[include_mask]
                        min_distances = min_distances[include_mask]

                        # Check if at least one of the latent patches is closer than the global closest patch
                        if min_distance_ixs.size()[0] > 0:
                            # Get patch dimensions for the current image
                            x = Image.open(imgs[i * batch_size + batch_i][0])
                            x_np = np.asarray(x)
                            x_np = np.float32(x_np) / 255
                            if x_np.ndim == 2:  # convert grayscale to RGB
                                x_np = np.stack((x_np,) * 3, axis=-1)
                            img_size = x_np.shape[:2]
                            patches_ix = []
                            for p_ix in range(W * H):
                                masked_similarity_map = np.zeros((W, H))
                                masked_similarity_map[p_ix // W, p_ix % W] = 1
                                upsampled_prototype_pattern = cv2.resize(masked_similarity_map,
                                                                         dsize=(img_size[1], img_size[0]),
                                                                         interpolation=cv2.INTER_CUBIC)
                                patch_indices = find_high_activation_crop(upsampled_prototype_pattern,
                                                                          args.upsample_threshold)
                                patches_ix.append(patch_indices)

                            # Image segmentation
                            try:
                                segvalues, _ = segment_image.segmentAsPascalvoc(imgs[i * batch_size + batch_i][0],
                                                                                overlay=True)
                            except ValueError:
                                print("Something wrong with image segmentation probably... skipping image for now")
                                continue

                            # Calculate overlap of patch with segmentation mask for each patch
                            try:
                                patches_segvalues = [segvalues["masks"][patch_ix[0]:patch_ix[1], patch_ix[2]:patch_ix[3]]
                                                     for patch_ix in patches_ix]
                            except IndexError:
                                print("Something wrong with image segmentation probably... skipping image for now")
                                continue
                            overlap_percentages = [round(np.sum(patch_segvalue) / np.size(patch_segvalue), 3)
                                                   for patch_segvalue in patches_segvalues]

                            # Check for the remaining patches whether they have enough overlap with the object mask
                            for min_distance, min_distance_ix in zip(min_distances, min_distance_ixs):
                                if overlap_percentages[min_distance_ix] > threshold:
                                    closest_patch = patches.view(D, W * H, W1, H1)[:, min_distance_ix, :, :]
                                    global_min_proto_dist[j] = min_distance
                                    global_min_patches[j] = closest_patch
                                    global_min_info[j] = {
                                        'input_image_ix': i * batch_size + batch_i,
                                        'patch_ix': min_distance_ix.item(),
                                        # Index in a flattened array of the feature map
                                        'W': W,
                                        'H': H,
                                        'W1': W1,
                                        'H1': H1,
                                        'distance': min_distance.item(),
                                        'nearest_input': torch.unsqueeze(xs[batch_i], 0),
                                        'node_ix': node.index,
                                        'overlap_percentages': overlap_percentages,
                                        'is_clever_hans': False
                                    }

            # Update the progress bar if required
            projection_iter.set_postfix_str(f'Batch: {i + 1}/{len(project_loader)}')

            del features_batch
            del distances_batch
            del out_map

        # Copy the patches to the prototype layer weights
        projection = torch.cat(tuple(global_min_patches[j].unsqueeze(0) for j in range(tree.num_prototypes)),
                               dim=0, out=tree.prototype_layer.prototype_vectors)
        del projection

    return global_min_info, tree


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
