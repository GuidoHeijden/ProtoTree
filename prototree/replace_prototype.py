import torch

from prototree.prototree import ProtoTree


def replace_prototype(tree: ProtoTree, project_info: dict, node_index: int, replacement_patch_index: int):
    """
    Given a ProtoTree and its project_info (the latter containing details of each prototype), a prototype
    can be replaced with another prototype from the same image. Note that the `tree` object is used as a reference here,
    i.e. the tree returned by this function is the same object as the input object.
    """
    # Retrieve the latent image for the prototype of the node at node_index
    prototype_info = project_info[tree._out_map[tree.nodes_by_index[node_index]]]
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


def replace_prototypes(tree: ProtoTree, project_info: dict, node_indices: list, replacement_patch_indices: list):
    """
    Given a ProtoTree and its project_info (the latter containing details of each prototype), replace
    multiple prototypes. Note that the `tree` object is used as a reference here, i.e. the tree returned by this
    function is the same object as the input object.
    """
    for node_index, new_prototype_index in zip(node_indices, replacement_patch_indices):
        # Retrieve the latent image for the prototype of the node at node_index
        prototype_info = project_info[tree._out_map[tree.nodes_by_index[node_index]]]
        W, H = prototype_info['W'], prototype_info['H']
        xs = prototype_info['nearest_input']
        latent_img_batch, _, _ = tree.forward_partial(xs)
        latent_img = latent_img_batch[0]

        # Replace the necessary info regarding the prototype: prototype_vector, patch_ix
        new_prototype_vector = torch.unsqueeze(torch.unsqueeze(
                latent_img[:, new_prototype_index // H, new_prototype_index % W],
                1), 1)
        tree.prototype_layer.prototype_vectors[tree._out_map[tree.nodes_by_index[node_index]]] = new_prototype_vector
        project_info[tree._out_map[tree.nodes_by_index[node_index]]]['patch_ix'] = new_prototype_index

    return tree, project_info
