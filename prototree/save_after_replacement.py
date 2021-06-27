import os
import argparse
import torch
import pickle

from util.data import get_dataloaders
from util.log import Log
from util.args import save_args

from prototree.upsample_after_replacement import upsample
from util.visualize import gen_vis
from prototree.prototree import ProtoTree


def save_after_replacement(tree: ProtoTree, project_info: dict, args: argparse.Namespace, which_nodes: list,
                           folder_name: str = "./replacements/prototree_replacement_0"):
    """
    Given a ProtoTree and its project_info (the latter containing details of each prototype), replace
    multiple prototypes. Note that the `tree` object is used as a reference here, i.e. the tree returned by this
    function is the same object as the input object.
    """
    # Do not overwrite previous replacement folders
    if not os.path.exists("./replacements"):
        os.mkdir("replacements")
    repl_folder_id = 0
    while os.path.exists(folder_name):
        folder_name = folder_name[:-len(str(repl_folder_id))]
        repl_folder_id += 1
        folder_name += str(repl_folder_id)

    # Upsample the prototree again for visualizing the prototypes
    args.log_dir = folder_name
    args.dir_for_saving_images = "upsampling_results"
    log = Log(args.log_dir)
    _, projectloader, _, classes, _ = get_dataloaders(args)
    upsample(tree, project_info, projectloader, folder_name="visualization", which_nodes=which_nodes, args=args, log=log)

    # TODO Visualize the tree with replacements (issues with Graphviz were inhibiting this from being useful)
    # gen_vis(tree, folder_name="visualization", args=args, classes=classes)

    # Save the new prototree for later reference
    save_args(args, folder_name + "/metadata")
    tree.save(folder_name + "/checkpoints")
    tree.save_state(folder_name + "/checkpoints")

    # Save the project_info
    with open(folder_name+'/project_info.pickle', 'wb') as handle:
        pickle.dump(project_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
