# Repairing Undesired Biases in ProtoTrees
This repository contains the code submission for the Research Experiments in Databases and Information Retrieval (REDI) course given at the University of Twente. This code extends upon the PyTorch code for Neural Prototype Trees (ProtoTrees) found in this repository: https://github.com/M-Nauta/ProtoTree. To learn more about ProtoTree, it is recommended reading the README of that repository and the research paper that it contains.

The aim of the REDI project was to repair undesired biases that were visualized through the ProtoTree's interpretable nature. The specific undesired bias considered is Clever Hans behaviour: the behaviour where a correct solution is found but through undesired reasons. An example of Clever Hans behaviour specific to this project is the prediction of a bird based on its surroundings rather than basing it on the bird itself. For this project, two research questions were formulated:
- **RQ1**: To what extent is it possible to (automatically) detect Clever Hans behaviour using ProtoTree as interpretable machine learning method?
- **RQ2**: To what degree does the correction of Clever Hans behaviour affect the prediction accuracy of a model?

To answer these research questions, some code was added/modified to this fork of the ProtoTree repository. The changes are summarized below:
- `prototree/find_high_similarity_replacement.py` contains two functions: a function for replacing a prototype with one within the same image; and a function for replacing multiple prototypes, each of which is replaced with a prototype from the restriction images. The latter function is very similar to `prototree/project.py`
- `prototree/identify_clever_hans.py` identifies Clever Hans prototypes based on the overlap of the prototype with the image mask.
- `prototree/measure_prototype_distance.py` measure the Eucledian distance for a list of prototypes between one ProtoTree and another.
- `prototree/replace_prototype.py` replace a prototype with a patch within the same image that can be entered as a function parameter; meant for manual replacement.

There were also some minor (technical) changes/additions:
- `prototree/upsample.py` was adapted to upsample images including an object mask made via image segmentation.
- `preprocess_data/download_birds.py` had a small bug fix.
- `prototree/evaluate_train.py` contains a function for evaluating a ProtoTree on the training set.
- `prototree/save_after_replacement.py` calls some functions to save a ProtoTree and is meant to be called on a ProtoTree that has just undergone replacements.
- `prototree/upsample_after_replacement.py` same as `upsample.py` but allows for upsampling a predefined set of nodes, which speeds up the upsampling process. Especially relevant to upsample ProtoTrees after replacement, since the prototypes that were not replaced do not need to be upsampled again.
