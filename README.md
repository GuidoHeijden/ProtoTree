# Repairing Undesired Biases in ProtoTrees
This repository contains the code submission for the Research Experiments in Databases and Information Retrieval (REDI) course given at the University of Twente. This code extends upon the PyTorch code for Neural Prototype Trees (ProtoTrees) found in this repository: https://github.com/M-Nauta/ProtoTree. To learn more about ProtoTree, it is recommended reading the README of that repository and the research paper that it contains.

The aim of the REDI project was to repair undesired biases that were visualized through the ProtoTree's interpretable nature. The specific undesired bias considered is Clever Hans behaviour: the behaviour where a correct solution is found but through undesired reasons. An example of Clever Hans behaviour specific to this project is the prediction of a bird based on its surroundings rather than basing it on the bird itself. For this project, two research questions were formulated:
- **RQ1**: To what extent is it possible to (automatically) detect Clever Hans behaviour using ProtoTree as interpretable machine learning method?
- **RQ2**: To what degree does the correction of Clever Hans behaviour affect the prediction accuracy of a model?

To answer these research questions, some code was added to this fork of the ProtoTree repository. The changes are summarized below:
- ....
