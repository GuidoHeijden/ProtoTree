import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.optim
from torch.utils.data import DataLoader

from prototree.prototree import ProtoTree
from util.log import Log


@torch.no_grad()
def eval_train(tree: ProtoTree,
               train_loader: DataLoader,
               num_batches,
               device,
               log: Log = None,
               sampling_strategy: str = 'distributed',
               progress_prefix: str = 'Training set evaluation'
               ) -> dict:
    tree = tree.to(device)

    # Keep an info dict about the procedure
    info = dict()
    if sampling_strategy != 'distributed':
        info['out_leaf_ix'] = []
    # Build a confusion matrix
    cm = np.zeros((tree._num_classes, tree._num_classes), dtype=int)

    # Make sure the model is in evaluation mode
    tree.eval()

    # Show progress on progress bar
    train_iter = tqdm(enumerate(train_loader),
                      total=min([len(train_loader),num_batches]),
                      desc=progress_prefix,
                      ncols=0)

    # Iterate through the train set
    for i, (xs, ys) in train_iter:
        if i == num_batches:
            break
        xs, ys = xs.to(device), ys.to(device)

        # Use the model to classify this batch of input data
        out, train_info = tree.forward(xs, sampling_strategy)
        ys_pred = torch.argmax(out, dim=1)

        # Update the confusion matrix
        cm_batch = np.zeros((tree._num_classes, tree._num_classes), dtype=int)
        for y_pred, y_true in zip(ys_pred, ys):
            cm[y_true][y_pred] += 1
            cm_batch[y_true][y_pred] += 1
        acc = acc_from_cm(cm_batch)
        train_iter.set_postfix_str(
            f'Batch [{i + 1}/{len(train_iter)}], Acc: {acc:.3f}'
        )

        # keep list of leaf indices where train sample ends up when deterministic routing is used.
        if sampling_strategy != 'distributed':
            info['out_leaf_ix'] += train_info['out_leaf_ix']
        del out
        del ys_pred
        del train_info

    info['confusion_matrix'] = cm
    info['train_accuracy'] = acc_from_cm(cm)
    log.log_message(
        "\nTrain accuracy with %s routing: " % sampling_strategy + str(info['train_accuracy']))
    return info


def acc_from_cm(cm: np.ndarray) -> float:
    """
    Compute the accuracy from the confusion matrix
    :param cm: confusion matrix
    :return: the accuracy score
    """
    assert len(cm.shape) == 2 and cm.shape[0] == cm.shape[1]

    correct = 0
    for i in range(len(cm)):
        correct += cm[i, i]

    total = np.sum(cm)
    if total == 0:
        return 1
    else:
        return correct / total
