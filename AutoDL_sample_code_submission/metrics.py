import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


def auc_metric(solution, prediction):
    if solution.sum(axis=0).min() == 0:
        return -1
    # single-class case
    if len(solution.shape) == 1 and prediction.shape[1] == 2:
        prediction = prediction[:, 1].reshape(-1)
    # multi-class case
    if not solution.shape == prediction.shape:
        solution = np.eye(prediction.shape[1])[solution]
    auc = roc_auc_score(solution, prediction, average='macro')
    return np.mean(auc * 2 - 1)


def acc_metric(solution, prediction):
    if solution.sum(axis=0).min() == 0:
        return np.nan
    acc = accuracy_score(solution, prediction)
    return acc