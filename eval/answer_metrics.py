"""Separate binary decision bias from threshold-independent discrimination."""
import torch


def binary_answer_metrics(scores, labels, predictions):
    scores = scores.detach().double().cpu()
    labels = labels.detach().cpu()
    predictions = predictions.detach().cpu()
    positive = labels.eq(1)
    npos, nneg = positive.sum(), (~positive).sum()
    if not npos or not nneg:
        raise ValueError('ROC AUC needs both labels')
    _, inverse, counts = scores.unique(sorted=True, return_inverse=True, return_counts=True)
    midranks = counts.cumsum(0).double() - (counts.double() - 1) / 2
    ranks = midranks[inverse]
    auc = (ranks[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    recall_no = predictions[~positive].eq(0).double().mean()
    recall_yes = predictions[positive].eq(1).double().mean()
    return {'yes_no_roc_auc': auc.item(), 'recall_no': recall_no.item(), 'recall_yes': recall_yes.item(), 'balanced_accuracy': ((recall_no + recall_yes)/2).item()}
