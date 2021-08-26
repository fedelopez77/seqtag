import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from seqtag.config import PAD_LABEL_ID, SUBTOKEN_LABEL_ID


def filter_artificial_labels(logits, labels):
    prediction = torch.argmax(logits, dim=-1).tolist()
    labels = labels.reshape(-1).tolist()
    assert len(prediction) == len(labels)

    final_pred, final_labels = [], []
    for i in range(len(prediction)):
        cur_label = labels[i]
        if cur_label != PAD_LABEL_ID and cur_label != SUBTOKEN_LABEL_ID:
            final_labels.append(labels[i])
            final_pred.append(prediction[i])

    return np.array(final_pred), np.array(final_labels)


def compute_metrics(logits, labels):
    """
    :param logits: tensor of (n, num_labels)
    :param labels: tensor of (n, 1)
    :return: dict with micro and macro precision, recall and f1
    """
    final_pred, final_labels = filter_artificial_labels(logits, labels)

    results = {}
    for avg in ["macro", "micro"]:
        results[avg] = {}
        for metric, fn in zip(["p", "r", "f1"],
                              [precision_score, recall_score, f1_score]):
            results[avg][metric] = fn(final_labels, final_pred, average=avg, zero_division=0)

    return results
