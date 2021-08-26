import random
import logging
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from seqtag import config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_logging(level=logging.DEBUG):
    log = logging.getLogger(__name__)
    log.parent.disabled = True
    log.propagate = False
    if log.handlers:
        return log
    log.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log


def count_labels(labels, id2label):
    """
    :param labels: tensor of (b, seq_len)
    :return: tensor of L labels with weights for loss
    """
    weights = torch.ones(len(id2label))
    for li in labels.view(-1):
        weights[li] += 1

    weight_sum = weights.sum() - (weights[config.PAD_LABEL_ID] + weights[config.SUBTOKEN_LABEL_ID])

    weights = weight_sum / weights
    # zeros weights for "artificial" labels
    weights[config.PAD_LABEL_ID] = 0
    weights[config.SUBTOKEN_LABEL_ID] = 0
    return weights


def write_results_to_file(results_file, results_data):
    results_data["timestamp"] = [datetime.now().strftime("%Y%m%d%H%M%S")]
    file = Path(results_file)
    pd.DataFrame.from_dict(results_data).to_csv(file, mode="a", header=not file.exists())
