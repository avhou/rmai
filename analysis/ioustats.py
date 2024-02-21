import pandas as pd
import numpy as np
from scipy import stats
from typing import *


def paired_samples_ttest(data_A: pd.DataFrame, data_B: pd.DataFrame, label: str): 
    ious_A = data_A[data_A["label_name"] == label]["IoU"].to_numpy()
    ious_B = data_B[data_B["label_name"] == label]["IoU"].to_numpy()
    if len(ious_A) != len(ious_B):
        raise "Paired samples of unequal length!"
    return stats.ttest_rel(ious_A, ious_B)

def t_test(data_A, data_B, label: str, alternative='both-sided'):
    ious_A = data_A[data_A["label_name"] == label]["IoU"].to_numpy()
    ious_B = data_B[data_B["label_name"] == label]["IoU"].to_numpy()

    _, double_p = stats.ttest_rel(ious_A, ious_B)
    if alternative == 'both-sided':
        pval = double_p
    elif alternative == 'greater':
        if np.mean(ious_A) > np.mean(ious_B):
            pval = double_p/2.
        else:
            pval = 1.0 - double_p/2.
    elif alternative == 'less':
        if np.mean(ious_A) < np.mean(ious_B):
            pval = double_p/2.
        else:
            pval = 1.0 - double_p/2.
    return pval