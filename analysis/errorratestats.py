import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

def compute_error_rates(predictions: pd.DataFrame, label: str) -> None:
    """ compute error rates

    Computes an error rate per row for the given label. The computed value is added to the given dataframe
    in column <label>_error_rate. The computation of the error rate is similar to the bit error rate (BER). 

    :param preditions: a dataframe containing ground truths and predictions
    :param label: the label for which to compute error rates
    :return: given dataframe with added column for error rates
    """
    ground_truth = label + '_ground_truth'
    predicted = label + '_predicted'
    bers = label + '_error_rate'
    BERs = []
    for idx, row in predictions.iterrows():
        denom = row[ground_truth] if row[ground_truth] >= row[predicted] else row[predicted]
        error_rate = (row[ground_truth] - row[predicted]) / denom if denom > 0 else 0
        BERs.append(error_rate)
    predictions[bers] = BERs


def summarize_error_rates(predictions: pd.DataFrame, *labels: str) -> pd.DataFrame:
    """ summarize error rates
    
    Returns mean and standard deviation of the error rate, for each given label.

    :param predictions: a dataframe containing error rates
    :param labels: one or more labels
    :return: summary dict with labels and their error rate mean and standard deviation 
    """
    summary = {}
    for label in labels:
        mean = predictions[label + "_error_rate"].mean()
        std = predictions[label + "_error_rate"].std()
        summary[label] = [mean, std]
    return  pd.DataFrame.from_dict(summary, orient = 'index', columns = ['mean', 'std'])

def check_normality_by_hist(data: pd.DataFrame, label: str) -> None:
    error_rates = data[label + "_error_rate"]
    plt.hist(error_rates, edgecolor='black')
    plt.title(label + "_error_rate")
    plt.xlabel('error_rate')   
    plt.ylabel('count') 
    plt.show()

def check_outliers_by_box(data: pd.DataFrame, label: str) -> None:
    sns.boxplot(data, y=label + "_error_rate")
    plt.show()

def mean_error_rate(data: pd.DataFrame, label: str):
    error_rates = data[label + "_error_rate"].to_numpy()
    return error_rates.mean()

def paired_samples_ttest(data_A: pd.DataFrame, data_B: pd.DataFrame, label: str): 
    error_rates_A = data_A[label + "_error_rate"].to_numpy()
    error_rates_B = data_B[label + "_error_rate"].to_numpy()
    return stats.ttest_rel(error_rates_A, error_rates_B)

def mc_nemar_test(data_A: pd.DataFrame, data_B: pd.DataFrame, label: str):
    complete_detect_A = data_A[label + "_error_rate"].apply(lambda x: 1 if x == 0 else 0).sum()
    complete_detect_B = data_B[label + "_error_rate"].apply(lambda x: 1 if x != 0 else 0).sum()
    incomplete_detect_A = len(data_A) - complete_detect_A
    incomplete_detect_B = len(data_B) - complete_detect_B
    data = [[complete_detect_A, incomplete_detect_A],
            [complete_detect_B, incomplete_detect_B]]
    return mcnemar(data, exact = True)


def t_test(data_A, data_B, label: str, alternative='both-sided'):
    error_rates_A = data_A[label + "_error_rate"].to_numpy()
    error_rates_B = data_B[label + "_error_rate"].to_numpy()

    _, double_p = stats.ttest_rel(error_rates_A, error_rates_B)
    if alternative == 'both-sided':
        pval = double_p
    elif alternative == 'greater':
        if np.mean(error_rates_A) > np.mean(error_rates_B):
            pval = double_p/2.
        else:
            pval = 1.0 - double_p/2.
    elif alternative == 'less':
        if np.mean(error_rates_A) < np.mean(error_rates_B):
            pval = double_p/2.
        else:
            pval = 1.0 - double_p/2.
    return pval