import pandas as pd
import numpy as np
import errorratestats as ut

def predict(seed: pd.DataFrame, label: str, factor: float) -> pd.DataFrame:
    ground_truth_col = label + '_ground_truth'
    predicted_col    = label + '_predicted'
    error_rate_col   = label + '_error_rate'
    if ~seed.columns.isin([ground_truth_col]).any():
        raise Exception('Missing columns in data frame')
    if ~seed.columns.isin([error_rate_col]).any():
        ut.compute_error_rates(seed, label)
    mock_from_seed = seed.loc[:, ~seed.columns.isin([predicted_col, error_rate_col])].copy()
    mock_predictions = []
    scale = seed[error_rate_col].std() * factor
    for idx, data in seed.iterrows():
        loc = data[ground_truth_col]
        pred = abs(round(np.random.normal(loc, scale)))
        mock_predictions.append(pred)
    mock_from_seed[predicted_col] = mock_predictions
    return  mock_from_seed

def chunkwise(t, size=2):
    it = iter(t)
    return zip(*[it]*size)

def mock_predict(*args, from_csv_file: str, to_csv_file: str) -> None:
    usecols = ["image"]
    for label, factor in chunkwise(args):
        usecols.append(label + "_ground_truth")
        usecols.append(label + "_predicted")
    seed = pd.read_csv(from_csv_file, usecols=usecols)
    for label, factor in chunkwise(args):
        seed = predict(seed, label, factor)
    seed.to_csv(to_csv_file)
