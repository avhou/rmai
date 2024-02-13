import utils as ut
import mockpredictor as mp
import pandas as pd
import os.path
import sys

path_base_scenario   = './base-scenario.csv'
path_mock_scenario_1 = './mock_scenario1.csv'
path_mock_scenario_2 = './mock_scenario2.csv'

labels = ["car", "bus"]


def generate_mock_predictions(from_scratch = False):
    if not os.path.isfile(path_base_scenario):
        raise Exception('Missing base scenarion csv file')
    if from_scratch and os.path.exists(path_mock_scenario_1): 
        os.remove(path_mock_scenario_1)
    if not os.path.isfile(path_mock_scenario_1):
        mp.mock_predict("car", 6.35, "bus", 2.7, from_csv_file=path_base_scenario, to_csv_file=path_mock_scenario_1)
    if from_scratch and os.path.exists(path_mock_scenario_2):
        os.remove(path_mock_scenario_2)
    if not os.path.isfile(path_mock_scenario_2):
        mp.mock_predict("car", 3.0, "bus", 2.0, from_csv_file=path_base_scenario, to_csv_file=path_mock_scenario_2)

def compute_error_rates(csv_file: str) -> pd.DataFrame:
    prediction_data = pd.read_csv(csv_file, index_col=0)
    for label in labels:
        ut.compute_error_rates(prediction_data, label)
    return prediction_data

# Generate mock predictions
generate_mock_predictions(from_scratch = True)

# Compute error rates
base_predictions   = compute_error_rates(path_base_scenario)
scen_1_predictions = compute_error_rates(path_mock_scenario_1)
scen_2_predictions = compute_error_rates(path_mock_scenario_2)

# Visual check of normality
#ut.check_normality_by_hist(base_predictions, "car")
#ut.check_normality_by_hist(base_predictions, "bus")

# Visual check of spread and outliers
#ut.check_outliers_by_box(base_predictions, "car")
#ut.check_outliers_by_box(base_predictions, "bus")

# Paired sample t-test
ttest_car = ut.paired_samples_ttest(base_predictions, scen_1_predictions, "car")
ttest_bus = ut.paired_samples_ttest(base_predictions, scen_1_predictions, "bus")
print(f"Paired sample t-test for car")
print(f"t-statistic: {round(ttest_car.statistic, 3)}")
print(f"p-value: {round(ttest_car.pvalue, 3)}")
print(f"Paired sample t-test for bus")
print(f"t-statistic: {round(ttest_bus.statistic, 3)}")
print(f"p-value: {round(ttest_bus.pvalue, 3)}")

# McNemar test
mcnemar_car = ut.mc_nemar_test(base_predictions, scen_1_predictions, "car")
mcnemar_bus = ut.mc_nemar_test(base_predictions, scen_1_predictions, "bus")
print(f"Mc Nemar for car")
print(mcnemar_car)
print(f"Mc Nemar for bus")
print(mcnemar_bus)
    