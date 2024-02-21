import errorratestats as ut
import plotutils as pu
import reportutils as ru
import imgutils as iu
import mockpredictor as mp
import pandas as pd
import os.path

path_base_scenario   = './base-scenario.csv'
path_scenario_1 = './scenario-rq1.csv'
path_scenario_2 = './scenario-rq2.csv'

labels = ["car", "bus"]


def generate_mock_predictions(from_scratch = False):
    if not os.path.isfile(path_base_scenario):
        raise Exception('Missing base scenarion csv file')
    if from_scratch and os.path.exists(path_scenario_1): 
        os.remove(path_scenario_1)
    if not os.path.isfile(path_scenario_1):
        mp.mock_predict("car", 6.35, "bus", 2.7, from_csv_file=path_base_scenario, to_csv_file=path_scenario_1)
    if from_scratch and os.path.exists(path_scenario_2):
        os.remove(path_scenario_2)
    if not os.path.isfile(path_scenario_2):
        mp.mock_predict("car", 3.0, "bus", 2.0, from_csv_file=path_base_scenario, to_csv_file=path_scenario_2)

def compute_error_rates(csv_file: str) -> pd.DataFrame:
    prediction_data = pd.read_csv(csv_file, index_col=0)
    for label in labels:
        ut.compute_error_rates(prediction_data, label)
    return prediction_data

def count_objects_predicted(scenario: str, qualifier: str, csv_file: str) -> pd.DataFrame:
    prediction_data = pd.read_csv(csv_file, index_col=0)
    list = []
    for label in labels:
        sum = prediction_data[label + "_" + qualifier].sum()
        list.append({"scenario": scenario, "label": label, "qualifier": qualifier.replace('_', ' '), "count": sum})
    return pd.DataFrame(list, columns = ["scenario", "label", "qualifier", "count"])    

def count_iou_based(scenario:str, qualifier: str, csv_file: str, threshold: float) -> pd.DataFrame:
    bounding_box_data = pd.read_csv(csv_file, sep='\t', index_col=0)
    list = []
    for label in labels:
        count = 0
        for idx, row in bounding_box_data.iterrows():
            if row["label_name"] == label and row["IoU"] > threshold:
                count += 1
        list.append({"scenario": scenario, "label": label, "qualifier": qualifier.replace('_', ' '), "count": count}) 
    return pd.DataFrame(list, columns = ["scenario", "label", "qualifier", "count"])

# ----------------------------------------
# Generate mock predictions
# (Only relevant while awaiting real data)
# ----------------------------------------
#generate_mock_predictions(from_scratch = True)

# -------------------
# Compute error rates
# -------------------
base_predictions   = compute_error_rates(path_base_scenario)
scen_1_predictions = compute_error_rates(path_scenario_1)
scen_2_predictions = compute_error_rates(path_scenario_2)

# ------------------------------------
# Compute unverified counts (from csv)
# ------------------------------------
# df_gt = count_objects_predicted("Ground truth", "ground_truth", path_base_scenario)
# df_base = count_objects_predicted("Baseline model", "predicted", path_base_scenario)
# df_sc1 = count_objects_predicted("DAWN-based model (actual images)", "predicted", path_scenario_1)
# df_sc2 = count_objects_predicted("DETRAC-based model (augmented images)", "predicted", path_scenario_2)
# df_counts = pd.concat([df_sc2, df_sc1, df_base, df_gt])
#print(df_counts)

# --------
# Graphing
# --------        
# Plot object detection counts for each label, per scenario (model) 
#pu.plot_horizontal_bar(df_counts)

# Visual check of normality
#ut.check_normality_by_hist(base_predictions, "car")
#ut.check_normality_by_hist(base_predictions, "bus")

# Visual check of spread and outliers
#ut.check_outliers_by_box(base_predictions, "car")
#ut.check_outliers_by_box(base_predictions, "bus")

# ----------------------------------------------------
# Statistics reporting of unverified counts (from csv)
# ----------------------------------------------------
ru.compare_error_rates("Base", "DAWN", {"Base": base_predictions, "DAWN": scen_1_predictions})
ru.compare_error_rates("Base", "DETRAC", {"Base": base_predictions, "DETRAC": scen_2_predictions})
ru.compare_error_rates("DETRAC", "DAWN", {"DAWN": scen_1_predictions, "DETRAC": scen_2_predictions})


# -----------------------------------------
# Detailed image based analysis (from json) 
# -----------------------------------------
# iou_threshold = 0.9
# out_file = iu.compute_IoUs( "base-scenario.json")
# count_base = count_iou_based("Baseline model", "predicted", out_file, iou_threshold)

# out_file = iu.compute_IoUs( "scenario-rq1.json")
# count_scen1 = count_iou_based("DAWN-based model (actual images)", "predicted", out_file, iou_threshold)

# out_file = iu.compute_IoUs("scenario-rq2.json")
# count_scen2 = count_iou_based("DETRAC-based model (augmented images)", "predicted", out_file, iou_threshold)

# df_gt = count_objects_predicted("Ground truth", "ground_truth", path_base_scenario)
# df_counts = pd.concat([count_scen2, count_scen1, count_base, df_gt])

# --------
# Graphing
# -------- 
# Plot object detection counts for each label, per scenario (model) 
#pu.plot_horizontal_bar(df_counts, f"Objects detected at IoU > {iou_threshold}")

# -----------------------------
# Statistics reporting of IoU's
# -----------------------------
base_ious  = pd.read_csv("base-scenario-IoU.csv", sep='\t', index_col=0)
scen1_ious = pd.read_csv("scenario-rq1-IoU.csv", sep="\t", index_col=0)
scen2_ious = pd.read_csv("scenario-rq2-IoU.csv", sep="\t", index_col=0)

ru.compare_ious("Base", "DAWN", {"Base": base_ious, "DAWN": scen1_ious})
ru.compare_ious("Base", "DETRAC", {"Base": base_ious, "DETRAC": scen2_ious})
ru.compare_ious("DETRAC", "DAWN", {"DAWN": scen1_ious, "DETRAC": scen2_ious})