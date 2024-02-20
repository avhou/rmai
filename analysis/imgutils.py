import pandas as pd
import numpy as np
import json
from typing import *
from pathlib import Path

label_name_to_code: Dict[str, int] = {
    "car": 0,
    "bus": 1
}

label_code_to_name: Dict[str, int] = {
    0: "car",
    1: "bus"
}

def compute_iou(bb1, bb2):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.

    :param bb1 : a dictionary with keys {'x1', 'x2', 'y1', 'y2'}.
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    :param bb2 : a dictionary with keys {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    :return: float in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def to_dict(bb_list: List) -> Dict:
    x1, y1, x2, y2 = bb_list
    return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}

col_image, col_label, col_label_name, col_ground_truth_bb, col_best_prediction_bb, col_IoU = 'image', 'label', 'label_name', 'ground_truth_bb', 'best_prediction_bb', 'IoU'

def init_data_frame(gt_data: Dict) -> pd.DataFrame:
    df = pd.DataFrame(columns = [col_image, col_label, col_label_name, col_ground_truth_bb, col_best_prediction_bb, col_IoU])
    for k, v in gt_data.items():
        for pr_bb in v:
            idx = len(df.index)
            df.loc[idx] = k
            df.at[idx, col_label] = pr_bb[0]
            df.at[idx, col_label_name] = label_code_to_name[pr_bb[0]]
            df.at[idx, col_ground_truth_bb] = pr_bb[1:]
            df.at[idx, col_best_prediction_bb] = None
            df.at[idx, col_IoU] = 0.0
    return df

def compute_best_predicted_bbs(pr_bbs: Dict, data_frame: pd.DataFrame) -> None:
    for label in label_code_to_name.keys():
        for k, v in pr_bbs.items():
            label_match = data_frame.loc[ (data_frame[col_image] == k) & (data_frame[col_label] == label)]
            for idx, row in label_match.iterrows():
                best_pr_bb, max_iou = None, 0.0
                gt_dict = to_dict(row[col_ground_truth_bb])
                for pr_bb in v:
                    if pr_bb[0] == label:
                        pr_bb = pr_bb[1:]
                        pr_dict = to_dict(pr_bb)
                        iou = compute_iou(gt_dict, pr_dict)
                        if iou > max_iou:
                            best_pr_bb = pr_bb
                            max_iou = iou
                data_frame.at[idx, col_best_prediction_bb] = best_pr_bb
                data_frame.at[idx, col_IoU] = max_iou

def compute_IoUs(json_file: str) -> Path:
    gt_dict = read_data(json_file, "ground_truth")
    pr_dict = read_data(json_file, "predicted")
    df = init_data_frame(gt_dict)
    compute_best_predicted_bbs(pr_dict, df)
    out_file = Path(json_file).stem + "-IoU.csv"
    df.to_csv(out_file, sep='\t', index=False, encoding='utf-8')
    return Path(out_file)

def read_data(file_name: str, qualifier: str) -> dict:
    """
    Read input from file and converts to expected format.

    :param file_name : name of the input file
    :param qualifier : one of "predicted" or "ground_truth"
    :return : dictionary with input data in expected format 
    """
    data_dict = {}
    with open(file_name, 'r') as file:
        data = json.load(file)
        for file_name, bbs in data.items():
            ground_truth_bbs = bbs[qualifier]
            bb_list = []
            for coords in ground_truth_bbs:
                if coords["name"] in label_name_to_code:
                    bb_list.append([
                        label_name_to_code[coords["name"]], 
                        coords["top_left_x"],
                        coords["top_left_y"],
                        coords["bottom_right_x"],
                        coords["bottom_right_y"]
                        ])                    
            data_dict[file_name] = bb_list            
    return data_dict    

# ---------------------------------
# Example of expected input format. 
# ---------------------------------
# Ground truth bounding boxes (with label at index 0)                
# gt_bbs = {
#     'snow_storm-049.jpg': [[0, 903, 248, 1274, 558], [0, 246, 405, 386, 508], [0, 537, 372, 744, 529], [1, 15, 311, 163, 491]],
#     'snow_storm-050.jpg': [[0, 211, 436, 391, 587], [0, 399, 425, 639, 623], [0, 0, 537, 250, 729], [0, 901, 384, 994, 471], [0, 845, 307, 938, 390], [0, 753, 381, 912, 516], [0, 588, 371, 713, 481], [0, 935, 271, 1000, 348], [1, 989, 208, 1145, 420]]
# }
# Predicted bounding boxes (with label at index 0)
# pr_bbs = {
#     'snow_storm-049.jpg': [[0, 881, 223, 1248, 512],[0, 554, 352, 733, 533],[0, 222, 384, 314, 484],[0, 44, 410, 155, 482],[1, 13, 320, 160, 485]],
#     'snow_storm-050.jpg': [[0, 210, 436, 391, 587], [0, 205, 440, 385, 560], [0, 901, 384, 994, 471], [0, 845, 307, 938, 390], [0, 753, 381, 912, 516], [0, 588, 371, 713, 481], [0, 935, 271, 1000, 348]]
# }
#
# -------------------------------
# Example of actual input format.
# -------------------------------
# {"image0001.jpg": 
#   {"ground_truth": [
#       {"top_left_x": 723, "top_left_y": 367, "bottom_right_x": 838, "bottom_right_y": 489, "name": "bus"}, 
#       {"top_left_x": 598, "top_left_y": 417, "bottom_right_x": 660, "bottom_right_y": 465, "name": "car"}, 
#       {"top_left_x": 871, "top_left_y": 425, "bottom_right_x": 935, "bottom_right_y": 467, "name": "car"}, 
#       {"top_left_x": 471, "top_left_y": 425, "bottom_right_x": 536, "bottom_right_y": 460, "name": "car"}, 
#       {"top_left_x": 1095, "top_left_y": 378, "bottom_right_x": 1279, "bottom_right_y": 546, "name": "car"}, 
#       {"top_left_x": 664, "top_left_y": 419, "bottom_right_x": 716, "bottom_right_y": 452, "name": "car"}, 
#       {"top_left_x": 140, "top_left_y": 413, "bottom_right_x": 270, "bottom_right_y": 453, "name": "car"}, 
#       {"top_left_x": 231, "top_left_y": 411, "bottom_right_x": 304, "bottom_right_y": 448, "name": "car"}, 
#       {"top_left_x": 532, "top_left_y": 411, "bottom_right_x": 581, "bottom_right_y": 454, "name": "car"}
#   ], 
#   "predicted": [
#       {"top_left_x": 0, "top_left_y": 354, "bottom_right_x": 131, "bottom_right_y": 694, "name": "person"},
#       {"top_left_x": 1103, "top_left_y": 380, "bottom_right_x": 1279, "bottom_right_y": 547, "name": "car"}, 
#       {"top_left_x": 597, "top_left_y": 419, "bottom_right_x": 662, "bottom_right_y": 466, "name": "car"}, 
#       {"top_left_x": 23, "top_left_y": 395, "bottom_right_x": 100, "bottom_right_y": 502, "name": "backpack"}, 
#       {"top_left_x": 471, "top_left_y": 411, "bottom_right_x": 538, "bottom_right_y": 463, "name": "car"}, 
#       {"top_left_x": 871, "top_left_y": 426, "bottom_right_x": 934, "bottom_right_y": 470, "name": "car"}, 
#       {"top_left_x": 723, "top_left_y": 368, "bottom_right_x": 843, "bottom_right_y": 489, "name": "truck"}, 
#       {"top_left_x": 1036, "top_left_y": 399, "bottom_right_x": 1055, "bottom_right_y": 446, "name": "person"}, 
#       {"top_left_x": 532, "top_left_y": 416, "bottom_right_x": 581, "bottom_right_y": 452, "name": "car"}, 
#       {"top_left_x": 663, "top_left_y": 416, "bottom_right_x": 718, "bottom_right_y": 452, "name": "car"}, 
#       {"top_left_x": 1017, "top_left_y": 397, "bottom_right_x": 1035, "bottom_right_y": 447, "name": "person"}, 
#       {"top_left_x": 142, "top_left_y": 413, "bottom_right_x": 237, "bottom_right_y": 452, "name": "car"}, 
#       {"top_left_x": 258, "top_left_y": 420, "bottom_right_x": 312, "bottom_right_y": 449, "name": "car"}, 
#       {"top_left_x": 113, "top_left_y": 534, "bottom_right_x": 146, "bottom_right_y": 608, "name": "handbag"}
#   ]
# }

#compute_IoUs("base-scenario.json")

# count = 0
# for idx, row in df.iterrows():
#     if row["label_name"] == "bus" and row["IoU"] > 0:
#         count += 1 

# print(f"Bus: {count}") 

# -----------------
# Handle scenario 1
# -----------------
# gt_dict = read_data("scenario-rq1.json", "ground_truth")
# pr_dict = read_data("scenario-rq1.json", "predicted")

# df = init_data_frame(gt_dict)
# compute_best_predicted_bbs(pr_dict, df)
# df.to_csv("scenario-rq1-IoU.csv", sep='\t', index=False, encoding='utf-8')

# count = 0
# for idx, row in df.iterrows():
#     if row["label_name"] == "bus" and row["IoU"] > 0:
#         count += 1 

# print(f"Bus: {count}")        

# -----------------
# Handle scenario 2
# -----------------
# gt_dict = read_data("scenario-rq2.json", "ground_truth")
# pr_dict = read_data("scenario-rq2.json", "predicted")

# df = init_data_frame(gt_dict)
# compute_best_predicted_bbs(pr_dict, df)
# df.to_csv("scenario-rq2-IoU.csv", sep='\t', index=False, encoding='utf-8')

# count = 0
# for idx, row in df.iterrows():
#     if row["label_name"] == "car" and row["IoU"] > 0:
#         count += 1        

# print(f"Cars: {count}")

