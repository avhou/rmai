import pandas as pd
import numpy as np
from typing import *

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

print(col_image)
print(col_label)
print(col_label_name)
print(col_ground_truth_bb)
print(col_best_prediction_bb)
print(col_IoU)


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

# ------------------------------------------------
# Example image: snow_storm-049.jpg
# Expected input format. To verify with Alexander.
# ------------------------------------------------
# Ground truth bounding boxes (with label at index 0)                
gt_bbs = {
    'snow_storm-049.jpg': [[0, 903, 248, 1274, 558], [0, 246, 405, 386, 508], [0, 537, 372, 744, 529], [1, 15, 311, 163, 491]],
    'snow_storm-050.jpg': [[0, 211, 436, 391, 587], [0, 399, 425, 639, 623], [0, 0, 537, 250, 729], [0, 901, 384, 994, 471], [0, 845, 307, 938, 390], [0, 753, 381, 912, 516], [0, 588, 371, 713, 481], [0, 935, 271, 1000, 348], [1, 989, 208, 1145, 420]]
}
# Predicted bounding boxes (with label at index 0)
pr_bbs = {
    'snow_storm-049.jpg': [[0, 881, 223, 1248, 512],[0, 554, 352, 733, 533],[0, 222, 384, 314, 484],[0, 44, 410, 155, 482],[1, 13, 320, 160, 485]],
    'snow_storm-050.jpg': [[0, 210, 436, 391, 587], [0, 205, 440, 385, 560], [0, 901, 384, 994, 471], [0, 845, 307, 938, 390], [0, 753, 381, 912, 516], [0, 588, 371, 713, 481], [0, 935, 271, 1000, 348]]
}


df = init_data_frame(gt_bbs)
compute_best_predicted_bbs(pr_bbs, df)

# The resulting dataframe df can now be used for further analysis

print("\n")
print(df)

