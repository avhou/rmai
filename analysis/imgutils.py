import pandas as pd
import numpy as np
from typing import *

label_map: Dict[str, int] = {
    "car": 0,
    "bus": 1
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
    label, x1, y1, x2, y2 = bb_list
    return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}

def compute_ious(ground_truths: List, predictions: List, label: str, threshold: float) -> Dict:
    # Filter by label
    gt_for_label = [i for i in ground_truths if i[0] == label_map[label]]
    pr_for_label = [i for i in predictions   if i[0] == label_map[label]]

    # Convert input to dict
    gt_dict = [to_dict(i) for i in gt_for_label]
    pr_dict = [to_dict(i) for i in pr_for_label]

    # Compute IoU's
    ious = []
    for ground in gt_dict:
        for prediction in pr_dict:
            ious.append(compute_iou(ground, prediction))
    
    # Filter out zero IoU's and split by threshold
    ious = [i for i in ious if i > 0]
    ious_infra = [i for i in ious if i < threshold]
    ious_supra = [i for i in ious if i >= threshold]
    
    return {
        'ious_infra': ious_infra,
        'ious_supra': ious_supra 
        }


# Example image: snow_storm-049.jpg
# Expected input format
data_gt = [[0, 903, 248, 1274, 558],[0, 246, 405, 386, 508],[0, 537, 372, 744, 529],[1, 15, 311, 163, 491]]
data_pr = [[0, 881, 223, 1248, 512],[0, 554, 352, 733, 533],[0, 222, 384, 314, 484],[0, 44, 410, 155, 482],[1, 13, 320, 160, 485]]


ious_car = compute_ious(data_gt, data_pr, "car", 0.5)
ious_bus = compute_ious(data_gt, data_pr, "bus", 0.5)

print("IoUs for label 'car'")
print(ious_car)

print("IoU for label 'bus'")
print(ious_bus)