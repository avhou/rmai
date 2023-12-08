import csv
import json
import os
from typing import *


def do_predict(model, validation_folder: str, result_file: str):
    """ do prediction of the images in validation_folder using the given model.  output is sent to a CSV in result_file

    :param validation_folder: the folder containing the validation dataset
    :param model: the yolov8 model to use
    :param result_file: the output CSV
    :return: nothing
    """

    label_map: Dict[int, str] = {
        0: "car",
        1: "bus"
    }
    validation_images = sorted([os.path.join(validation_folder, f) for f in os.listdir(validation_folder) if os.path.isfile(os.path.join(validation_folder, f)) and (f.lower().endswith("jpg") or f.lower().endswith("jpeg"))])
    annotation_files = [annotation_file(f) for f in validation_images]
    print(f"will process images : {validation_images}")
    print(f"will use annotation_files : {annotation_files}")
    with open(result_file, 'w') as csvfile:
        csvwriter =csv.writer(csvfile)
        csvwriter.writerow(["image", "car_ground_truth", "bus_ground_truth", "car_predicted", "bus_predicted", "car_percentage", "bus_percentage"])
        for image, annotation in zip(validation_images, annotation_files):
            results = model.predict(image, save=True)
            print(f"# results : {len(results)}")
            result = results[0]
            predicted_labels = Counter([pred["name"] for pred in json.loads(result.tojson()) if pred["name"] in label_map.values()])
            print(f"predicted labels {predicted_labels}")
            ground_truth_labels = actual_labels(annotation, label_map)
            print(f"ground truth labels {ground_truth_labels}")
            csvwriter.writerow([
                image,
                ground_truth_labels["car"],
                ground_truth_labels["bus"],
                predicted_labels["car"],
                predicted_labels["bus"],
                0 if ground_truth_labels["car"] == 0 else predicted_labels["car"] / ground_truth_labels["car"],
                0 if ground_truth_labels["bus"] == 0 else predicted_labels["bus"] / ground_truth_labels["bus"],
            ])


def annotation_file(image_file: str) -> str:
    return f"{image_file[:-4]}.txt"

def actual_labels(annotation_file: str, label_map: Dict[int, str]) -> Counter:
    with open(annotation_file, "r") as f:
        return Counter([label_map[int(line.split(" ")[0])] for line in f.readlines()])
