import json
import sys
from ultralytics import YOLO
import os
from pathlib import Path
from typing import *
from json import *
import csv


def do_base_scenario(dawn_folder: str):
    """ base scenario

    Use the base yolo model and validate against the dawn validation set

    :param dawn_folder: the folder containing the dawn validation dataset
    :return: nothing
    """

    label_map: Dict[int, str] = {
        0: "car",
        1: "bus"
    }
    model = YOLO("yolov8n.pt")
    validation_images = sorted([os.path.join(dawn_folder, f) for f in os.listdir(dawn_folder) if os.path.isfile(os.path.join(dawn_folder, f)) and (f.lower().endswith("jpg") or f.lower().endswith("jpeg"))])
    annotation_files = [annotation_file(f) for f in validation_images]
    print(f"will process images : {validation_images}")
    print(f"will use annotation_files : {annotation_files}")
    with open('base-scenario.csv', 'w') as csvfile:
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
            # for result in results:
                # this will save an annotation file with the detected matches
                # result.save_txt(os.path.join(Path(dawn_folder).parent, f"{os.path.basename(result.path)}-results.txt"))
                # print(f"names : {result.names}")
                # print(f"path : {result.path}")
                # print(f"tojson() : {result.tojson()}")
                # print(f"boxes : {result.boxes}")
                # print(f"probs : {result.probs}")


    # print("training scenario1")
    # # the yaml describes the config of the yolo model (which backbone to use, ...), the pt are the pretrained weights
    # model = YOLO("yolov8.yaml").load("yolov8n.pt")
    # # the data yaml describes training and validation datasets
    # results = model.train(data="data/DAWN/dawn-train-dataset.yml", epochs=100, imgsz=640, device="mps", verbose=True, plots=True)
    # print(results)

    # print("do validation")
    # # the yaml describes the config of the yolo model (which backbone to use, ...), the pt are the pretrained weights
    # # strange : the x models results are way worse than the n models results
    # model = YOLO("yolov8.yaml").load("yolov8n.pt")
    # # the data yaml describes training and validation datasets
    # metrics = model.val(data="data/DAWN/dawn-train-dataset.yml", device="mps", save_json=True, plots=True)
    # print(metrics.box.map)  # map50-95
    # print(metrics.box.map50)  # map50
    # print(metrics.box.map75)  # map75
    # print(metrics.box.maps)  # a list contains map50-95 of each category

def annotation_file(image_file: str) -> str:
    return f"{image_file[:-4]}.txt"

def actual_labels(annotation_file: str, label_map: Dict[int, str]) -> Counter:
    with open(annotation_file, "r") as f:
        return Counter([label_map[int(line.split(" ")[0])] for line in f.readlines()])

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        raise Exception("please specify dawn validation folder")

    do_base_scenario(sys.argv[1])
