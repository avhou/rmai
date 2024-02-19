import csv
import json
import os
from typing import *
import imageio as iio


class Annotation:
    """ Keeps track of information about annotations """
    name: str
    center_x: float
    center_y: float
    width: float
    height: float

    def __init__(self, name:str, center_x: float, center_y: float, width: float, height: float):
        self.name = name
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height

    def from_line(line: str, label_map: Dict[int, str]):
        values = line.split()
        return Annotation(label_map.get(int(values[0]), "Ongekend!"), float(values[1]), float(values[2]), float(values[3]), float(values[4]))

    def __repr__(self) -> str:
        return f"center_x {self.center_x}, center_y {self.center_y}, width {self.width}, height {self.height}"

    def __str__(self) -> str:
        return repr(self)

    def to_bounding_box(self, image_width: float, image_height: float) -> dict:
        result = {}
        result["top_left_x"] = int((self.center_x - (self.width / 2)) * image_width)
        result["top_left_y"] = int((self.center_y - (self.height / 2)) * image_height)
        result["bottom_right_x"] = int((self.center_x + (self.width / 2)) * image_width)
        result["bottom_right_y"] = int((self.center_y + (self.height / 2)) * image_height)
        result["name"] = self.name
        return result

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
    bounding_boxes = {}

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
            (height, width, _) = iio.v2.imread(image).shape
            bounding_boxes[os.path.basename(image)] = {
                "ground_truth" : [a.to_bounding_box(width, height) for a in all_annotations(annotation, label_map)],
                "predicted": [{
                    "top_left_x": int(r["box"]["x1"]),
                    "top_left_y": int(r["box"]["y1"]),
                    "bottom_right_x": int(r["box"]["x2"]),
                    "bottom_right_y": int(r["box"]["y2"]),
                    "name": r["name"]
                } for r in json.loads(result.tojson())],
            }
            csvwriter.writerow([
                image,
                ground_truth_labels["car"],
                ground_truth_labels["bus"],
                predicted_labels["car"],
                predicted_labels["bus"],
                0 if ground_truth_labels["car"] == 0 else predicted_labels["car"] / ground_truth_labels["car"],
                0 if ground_truth_labels["bus"] == 0 else predicted_labels["bus"] / ground_truth_labels["bus"],
            ])
    jsonfile = os.path.splitext(result_file)[0] + ".json"
    with open(jsonfile, 'w') as jsonfile:
       json.dump(bounding_boxes, jsonfile)


def annotation_file(image_file: str) -> str:
    return f"{image_file[:-4]}.txt"

def actual_labels(annotation_file: str, label_map: Dict[int, str]) -> Counter:
    with open(annotation_file, "r") as f:
        return Counter([label_map[int(line.split(" ")[0])] for line in f.readlines()])

def all_annotations(annotation_file: str, label_map: Dict[int, str]) -> List[Annotation]:
    with open(annotation_file, "r") as f:
        return [Annotation.from_line(line, label_map) for line in f.readlines()]
