import sys
import os
from typing import List, Set
import xml.etree.ElementTree as ET
import imageio as iio
import itertools
from collections import defaultdict


class Annotation:
    num: int
    left: float
    top: float
    width: float
    height: float
    vehicle_type: str
    dataset: str

    def __init__(self, num: int, dataset: str, left: float, top: float, width: float, height: float, vehicle_type: str):
        self.num = num
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.vehicle_type = vehicle_type
        self.dataset = dataset

    def __repr__(self) -> str:
        return f"num {self.num}, dataset {self.dataset}, left {self.left}, top {self.top}, width {self.width}, height {self.height}, vehicle_type {self.vehicle_type}"

    def __str__(self) -> str:
        return f"num {self.num}, dataset {self.dataset}, left {self.left}, top {self.top}, width {self.width}, height {self.height}, vehicle_type {self.vehicle_type}"

    def annotate(self, image_width: float, image_height: float, annotation_class: int) -> str:
        center_x = (self.left + (self.width / 2)) / image_width
        center_y = (self.top + (self.height / 2)) / image_height
        width = self.width / image_width
        height = self.height / image_height
        return f"{annotation_class} {center_x} {center_y} {width} {height}{os.linesep}"


def read_annotations(annotation_file: str, annotation_filename: str) -> (Set[str], List[Annotation]):
    result = []
    sunny_datasets = set()
    datasetname = os.path.splitext(annotation_filename)[0]
    tree = ET.parse(annotation_file)
    root = tree.getroot()
    for sequence_attribute in root.iter("sequence_attribute"):
        if str(sequence_attribute.attrib["sence_weather"]).lower() == "sunny":
            sunny_datasets.add(datasetname)
            print(f"dataset {datasetname} was added to the list of sunny datasets")

    for frame in root.iter("frame"):
        num = int(frame.attrib['num'])
        for target in frame.iter("target"):
            box = target.find("box")
            attribute = target.find("attribute")
            if box is not None and attribute is not None:
                left = float(box.attrib["left"])
                top = float(box.attrib["top"])
                width = float(box.attrib["width"])
                height = float(box.attrib["height"])
                vehicle_type = str(attribute.attrib["vehicle_type"])
                result.append(Annotation(num, datasetname, left, top, width, height, vehicle_type))
    return (sunny_datasets, result)


def convert_detrac(image_folder: str, annotation_folder: str):
    annotations = []
    sunny_datasets = set()
    for dirpath, _, files in os.walk(annotation_folder):
        for filename in files:
            if filename.lower().endswith("xml"):
                fname = os.path.join(dirpath, filename)
                print(f"read annotation file {fname}")
                (sunny_datasets_for_file, extra_annotations) = read_annotations(fname, filename)
                annotations.extend(extra_annotations)
                sunny_datasets.update(sunny_datasets_for_file)

    grouped_annotations = defaultdict(list)
    for key, values in itertools.groupby(annotations, lambda annotation: (annotation.num, annotation.dataset)):
        grouped_annotations[key] = list(values)

    labelmap = {}
    total_images = 0
    total_sunny_images = 0

    for dirpath, _, files in os.walk(image_folder):
        for filename in files:
            if filename.lower().endswith("jpg") or filename.lower().endswith("jpeg"):
                fname = os.path.join(dirpath, filename)
                image = iio.v2.imread(fname)
                (height, width, _) = image.shape
                image_num = int("".join(c for c in filename if c.isdigit()))
                image_dataset = os.path.basename(os.path.dirname(fname))
                annotations_for_image = [a for a in grouped_annotations[(image_num, image_dataset)]]
                annotation_filename = f"{os.path.splitext(fname)[0]}.txt"
                print(f"write annotation file {annotation_filename}, image_num {image_num}, image_dataset {image_dataset}")
                with open(annotation_filename, "w") as f:
                    for annotation in annotations_for_image:
                        if annotation.vehicle_type not in labelmap:
                            value = 0 if len(labelmap) == 0 else max(labelmap.values())
                            labelmap[annotation.vehicle_type] = value + 1
                        f.write(annotation.annotate(width, height, labelmap[annotation.vehicle_type]))
                total_images += 1
                if image_dataset in sunny_datasets:
                    total_sunny_images += 1

    print(f"totaal aantal images : {total_images}")
    print(f"totaal aantal sunny images : {total_sunny_images}")
    print("labelmap : ")
    print(labelmap)




if __name__ == '__main__':
    if (len(sys.argv) < 3):
        raise Exception("please specify image folder and annotation folder")

    convert_detrac(sys.argv[1], sys.argv[2])
    print("done")
