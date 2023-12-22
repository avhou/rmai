import sys
import os
from typing import *
import xml.etree.ElementTree as ET
import imageio as iio
import itertools
from collections import defaultdict
import tqdm
import random
from pathlib import Path
import shutil


class Annotation:
    """ Keeps track of information about annotations """
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
    """ Parses one detrac XML file

    :param annotation_file:  the filename
    :param annotation_filename: the absolute filename
    :return: a tuple containing (sunny datasets, all annotations) found in this XML file
    """
    annotations = []
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
                annotations.append(Annotation(num, datasetname, left, top, width, height, vehicle_type))
    return (sunny_datasets, annotations)


def process_detrac(image_folder: str, annotation_folder: str):
    """ process the detract image and annotation folders

    will determine the sunny datasets, convert those sunny images to yolo format and randomly choose 167 training images and 33 validation images of those sunny images
    these images will be augmented 3 times so we end up with 501 training images and 99 validation images

    :param image_folder: folder containing the images (in subdirectories per dataset / traffic intersection)
    :param annotation_folder: folder containing the annotations (in subdirectories per dataset / traffic intersection)
    :return: nothing
    """
    annotations: List[Annotation] = []
    sunny_datasets: Set[str] = set()
    for dirpath, _, files in os.walk(annotation_folder):
        for filename in files:
            if filename.lower().endswith("xml"):
                fname = os.path.join(dirpath, filename)
                print(f"read annotation file {fname}")
                (sunny_datasets_for_file, extra_annotations) = read_annotations(fname, filename)
                annotations.extend(extra_annotations)
                sunny_datasets.update(sunny_datasets_for_file)


    print(f"all vehicle types before filtering : {set((a.vehicle_type for a in annotations))}")
    # we only want to keep annotations containing car and truck vehicle types
    filtered_annotations: List[Annotation] = [a for a in annotations if a.vehicle_type == "car" or a.vehicle_type == "truck"]
    print(f"total # of annotations is {len(annotations)}, filtered # of annotations is {len(filtered_annotations)}")
    print(f"all vehicle types after filtering : {set((a.vehicle_type for a in annotations))}")
    grouped_annotations = defaultdict(list)
    for key, values in itertools.groupby(filtered_annotations, lambda annotation: (annotation.num, annotation.dataset)):
        grouped_annotations[key] = list(values)

    labelmap: Dict[str, int] = {
        "car": 0,
        "bus": 1
    }
    sunny_images = []

    for dirpath, _, files in os.walk(image_folder):
        current_dir = os.path.basename(dirpath)
        if current_dir in sunny_datasets and "train-dataset" not in dirpath.lower() and "validation-dataset" not in dirpath.lower():
            for filename in tqdm.tqdm(files, desc=f"Process detrac files in folder {dirpath}"):
                if filename.lower().endswith("jpg") or filename.lower().endswith("jpeg"):
                    fname = os.path.join(dirpath, filename)
                    image = iio.v2.imread(fname)
                    (height, width, _) = image.shape
                    image_num = int("".join(c for c in filename if c.isdigit()))
                    image_dataset = os.path.basename(os.path.dirname(fname))
                    annotations_for_image = [a for a in grouped_annotations[(image_num, image_dataset)]]
                    if len(annotations_for_image) > 0:
                        annotation_filename = f"{os.path.splitext(fname)[0]}.txt"
                        with open(annotation_filename, "w") as f:
                            for annotation in annotations_for_image:
                                f.write(annotation.annotate(width, height, labelmap[annotation.vehicle_type]))
                        sunny_images.append(fname)

    print(f"totaal aantal sunny images : {len(sunny_images)}")
    random.shuffle(sunny_images)

    # restart from scratch for the train and validation datasets
    train_dataset_folder = os.path.join(Path(image_folder).parent, "train-dataset")
    if os.path.exists(train_dataset_folder):
        shutil.rmtree(train_dataset_folder)
    os.mkdir(train_dataset_folder)

    validation_dataset_folder = os.path.join(Path(image_folder).parent, "validation-dataset")
    if os.path.exists(validation_dataset_folder):
        shutil.rmtree(validation_dataset_folder)
    os.mkdir(validation_dataset_folder)

    training = sunny_images[:167]
    validation = sunny_images[167:167+33]
    print(f"# files {len(sunny_images)}, # training files {len(training)}, # validation files {len(validation)}")
    print(f"copy training files")
    index = 0
    for file in training:
        index += 1
        basename = f"image{str(index).zfill(4)}"
        annotation_file = f"{os.path.splitext(file)[0]}.txt"
        shutil.copy(file, f"{train_dataset_folder}/{basename}.jpg")
        shutil.copy(annotation_file, f"{train_dataset_folder}/{basename}.txt")

    print(f"copy validation files")
    index = 0
    for file in validation:
        index += 1
        basename = f"image{str(index).zfill(4)}"
        annotation_file = f"{os.path.splitext(file)[0]}.txt"
        shutil.copy(file, f"{validation_dataset_folder}/{basename}.jpg")
        shutil.copy(annotation_file, f"{validation_dataset_folder}/{basename}.txt")

    print(f"labelmap : {labelmap}")


    print(f"generate yolov8 dataset yaml file - colab")
    yolo_training_dataset = os.path.join(Path(image_folder).parent, "detrac-train-dataset-colab.yml")
    with open(yolo_training_dataset, "w") as f:
        f.write(f"path: /datasets/detrac{os.linesep}")
        f.write(f"train: /datasets/detrac/train-dataset-augmented{os.linesep}")
        f.write(f"val: /datasets/detrac/validation-dataset-augmented{os.linesep}")
        f.write(f"names:{os.linesep}")
        f.write(f"  0: car{os.linesep}")
        f.write(f"  1: bus{os.linesep}")



if __name__ == '__main__':
    if (len(sys.argv) < 3):
        raise Exception("please specify image folder and annotation folder")

    process_detrac(sys.argv[1], sys.argv[2])
    print("done")
