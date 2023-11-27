import sys
import os
import random
import shutil
from typing import List
import xml.etree.ElementTree as ET


def process_dawn(dawn_folder: str):
    """ processes the dawn dataset

    Does not keep the sandy images.  Takes all other images + the yolo annotations, shuffles them and generates a training and validation dataset

    :param dawn_folder: the folder containing the 4 unzipped dawn datasets (Fog, Rain, Sand, Snow).  Sand will be filtered out, the other images will be kept
                        and shuffled to prepare a training and validation dataset
    :return: nothing
    """

    # contains tuples (relative) filename, (relative) yolo annotation file
    all_files = []
    # contains the label_map, mappings from numeric id to human readable label
    label_map = dict()
    for dirpath, _, files in os.walk(dawn_folder):
        for file in files:
            if not file.lower().endswith("zip") and (file.lower().endswith("jpg") or file.lower().endswith("jpeg")) and not "sand" in dirpath.lower() and not "train-dataset" in dirpath.lower() and not "validation-dataset" in dirpath.lower():
                fname = os.path.join(dirpath, file)
                fbasename = os.path.splitext(os.path.basename(fname))[0]
                yolo_annotation_file = os.path.join(dirpath, f"{os.path.basename(dirpath)}_YOLO_darknet/{fbasename}.txt")
                voc_annotation_file = os.path.join(dirpath, f"{os.path.basename(dirpath)}_PASCAL_VOC/{fbasename}.xml")
                print(f"found file {file}, in dirname {dirpath}, yolo annotation file {yolo_annotation_file}, voc annotation file {voc_annotation_file}")
                all_files.append((fname, yolo_annotation_file))
                for yolo_annotation, voc_annotation in zip(read_yolo_annotations(yolo_annotation_file), read_voc_annotations(voc_annotation_file)):
                    label_map[yolo_annotation] = voc_annotation
    # the label_map needs some manipulations.  we dont have all labels in our datasets, and yolo expects continuous ranges of labels
    max_label = sorted(label_map)[-1]
    for key in range(max_label):
        if key not in label_map:
            label_map[key] = "Unknown"
    print(f"final label_map : {label_map}")


    # restart from scratch for the train and validation datasets
    train_dataset_folder = os.path.join(dawn_folder, "train-dataset")
    if os.path.exists(train_dataset_folder):
        shutil.rmtree(train_dataset_folder)
    os.mkdir(train_dataset_folder)

    validation_dataset_folder = os.path.join(os.path.join(dawn_folder, "validation-dataset"))
    if os.path.exists(validation_dataset_folder):
        shutil.rmtree(validation_dataset_folder)
    os.mkdir(validation_dataset_folder)

    random.shuffle(all_files)

    training = all_files[:600]
    validation = all_files[600:]
    print(f"# training files {len(training)}, # validation files {len(validation)}")
    print(f"copy training files")
    index = 0
    for file, annotation_file in training:
        index += 1
        basename = f"image{str(index).zfill(4)}"
        shutil.copy(file, f"{train_dataset_folder}/{basename}.jpg")
        shutil.copy(annotation_file, f"{train_dataset_folder}/{basename}.txt")
    print(f"copy validation files")
    index = 0
    for file, annotation_file in validation:
        index += 1
        basename = f"image{str(index).zfill(4)}"
        shutil.copy(file, f"{validation_dataset_folder}/{basename}.jpg")
        shutil.copy(annotation_file, f"{validation_dataset_folder}/{basename}.txt")

    print(f"generate yolov8 dataset yaml file")
    yolo_training_dataset = os.path.join(dawn_folder, "dawn-train-dataset.yml")
    with open(yolo_training_dataset, "w") as f:
        f.write(f"path: {os.path.abspath(dawn_folder)}{os.linesep}")
        f.write(f"train: {os.path.abspath(train_dataset_folder)}{os.linesep}")
        f.write(f"val: {os.path.abspath(validation_dataset_folder)}{os.linesep}")
        f.write(f"names:{os.linesep}")
        for key in sorted(label_map):
            f.write(f"  {key}: {label_map[key]}{os.linesep}")

    print(f"Done")



def read_voc_annotations(annotation_file: str) -> List[str]:
    result = []
    tree = ET.parse(annotation_file)
    root = tree.getroot()
    for object in root.iter("object"):
        result.append(object.find("name").text)
    return result

def read_yolo_annotations(annotation_file: str) -> List[int]:
    result = []
    with open(annotation_file, "r") as f:
        for line in f.readlines():
            result.append(int(line.split(" ")[0]))
    return result


if __name__ == '__main__':
    if (len(sys.argv) < 2):
        raise Exception("please specify dawn folder")

    process_dawn(sys.argv[1])
