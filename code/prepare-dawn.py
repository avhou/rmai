import sys
import os
import random
import shutil
from typing import *
import xml.etree.ElementTree as ET


def process_dawn(dawn_folder: str):
    """ processes the dawn dataset

    Does not keep the sandy images.  Takes all other images + the yolo annotations, shuffles them and generates a training, validation and test dataset

    :param dawn_folder: the folder containing the 4 unzipped dawn datasets (Fog, Rain, Sand, Snow).  Sand will be filtered out, the other images will be kept
                        and shuffled to prepare a training and validation dataset
    :return: nothing
    """

    # contains tuples (relative) filename, (relative) yolo annotation file
    all_files = []
    # label_map of labels of interest to us
    label_map: Dict[str, int] = { "car": 0, "bus": 1}
    dawn_label_map: Dict[int, str] = {}
    for dirpath, _, files in os.walk(dawn_folder):
        for file in files:
            if not file.lower().endswith("zip") and (file.lower().endswith("jpg") or file.lower().endswith("jpeg")) and not "sand" in dirpath.lower() and not "train-dataset" in dirpath.lower() and not "validation-dataset" in dirpath.lower() and not "test-dataset" in dirpath.lower():
                fname = os.path.join(dirpath, file)
                fbasename = os.path.splitext(os.path.basename(fname))[0]
                yolo_annotation_file = os.path.join(dirpath, f"{os.path.basename(dirpath)}_YOLO_darknet/{fbasename}.txt")
                voc_annotation_file = os.path.join(dirpath, f"{os.path.basename(dirpath)}_PASCAL_VOC/{fbasename}.xml")
                print(f"found file {file}, in dirname {dirpath}, yolo annotation file {yolo_annotation_file}, voc annotation file {voc_annotation_file}")
                all_voc_annotations = set(read_voc_annotations(voc_annotation_file))
                if len(all_voc_annotations.intersection(set(label_map.keys()))) == 0:
                   print(f"no relevant labels found for file {fname}, we will skip it")
                else:
                    all_files.append((fname, yolo_annotation_file))
                    for yolo_annotation, voc_annotation in zip(read_yolo_annotations(yolo_annotation_file), read_voc_annotations(voc_annotation_file)):
                        dawn_label_map[yolo_annotation] = voc_annotation

    print(f"final dawn label map {dawn_label_map}")

    # restart from scratch for the train and validation datasets
    train_dataset_folder = os.path.join(dawn_folder, "train-dataset")
    if os.path.exists(train_dataset_folder):
        shutil.rmtree(train_dataset_folder)
    os.mkdir(train_dataset_folder)

    validation_dataset_folder = os.path.join(os.path.join(dawn_folder, "validation-dataset"))
    if os.path.exists(validation_dataset_folder):
        shutil.rmtree(validation_dataset_folder)
    os.mkdir(validation_dataset_folder)

    test_dataset_folder = os.path.join(dawn_folder, "test-dataset")
    if os.path.exists(test_dataset_folder):
        shutil.rmtree(test_dataset_folder)
    os.mkdir(test_dataset_folder)
    random.shuffle(all_files)

    # split 501 (3*167) training, 99 (3*33) validation, 98 test
    training = all_files[:501]
    validation = all_files[501:501+99]
    test = all_files[501+99:]
    print(f"# files {len(all_files)}, # training files {len(training)}, # validation files {len(validation)}, # test files {len(test)}")
    print(f"copy training files")
    index = 0
    for file, annotation_file in training:
        index += 1
        basename = f"image{str(index).zfill(4)}"
        shutil.copy(file, f"{train_dataset_folder}/{basename}.jpg")
        filter_yolo_annotations(annotation_file, f"{train_dataset_folder}/{basename}.txt", dawn_label_map, label_map)

    print(f"copy validation files")
    index = 0
    for file, annotation_file in validation:
        index += 1
        basename = f"image{str(index).zfill(4)}"
        shutil.copy(file, f"{validation_dataset_folder}/{basename}.jpg")
        filter_yolo_annotations(annotation_file, f"{validation_dataset_folder}/{basename}.txt", dawn_label_map, label_map)

    print(f"copy test files")
    index = 0
    for file, annotation_file in test:
        index += 1
        basename = f"image{str(index).zfill(4)}"
        shutil.copy(file, f"{test_dataset_folder}/{basename}.jpg")
        filter_yolo_annotations(annotation_file, f"{test_dataset_folder}/{basename}.txt", dawn_label_map, label_map)

    print(f"generate yolov8 dataset yaml file")
    yolo_training_dataset = os.path.join(dawn_folder, "dawn-train-dataset.yml")
    with open(yolo_training_dataset, "w") as f:
        f.write(f"path: {os.path.abspath(dawn_folder)}{os.linesep}")
        f.write(f"train: {os.path.abspath(train_dataset_folder)}{os.linesep}")
        f.write(f"val: {os.path.abspath(validation_dataset_folder)}{os.linesep}")
        f.write(f"test: {os.path.abspath(test_dataset_folder)}{os.linesep}")
        f.write(f"names:{os.linesep}")
        f.write(f"  0: car{os.linesep}")
        f.write(f"  1: bus{os.linesep}")

    print(f"generate yolov8 dataset yaml file - colab")
    yolo_training_dataset = os.path.join(dawn_folder, "dawn-train-dataset-colab.yml")
    with open(yolo_training_dataset, "w") as f:
        f.write(f"path: /datasets/dawn{os.linesep}")
        f.write(f"train: /datasets/dawn/train-dataset{os.linesep}")
        f.write(f"val: /datasets/dawn/validation-dataset{os.linesep}")
        f.write(f"test: /datasets/dawn/test-dataset{os.linesep}")
        f.write(f"names:{os.linesep}")
        f.write(f"  0: car{os.linesep}")
        f.write(f"  1: bus{os.linesep}")

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

def filter_yolo_annotations(src: str, dest: str, dawn_label_map: Dict[int, str], yolo_label_map: Dict[str, int]):
    number_annotations_written = 0
    with open(src, "r") as r:
        with open(dest, "w") as w:
            for line in r.readlines():
                words = line.split(" ")
                label = int(words[0])
                rest = " ".join(words[1:])
                label_str = dawn_label_map[label]
                if label_str in yolo_label_map:
                    w.write(f"{yolo_label_map[label_str]} {rest}")
                    number_annotations_written += 1
    # make sure at least one annotation was written
    if number_annotations_written == 0:
        raise Exception(f"we did not write any annotation for file {src}")

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        raise Exception("please specify dawn folder")

    process_dawn(sys.argv[1])
