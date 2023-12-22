import json
import sys
from ultralytics import YOLO
import os
from pathlib import Path
from typing import *
from json import *
import csv
import predict


def do_base_scenario(dawn_folder: str):
    """ base scenario

    Use the base yolo model and predict against the dawn test set

    :param dawn_folder: the folder containing the dawn test dataset
    :return: nothing
    """

    model = YOLO("yolov8n.pt")
    predict.do_predict(model, dawn_folder, 'base-scenario.csv')

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

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        raise Exception("please specify dawn test folder")

    do_base_scenario(sys.argv[1])
