from typing import *

from ultralytics import YOLO
import predict
import sys


def do_scenario_rq1(dawn_folder: str, pre_trained_model: str):
    """ scenario rq1

    train yolo with dawn images and evaluate against dawn images
    the training of the model is done in google colab (scenario-rq1.ipynb).  This gives us a .pt pretrained model containing the necessary wheights.
    we can then use this model to do predictions
    """

    model = YOLO(pre_trained_model)
    predict.do_predict(model, dawn_folder, 'scenario-rq1.csv')


if __name__ == '__main__':
    if (len(sys.argv) < 3):
        raise Exception("please specify dawn test folder and pretrained model")

    do_scenario_rq1(sys.argv[1], sys.argv[2])
