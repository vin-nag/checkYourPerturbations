"""
This file creates benchmarks that are used to evaluate the various generators. A benchmark consists of one or more
DNN models (in keras format) for which we have one or more tuples of <images, labels>.

Date:
    July 5, 2020

Project:
    ECE653 Final Project: Check Your Perturbations

Authors:
    name: Vineel Nagisetty, Laura Graves, Joseph Scott
    contact: vineel.nagisetty@uwaterloo.ca
"""

from enum import Enum
from tensorflow import keras
import numpy as np
import pandas as pd


class BenchmarkEnums(Enum):
    """ This is an enum that contains all the different benchmarks. """

    Demo = {
        "models": ["./../src/data/models/MNIST/regularFCNN"],
        "images": "./../src/data/images/MNIST/demo.npy",
        "similarityType": "l2",
        "similarityMeasure": 10,
        "timeLimit": 50
    }

    MainSimilar = {
        "models": ["./../src/data/models/MNIST/regularFCNN", "./../src/data/models/MNIST/robustFCNN"],
        "images": "./../src/data/images/MNIST/demo.npy",
        "similarityType": "l2",
        "similarityMeasure": 2.5,
        "timeLimit": 600
    }

    Main = {
        "models": ["./../src/data/models/MNIST/regularFCNN", "./../src/data/models/MNIST/robustFCNN"],
        "images": "./../src/data/images/MNIST/demo.npy",
        "similarityType": "l2",
        "similarityMeasure": 5,
        "timeLimit": 600
    }

    MainDissimilar = {
        "models": ["./../src/data/models/MNIST/regularFCNN", "./../src/data/models/MNIST/robustFCNN"],
        "images": "./../src/data/images/MNIST/demo.npy",
        "similarityType": "l2",
        "similarityMeasure": 7.5,
        "timeLimit": 600
    }

    All = {
        "models":
            [
                "./../src/data/models/MNIST/regularFCNN", "./../src/data/models/MNIST/robustFCNN",
                "./../src/data/models/MNIST/regularCNN", "./../src/data/models/MNIST/robustCNN",
                "./../src/data/models/MNIST/thermometerCNN"
            ],
        "images": "./../src/data/images/MNIST/demo.npy",
        "similarityType": "l2",
        "similarityMeasure": 5,
        "timeLimit": 600
    }

    Thermometer = {
        "models": ["./../src/data/models/MNIST/thermometerCNN"],
        "images": "./../src/data/images/MNIST/demo.npy",
        "similarityType": "l2",
        "similarityMeasure": 5
    }

    CNNs = {
        "models": ["./../src/data/models/MNIST/regularCNN", "./../src/data/models/MNIST/robustCNN,"
                                                            "./../src/data/models/MNIST/thermometerCNN"],
        "images": "./../src/data/images/MNIST/demo.npy",
        "similarityType": "l2",
        "similarityMeasure": 5,
        "timeLimit": 600
    }

    def __str__(self):
        return self.value


class Benchmark:
    """ This class contains a given benchmark. """

    def __init__(self, benchmarkType, similarityType=None, similarityMeasure=None, verbose=False):
        """
        Standard init function.
        :param benchmarkType: Enum that is found in BenchmarkEnums.
        """
        if benchmarkType not in BenchmarkEnums:
            raise Exception(f"type: {benchmarkType} not in benchmark.")
        self.name = benchmarkType.name
        self.type = benchmarkType.value
        self.data = pd.DataFrame(columns=['modelName', 'model', 'image', 'label'])
        self.numImages = 0
        self.timeLimit = self.type["timeLimit"]
        self.similarityType = self.type["similarityType"] if similarityType is None else similarityType
        self.similarityMeasure = self.type["similarityMeasure"] if similarityMeasure is None else similarityMeasure
        self.verbose = verbose
        self.createBenchmark()

    def createBenchmark(self):
        """
        This function creates a benchmark, loading the models and images from their file names.
        :return: None
        """
        i = 0
        if self.verbose:
            print(f"Creating benchmark: {self.name}")
        for modelName in self.type["models"]:
            model = keras.models.load_model(modelName)
            onlyModelName = modelName[modelName.rfind("/") + 1:]
            if self.verbose:
                print(f"Loaded model: {onlyModelName}")
            imageSets = self.type["images"]
            images = np.load(imageSets, allow_pickle=True)
            size = images.shape[0]
            for index in range(size):
                image, label = images[index, 0], images[index, 1]
                pred = np.argmax(model.predict(image), axis=1)[0]
                if pred == label:
                    self.data.loc[i] = [onlyModelName, model, image, label]
                    i += 1
            self.numImages = i
        print(f"Created benchmark: {self.name} with shape: {self.data.shape}.")

    def getData(self):
        """
        This function returns the benchmark data
        :return: pandas dataframe with each row consisting of model, image and label.
        """
        return self.data
