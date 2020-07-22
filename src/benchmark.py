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
    """ This is an enum that contains all the different benchmarks. TODO: Add new Benchmarks. """
    Demo = {"./../src/data/models/fullyConnected.h5": [
        ("./../src/data/images/MNIST/nine.npy", 9),
        ("./../src/data/images/MNIST/one.npy", 1),
        ("./../src/data/images/MNIST/three.npy", 3),
        ("./../src/data/images/MNIST/seven.npy", 7)
    ]}


class Benchmark:
    """ This class contains a given benchmark. """

    def __init__(self, benchmarkType):
        """
        Standard init function.
        :param benchmarkType: Enum that is found in BenchmarkEnums.
        """
        if benchmarkType not in BenchmarkEnums:
            raise Exception(f"type: {benchmarkType} not in benchmark.")
        self.name = benchmarkType.name
        self.type = benchmarkType.value
        self.data = pd.DataFrame(columns=['model', 'image', 'label'])
        self.createBenchmark()

    def createBenchmark(self):
        """
        This function creates a benchmark, loading the models and images from their file names.
        :return: None
        """
        i = 0
        for modelName in self.type:
            model = keras.models.load_model(modelName)
            for imageName, label in self.type[modelName]:
                image = np.load(imageName)
                self.data.loc[i] = [model, image, label]
                i += 1

    def getData(self):
        """
        This function returns the benchmark data
        :return: pandas dataframe with each row consisting of model, image and label.
        """
        return self.data
