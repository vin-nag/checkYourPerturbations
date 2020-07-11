"""
This file contains the benchmark code.

Date:
    July 5, 2020

Project:
    ECE653 Final Project

Authors:
    name: Vineel Nagisetty, Laura Graves, Joseph Scott
    contact: vineel.nagisetty@uwaterloo.ca
"""

from enum import Enum
from tensorflow import keras
import numpy as np
import pandas as pd


class BenchmarkEnums(Enum):
    Demo = {"./../src/data/models/content/model": [("./../src/data/images/image.npy", 1), ]}


class Benchmark:
    def __init__(self, benchmarkType):
        if benchmarkType not in BenchmarkEnums:
            raise Exception(f"type: {benchmarkType} not in benchmark.")
        self.name = benchmarkType.name
        self.type = benchmarkType.value
        self.data = pd.DataFrame(columns=['model', 'image', 'label'])
        self.createBenchmark()

    def createBenchmark(self):
        i = 0
        for modelName in self.type:
            model = keras.models.load_model(modelName)
            for imageName, label in self.type[modelName]:
                image = np.load(imageName)
                self.data.loc[i] = [model, image, label]
                i += 1

    def getData(self):
        return self.data
