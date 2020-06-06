"""
This file contains the stub for benchmark code.

Date:
    June 5, 2020

Project:
    ECE653 Final Project

Authors:
    name: Vineel Nagisetty, Laura Graves, Joseph Scott
    contact: vineel.nagisetty@uwaterloo.ca
"""

import enum

normal = {
    "model.h5": [("img.png", 1)],
    "model2.h5": [("img2.png", 1), ("img.png", 1)]
}

adversarial = {
    "model2.h5": ("img2.png", 1)
}


class BenchmarkEnums(enum.Enum):
    Normal = normal
    Adversarial = adversarial


class Benchmark:
    def __init__(self):
        self.data = BenchmarkEnums.Normal

    def createBenchmark(self):
        pass

    def getData(self):
        return self.data

    def setData(self, data):
        self.data = data
