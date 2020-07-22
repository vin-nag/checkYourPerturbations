"""
This file runs the evaluation of various generators on a given benchmark.

Date:
    July 5, 2020

Project:
    ECE653 Final Project: Check Your Perturbations

Authors:
    name: Vineel Nagisetty, Laura Graves, Joseph Scott
    contact: vineel.nagisetty@uwaterloo.ca
"""

import time
from func_timeout import func_timeout, FunctionTimedOut
from sklearn.preprocessing import normalize
from src.plotter import displayPerturbedImages
from src.utils import par2scores, calculateSimilarity
import numpy as np
import pandas as pd


class Evaluator:
    """ This class performs the evaluation of various generators on a given benchmark. """

    def __init__(self, benchmark, generators):
        """
        Standard init function.
        :param benchmark: pandas dataframe with each row consisting of <model, image, label> data.
        :param generators: dictionary consisting of <generator name: generator class>
        """
        self.benchmark = benchmark
        self.generators = generators
        self.results = pd.DataFrame(columns=('generatorName', 'generatorObj', 'model', 'image', 'label',
                                             'perturbed image', 'perturbed label', 'time', 'similarity'))

    def evaluate(self, timeMax=10, similarityType="l2", similarityMeasure=10):
        """
        This function performs evaluation and records par2scores and similarity measures.
        TODO: Possibly add other measures of evaluation.
        :param timeMax: time (in seconds) of time out default=10.
        :param similarityType: str type of similarity measurement. default=l2
        :param similarityMeasure: float maximum allowable similarity score for the perturbed image. Currently not used.
        :return: None
        """
        i = 0
        for generatorName in self.generators:
            for index, row in self.benchmark.getData().iterrows():
                # initialize generator object
                generatorObj = self.generators[generatorName](generatorName, row['model'], row['image'], row['label'],
                                                              similarityType, similarityMeasure)
                # calculate par2score
                timeTaken, perturbedImg, perturbedPrediction = par2scores(generator=generatorObj, timeMax=timeMax)
                # calculate similarity score if generator provides one
                if perturbedImg is not None:
                    similarity = calculateSimilarity(row['image'], perturbedImg, similarityType)
                    print(f"Results for: {generatorName} true label: {row['label']}, perturbed label: "
                          f"{perturbedPrediction}, time: {round(timeTaken, 4)}, similarity: {round(similarity, 4)}.")
                    displayPerturbedImages(row['image'], "original", row['label'], perturbedImg, generatorName,
                                           perturbedPrediction)
                else:
                    similarity = None
                    print(f"Results for: {generatorName} true label: {row['label']} timed out.")
                # record data and print
                self.results.loc[i] = [generatorName, generatorObj, row['model'], row['image'], row['label'],
                                       perturbedImg, perturbedPrediction, timeTaken, similarity]
        print("Completed Evaluation.")

