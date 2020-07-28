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

from src.plotter import displayPerturbedImages
from src.utils import par2scores, calculateSimilarity
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

    def evaluate(self, timeMax=50, similarityType="l2", similarityMeasure=10, verbose=True):
        """
        This function performs evaluation and records par2scores and similarity measures.
        :param verbose: bool whether to print statements
        :param timeMax: time (in seconds) of time out default=10.
        :param similarityType: str type of similarity measurement. default=l2
        :param similarityMeasure: float maximum allowable similarity score for the perturbed image. Currently not used.
        :return: None
        """
        for generatorName in self.generators:
            if verbose:
                print(f"Starting evaluation for {generatorName}:")
            for index, row in self.benchmark.getData().iterrows():
                if verbose:
                    print(f"\tEvaluating model: {row['modelName']} for true label: {row['label']}")
                # initialize generator object
                generatorObj = self.generators[generatorName](generatorName, row['model'], row['modelName'],
                                    row['image'], row['label'], similarityType, similarityMeasure)
                # calculate par2score
                par2scores(generator=generatorObj, timeMax=timeMax, similarityMeasure=similarityMeasure)
                # calculate similarity score if generator provides one
                if generatorObj.completed:
                    similarity = calculateSimilarity(row['image'], generatorObj.advImage, similarityType)
                    if verbose:
                        print(f"\t\tResult: new label: {generatorObj.advLabel}, time: {round(generatorObj.time, 4)}, "
                              f"similarity: {round(similarity, 4)}.")
                        displayPerturbedImages(row['image'], "original", row['label'], generatorObj.advImage,
                                               generatorName, generatorObj.advLabel)
                else:
                    similarity = None
                    if verbose:
                        print(f"\t\tResult: timed out.")
                self.results.append([generatorName, generatorObj, row['modelName'], row['image'], row['label'],
                                     generatorObj.advImage, generatorObj.advLabel, generatorObj.time, similarity])
        if verbose:
            print("Completed Evaluation.")
