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
                timeTaken, perturbedImg, perturbedPrediction = self.par2scores(generator=generatorObj, timeMax=timeMax)
                # calculate similarity score if generator provides one
                if perturbedImg is not None:
                    similarity = Evaluator.calculateSimilarity(row['image'], perturbedImg, similarityType)
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

    @staticmethod
    def par2scores(generator, timeMax, similarityType="l2", similarityMeasure=10):
        """
        This function calculates the par2scores for a generator.
        :param generator: GeneratorTemplate object
        :param timeMax: float timeout
        :param similarityMeasure: float default: 10
        :param similarityType: str default: "l2"
        :return: tuple of (time taken, fuzzed image, fuzzed prediction). Note the last two elements are None if timeout.
        """
        try:
            start_time = time.time()
            # create new process to measure time taken.
            perturbedImg = func_timeout(timeout=timeMax, func=generator.generateAdversarialExample)
            perturbedPrediction = np.argmax(generator.model.predict(perturbedImg), axis=1)[0]
            # verify results
            assert generator.label != perturbedPrediction, "perturbed image label is the same as the original image."
            assert Evaluator.areSimilar(perturbedImg, generator.image, similarityType, similarityMeasure), \
                "perturbed image is not similar to original."
            timeTaken = time.time() - start_time
            return timeTaken, perturbedImg, perturbedPrediction
        except FunctionTimedOut:
            return timeMax, None, None

    @staticmethod
    def areSimilar(img1, img2, similarityType="l2", similarityMeasure=10):
        """
        This function checks whether two images are similar based on the similarity type and score
        :param img1: np.array
        :param img2: np.array
        :param similarityType: str default: "l2"
        :param similarityMeasure: float default: 10
        :return: bool - whether the two images are similar
        """
        return Evaluator.calculateSimilarity(img1, img2, similarityType) < similarityMeasure

    @staticmethod
    def calculateSimilarity(img1, img2, similarityType="l2"):
        """
        This method calculates the similarity of two images given the type of similariy function (such as l2 distance).
        :param img1: np.array
        :param img2: np.array
        :param similarityType: str default: "l2"
        :return: float - distance between the images
        """
        img1 = normalize(img1.astype(np.float64).squeeze(), norm=similarityType, axis=1)
        img2 = normalize(img2.astype(np.float64).squeeze(), norm=similarityType, axis=1)
        return np.linalg.norm(img1 - img2)
