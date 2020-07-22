"""
This file contains functions that are needed by other python files.

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
import numpy as np


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
        assert areSimilar(perturbedImg, generator.image, similarityType, similarityMeasure), \
            "perturbed image is not similar to original."
        timeTaken = time.time() - start_time
        return timeTaken, perturbedImg, perturbedPrediction
    except FunctionTimedOut:
        return timeMax, None, None


def areSimilar(img1, img2, similarityType="l2", similarityMeasure=10):
    """
    This function checks whether two images are similar based on the similarity type and score
    :param img1: np.array
    :param img2: np.array
    :param similarityType: str default: "l2"
    :param similarityMeasure: float default: 10
    :return: bool - whether the two images are similar
    """
    return calculateSimilarity(img1, img2, similarityType) < similarityMeasure


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
