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

from sklearn.preprocessing import normalize
import tensorflow as tf
import numpy as np


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
