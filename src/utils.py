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


def get_or_guess_labels(model_fn, x, y=None, targeted=False):
    """
    Helper function to get the label to use in generating an
    adversarial example for x.
    If 'y' is not None, then use these labels.
    If 'targeted' is True, then assume it's a targeted attack
    and y must be set.
    Otherwise, use the model's prediction as the label and perform an
    untargeted attack
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    """
    if targeted is True and y is None:
        raise ValueError("Must provide y for a targeted attack!")

    preds = model_fn(x)
    nb_classes = preds.shape[-1]

    # labels set by the user
    if y is not None:
        y = np.asarray(y)

        if len(y.shape) == 1:
            # the user provided a list/1D-array
            idx = y.reshape([-1, 1])
            y = np.zeros_like(preds)
            y[:, idx] = 1

        y = tf.cast(y, x.dtype)
        return y, nb_classes

    # must be an untargeted attack
    labels = tf.cast(tf.equal(tf.reduce_max(
        preds, axis=1, keepdims=True), preds), x.dtype)

    return labels, nb_classes


def set_with_mask(x, x_other, mask):
    """Helper function which returns a tensor similar to x with all the values
     of x replaced by x_other where the mask evaluates to true.
    """
    mask = tf.cast(mask, x.dtype)
    ones = tf.ones_like(mask, dtype=x.dtype)
    return x_other * mask + x * (ones - mask)