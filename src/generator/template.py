"""
This file contains the template of an abstract generator object.

Date:
    July 5, 2020

Project:
    ECE653 Final Project: Check Your Perturbations

Authors:
    name: Vineel Nagisetty, Laura Graves, Joseph Scott
    contact: vineel.nagisetty@uwaterloo.ca
"""

import numpy as np


class GeneratorTemplate:
    """ This class is a template for a Generator """

    def __init__(self, name, model, image, label, similarityType="l2", similarityMeasure=10):
        """
        Standard init function
        :param name: str name of the generator
        :param model: keras model of a DNN
        :param image: np.array of an input for the keras model
        :param label: int the true label of the image
        :param similarityType: str the type of similarity distance function (default: "l2")
        :param similarityMeasure: float the minimum allowable similarity for the perturbed image.
        """
        assert np.argmax(model.predict(image), axis=1)[0] == label, "the label provided is not correct"
        self.name = name
        self.model = model
        self.image = image
        self.label = label
        self.similarityType = similarityType
        self.similarityMeasure = similarityMeasure
        self.imageShape = image.shape

    def generateAdversarialExample(self):
        """
        This function needs to be overridden by the different generators
        :return:
        """
        pass