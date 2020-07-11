"""
This file contains the template for a generator.

Date:
    July 5, 2020

Project:
    ECE653 Final Project

Authors:
    name: Vineel Nagisetty, Laura Graves, Joseph Scott
    contact: vineel.nagisetty@uwaterloo.ca
"""


class GeneratorTemplate:
    """ This class is a template for a Generator """

    def __init__(self, name, model, image, label, similarityType="l2", similarityMeasure=10):
        """
        Standard init function
        :param name:
        :param model:
        :param image:
        :param label:
        :param similarityType:
        :param similarityMeasure:
        """
        assert model(image) == label, "the label provided is not correct"
        self.name = name
        self.model = model
        self.image = image
        self.label = label
        self.similarityType = similarityType
        self.similarityMeasure = similarityMeasure

    def generateAdversarialExample(self):
        """
        This function needs to be overridden by the different generators
        :return:
        """
        pass
