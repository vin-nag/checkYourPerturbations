"""
This file contains the template of a CleverHans generator object.

Date:
    July 5, 2020

Project:
    ECE653 Final Project: Check Your Perturbations

Authors:
    name: Vineel Nagisetty, Laura Graves, Joseph Scott
    contact: vineel.nagisetty@uwaterloo.ca
"""

from src.generator.template import GeneratorTemplate


class CleverHansTemplate(GeneratorTemplate):
    """ This class is the template of a CleverHansAttack. It needs to be extended."""

    def generateAdversarialExample(self):
        """
        This overrides the function from the GeneratorTemplate class.
        :return: np.array representing fuzzed image.
        """
        pass

