"""
This file contains the stub for symbolic execution code.

Date:
    June 5, 2020

Project:
    ECE653 Final Project: Check Your Perturbations

Authors:
    name: Vineel Nagisetty, Laura Graves, Joseph Scott
    contact: vineel.nagisetty@uwaterloo.ca
"""

from src.generator.template import GeneratorTemplate


class SymbolicExecutioner(GeneratorTemplate):
    """ This class is the stub for our symbolic execution implementation."""
    def __init__(self, name, model, image, label, similarityType="l2", similarityMeasure=10, verbose=True):
        super().__init__(name, model, image, label, similarityType, similarityMeasure, verbose)

    def generateAdversarialExample(self):
        """ This method needs to be implemented. TODO: Implement this method. """
        pass
