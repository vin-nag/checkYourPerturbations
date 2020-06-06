"""
This file contains the stub for fuzzing code.

Date:
    June 5, 2020

Project:
    ECE653 Final Project

Authors:
    name: Vineel Nagisetty, Laura Graves, Joseph Scott
    contact: vineel.nagisetty@uwaterloo.ca
"""

from src.generator.factory import AbstractGenerator


class Fuzzer(AbstractGenerator):
    def __init__(self, **kwargs):
        self.model = kwargs.model
        self.input = kwargs.input
        self.label = kwargs.label
        if "type" in kwargs:
            self.type = kwargs.type
        else:
            self.type = "random"

    def generateAdversarialExample(self):
        pass
