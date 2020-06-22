"""
This file contains factory design pattern to create generator classes.

Date:
    June 5, 2020

Project:
    ECE653 Final Project

Authors:
    name: Vineel Nagisetty, Laura Graves, Joseph Scott
    contact: vineel.nagisetty@uwaterloo.ca
"""


class Generator:
    def __init__(self, factory):
        self.factory = factory
        self.generate = factory.generateAdversarialExample()


class AbstractGenerator:
    def generateAdversarialExample(self, image, method):
        pass
