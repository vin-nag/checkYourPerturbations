"""
This file contains the factory design pattern to create various generators.

Date:
    July 5, 2020

Project:
    ECE653 Final Project: Check Your Perturbations

Authors:
    name: Vineel Nagisetty, Laura Graves, Joseph Scott
    contact: vineel.nagisetty@uwaterloo.ca
"""

from enum import Enum
from src.generator.fuzzer import StepFuzzer, NormFuzzer, LaplaceFuzzer, VinFuzzer
from src.generator.cleverHans import CleverHansTemplate


class GeneratorTypes(Enum):
    """ This enum contains the various generators we wish to evaluate. """
    #StepFuzz = StepFuzzer
    #NormFuzz = NormFuzzer
    #LaplaceFuzz = LaplaceFuzzer
    Clever = CleverHansTemplate
    #VinFuzz = VinFuzzer


class GeneratorSelector:
    """ This class registers new generators and retrieves them. """

    def __init__(self):
        """
        Standard init function.
        """
        self.types = {}
        self.registerAllGenerators()

    def registerNewGenerator(self, name, obj):
        """
        This function registers new generator. Not needed if that generator is found in the GeneratorTypes enum.
        :param name: str name of the generator
        :param obj: class GeneratorTemplate class (not object)
        :return: None
        """
        if name in self.types:
            raise Exception(f"name: {name} already in generator types")
        else:
            self.types[name] = obj

    def registerAllGenerators(self):
        """
        This function registers all generators listed in the GeneratorTypes enum.
        :return: None
        """
        for generator in GeneratorTypes:
            self.registerNewGenerator(generator.name, generator.value)

    def getGenerator(self, name):
        """
        This function returns a specific generator
        :param name: str name of the generator
        :return: GeneratorTemplate (or its sub class)
        """
        if name not in self.types:
            raise Exception(f"name: {name} not in generator types")
        else:
            return self.types[name]

    def getAllGenerators(self):
        """
        This function returns the entire dictionary of generators.
        :return: dict {name: class} of generators.
        """
        return self.types
