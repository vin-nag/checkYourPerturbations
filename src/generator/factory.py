"""
This file implements the factory design pattern to create generator classes.

Date:
    July 10, 2020

Project:
    ECE653 Final Project

Authors:
    name: Vineel Nagisetty, Laura Graves, Joseph Scott
    contact: vineel.nagisetty@uwaterloo.ca
"""

from enum import Enum
from src.generator.fuzzer import Fuzzer


class GeneratorTypes(Enum):
    FUZZ = Fuzzer


class GeneratorSelector:
    """ This class registers new generators and retrieves them """

    def __init__(self):
        self.types = {}
        self.registerAllGenerators()

    def registerNewGenerator(self, name, obj):
        if name in self.types:
            raise Exception(f"name: {name} already in generator types")
        else:
            self.types[name] = obj

    def registerAllGenerators(self):
        for generator in GeneratorTypes:
            self.types[generator.name] = generator.value

    def getGenerator(self, name):
        if name not in self.types:
            raise Exception(f"name: {name} not in generator types")
        else:
            return self.types[name]

