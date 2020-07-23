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
import pdb
from src.generator.template import GeneratorTemplate
import tensorflow as tf

class SymbolicExecutioner(GeneratorTemplate):
    """ This class is the stub for our symbolic execution implementation."""
    def __init__(self, name, model, image, label, similarityType="l2", similarityMeasure=10):
        super().__init__(name, model, image, label, similarityType, similarityMeasure)
        self.weights = []
        self.biases = []
        self._parse_model_mlp()
    def _parse_model_mlp(self):
        self.model = tf.keras.models.load_model('data/models/mnist_model')
        for layer in self.model.layers:
            print(layer.weights)
            if len(layer.weights) == 0: continue
            else:
                print(len(layer.weights[0].numpy()), len(layer.weights[1].numpy()))
                self.weights.append(layer.weights[0].numpy())
                self.biases.append(layer.weights[1].numpy())
    def generateAdversarialExample(self):
        """ This method needs to be implemented. TODO: Implement this method. """
        self.solve()
