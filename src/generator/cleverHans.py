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
from cleverhans.future.tf2.attacks import fast_gradient_method, basic_iterative_method
from src.utils import areSimilar
import tensorflow as tf
import numpy as np


class CleverHansTemplate(GeneratorTemplate):
    """ This class is the template of a CleverHansAttack. It needs to be extended."""

    def advStep(self,  model_fn, image, epsilon, clip_min=-1, clip_max=1, norm=2):
        """ This function needs to be overridden by the classes extending CleverHansTemplate. """
        return self.image

    def generateAdversarialExample(self, epsilon=0.2):
        """
        This overrides the function from the GeneratorTemplate class.
        :param epsilon: the value of each perturbation step.
        :return: np.array representing adversarial image.
        """
        advImage = self.image
        logits_model = tf.keras.Model(self.model.input, self.model.layers[-1].output)
        if self.similarityType == "l2":
            norm = 2
        elif self.similarityType == "l1":
            norm = 1
        else:
            norm = np.inf
        while True:
            advPrediction = np.argmax(self.model.predict(advImage), axis=1)[0]
            if advPrediction != self.label and areSimilar(self.image, advImage.numpy()):
                break
            advImage = self.advStep(model_fn=logits_model, image=advImage, epsilon=epsilon, norm=norm)
        return advImage.numpy()


class BIM(CleverHansTemplate):
    """ This class implements the Basic Iterative Method (using CleverHans)."""
    def advStep(self,  model_fn, image, epsilon, clip_min=-1, clip_max=1, norm=2):
        return basic_iterative_method(model_fn, image, self.similarityMeasure, epsilon, 20, norm, clip_min, clip_max,
                                      sanity_checks=False)


class FGSM(CleverHansTemplate):
    """ This class implements the Fast Gradient Sign Method (using CleverHans)."""
    def advStep(self,  model_fn, image, epsilon, clip_min=-1, clip_max=1, norm=2):
        return fast_gradient_method(model_fn, image, epsilon, norm, clip_min, clip_max)
