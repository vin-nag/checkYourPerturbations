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
from cleverhans.future.tf2.attacks import fast_gradient_method
from src.utils import areSimilar
import tensorflow as tf


class CleverHansTemplate(GeneratorTemplate):
    """ This class is the template of a CleverHansAttack. It needs to be extended."""

    @staticmethod
    def method(model_fn, x, eps, norm, clip_min, clip_max):
        return x

    def generateAdversarialExample(self):
        """
        This overrides the function from the GeneratorTemplate class.
        :return: np.array representing fuzzed image.
        """
        logits_model = tf.keras.Model(self.model.input, self.model.layers[-1].output)
        advImage = self.method(logits_model, self.image, eps=0.2, norm=2, clip_min=-1, clip_max=1)
        assert self.label != self.model.predict(advImage), "perturbed image label is the same as the original image."
        assert areSimilar(advImage, self.image, self.similarityType, self.similarityMeasure), \
            "perturbed image is not similar to original."
        return advImage



