"""
This file contains the template of an abstract generator object.

Date:
    July 5, 2020

Project:
    ECE653 Final Project: Check Your Perturbations

Authors:
    name: Vineel Nagisetty, Laura Graves, Joseph Scott
    contact: vineel.nagisetty@uwaterloo.ca
"""

from src.generator.template import GeneratorTemplate
import tensorflow as tf
from tensorflow.keras import utils, losses
import numpy as np


class Fuzzer(GeneratorTemplate):
    """ This class is the template of a Fuzzer. It needs to be extended and the fuzzStep
    function needs to be overridden."""

    def fuzzStep(self, image, epsilon):
        """ This function needs to be overridden by the classes extending Fuzzer. """
        return self.image

    def generateAdversarialExample(self, epsilon=5/255):
        """
        This overrides the function from the GeneratorTemplate class.
        :param epsilon: the value of each fuzzing step.
        :return: np.array representing fuzzed image.
        """
        fuzzImage = self.image
        while True:
            fuzzPrediction = np.argmax(self.model.predict(fuzzImage), axis=1)[0]
            if fuzzPrediction != self.label:
                break
            fuzzImage = self.fuzzStep(fuzzImage, epsilon=epsilon)
        return fuzzImage


class StepFuzzer(Fuzzer):
    """ This class extends the Fuzzer class. """

    def fuzzStep(self, image, epsilon):
        fuzzArray = np.random.randint(-1, 2, self.imageShape)
        return np.clip((image + epsilon * fuzzArray), -1, 1)


class NormFuzzer(Fuzzer):
    """ This class extends the Fuzzer class. """

    def fuzzStep(self, image, epsilon):
        """
        This method overrides the function in Fuzzer class.
        :param image: np.array of the image to fuzz
        :param epsilon: float the step size to fuzz each time
        :return: np.array the fuzzed image.
        """
        fuzzArray = np.random.normal(0, epsilon, self.imageShape)
        return np.clip((image + fuzzArray), -1, 1)


class LaplaceFuzzer(Fuzzer):
    """ This class extends the Fuzzer class. """

    def fuzzStep(self, image, epsilon):
        """
        This method overrides the function in Fuzzer class.
        :param image: np.array of the image to fuzz
        :param epsilon: float the step size to fuzz each time
        :return: np.array the fuzzed image.
        """
        fuzzArray = np.random.laplace(0, epsilon, self.imageShape)
        return np.clip((image + fuzzArray), -1, 1)


class VinFuzzer(Fuzzer):
    """ This class extends the Fuzzer class. """

    def fuzzStep(self, image, epsilon, numIters=20):
        """
        This method overrides the function in Fuzzer class.
        :param numIters: int the number of iterations to perform this fuzzing
        :param image: np.array of the image to fuzz
        :param epsilon: float the step size to fuzz each time
        :return: np.array the fuzzed image.
        """
        i = 0
        lower = np.clip(image - epsilon, -1, 1)
        upper = np.clip(image + epsilon, -1, 1)
        # set perturbation ranges
        lossfn = losses.CategoricalCrossentropy()
        while i < numIters:
            rand = np.random.random_sample(image.shape)
            newImage = tf.convert_to_tensor(rand * (upper - lower) + lower, dtype=np.float32)
            with tf.GradientTape() as tape:
                tape.watch(newImage)
                newpred = tf.squeeze(self.model(newImage))
                loss = lossfn(newpred, utils.to_categorical(self.label, num_classes=10))
            grad = tape.gradient(loss, newImage)
            grad = tf.sign(grad)
            for i in range(len(grad)):
                for j in range(len(grad[0, 0])):
                    for k in range(len(grad[0, 0, 0])):
                        if grad[i, 0, j, k] == 1:
                            lower[i, 0, j, k] = newImage[i, 0, j, k]
                        if grad[i, 0, j, k] == -1:
                            upper[i, 0, j, k] = newImage[i, 0, j, k]
        return newImage.numpy()
