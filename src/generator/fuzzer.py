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
from src.utils import areSimilar
import time


class Fuzzer(GeneratorTemplate):
    """ This class is the template of a Fuzzer. It needs to be extended and the fuzzStep
    function needs to be overridden."""

    def fuzzStep(self, image, epsilon, iters):
        """ This function needs to be overridden by the classes extending Fuzzer. """
        return self.image

    def generateAdversarialExample(self):
        """
        This overrides the function from the GeneratorTemplate class.
        :param epsilon: the value of each fuzzing step.
        :return: np.array representing fuzzed image.
        """
        start_time = time.time()
        i = 0
        self.advImage = self.image.copy()
        epsilon = np.sqrt((self.similarityMeasure**2)/784)*0.99
        # print("sim", self.similarityMeasure, "eps", epsilon)
        while True:
            self.advLabel = np.argmax(self.model.predict(self.advImage), axis=1)[0]
            i += 1
            if self.advLabel != self.label and areSimilar(self.image, self.advImage,
                                                          similarityMeasure=self.similarityMeasure):
                break
            if i > 500:
                self.advImage = self.image.copy()
                if self.verbose:
                    print("\t\treset image")
                i = 0
            else:
                self.advImage = self.fuzzStep(self.advImage, epsilon=epsilon, iters=i)
        end_time = time.time()
        self.time = end_time - start_time
        self.completed = True


class StepFuzzer(Fuzzer):
    """ This class extends the Fuzzer class. """

    def fuzzStep(self, image, epsilon, iters):
        fuzzArray = np.random.randint(-1, 2, self.imageShape)
        return np.clip((image + epsilon * fuzzArray), -0.5, 5)


class NormFuzzer(Fuzzer):
    """ This class extends the Fuzzer class. """

    def fuzzStep(self, image, epsilon, iters):
        """
        This method overrides the function in Fuzzer class.
        :param image: np.array of the image to fuzz
        :param epsilon: float the step size to fuzz each time
        :return: np.array the fuzzed image.
        """
        fuzzArray = np.random.normal(0, epsilon, self.imageShape)
        return np.clip((image + fuzzArray), -0.5, 5)


class LaplaceFuzzer(Fuzzer):
    """ This class extends the Fuzzer class. """

    def fuzzStep(self, image, epsilon, iters):
        """
        This method overrides the function in Fuzzer class.
        :param image: np.array of the image to fuzz
        :param epsilon: float the step size to fuzz each time
        :return: np.array the fuzzed image.
        """
        fuzzArray = np.random.laplace(0, epsilon, self.imageShape)
        return np.clip((image + fuzzArray), -0.5, 5)


class VinFuzzer(Fuzzer):
    """ This class extends the Fuzzer class. """

    def __init__(self, name, model, modelName, image, label, similarityType="l2", similarityMeasure=10, verbose=True):
        super().__init__(name, model, modelName, image, label, similarityType, similarityMeasure, verbose)
        self.lowerBound = np.clip(self.image - 0.2, -1, 1)
        self.upperBound = np.clip(self.image + 0.2, -1, 1)

    def fuzzStep(self, image, epsilon, iters):
        """
        This method overrides the function in Fuzzer class.
        :param image: np.array of the image to fuzz
        :param epsilon: float the step size to fuzz each time
        :return: np.array the fuzzed image.
        """
        # set perturbation ranges
        lossfn = losses.categorical_crossentropy
        rand = np.random.random_sample(image.shape)
        newImage = tf.convert_to_tensor(rand * (self.upperBound - self.lowerBound) + self.lowerBound,
                                        dtype=np.float32)
        with tf.GradientTape() as tape:
            tape.watch(newImage)
            newpred = tf.squeeze(self.model(newImage))
            loss = lossfn(newpred, utils.to_categorical(self.label, num_classes=10))
        grad = tape.gradient(loss, newImage)
        self.lowerBound = np.clip(newImage + grad, self.lowerBound, 1)
        self.upperBound = np.clip(newImage - grad, -1, self.upperBound)
        if iters > 500 or np.equal(self.lowerBound, self.upperBound).any():
            self.lowerBound = np.clip(self.image - 0.2, -1, 1)
            self.upperBound = np.clip(self.image + 0.2, -1, 1)
            if self.verbose:
                print("\t\treset bounds.")
        return newImage.numpy()
