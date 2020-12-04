"""
This file has several ways of constrained gradient descent approaches to adversarial example generation

Date:
    November 10, 2020

Project:
    Constrained Gradient Descent

Authors:
    name: Vineel Nagisetty, Laura Graves
    contact: vineel.nagisetty@uwaterloo.ca
"""

import time

import numpy as np
from tensorflow import GradientTape, sign, norm, Tensor
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
import tensorflow as tf

from src.generator.template import GeneratorTemplate
from src.utils import areSimilar


class CGDTemplate(GeneratorTemplate):
    """ This class is the template of a Genetic Algorithm. It needs to be extended and the various
    functions needs to be overridden."""

    def __init__(self, name, model, modelName, image, label, similarityType="l2", similarityMeasure=10, verbose=True):
        super().__init__(name, model, modelName, image, label, similarityType, similarityMeasure, verbose)
        self.advImage = tf.random.normal(self.image.shape, mean=0.0, stddev=1.0, dtype=tf.float32)
        self.updateEpsilon = 0.01
        self.convertedLabel = tf.reshape(tf.one_hot(self.label, 10), (1, 10))
        self.loss_object = tf.losses.CategoricalCrossentropy()

    def generateAdversarialExample(self):
        """
        This overrides the function from the GeneratorTemplate class.
        :return: None.
        """
        start_time = time.time()
        while True:
            self.advLabel = np.argmax(self.model.predict(self.advImage), axis=1)[0]
            if self.advLabel != self.label and areSimilar(self.image, self.advImage.numpy(),
                                                          similarityMeasure=self.similarityMeasure):
                break
            self.advImage = tf.clip_by_value(self.advImage + self.updateEpsilon * self.solve(), -1.0, 1.0)
        end_time = time.time()
        self.time = end_time - start_time
        self.advImage = self.advImage.numpy()
        self.completed = True

    def solve(self):
        with GradientTape() as tape:
            tape.watch(self.advImage)
            prediction = self.model(self.advImage)
            loss = self.cgdLoss(prediction)

        gradient = tape.gradient(loss, self.advImage)
        signed_grad = sign(gradient)
        return signed_grad

    def cgdLoss(self, prediction):
        loss = CategoricalCrossentropy()
        return loss(self.convertedLabel, prediction)


class CGDSimilar(CGDTemplate):
    def cgdLoss(self, prediction):
        loss = self.loss_object(self.convertedLabel, prediction) - norm(self.advImage - self.image, ord='euclidean')
        return loss
