"""
This file contains the templates of abstract (and concrete) generator objects that use genetic algorithms. Code from
https://github.com/ptyshevs/ga_adv/blob/master/GeneticSolver.py

Date:
    August 5, 2020

Project:
    ECE653 Final Project: Check Your Perturbations

Authors:
    name: Vineel Nagisetty, Laura Graves, Joseph Scott
    contact: vineel.nagisetty@uwaterloo.ca
"""

import numpy as np
from src.utils import calculateSimilarity


def standardPixelSwapCrossover(parentOne, parentTwo, shape):
    """
    This stub function is intended to perform a cross over to produce two children given two parents. This needs to
    be overridden.
    :param parentOne: np.array representing parent one
    :param parentTwo: np.array representing parent two
    :param shape: np.array shape of the image
    :return (np.array, np.array) representing two children
    """
    parentOne = np.expand_dims(parentOne, axis=0)
    parentTwo = np.expand_dims(parentTwo, axis=0)
    select_mask = np.random.binomial(1, 0.5, size=shape).astype('bool')
    child1, child2 = np.copy(parentOne), np.copy(parentTwo)
    child1[select_mask] = parentTwo[select_mask]
    child2[select_mask] = parentOne[select_mask]
    return child1, child2


def pixelIntensityMutation(individual, shape):
    """
    This function performs the mutation for a given individual. This needs to be overridden.
    :param individual: np.array representing the image
    :return np.array representing the mutated individual
    """
    individual = np.expand_dims(individual, axis=0)
    a = np.random.binomial(1, 0.1, size=shape).astype('bool')
    individual[a] = np.clip(individual[a] + np.random.randn(*individual[a].shape) * 0.1, 0, 1)
    return individual


def evaluateSimilarityFitness(model, image, population, label):
    """
    This function overrides from the Genetic class
    :param model: tensorflow Model
    :param image image: np.array of original image
    :param population: np.array of population maintained by genetic algorithm
    :param label: int true label of the original image
    :return np.array representing the fitness scores over the population
    """
    y_targets = model.predict(population)[:, label]
    similarities = [calculateSimilarity(image, population[x]) for x in range(population.shape[0])]
    # fitness = prediction on true label * similarity to original image
    return -0.5 * y_targets + 0.5 * np.array(similarities)
