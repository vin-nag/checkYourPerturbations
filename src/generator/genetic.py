"""
This file contains the templates of abstract (and concrete) generator objects that use genetic algorithms. Code from
https://github.com/ptyshevs/ga_adv/blob/master/GeneticSolver.py but modified

Date:
    August 5, 2020

Project:
    ECE653 Final Project: Check Your Perturbations

Authors:
    name: Vineel Nagisetty, Laura Graves, Joseph Scott
    contact: vineel.nagisetty@uwaterloo.ca
"""

from src.generator.template import GeneratorTemplate
from src.utils import calculateSimilarity, areSimilar
import numpy as np
import time


class Genetic(GeneratorTemplate):
    """ This class is the template of a Genetic Algorithm. It needs to be extended and the various
    functions needs to be overridden."""

    def __init__(self, name, model, modelName, image, label, similarityType="l2", similarityMeasure=10, verbose=True):
        super().__init__(name, model, modelName, image, label, similarityType, similarityMeasure, verbose)
        self.numGenerations = 50
        self.populationSize = 1000
        self.mutationRate = 0.1
        self.retainBest = 0.6
        self.advImage = self.image.copy()
        self.population = self.generateInitialPopulation(self.populationSize)
        self.scores = None

    def generateInitialPopulation(self, number):
        """
        Generating initial population of individual solutions
        :return: np.array initial population as an array
        """
        rand = np.random.rand(number, self.imageShape[1], self.imageShape[2], self.imageShape[3])
        repeats = np.repeat(self.image, self.populationSize, axis=0)
        return repeats + rand

    def generateAdversarialExample(self):
        """
        This overrides the function from the GeneratorTemplate class.
        :return: None.
        """
        start_time = time.time()
        while True:
            self.advLabel = np.argmax(self.model.predict(self.advImage), axis=1)[0]
            if self.advLabel != self.label and areSimilar(self.image, self.advImage,
                                                          similarityMeasure=self.similarityMeasure):
                break
            self.advImage = self.evolve() + self.image
        end_time = time.time()
        self.time = end_time - start_time
        self.completed = True

    def evolve(self):
        """
        This function performs genetic algorithm evolution
        """
        for i in range(self.numGenerations):
            self.scores = self.evaluateFitness()

            # keep the best individuals
            retain_len = int(len(self.scores) * self.retainBest)
            sorted_indices = np.argsort(self.scores)
            self.population = self.population[sorted_indices]
            parents = self.population[:retain_len]

            # mutate all individuals (except for the top ten)
            for idx in range(10, retain_len):
                if np.random.rand() < self.mutationRate:
                    self.mutate(parents[idx])

            # fill up the remaining population by recombination to create children
            places_left = self.populationSize - retain_len
            children = []
            while len(children) < places_left:
                parentOne, parentTwo = np.random.choice(retain_len - 1, size=2, replace=False)
                child1, child2 = self.crossover(parents[parentOne], parents[parentTwo])
                children.append(child1)
                if len(children) < places_left:
                    children.append(child2)

            # add the values to population
            children = np.array(children)
            parents = np.append(parents, children, axis=0)
            self.population = parents
        return self.population[0]

    def crossover(self, parentOne, parentTwo):
        """
        This stub function is intended to perform a cross over to produce two children given two parents. This needs to
        be overridden.
        :param parentOne: np.array representing parent one
        :param parentTwo: np.array representing parent two
        :return (np.array, np.array) representing two children
        """
        return parentOne, parentTwo

    def mutate(self, individual):
        """
        This function performs the mutation for a given individual. This needs to be overridden.
        :param individual: np.array representing the image
        :return np.array representing the mutated individual
        """
        return individual

    def evaluateFitness(self):
        """
        This function performs evaluation over the population
        :return np.array representing the fitness scores over the population
        """
        return self.scores


class SimilarGenetic(Genetic):
    """ This class extends the Genetic class. """

    def crossover(self, parentOne, parentTwo):
        """
        This function implements a standard 2 point crossover to produce two children given two parents.
        :param parentOne: np.array representing parent one
        :param parentTwo: np.array representing parent two
        :return (np.array, np.array) representing two children
        """
        parentOneCopy = parentOne.flatten()
        parentTwoCopy = parentTwo.flatten()
        bounds = np.sort(np.random.choice(parentOneCopy.shape[0], size=2, replace=False))
        child1, child2 = np.copy(parentOneCopy), np.copy(parentTwoCopy)
        child1[bounds[0]: bounds[1]] = parentTwoCopy[bounds[0]: bounds[1]]
        child2[bounds[0]: bounds[1]] = parentOneCopy[bounds[0]: bounds[1]]
        return child1.reshape(self.imageShape[1:]), child2.reshape(self.imageShape[1:])

    def mutate(self, individual):
        """
        This function performs the mutation for a given individual. It randomly selects pixels and perturbs them
        :param individual: np.array representing the image
        :return np.array representing the mutated individual
        """
        individual = np.expand_dims(individual, axis=0)
        a = np.random.normal(1, 0.1, size=self.imageShape).astype('bool')
        individual[a] = np.clip(individual[a] + np.random.randn(*individual[a].shape) * 0.1, -1, 1)
        return individual

    def evaluateFitness(self):
        """
        This function overrides from the Genetic class. It calculates fitness as
        :return np.array representing the fitness scores over the population
        """
        yTargets = self.model.predict(self.population)[:, self.label]
        similarityDistance = np.array([calculateSimilarity(self.image, self.population[x]) for x in
                                 range(self.population.shape[0])])
        return yTargets * similarityDistance
