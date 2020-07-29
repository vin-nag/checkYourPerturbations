"""
This file runs the evaluation of various generators on a given benchmark.

Date:
    July 5, 2020

Project:
    ECE653 Final Project: Check Your Perturbations

Authors:
    name: Vineel Nagisetty, Laura Graves, Joseph Scott
    contact: vineel.nagisetty@uwaterloo.ca
"""

from src.plotter import displayPerturbedImages, createCactusPlot, displayPerturbedImagesDF
from src.utils import calculateSimilarity
from func_timeout import func_timeout, FunctionTimedOut
import pandas as pd


class Evaluator:
    """ This class performs the evaluation of various generators on a given benchmark. """

    def __init__(self, benchmark, generators, timeLimit=25, similarityType="l2", similarityMeasure=10, verbose=True):
        """
        Standard init function.
        :param benchmark: pandas dataframe with each row consisting of <model, image, label> data.
        :param generators: dictionary consisting of <generator name: generator class>
        """
        self.benchmark = benchmark
        self.generators = generators
        self.timeLimit = timeLimit
        self.similarityType = similarityType
        self.similarityMeasure = similarityMeasure
        self.verbose = verbose

    @staticmethod
    def runGenerator(generator, timeLimit=25, verbose=False):
        """
        This wrapper function runs the generateAdversarialExample on each generator
        :param generator: GeneratorTemplate object
        """
        # adding thread scope value so that keras works
        import keras.backend.tensorflow_backend as tb
        tb._SYMBOLIC_SCOPE.value = True

        try:
            generator.generateAdversarialExample()
        except Exception as e:
            generator.time = timeLimit * 2
            if verbose:
                print(f"\t\tResult: Error ({e})")

    def evaluateEach(self, generator):
        """
        This function calculates the par2scores for a generator.
        :param generator: GeneratorTemplate object
        :param timeMax: float timeout
        :param similarityLimit: float default: 10
        :param similarityType: str default: "l2"
        :param verbose: bool default: true
        :return: tuple of (time taken, fuzzed image, fuzzed prediction). Note the last two elements are None if timeout.
        """
        # create new process to measure time taken.
        try:
            func_timeout(timeout=self.timeLimit, func=Evaluator.runGenerator, args=(generator, self.timeLimit,
                                                                                    self.verbose))
            if generator.completed:
                generator.similarity = calculateSimilarity(generator.advImage, generator.image,
                                                           generator.similarityType)
                # verify results
                assert generator.label != generator.advLabel, "perturbed image label is the same as the original image."
                assert generator.similarity < self.similarityMeasure, "perturbed image is not similar to original."

                if self.verbose:
                    print(f"\t\tResult: new label: {generator.advLabel}, time: {round(generator.time, 4)}, "
                          f"similarity: {round(generator.similarity, 4)}.")

        except FunctionTimedOut:
            generator.time = self.timeLimit * 2
            if self.verbose:
                print(f"\t\tResult: timed out.")

    def evaluate(self, display=False):
        """
        This function performs evaluation and records par2scores and similarity measures.
        :return: None
        """
        results = pd.DataFrame(columns=('generatorName', 'modelName', 'image', 'label',
                                        'advImage', 'advLabel', 'time', 'similarity', 'completed'))
        i = 0
        for generatorName in self.generators:
            if self.verbose:
                print(f"Starting evaluation for {generatorName}:")
            for index, row in self.benchmark.getData().iterrows():
                if self.verbose:
                    print(f"\tEvaluating model: {row['modelName']} for true label: {row['label']}")
                # initialize generator object
                generatorObj = self.generators[generatorName](name=generatorName, model=row['model'],
                                                              modelName=row['modelName'], image=row['image'].copy(),
                                                              label=row['label'], similarityType=self.similarityType,
                                                              similarityMeasure=self.similarityMeasure)
                # run evaluation on each generator
                self.evaluateEach(generator=generatorObj)
                if display:
                    if generatorObj.completed:
                        displayPerturbedImages(generatorObj.image, "original", generatorObj.label,
                                               generatorObj.advImage, generatorName, generatorObj.advLabel)

                results.loc[i] = [generatorName, row['modelName'], generatorObj.image, generatorObj.label,
                                  generatorObj.advImage, generatorObj.advLabel, generatorObj.time,
                                  generatorObj.similarity, generatorObj.completed]
                i += 1
        if self.verbose:
            print("Completed Evaluation.")

        createCactusPlot(df=results, size=self.benchmark.numImages, timeout=self.timeLimit)
        displayPerturbedImagesDF(df=results)
        return results
