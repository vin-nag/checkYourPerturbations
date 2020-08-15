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

from src.utils import calculateSimilarity, saveDF
from func_timeout import func_timeout, FunctionTimedOut
import pandas as pd
from datetime import date
from tqdm import tqdm


class Evaluator:
    """ This class performs the evaluation of various generators on a given benchmark. """

    def __init__(self, benchmark, generators, timeLimit=25, verbose=False):
        """
        Standard init function.
        :param benchmark: pandas dataframe with each row consisting of <model, image, label> data.
        :param generators: dictionary consisting of <generator name: generator class>
        """
        self.benchmark = benchmark
        self.generators = generators
        self.timeLimit = timeLimit
        self.similarityType = self.benchmark.similarityType
        self.similarityMeasure = self.benchmark.similarityMeasure
        self.verbose = verbose
        self.date = date.today().strftime("%b-%d-%Y")

    @staticmethod
    def runGenerator(generator, timeLimit=25, verbose=False):
        """
        This wrapper function runs the generateAdversarialExample on each generator
        :param generator: GeneratorTemplate object
        :param timeLimit: float time limit
        :param verbose: bool whether to display debug statements
        """
        # adding thread scope value so that keras works
        import keras.backend.tensorflow_backend as tb
        tb._SYMBOLIC_SCOPE.value = True

        try:
            generator.generateAdversarialExample()
        except Exception as e:
            generator.time = 2 * timeLimit
            if verbose:
                print(f"\t\tResult: Error ({e})")

    def evaluateEach(self, generator):
        """
        This function calculates the par2scores for a generator.
        :param generator: GeneratorTemplate object
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
                    tqdm.write(f"\t\tResult: new label: {generator.advLabel}, time: {round(generator.time, 4)}, "
                               f"similarity: {round(generator.similarity, 4)}.")

        except FunctionTimedOut:
            generator.time = 2 * self.timeLimit
            generator.similarity = 2 * self.similarityMeasure
            if self.verbose:
                tqdm.write(f"\t\tResult: timed out.")

    def evaluate(self):
        """
        This function performs evaluation and records par2scores and similarity measures.
        :return: None
        """
        i = 0
        with tqdm(total=self.benchmark.data.shape[0] * len(self.generators)) as progressBar:
            for generatorName in self.generators:
                results = pd.DataFrame(columns=('generatorName', 'modelName', 'image', 'label',
                                                'advImage', 'advLabel', 'time', 'similarity', 'completed'))
                for index, row in self.benchmark.getData().iterrows():
                    progressBar.set_description(f"\tEvaluating generator: {generatorName} on model: {row['modelName']} "
                                                f"for true label: {row['label']}")
                    # initialize generator object
                    generatorObj = self.generators[generatorName](name=generatorName, model=row['model'],
                                                                  modelName=row['modelName'], image=row['image'].copy(),
                                                                  label=row['label'],
                                                                  similarityType=self.similarityType,
                                                                  similarityMeasure=self.similarityMeasure,
                                                                  verbose=self.verbose)
                    # run evaluation on each generator
                    self.evaluateEach(generator=generatorObj)

                    results.loc[i] = [generatorName, row['modelName'], generatorObj.image, generatorObj.label,
                                      generatorObj.advImage, generatorObj.advLabel, generatorObj.time,
                                      generatorObj.similarity, generatorObj.completed]
                    i += 1
                    progressBar.update()

                saveDF(f"./../src/results/data/{self.benchmark.name}/{self.date}/", f"{generatorName}.pkl", results)

        if self.verbose:
            print("Completed Evaluation.")
        return
