"""
This file contains the stub for the evaluator code.

Date:
    June 5, 2020

Project:
    ECE653 Final Project

Authors:
    name: Vineel Nagisetty, Laura Graves, Joseph Scott
    contact: vineel.nagisetty@uwaterloo.ca
"""
from src.benchmark import Benchmark
from multiprocessing import Process
import time
from func_timeout import func_timeout, FunctionTimedOut
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd


class Evaluator:
    def __init__(self, benchmark, generators):
        self.benchmark = benchmark
        self.generators = generators
        self.results = pd.DataFrame(columns=('generator', 'model', 'image', 'label', 'perturbed image',
                                             'perturbed label', 'time'))

    def evaluate(self, timeMax=1000, similarityType="l2", similarityScore=1):
        i = 0
        for generator in self.generators:
            for index, row in self.benchmark.getData().iterrows():
                print(row)
                timeTaken, perturbedImg = self.par2scores(generator=generator, model=row['model'], img=row['image'],
                                                          trueLabel=row['label'], timeMax=timeMax)
                self.results.loc[i] = [generator, row['model'], row['image'], row['label'], perturbedImg,
                                       row['model'](perturbedImg), timeTaken]

    @staticmethod
    def par2scores(generator, model, img, trueLabel, timeMax):
        try:
            start_time = time.time()
            perturbedImg = func_timeout(timeout=timeMax, func=generator.generateAdversarialExample,
                                        args=(model, img))

            assert trueLabel != model(perturbedImg) and Evaluator.areSimilar(perturbedImg, img)
            timeTaken = time.time() - start_time
            return timeTaken, perturbedImg

        except FunctionTimedOut:
            return timeMax, None

    @staticmethod
    def areSimilar(img1, img2, similarityType="l2", similarityScore=1):
        img1 = normalize(img1.astype(np.float64).flatten(), norm=similarityType, axis=1)
        img2 = normalize(img2.astype(np.float64).flatten(), norm=similarityType, axis=1)
        return np.linalg.norm(img1 - img2) < similarityScore



