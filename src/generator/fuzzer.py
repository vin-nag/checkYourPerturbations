"""
This file contains the stub for fuzzing code.

Date:
    June 5, 2020

Project:
    ECE653 Final Project

Authors:
    name: Vineel Nagisetty, Laura Graves, Joseph Scott
    contact: vineel.nagisetty@uwaterloo.ca
"""

from src.generator.factory import AbstractGenerator
import numpy as np
import torch

class Fuzzer(AbstractGenerator):
    def __init__(self, **kwargs):
        self.model = kwargs.model
        self.input = kwargs.input
        self.label = kwargs.label
        if "type" in kwargs:
            self.type = kwargs.type
        else:
            self.type = "random"

    def generateAdversarialExample(self, epsilon = 2/255, method="vin"):
        nIters = 0

        output = self.model(self.image.float())
        pred = output.data.max(1, keepdim=True)[1]

        if method == "step":
            fuzzimage = self.stepFuzz(epsilon, zero=True)
        if method == "norm":
            fuzzimage = self.normFuzz(epsilon)
        if method == "laplace":
            fuzzimage = self.laplaceFuzz(epsilon)
        if method == "vin":
            target = torch.tensor(np.random.randint(10, size=[64]))
            fuzzimage = self.vinFuzz(target, epsilon)

        fuzzoutput = self.model(fuzzimage.float())
        fuzzpred = fuzzoutput.data.max(1, keepdim=True)[1]

        correctclass = pred.eq(self.label.data.view_as(pred))
        adversarialclass = ~pred.eq(fuzzpred.data.view_as(pred))

        while (correctclass & adversarialclass).sum() < 1:

            nIters += 1

            if method == "step":
                fuzzimage = self.stepFuzz(epsilon, zero=True)
            if method == "norm":
                fuzzimage = self.normFuzz(epsilon)
            if method == "laplace":
                fuzzimage = self.laplaceFuzz(epsilon)
            if method == "vin":
                fuzzimage = self.vinFuzz(epsilon)

            fuzzoutput = self.model(fuzzimage.float())
            fuzzpred = fuzzoutput.data.max(1, keepdim=True)[1]

            correctclass = pred.eq(self.label.data.view_as(pred))
            adversarialclass = ~pred.eq(fuzzpred.data.view_as(pred))

    def stepFuzz(self, epsilon=2 / 255, zero=True):
        if zero:
            fuzzarray = np.random.randint(-1, 2, [64, 1, 28, 28])
        else:
            fuzzarray = np.random.choice([-1, 1], [64, 1, 28, 28])
        fuzzimage = torch.clamp((self.image + epsilon * fuzzarray), -1, 1)
        return fuzzimage

    def normFuzz(self, epsilon=2 / 255):
        fuzzarray = np.random.normal(0, epsilon, [64, 1, 28, 28])
        fuzzimage = torch.clamp((self.image + fuzzarray), -1, 1)
        return fuzzimage

    def laplaceFuzz(self, epsilon=2 / 255):
        fuzzarray = np.random.laplace(0, epsilon, [64, 1, 28, 28])
        fuzzimage = np.clip((self.image + fuzzarray), -1, 1)
        return fuzzimage

    # adds fuzz to an image based on the Nagisetty method
    def vinFuzz(self, target, epsilon=2 / 255, numIters=20):
        criterion = torch.nn.functional.nll_loss
        i = 0
        lower = np.clip(self.image - epsilon, -1, 1)
        upper = np.clip(self.image + epsilon, -1, 1)
        # set perturbation ranges
        rand = torch.rand(self.image.size())
        newimage = rand * (upper - lower) + lower
        newimage = torch.autograd.Variable(newimage.data.float(), requires_grad=True)
        while (i < numIters):
            newpred = self.model(newimage)
            loss = criterion(newpred, target)
            loss.backward()
            grad = newimage.grad.sign()
            for i in range(len(grad)):
                for j in range(len(grad[0, 0])):
                    for k in range(len(grad[0, 0, 0])):
                        if grad[i, 0, j, k] == 1:
                            lower[i, 0, j, k] = newimage[i, 0, j, k]
                        if grad[i, 0, j, k] == -1:
                            upper[i, 0, j, k] = newimage[i, 0, j, k]
            rand = torch.rand(self.image.size())
            newimage = rand * (upper - lower) + lower
            newimage = torch.autograd.Variable(newimage.data.float(), requires_grad=True)
        return newimage
