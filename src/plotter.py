"""
This file plots the results of evaluation.

Date:
    July 5, 2020

Project:
    ECE653 Final Project: Check Your Perturbations

Authors:
    name: Vineel Nagisetty, Laura Graves, Joseph Scott
    contact: vineel.nagisetty@uwaterloo.ca
"""

import matplotlib.pyplot as plt


def createCactusPlot(df):
    """
    This function should create a cactus plot given a dataframe.
    :param df:
    :return:
    """
    # TODO: Implement this.
    pass


def displayPerturbedImagesDF(df):
    """
    This function should display the perturbed images given a dataframe.
    :param df:
    :return:
    """
    # TODO: Implement this.
    pass


def displayPerturbedImages(img1, name1, label1, img2, name2, label2, fname=None):
    """
    This function should display the perturbed images given some details.
    :return:
    """
    # TODO: Implement this.
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img1.reshape(28, 28))
    plt.xlabel(label1)
    plt.subplot(1, 2, 2)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img2.reshape(28, 28))
    plt.xlabel(label2)
    plt.title(f"Plotting Images for {name1} (left) and {name2} (right)")
    plt.show()
