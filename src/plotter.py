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
import numpy as np


def getTimesFromDataFrame(df):
    """
    This function takes in a DataFrame object and returns the relevant information as a dictionary
    :param df: Pandas.DataFrame
    :return: dictionary
    """
    d = {}
    for name in df.generatorName.unique():
        tmp = df.loc[df['generatorName'] == name]['time']
        lst = np.array([x for _,x in tmp.iteritems()])
        d[name] = np.cumsum(lst)
    return d


def createCactusPlot(df, title="Cactus Plot of Adversarial Generators", fname=None):
    """
    This function should create a cactus plot given a dataframe.
    :param df: Pandas.DataFrame
    :param title: string
    :param fname: string
    :return: None
    """
    plt.title(title)
    plt.xlabel("Number of Instances (#)")
    plt.ylabel("Time Taken (seconds)")

    d = getTimesFromDataFrame(df)

    for key in d.keys():
        plt.plot(d[key], label=key, marker='x')

    plt.grid(False)
    plt.legend()

    if fname is not None:
        plt.savefig(fname)
    plt.show()


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
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img1.reshape(28, 28), cmap='Greys_r', vmin=-0.5, vmax=0.5)
    plt.xlabel(label1)
    plt.subplot(1, 2, 2)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img2.reshape(28, 28), cmap='Greys_r', vmin=-0.5, vmax=0.5)
    plt.xlabel(label2)
    plt.title(f"Plotting Images for {name1} (left) and {name2} (right)")
    plt.show()


def plotImage(img):
    """
    This function plots a single image
    :param img: numpy array of input to the model
    :return: None
    """
    plt.figure(figsize=(8, 4))
    plt.grid(False)
    plt.imshow(img)
    plt.title(f"Plotting Image")
    plt.show()
