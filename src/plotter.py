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
        lst = np.array([x for _, x in tmp.iteritems()])
        d[name] = np.cumsum(lst)
    return d


def createCactusPlot(df, size=3, timeout=25, fname=None):
    """
    This function should create a cactus plot given a dataframe.
    :param df: Pandas.DataFrame
    :param title: string
    :param fname: string
    :return: None
    """
    plt.title(f"Cactus Plot of Adversarial Generators (Timeout: {timeout})")
    plt.xlabel("Number of Instances (#)")
    plt.ylabel("Time Taken (seconds)")

    d = getTimesFromDataFrame(df)
    xAxis = [x for x in range(1, size+1)]
    plt.xticks(xAxis)

    for key in d.keys():
        plt.plot(xAxis, d[key], label=key, marker='x')

    plt.grid(False)
    plt.legend()

    if fname is not None:
        plt.savefig(fname)
    plt.show()


def plotEachModel(modelName, image, origLabel, lst):
    """
    This function plots the generated images for each model.
    :param: modelName: string
    :param: image: numpy ndarray
    :param: origLabel: string
    :param: lst: List of lists
    :return: None
    """
    fig, axs = plt.subplots(1, len(lst) + 1, figsize=(10, 4))
    fig.suptitle(f"Plotting Generated Adversarial Images for Model Name: {modelName}", fontsize=14)

    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].grid(False)
    axs[0].set_title("Original")
    axs[0].imshow(image.squeeze(), cmap='Greys_r', vmin=-0.5, vmax=0.5)
    axs[0].set_xlabel(f"label:{origLabel}")

    for i in range(len(lst)):
        axs[i + 1].set_xticks([])
        axs[i + 1].set_yticks([])
        axs[i + 1].grid(False)
        axs[i + 1].set_title(f"{lst[i][0]}")
        axs[i + 1].imshow(lst[i][1].squeeze(), cmap='Greys_r', vmin=-0.5, vmax=0.5)
        axs[i + 1].set_xlabel(f"label:{lst[i][2]}, sim:{round(lst[i][3], 2)}")
    plt.show()


def displayPerturbedImagesDF(df):
    """
    This function should display the perturbed images given a dataframe.
    :param df: Pandas.DataFrame
    :return:
    """
    for name in df.modelName.unique():
        tmp = df[df['modelName'] == name]
        for label in tmp.label.unique():
            lst = []
            newTmp = tmp[tmp['label'] == label]
            image = newTmp.iloc[0]['image']
            for i, row in newTmp.iterrows():
                if row['completed']:
                    lst.append([row['generatorName'], row['advImage'], row['advLabel'], row['similarity']])
            plotEachModel(name, image, label, lst)


def displayPerturbedImages(img1, name1, label1, img2, name2, label2, fname=None):
    """
    This function should display the perturbed images given some details.
    :return: None
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
