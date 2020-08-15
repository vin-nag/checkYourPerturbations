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

from prettytable import PrettyTable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import sys
import os


def getSimilaritiesFromDataFrame(df, similarityLimit=10):
    """
    This function takes in a DataFrame object and returns the relevant information as a dictionary
    :param df: Pandas.DataFrame
    :param similarityLimit int the similarity limit of the benchmark
    :return: dictionary
    """
    d = {}
    for name in df.generatorName.unique():
        tmp = df.loc[df['generatorName'] == name]['similarity']
        lst = np.array([x if x is not None else 2 * similarityLimit for _, x in tmp.iteritems()])
        d[name] = np.sort(lst)
    return d


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
        d[name] = np.sort(lst)
    return d


def printPar2Scores(d1, d2):
    table = PrettyTable(['Generator', 'Time (seconds)', 'Similarity Score (l2 distance)'])
    for key in d1.keys():
        table.add_row([key, round(sum(d1[key]), 4), round(sum(d2[key]), 4)])
    print(table)


def createCactusPlot(d, timeout, fname=None):
    """
    This function should create a cactus plot given a dataframe.
    :param d: dictionary
    :param timeout: int
    :param fname: string
    :return: None
    """
    plt.figure(figsize=(10, 10))
    plt.title(f"Cactus Plot of Adversarial Generators (Timeout: {timeout} seconds)")
    plt.xlabel("Number of Instances (#)")
    plt.ylabel("Time Taken (seconds)")

    for key in d.keys():
        lst = [x for x in d[key] if x < timeout]
        size = len(lst)
        xAxis = [x for x in range(1, size+1)]
        plt.xticks(xAxis)
        plt.plot(xAxis, lst, '^-', label=key)

    plt.grid(False)
    plt.legend()

    if fname is not None:
        plt.savefig(fname)
    plt.show()


def plotEachModel(modelName, image, origLabel, lst, fname=None):
    """
    This function plots the generated images for each model.
    :param: modelName: string
    :param: image: numpy ndarray
    :param: origLabel: string
    :param: lst: List of lists
    :return: None
    """
    fig, axs = plt.subplots(1, len(lst) + 1, figsize=(15, 5))
    fig.suptitle(f"Plotting Generated Adversarial Images for Model Name: {modelName}", fontsize=12)

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

    if fname is not None:
        plt.savefig(fname)
    plt.show()


def displayPerturbedImagesDF(df, fname=None):
    """
    This function should display the perturbed images given a dataframe.
    :param df: Pandas.DataFrame
    :param fname: str name of the file to save
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
            plotEachModel(name, image, label, lst, fname)


def displayPerturbedImages(img1, name1, label1, img2, name2, label2):
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
    plt.suptitle(f"Plotting Images for {name1} (left) and {name2} (right)")
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


def plotResults(args):
    inputFolder = args.input

    df = pd.DataFrame()
    for filename in os.listdir(inputFolder):
        data = pd.read_pickle(inputFolder+filename)
        df = df.append(data)

    d = getTimesFromDataFrame(df)
    d2 = getSimilaritiesFromDataFrame(df)
    createCactusPlot(d, args.timeout, args.output)
    printPar2Scores(d, d2)
    sys.exit(0)


def main():
    """
    The main function that parses arguments
    :return: None
    """
    parser = argparse.ArgumentParser(description="Plot results of an evaluation.")
    parser.add_argument("-i", "--input", help="Input folder where the dataframes are saved during evaluation.")
    parser.add_argument("-o", "--output", help="Output filename to save the cactus plot.")
    parser.add_argument("-t", "--timeout", help="Timeout of that evaluation.", type=int)
    try:
        args = parser.parse_args()
        plotResults(args)
    except Exception as e:
        print(sys.stderr, e)


if __name__ == "__main__":
    main()
