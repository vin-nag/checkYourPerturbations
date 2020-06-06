"""
This file is the main function that parses user argument and runs experiments.

Date:
    November 6, 2019

Project:
    LogicGAN

Authors:
    name: Vineel Nagisetty, Laura Graves, Joseph Scott, Vijay Ganesh
    contact: vineel.nagisetty@uwaterloo.ca
"""

import argparse
import sys


def main():
    """
    The main function that parses arguments
    :return:
    """
    parser = argparse.ArgumentParser(description="run the experiment using regular and logic GANs")
    try:
        args = parser.parse_args()
        run(args)
    except Exception as e:
        print(sys.stderr, e)


def run(args:argparse.Namespace) -> None:
    pass

