"""
This file is the main function that parses user argument and runs experiments.

Date:
    June 5, 2020

Project:
    ECE653 Final Project

Authors:
    name: Vineel Nagisetty, Laura Graves, Joseph Scott
    contact: vineel.nagisetty@uwaterloo.ca
"""

import argparse
import sys


def main():
    """
    The main function that parses arguments
    :return:
    """
    parser = argparse.ArgumentParser(description="run the fuzzer on ")
    try:
        args = parser.parse_args()
        run(args)
    except Exception as e:
        print(sys.stderr, e)


def run(args: argparse.Namespace) -> None:
    pass
