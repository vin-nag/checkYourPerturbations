"""
This file is the main function that parses user arguments and runs the experiment.

Date:
    July 5, 2020

Project:
    ECE653 Final Project: Check Your Perturbations

Authors:
    name: Vineel Nagisetty, Laura Graves, Joseph Scott
    contact: vineel.nagisetty@uwaterloo.ca
"""

import argparse
import sys
[sys.path.append(i) for i in ['.', '..']]
from src.generator.factory import GeneratorSelector
from src.benchmark import BenchmarkEnums, Benchmark
from src.evaluator import Evaluator


def main():
    """
    The main function that parses arguments
    :return: None
    """
    parser = argparse.ArgumentParser(description="Evaluate adversarial example generators on a given benchmark.")
    parser.add_argument('--benchmark', type=BenchmarkEnums, default=BenchmarkEnums.Demo)
    parser.add_argument('--verbose', action='store_true', default=False)
    try:
        args = parser.parse_args()
        run(args)
    except Exception as e:
        print(sys.stderr, e)


def run(args) -> None:
    """
    This function parses the arguments provided and runs the experiment
    :param args: the arguments provided by the user.
    :return: None
    """
    selector = GeneratorSelector()
    generators = selector.getAllGenerators()
    benchmark = Benchmark(args.benchmark)
    evaluator = Evaluator(benchmark=benchmark, generators=generators, timeLimit=benchmark.timeLimit,
                          verbose=args.verbose)
    evaluator.evaluate()
    sys.exit(0)


if __name__ == "__main__":
    main()
