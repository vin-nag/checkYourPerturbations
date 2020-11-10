# Check Your Perturbations
## By Laura Graves, Vineel Nagisetty, and Joseph Scott

## Table of Contents
* Introduction
* Usage
* Report

## Introduction
Adversarial examples are specifically crafted inputs that are able to cause a neural network to misclassify them. The challenge in creating these examples is that simple gradient descent is often defeated by defense methods. In this work we propose a suite of non-gradient-based methods for creating adversarial examples, including methods based on fuzzing, genetic algorithms, and symbolic execution and compare them with gradient based fuzzers. We find that, while gradient descent outperforms other methods when attacking undefended fully connected models, the non-gradient methods outperform it against convolutional models. This shows promise for finding adversarial examples against defended models as well as highlights the insufficiency of current defense methods, showing the need for greater research into non-gradient based defenses. This is done as part of the final project for University of Waterloo's ECE 653: Testing, Quality Assurance, and Maintenance course.

## Usage
You are free to clone, run and modify this file as you see fit. 

### Source Code
The code of the project is found in the `/src/` directory and run using the main.py file. 

### Reproduce Results
To reproduce the results, clone the latest version of cleverhans and follow these steps:
1. Clone and install the latest stable version of Cleverhans: `git clone https://github.com/tensorflow/cleverhans && cd cleverhans && pip install .`
2. Run the main file: `cd ../src/ && python3 main.py`
3. Plot and save the results: `python3 plotter.py -i ./results/data/{Benchmark Name}/{Date} -o {location to save cactus plot} -t <time out limit>`

Please note that you need to install tensorflow 2.2.0 to run these files. The full requirements are found in `requirements.txt`. 

## Report
The final report for this project is found [here](https://github.com/vin-nag/checkYourPerturbations/blob/master/documentation/project_proposal.pdf).

