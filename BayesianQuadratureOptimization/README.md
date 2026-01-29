# QC-AcquisitionFunction
## Introduction
This repository contains the code for the Bayesian quadrature optimization in molecular dynamics chapter of the internship report. This is modified code from the following paper: 'Quantum Bayesian Optimization' (https://arxiv.org/abs/2310.05373).

## Building a suited environment
Follow the steps below to build a virtual environment using Conda and then install the right packages using Pip:

1. ***Install Python***: Make sure you have Python 3.11 installed, as this was the version used when the code was modified, although other versions may also work.
2. ***Create environment***: Run the following command to create and activate an environment. Make sure you replace '[environment_name]' with the name you want to use, make sure you use a different name then your other environments.
```bash
conda create -n [environment_name]
conda activate [environment_name]
```
3. ***Installing required packages***: Run the following code to install the necessary packages:
```bash
pip install -r requirements.txt
```

## Project Structure
BayesianQuadratureOptimization/ 
- README.md 
- requirements.txt # Packages required to run the code
- QuantumBayesianOptimization/ # The quantum bayesian optimization algorithm
- RBFKernelApproximation/ # The approximation of the kernel used, 1D and 2D