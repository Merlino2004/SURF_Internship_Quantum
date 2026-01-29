# QC-AcquisitionFunction
## Introduction
This repository contains the code for the QKNN application discussed in the internship report. 

## Building a suited environment
Follow the steps below to build a virtual environment using Conda and then install the right packages using Pip:

1. ***Install Python***: Make sure you have Python 3.11 installed, as this was the version used during the code's modification, though other versions might also work.
2. ***Install Conda***: Make sure you have Conda installed to create an environment. The version does not matter.
3. ***Create environment***: Run the following command to create and activate an environment. Make sure you replace '[environment_name]' with the name you want to use.
```bash
conda create -n [environment_name]
conda activate [environment_name]
```
4. ***Installing required packages***: Run the following code to install the necessary packages:
```bash
pip install -r requirements.txt
```

## Project Structure
qc-acquisition_function/ 
- README.md 
- requirements.txt
- QuantumBayesianOptimization/ # The quantum bayesian optimization algorithm
- RBFKernelApproximation/ # The approximation of the kernel used, 1D and 2D
