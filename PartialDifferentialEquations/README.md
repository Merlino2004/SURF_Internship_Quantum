# PartialDifferentialEquations
## Introduction
This repository contains the tutorial, including code, for the classical solution and quantum solution on how to solve linear PDEs with code. The tutorial is in a Jupyter Notebook, because the code was edited in Visual Studio Code, the requirements file won't install Jupyter Notebook. Make sure you have Jupyter Notebook already installed, or run the code via VSC. 

## Building a suited environment
Follow the steps below to build a virtual environment using Conda and then install the right packages using Pip:

1. ***Install Python***: Make sure you have Python 3.11 installed, as this was the version used when the code was modified, although other versions may also work.
2. ***Create environment***: Run the following command to create and activate an environment. Make sure you replace '[environment_name]' with the name you want to use, make sure you use a different name then your other environments.
```bash
conda create -n [environment_name]
conda activate [environment_name]
```
3. ***Installing required packages***: Run the following code to install the necessary packages.
```bash
pip install -r requirements.txt
```

## Project Structure
PartialDifferentialEquations/ 
- README.md 
- requirements.txt # Packages required to run the code.
- VQLStutorialClassicalvsQuantum.ipynb # The Jupyter Notebook containing the tutorial.
- Figures / # Figures used in the tutorial.