# QCNN
## Introduction
This repository contains the code for the QCNN application discussed in the internship report. 

## Building a suited environment
Follow the steps below to build a virtual environment using Conda and then install the right packages using Pip:

1. ***Install Python***: Make sure you have Python 3.11.13 installed, as this was the version used during the code's modification, though other versions might also work.
2. ***Create environment***: Run the following command to create and activate an environment. Make sure you replace '[environment_name]' with the name you want to use and use a different name then your other environments.
```bash
conda create -n [environment_name]
conda activate [environment_name]
```
3. ***Installing required packages***: Run the following code to install the necessary packages, if you have a windows system:
```bash
pip install -r SURF_Internship_Quantum\MedicalImaging\QCNN\requirements.txt
```
Otherwise run:
```bash
pip install -r SURF_Internship_Quantum/MedicalImaging/QCNN/requirements.txt
```

4. ***Settings***: Open the '[config.yaml]' file in the configs folder and change the settings to your desired settings.

5. ***Train the model***: Run the following code to train the QCNN with the chosen dataset. Choose between classical and quantum mode by changing the `--mode` flag, if you have a windows system:
Use the following code to run the classical head.
```bash
python SURF_Internship_Quantum\MedicalImaging\QCNN\src\train.py --config SURF_Internship_Quantum\MedicalImaging\QCNN\configs\config.yaml --mode classical
```
And the following code for the quantum head.
```bash
python SURF_Internship_Quantum\MedicalImaging\QCNN\src\train.py --config SURF_Internship_Quantum\MedicalImaging\QCNN\configs\config.yaml --mode quantum
```

If you have an other operating system:
Use the following code to run the classical head.
```bash
python SURF_Internship_Quantum/MedicalImaging/QCNN/src/train.py --config SURF_Internship_Quantum/MedicalImaging/QCNN/configs/config.yaml --mode classical
```
And the following code for the quantum head.
```bash
python SURF_Internship_Quantum/MedicalImaging/QCNN/src/train.py --config SURF_Internship_Quantum/MedicalImaging/QCNN/configs/config.yaml --mode quantum
```
Make sure you run the application in the newly created environment! Otherwise, errors will be displayed.

## Configuration (`configs/config.yaml`)

The training behavior is controlled by `config.yaml`. Here is a brief overview of the key parameters:

* **`dataset_type`**: Toggles between `'pcam'` (binary image classification) and `'tcga'` (multiclass embedding classification).
* **`model`**:
    * **`latent_dim`**: The size of the embedding vector produced by the backbone.
    * **`n_qubits`**: The number of qubits to use in the quantum head.
    * **`n_quantum_layers`**: The number of repeated layers in the quantum circuit's ansatz.
    * **`num_classes`**: Number of unique classes for the TCGA dataset.
    * **`quantum_head_type`**: The type of quantum embedding to use, either `'amplitude'` or `'angle'`.
    * **`entangling_layer`**: The type of entangling layer in the quantum circuit, either `'strong'` or `'basic'`.
* **`training`**:
    * **`batch_size`**: The number of samples per batch.
    * **`lr`**: The learning rate for the Adam optimizer.
    * **`epochs`**: The total number of training epochs.

## Project Structure
QCNN/ 
- README.md 
- requirements.txt # Packages required to run the code.
- Plotting.py # Code for plotting accuracies and training loss from the logs of the trained model.
- coherence_time2.py # Calculate the coherence time of the quantum head.
- src/ # Contains code for training the model.
- slurm/ # Contains a sbatch file to train on a HPC.
- plots/ # Plots get written in this folder.
- models/ # Trained models get written in this folder.
- logs/ # Logs of model training get written in this folder.
- configs/ # Contains the configuration file that contains the settings of the QCNN.
