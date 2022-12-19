# eDICE
Epigenomic Data Imputation via Contextualised Embeddings

![eDICE architecture](eDICE_architecture_s.png "epigenomic Data Imputation through Contextualised Embeddings (eDICE)")

This repository contains the code for the model presented in the paper [Getting Personal with Epigenetics: Towards Machine-Learning-Assisted Precision Epigenomics](https://www.biorxiv.org/content/10.1101/2022.02.11.479115v1).

# Overview

The _edice_ folder contains the source code used to perform the experiments presented in the paper.

The _scripts_ folder contains the code to train eDICE on the Roadmap dataset as well as the code used to apply transfer learning for individualized predictions on the ENTEx dataset.

The _r_ folder contains the code used to perform the differential peak analysis using R and DiffBind.


# System requirements

## Hardware requirements 

eDICE requires only a standard computer with enough RAM to support the in-memory operations. However, the use of a GPU accelerator is recommended for the analysis of larger datasets.


## Software requirements

The eDICE models were trained on computers operating on Ubuntu 16.04 and Ubuntu 22.04.

eDICE was developed using python 3.9. We recommend setting up a suitable environment using [Anaconda](https://www.anaconda.com/). 
The environment can be created as follows

     conda create -n edice python==3.9
     conda activate edice
     python setup.py install

This operation will install all the package dependencies for eDICE, which should require only a few minutes on a typical computer. Alternatively, the requirements.txt file lists the dependencies used to perform the experiments presented in the paper.

The _r_ folder contains the pipeline used for the differential peak analysis, which was performed using R version 4.1.0 (2021-05-18) and DiffBind version 3.2.7.


# Demo

Sample data for a minimal run of the training script is provided in the folder edice/data/roadmap.

Ensure that a suitable environment is setup and active (see Software requirements).
 
To run the sample training script, the command is:

     python scripts/train_roadmap.py --experiment_name "myExperiment" --train_splits "train" --epochs 20 --transformation "arcsinh" --embed_dim 256 --lr 0.0003 --n_targets 120

The sample script produces a trained edice model located in the _oputputs_ folder, as well as saving the predictions for the test tracks as a ? file. 
A typical run of this example script requires approximately ? minutes on a standard laptop.

Full data and trained models to run the Roadmap training and ENTEx transfer learning scripts are available on request.


# License

This project is covered under the MIT License.


# Citation

For usege of this package please cite the original paper [Getting Personal with Epigenetics: Towards Machine-Learning-Assisted Precision Epigenomics](https://www.biorxiv.org/content/10.1101/2022.02.11.479115v1).