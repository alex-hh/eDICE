# eDICE
Epigenomic Data Imputation via Contextualised Embeddings

![eDICE architecture](eDICE_architecture_s.png "epigenomic Data Imputation through Contextualised Embeddings (eDICE)")

This repository contains the code for the model presented in the paper [Getting Personal with Epigenetics: Towards Machine-Learning-Assisted Precision Epigenomics](https://www.biorxiv.org/content/10.1101/2022.02.11.479115v1).

# Overview

The _scripts_ folder contains the code to train eDICE on the Roadmap dataset as well as the code used to apply transfer learning for individualized predictions on the ENTEx dataset.





# System requirements

## Hardware requirements 

## Software requirements



## Python dependencies




# Demo

Sample data for a minimal run of the training script is provided in the folder edice/data/roadmap.

Setup the edice package using 

     python setup.py install
 
To run the sample training script, the command is:

     python scripts/train_roadmap.py --experiment_name "myExperiment" --train_splits "train" --epochs 20 --transformation "arcsinh" --embed_dim 256 --lr 0.0003 --n_targets 120


Full data and trained models to run the transfer learning scripts are available on request


# License