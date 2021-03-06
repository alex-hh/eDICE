# eDICE
Epigenomic Data Imputation via Contextualised Embeddings

![eDICE architecture](eDICE_architecture_s.png "epigenomic Data Imputation through Contextualised Embeddings (eDICE)")

This repository contains the code for the model presented in the paper [Getting Personal with Epigenetics: Towards Machine-Learning-Assisted Precision Epigenomics](https://www.biorxiv.org/content/10.1101/2022.02.11.479115v1).

The _scripts_ folder contains the code to train eDICE on the Roadmap dataset as well as to apply transfer learning for individualized predictions on the ENTEx dataset.

Sample data for a minimal run of the training script is provided in the _sample\_data_ folder. 
To run the sample training script, the command is:

     python3 scripts/train_roadmap.py --experiment_name "myExperiment" --experiment_group "example" --train_splits "train" --epochs 20 --transformation "arcsinh" --embed_dim 256 --loss "mse" --lr 0.0003 --n_targets 120


Full data and trained models to run the transfer learning scripts are available on request
