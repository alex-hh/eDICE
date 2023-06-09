# eDICE

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8017391.svg)](https://doi.org/10.5281/zenodo.8017391)

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
The environment and the package can be setup from the cloned eDICE folder as follows


```bash
conda create -n edice python==3.9
conda activate edice
pip install -r requirements.txt
python setup.py install
```

This operation will install all the package dependencies for eDICE, which should require only a few minutes on a typical computer. Alternatively, the requirements.txt file lists the dependencies used to perform the experiments presented in the paper.

The _r_ folder contains the pipeline used for the differential peak analysis, which was performed using R version 4.1.0 (2021-05-18) and DiffBind version 3.2.7.


# Demo

Sample data for a minimal run of the training script is provided in the folder edice/data/roadmap.

Ensure that a suitable environment is setup and active (see Software requirements).
 
To run the sample training script, the command is:

```bash
python scripts/train_roadmap.py --experiment_name "myRoadmapExperiment" --train_splits "train" --epochs 20 --transformation "arcsinh" --embed_dim 256 --lr 0.0003 --n_targets 120
```

The sample script produces a trained edice model located in the _oputputs_ folder, as well as saving the predictions for the test tracks as a .npz file. 
A typical run of this example script requires approximately 40 minutes on a standard laptop.

Full data and trained models to run the Roadmap training and ENTEx transfer learning scripts are available at [Data for reproducing the training of eDICE model](https://doi.org/10.17617/3.VKEFB6). 

To reproduce the model used for validation on the Roadmap dataset, download the `roadmap_tracks_shuffled.h5` file from the linked dataset, move it to a data directory e.g. `data/roadmap/roadmap_tracks_shuffled.h5` together with the `idmap.json` and the `predict_splits.json` files, include the `annotations` folder in the data folder, and run:

```bash
python scripts/train_roadmap.py --experiment_name "eDICE_Roadmap" --dataset "RoadmapRnd" --data_dir "data" --split_file "data/roadmap/predictd_splits.json" --train_splits "train" "val" --epochs 50 --transformation "arcsinh" --embed_dim 256 --lr 0.0003 --n_targets 120
```

# Run eDICE on your data

To run eDICE on custom data, the epigenomic tracks must be provided in a suitable HDF5 format. Utility functions to preprocess data in a suitable manner are under development. Once the data is processed, run the `train_eDICE.py` script as: 


```bash
python scripts/train_eDICE.py --experiment_name "myCustomExperiment" --dataset_filepath "roadmap/SAMPLE_chr21_roadmap_train.h5" --data_dir "sample_data" --idmap "sample_data/roadmap/idmap.json" --dataset_name "mySampleRoadmap" --split_file "sample_data/roadmap/predictd_splits.json" --gap_file "annotations/hg19gap.txt" --blacklist_file "annotations/hg19-blacklist.v2.bed" --train_splits "train" --epochs 20 --transformation "arcsinh" --embed_dim 256 --lr 0.0003 --n_targets 120
```



# License

This project is covered under the MIT License.


# Citation

For usage of this package please cite the original paper [Getting Personal with Epigenetics: Towards Machine-Learning-Assisted Precision Epigenomics](https://www.biorxiv.org/content/10.1101/2022.02.11.479115v1).