import copy

from data_loaders.hdf5_datasets import HDF5Dataset
from data_loaders.metadata import *


class RoadmapDataset(HDF5Dataset, RoadmapMetadata):
    pass


dataset_configs = {}

# A dataset class, filepath, idmap, (split) combination define a dataset
ROADMAP = {"dataset_class": RoadmapDataset,
           "filepath": "roadmap/roadmap_tracks_shuffled.h5",  # relative to data_dir
           "idmap": "data/roadmap/idmap.json",
           "name": "RoadmapRnd"}  # relative to repo base
dataset_configs['RoadmapRnd'] = ROADMAP

ROADMAPCHR21 = copy.deepcopy(ROADMAP)
ROADMAPCHR21["filepath"] = "roadmap/chr21_roadmap_tracks.h5"
ROADMAPCHR21["chromosome"] = "chr21"
ROADMAPCHR21["name"] = "RoadmapChr21"
dataset_configs['RoadmapChr21'] = ROADMAPCHR21

PREDICTD = copy.deepcopy(ROADMAP)
PREDICTD["splits"] = "data/roadmap/predictd_splits.json"  # relative to repo base
PREDICTD["name"] = "PredictdRnd"
dataset_configs['PredictdRnd'] = PREDICTD

PREDICTDCHR21 = copy.deepcopy(PREDICTD)
PREDICTDCHR21["filepath"] = "roadmap/chr21_roadmap_tracks.h5"
PREDICTDCHR21["chromosome"] = "chr21"
PREDICTDCHR21["name"] = "PredictdChr21"
dataset_configs['PredictdChr21'] = PREDICTDCHR21


def load_dataset(dataset_name, **kwargs):
    config = dataset_configs[dataset_name].copy()
    dataset = config.pop('dataset_class')
    config.update(kwargs)
    return dataset(**config)


def load_metadata(dataset_name):
    config = dataset_configs[dataset_name]
    raise NotImplementedError()
