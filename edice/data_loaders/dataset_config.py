import copy

from edice.data_loaders.hdf5_datasets import HDF5Dataset
from edice.data_loaders.metadata import *


class RoadmapDataset(HDF5Dataset, RoadmapMetadata):
    pass


dataset_configs = {}

# A dataset class, filepath, idmap, (split) combination define a dataset
ROADMAP = {"dataset_class": RoadmapDataset,
           "filepath": "roadmap/roadmap_tracks_shuffled.h5",  # relative to data_dir
           "idmap": "data/roadmap/idmap.json",
           "name": "RoadmapRnd"}  # relative to repo base
dataset_configs['RoadmapRnd'] = ROADMAP

PREDICTD = copy.deepcopy(ROADMAP)
# relative to repo base
PREDICTD["splits"] = "data/roadmap/predictd_splits.json"
PREDICTD["name"] = "PredictdRnd"
dataset_configs['PredictdRnd'] = PREDICTD

ROADMAP_SAMPLE = {"dataset_class": RoadmapDataset,
                  "data_dir": "sample_data",
                  "filepath": "roadmap/SAMPLE_chr21_roadmap_train.h5",  # relative to data_dir
                  "idmap": "sample_data/roadmap/idmap.json",
                  "name": "RoadmapSample"}
dataset_configs['RoadmapSample'] = ROADMAP_SAMPLE

ROADMAPCHR21 = copy.deepcopy(ROADMAP)
ROADMAPCHR21["filepath"] = "roadmap/chr21_roadmap_tracks.h5"
ROADMAPCHR21["chromosome"] = "chr21"
ROADMAPCHR21["name"] = "RoadmapChr21"
dataset_configs['RoadmapChr21'] = ROADMAPCHR21

PREDICTDCHR21 = copy.deepcopy(PREDICTD)
PREDICTDCHR21["filepath"] = "roadmap/chr21_roadmap_tracks.h5"
PREDICTDCHR21["chromosome"] = "chr21"
PREDICTDCHR21["name"] = "PredictdChr21"
dataset_configs['PredictdChr21'] = PREDICTDCHR21

ROADMAPCHR1 = copy.deepcopy(ROADMAP)
ROADMAPCHR1["filepath"] = "roadmap/chr1_roadmap_tracks.h5"
ROADMAPCHR1["chromosome"] = "chr1"
ROADMAPCHR1["name"] = "RoadmapChr1"
dataset_configs['RoadmapChr1'] = ROADMAPCHR1

PREDICTDCHR1 = copy.deepcopy(PREDICTD)
PREDICTDCHR1["filepath"] = "roadmap/chr1_roadmap_tracks.h5"
PREDICTDCHR1["chromosome"] = "chr1"
PREDICTDCHR1["name"] = "PredictdChr1"
dataset_configs['PredictdChr1'] = PREDICTDCHR1

ROADMAPCHR4 = copy.deepcopy(ROADMAP)
ROADMAPCHR4["filepath"] = "roadmap/chr4_roadmap_tracks.h5"
ROADMAPCHR4["chromosome"] = "chr4"
ROADMAPCHR4["name"] = "RoadmapChr4"
dataset_configs['RoadmapChr4'] = ROADMAPCHR4

PREDICTDCHR4 = copy.deepcopy(PREDICTD)
PREDICTDCHR4["filepath"] = "roadmap/chr4_roadmap_tracks.h5"
PREDICTDCHR4["chromosome"] = "chr4"
PREDICTDCHR4["name"] = "PredictdChr4"
dataset_configs['PredictdChr4'] = PREDICTDCHR4


def load_dataset(dataset_name, **kwargs):
    config = dataset_configs[dataset_name].copy()
    dataset = config.pop('dataset_class')
    config.update(kwargs)
    print(config)
    return dataset(**config)


def load_metadata(dataset_name):
    config = dataset_configs[dataset_name]
    raise NotImplementedError()
