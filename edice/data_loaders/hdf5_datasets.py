"""
    Classes and utils for holding and working with dataset metadata 
        (track names and cell/assay id mapping) and data (signal values)

    Currently supports data stored in HDF5 file (HDF5Dataset)
"""
import gc
import os

from pathlib import Path

import h5py
import numpy as np

from edice.models import metrics
from edice.data_loaders.data_generators import TrainInMemGenerator, ValInMemGenerator, TestInMemGenerator
from edice.data_loaders import hdf5_utils
from edice.data_loaders.annotations import IntervalAnnotation

from edice.utils.CONSTANTS import DATA_DIR


class HDF5Dataset:

    def __init__(self, filepath, idmap=None, tracks=None, 
                 splits=None, total_bins=None, data_dir=None,
                 chromosome=None, name=None):
        data_dir = Path(data_dir or DATA_DIR)
        self.hdf5_file = str(data_dir/filepath)
        self.chromosome = chromosome
        self.name = name or self.__class__.name

        if tracks is None:
            with h5py.File(self.hdf5_file, 'r') as h5f:
                tracks = [t.decode() for t in h5f['track_names'][:]]
                self.total_bins = total_bins or h5f['targets'].shape[0]

        super().__init__(tracks=tracks, idmap=idmap, splits=splits)

    def load_gaps(self):
        self.gaps = IntervalAnnotation.from_gap(os.path.join(DATA_DIR, self.gap_file))

    def load_blacklist(self):
        self.blacklist = IntervalAnnotation.from_bed(os.path.join(DATA_DIR, self.blacklist_file), extra_cols=["reason"])

    def _get_bins_with_gaps(self):
        if not hasattr(self, "gaps"):
            self.load_gaps()
        bins_with_gaps = self.gaps.get_chrom_annotated_bins(self.chromosome, bin_size=self.track_resolution)
        return [b for b in bins_with_gaps if b < self.total_bins]

    def _get_blacklist_bins(self):
        if not hasattr(self, "blacklist"):
            self.load_blacklist()
        blacklist_bins = self.blacklist.get_chrom_annotated_bins(self.chromosome, bin_size=self.track_resolution)
        return [b for b in blacklist_bins if b < self.total_bins]

    def _get_bin_mask(self,
                      start_bin=0,
                      total_bins=None,
                      exclude_gaps=False,
                      exclude_blacklist=False):
        total_bins = total_bins or self.total_bins
        # first get a mask covering all bins from 0 to start_bin + total_bins
        # then slice to get mask for the bins start_bin:start_bin + total_bins
        mask = np.ones(start_bin+total_bins, dtype=bool)
        if exclude_gaps:
            assert self.chromosome is not None and self.gap_file is not None,\
                ("Exclude gaps can only be specified on a chromosome"
                 "dataset with a gap_file attribute")
            bins_with_gaps = self._get_bins_with_gaps()
            # Set mask elements corresponding to gap bins to False
            mask[bins_with_gaps] = False
            print(f"Excluding {mask.shape[0]} - {mask.sum()} bins with gaps (starting from 0)")
        
        if exclude_blacklist:
            assert self.chromosome is not None and self.blacklist_file is not None,\
                ("Exclude blacklist can only be specified on a chromosome"
                 "dataset with a blacklist_file attribute")
            blacklist_bins = self._get_blacklist_bins()
            mask[blacklist_bins] = False
            print(f"Excluding {mask.shape[0]} - {mask.sum()} bins with gaps or blacklist (starting from 0)")

        mask = mask[start_bin:]  # slice back to (total_bins,)
        print(f"Keeping {mask.sum()} bins of {mask.shape[0]} (starting from {start_bin})")
        return mask

    def load_tracks(self, tracks, start_bin=0, total_bins=None,
                    exclude_gaps=False, exclude_blacklist=False):
        mask = self._get_bin_mask(
            start_bin=start_bin, total_bins=total_bins,
            exclude_gaps=exclude_gaps, exclude_blacklist=exclude_blacklist
        )
        assert mask.shape[0] == (total_bins or self.total_bins)
        print("Loading data from h5 file", flush=True)
        track_ids = np.asarray([self.track2id[t] for t in tracks])
        return hdf5_utils.load_tracks_from_hdf5_by_id(
            self.hdf5_file, track_ids, start_bin,
            mask.shape[0], mask)

    def prepare_data(self,
                     support_tracks,
                     target_tracks=None,
                     return_track_ids=False,
                     exclude_gaps=False,
                     exclude_blacklist=False):
        """
        support_tracks, target_tracks: lists of tracks
        return_track_ids: return track ids rather than loading values
        for tracks (for use with ValHDF5Generator)
        """
        if return_track_ids:
            supports = np.asarray([self.track2id[t] for t in support_tracks])
        else:
            supports = self.load_tracks(support_tracks,
                                        exclude_gaps=exclude_gaps,
                                        exclude_blacklist=exclude_blacklist)
        support_cell_ids = [self.cell2id[self.get_track_cell(t)] 
                            for t in support_tracks]
        support_assay_ids = [self.assay2id[self.get_track_assay(t)]
                             for t in support_tracks]

        if target_tracks is not None:
            if return_track_ids:
                targets = np.asarray([self.track2id[t] for t in target_tracks])
            else:
                targets = self.load_tracks(target_tracks,
                                           exclude_gaps=exclude_gaps,
                                           exclude_blacklist=exclude_blacklist)
            target_cell_ids = [self.cell2id[self.get_track_cell(t)]
                               for t in target_tracks]
            target_assay_ids = [self.assay2id[self.get_track_assay(t)]
                                for t in target_tracks]
        else:
            targets = None
            target_cell_ids = None
            target_assay_ids = None
        return supports, support_cell_ids, support_assay_ids, targets,\
                target_cell_ids, target_assay_ids

    def get_train_generator(self, train_tracks=None, batch_size=256,
                            transform=None, shuffle=True, n_targets=50,
                            min_targets=None, max_targets=None,
                            exclude_gaps=False, exclude_blacklist=False,
                            return_dict=False):
        if train_tracks is None:
            assert hasattr(self, 'splits') and isinstance(self.splits, dict),\
                'Splits not passed to dataset constructor'
            train_tracks = self.splits['train']
        n_target_str = (f"{min_targets} - {max_targets}"
                        if max_targets is not None
                        else n_targets)
        print(f"loading train generator on {len(train_tracks)} tracks using "
              f"n targets {n_target_str} at each bin and {transform} transform")
        supports, cell_ids, assay_ids, _ , _ , _ = self.prepare_data(
            train_tracks, exclude_gaps=exclude_gaps, exclude_blacklist=exclude_blacklist
        )
        return TrainInMemGenerator(supports, cell_ids, assay_ids,
                                   transform=transform, batch_size=batch_size,
                                   shuffle=shuffle, n_targets=n_targets,
                                   min_targets=min_targets, max_targets=max_targets,
                                   return_dict=return_dict)

    def get_supports_targets_generator(self,
                                       support_tracks,
                                       target_tracks,
                                       batch_size=256,
                                       transform=None,
                                       exclude_gaps=False,
                                       exclude_blacklist=False,
                                       return_dict=False,
                                       n_bins_to_load=100000):
        """
        Instead of specifying splits as in get_val_generator
        TODO: remove redundant return_dict arg
        """
        print(f"loading val generator with {len(support_tracks)} supports and "
              f"{len(target_tracks)} targets; using {transform} transform")
        (supports, cell_ids, assay_ids,
         targets, target_cell_ids, target_assay_ids) = self.prepare_data(
            support_tracks, target_tracks, exclude_gaps=exclude_gaps,
            exclude_blacklist=exclude_blacklist, return_track_ids=False)
        print("loading all support / target tracks into memory")
        return ValInMemGenerator(supports, cell_ids, assay_ids,
                                 targets, target_cell_ids, target_assay_ids,
                                 batch_size=batch_size, transform=transform,
                                 return_dict=return_dict)
        

    def get_val_generator(self,
                          support_splits=None,
                          target_split="val",
                          transform=None,
                          **kwargs):
        """
        Instantiate an in mem val generator using the tracks
        from the specified splits as supports / targets.
        """
        assert (hasattr(self, 'splits') and
                isinstance(self.splits, dict)), "Dataset has no splits attribute"
        if support_splits is None:
            support_splits = ["train"]
        support_tracks = [t for s in support_splits for t in self.splits[s]]
        target_tracks = self.splits[target_split]
        return self.get_supports_targets_generator(
            support_tracks, target_tracks, transform=transform,
            **kwargs)

    def get_targets_metrics(self,
                            target_tracks,
                            grouped_assay_metrics=True,
                            per_track_metrics=False):

        track_ids_by_assay = self.tracks_by_type(
            target_tracks, "assay", return_track_ids=True)
        n_outputs_val = len(target_tracks)
        val_metrics = [metrics.PearsonCorrelation(n_outputs_val),
                       metrics.MeanSquaredError(n_outputs_val)]
        if grouped_assay_metrics:
            val_metrics += [metrics.AssayLevelMSE(ids, subset_name=a)
                            for a, ids in track_ids_by_assay.items()]
            val_metrics += [metrics.AssayLevelCorr(ids, subset_name=a)
                            for a, ids in track_ids_by_assay.items()]
        if per_track_metrics:
            # I think the problem I have is that I'm getting ids relative
            # to the full dataset, when I need ids relative to val_targets
            val_metrics += [metrics.TrackLevelMSE(i, track_name=t)
                            for i, t in enumerate(target_tracks)]
            val_metrics += [metrics.TrackLevelCorr(i, track_name=t)
                            for i, t in enumerate(target_tracks)]

        return val_metrics

    def get_val_metrics(self, val_split="val", **kwargs):
        assert (hasattr(self, 'splits') and
                isinstance(self.splits, dict)), "Dataset has no splits attribute"
        val_tracks = self.splits[val_split]
        return self.get_targets_metrics(val_tracks, **kwargs)
