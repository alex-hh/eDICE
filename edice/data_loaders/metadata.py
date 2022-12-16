import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def load_saved_idmap(idmap):
    assert isinstance(idmap, str), 'idmap must be dict, str or None'
    with open(idmap, 'r') as f:
        idmap = json.load(f)
    cell2id = idmap['cell2id']
    assay2id = idmap['assay2id']
    # CHECK COMPATIBILITY WITH CELLS AND ASSAYS
    return cell2id, assay2id


def build_new_idmap(cells, assays):
    cell2id = {c: i for i, c in enumerate(cells)}
    assay2id = {a: i for i, a in enumerate(assays)}
    return cell2id, assay2id


def check_idmap_compat(idmap, cells, assays):
    for cell in cells:
        assert cell in idmap['cell2id']
    for assay in assays:
        assert assay in idmap['assay2id']


def read_splits_csv(csvfile):
    """
    first column ('name'): track names
    second column ('set'): T/V/B
    """
    splits_df = pd.read_csv(csvfile)  # TODO: remove pandas dependency...
    train_expt_names = [t for t in splits_df[splits_df['set']=='T']['name'].values]
    val_expt_names = [t for t in splits_df[splits_df['set']=='V']['name'].values]
    return {'train': train_expt_names, 'val': val_expt_names}


def read_splits_json(jsonfile):
    """
    json dict of train: train_tracks, val: val_tracks, test: test_tracks
    """
    with open(jsonfile, 'r') as jf:
        splits = json.load(jf)
    return splits


class TrackMappingMixin:

    def __init__(self, tracks, idmap=None, splits=None):
        # print('init meta')
        if isinstance(tracks, str):
            # n.b. we first removed .pval.signal suffix using following
            # ['.'.join(t.split('.')[:-2]) for t in text.split('\n') if t]
            tracks = [t for t in Path(tracks).read_text().split('\n') if t]
        else:
            assert isinstance(tracks, list),\
                "tracks must be str or list"
        
        self.tracks = tracks
        self.track2id = {t: i for i, t in enumerate(self.tracks)}
        
        if splits is not None:
            if isinstance(splits, str):
                suffix = splits.split(".")[-1]
                if suffix == "json":
                    splits = read_splits_json(splits)
                else:
                    raise NotImplementedError()
            else:
                assert isinstance(splits, dict), "splits must be str or dict"
        else:
            assert idmap is not None, "Must provide splits or idmap or both"
        self.splits = splits

        if idmap is not None:
            self.cell2id, self.assay2id = load_saved_idmap(idmap)
            self.cells = self.sort_cells(list(self.cell2id.keys()))
            self.assays = self.sort_assays(list(self.assay2id.keys()))

        else:
            train_tracks = self.splits["train"]
            cells = list(set([self.get_track_cell(t) for t in train_tracks]))
            self.cells = self.sort_cells(cells)
            assays = list(set([self.get_track_assay(t) for t in train_tracks]))
            self.assays = self.sort_assays(assays)
        
            # either load idmap or build one based on track list
            self.cell2id, self.assay2id = build_new_idmap(self.cells, self.assays)
        
        check_idmap_compat({'cell2id': self.cell2id,
                            'assay2id': self.assay2id},
                            self.cells, self.assays)

    def save_idmap(self, filepath):
        with open(filepath, 'w') as jf:
            json.dump({'cell2id': self.cell2id,
                       'assay2id': self.assay2id}, jf, indent=2)    

    def tracks_by_type(self, tracks, type_,
                       return_group_ids=False,
                       return_track_ids=False):
        """
        This is for use for constructing assay-level metrics, which need
        to have lists of track ids by assay, where ids are relative
        to a given set of tracks, whereas self.track2id returns ids
        relative to the hdf5 targets array

        return_group_ids:
            whether keys should be ids rather than cell / assay names
        return_track_ids:
            whether elements in value arrays should be ids rather
                than cell / assay names
        """
        assert type_ in ["cell", "assay"], "type_ must be cell or assay"
        tracks_by_group = defaultdict(list)
        
        for i, t in enumerate(tracks):

            if t not in self.tracks:
                print(f"Skipping track: {t} in splits but not in tracks")
                continue
        
            if type_ == "cell":
                group_code = self.get_track_cell(t)
                if return_group_ids:
                    group_code = self.cell2id[group_code]
        
            elif type_ == "assay":
                group_code = self.get_track_assay(t)
                if return_group_ids:
                    group_code = self.assay2id[group_code]

            if return_track_ids:
                tracks_by_group[group_code].append(i)
            else:
                tracks_by_group[group_code].append(t)
 
        return tracks_by_group

    def split_tracks_by_type(self, type_, split=None,
                             return_group_ids=False,
                             return_track_ids=False):
        """
        see tracks_by_type
        """
        if split is None:
            tracks = self.tracks
        else:
            assert split in self.splits, f"{split} is not a known split"
            tracks = self.splits[split]
        return self.tracks_by_type(tracks, type_,
                                   return_group_ids=return_group_ids,
                                   return_track_ids=return_track_ids)

    def split_counts_by_type(self, split=None, type_="cell"):
        """
        It is useful to know whether e.g. there are any cells that
        are not represented in the training set of a given train/val
        split, since then the model will be unable to make predictions
        for them.
        """
        assert type_ in ["cell", "assay"], "group must be cell or assay"
        if split is None:
            count_dict = {}
            for split in self.splits:
                count_dict[split] = self.split_counts_by_type(split=split,
                                                              type_=type_)
        else:
            group_getter = getattr(self, f"get_track_{type_}")
            count_dict = {g: 0 for g in getattr(self, type_+"s")}
            for track in self.splits[split]:
                count_dict[group_getter(track)] += 1
        return count_dict

    def sort_cells(self, cells):
        """
        Override for non-arbitrary sorting
        """
        return sorted(cells)

    def sort_assays(self, assays):
        """
        Override for non-arbitrary sorting
        """
        return sorted(assays)

    def get_track_cell(self):
        raise NotImplementedError('get_track_cell method must be implemented for this dataset')

    def get_track_assay(self):
        raise NotImplementedError('get_track_cell method must be implemented for this dataset')


class RoadmapMetadata(TrackMappingMixin):

    gap_file = "annotations/hg19gap.txt"
    blacklist_file = "annotations/hg19-blacklist.v2.bed"
    track_resolution = 25

    @staticmethod
    def get_track_cell(track):
        """
        Roadmap naming convention: Ecellnum-assayname
        """
        return track.split('-')[0]

    @staticmethod
    def get_track_assay(track):
        """
        Roadmap naming convention: E<cellId>-<assayName>
        """
        return track.split('-')[1]

    def sort_cells(self, cells):
        return sorted(cells, key=lambda x: int(x[1:]))
