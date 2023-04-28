import h5py
import numpy as np


def load_tracks_from_hdf5_by_id(hdf5_file, track_ids, start_bin,
                                total_bins, boolean_mask):
    """
      boolean_mask: a boolean array with True values at bins
                    which should be kept. can be used e.g. to indicate 
                    presence of gap/blacklist regions
    """
    if isinstance(track_ids, list):
        track_ids = np.asarray(track_ids)
    sortperm = track_ids.argsort()
    # https://arogozhnikov.github.io/2015/09/29/NumpyTipsAndTricks1.html#Computing-inverse-of-permutation
    invsortperm = sortperm.argsort()
    sorted_track_ids = track_ids[sortperm]
    # TODO - comment out...
    np.testing.assert_equal(sorted_track_ids[invsortperm],
                            track_ids)
    # https://docs.h5py.org/en/stable/high/dataset.html#fancy-indexing
    with h5py.File(hdf5_file, 'r') as hdf5:
        # for b in range(hdf5['targets'].)
        data = hdf5['targets'][start_bin:start_bin+total_bins, sorted_track_ids]

    data = data[:, invsortperm]
    # data = data[include_bins]
    data = data[boolean_mask]
    # print("Data shape after filtering out gaps", data.shape, flush=True)
    return data
