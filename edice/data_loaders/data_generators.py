import math
import tensorflow as tf
import numpy as np

from edice.data_loaders import hdf5_utils
from edice.utils.transforms import apply_transformation


def format_inputs(inputs, return_dict=False):
    if return_dict:
        return inputs
    else:
        input_names = ['supports', 'support_cell_ids',
                       'support_assay_ids', 'target_cell_ids',
                       'target_assay_ids']
        return [inputs[name] for name in input_names]


def _get_n_batch_targets(n_targets, min_targets, max_targets):
    # max, min override n_targets. max_targets is one above the highest number
    # that can be drawn; min targets is the lowest number that can be drawn
    if max_targets is not None:
        assert min_targets is not None, "If max targets is specified, min must be too"
        return np.random.randint(min_targets, max_targets)
    else:
        return n_targets


class TrainInMemGenerator(tf.keras.utils.Sequence):

    """
    TODO: look at NPs data generator for inspiration - 
          they have this random target context split
          maybe that tensorflow meta learning repo too...
    """

    def __init__(self,
                 signal_values,
                 cellids,
                 assayids,
                 transform=None,
                 shuffle=True,
                 batch_size=256,
                 n_targets=50,
                 max_targets=None,
                 min_targets=None,
                 return_dict=False):
        assert max_targets or n_targets, "Must specify n_targets or max_targets" 
        self.X = apply_transformation(transform, signal_values)
        self.cellids = np.asarray(cellids)
        self.assayids = np.asarray(assayids)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n_targets = n_targets
        self.n_tracks = self.X.shape[1]
        self.transform = transform
        self.return_dict = return_dict
        self.max_targets = max_targets
        self.min_targets = min_targets
        assert self.n_tracks == len(self.cellids), \
            "Length of cell ids must match number of tracks"
        assert self.n_tracks == len(self.assayids), \
            "Length of assay ids must match number of tracks"
        if self.shuffle:
            print('Shuffling data', flush=True)
            self.perm = np.random.permutation(self.X.shape[0])
            self.X = self.X[self.perm]

    def __len__(self):
        return math.ceil(self.X.shape[0]/self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            print('Shuffling data', flush=True)
            self.perm = np.random.permutation(self.X.shape[0])
            self.X = self.X[self.perm]

    def __getitem__(self, batch_index):
        """
        The idea of this implementation vs the previous one is:
          since it's a partition, we can permute and slice.

        This also makes it quite similar to the val/test version,
            which just dont permute/partition
        """
        batch_start = batch_index*self.batch_size
        batch_stop = (batch_index+1)*self.batch_size
        batch_X = self.X[batch_start: batch_stop].copy()
        batch_size = batch_X.shape[0]

        # each (batch_size, n_tracks) same as batch_X
        batch_cellids = np.tile(self.cellids, (batch_size, 1))
        batch_assayids = np.tile(self.assayids, (batch_size, 1))

        batch_n_targets = _get_n_batch_targets(self.n_targets,
                                               self.min_targets,
                                               self.max_targets)

        # here a batched_gather i.e. tf.gather with batch_dims 1 would help
        for i in range(batch_size):
            perm = np.random.permutation(self.n_tracks)
            batch_X[i,:] = batch_X[i, perm]
            batch_cellids[i, :] = batch_cellids[i, perm]
            batch_assayids[i, :] = batch_assayids[i, perm]
        
        inp = {'supports': batch_X[:, batch_n_targets:],
               'support_cell_ids': batch_cellids[:, batch_n_targets:],
               'support_assay_ids': batch_assayids[:, batch_n_targets:],
               'target_cell_ids': batch_cellids[:, :batch_n_targets],
               'target_assay_ids': batch_assayids[:, :batch_n_targets]}
        targets = batch_X[:, :batch_n_targets]
        
        return format_inputs(inp, self.return_dict), targets


class ValInMemGenerator(tf.keras.utils.Sequence):

    def __init__(self,
                 supports,
                 cellids,
                 assayids,
                 targets,
                 target_cellids,
                 target_assayids,
                 transform=None,
                 batch_size=256,
                 return_dict=False):
        self.X = apply_transformation(transform, supports)
        self.cellids = cellids
        self.assayids = assayids
        self.targets = apply_transformation(transform, targets)
        self.target_cellids = target_cellids
        self.target_assayids = target_assayids
        self.batch_size = batch_size
        self.transform = transform
        self.return_dict = return_dict

    def __len__(self):
        return math.ceil(self.X.shape[0]/self.batch_size)

    def get_batch_data(self, batch_start):
        batch_stop = batch_start+self.batch_size
        batch_X = self.X[batch_start: batch_stop]
        batch_y = self.targets[batch_start: batch_stop]
        batch_size = batch_X.shape[0]

        # each (batch_size, n_tracks) same as batch_X
        batch_cellids = np.tile(self.cellids, (batch_size, 1))
        batch_assayids = np.tile(self.assayids, (batch_size, 1))

        batch_target_cellids = np.tile(self.target_cellids,
                                       (batch_size, 1))
        batch_target_assayids = np.tile(self.target_assayids,
                                       (batch_size, 1))

        inps = {'supports': batch_X,
                'support_cell_ids': batch_cellids,
                'support_assay_ids': batch_assayids,
                'target_cell_ids': batch_target_cellids,
                'target_assay_ids': batch_target_assayids}
        return format_inputs(inps, self.return_dict), batch_y

    def __getitem__(self, batch_index):
        batch_start = batch_index*self.batch_size
        return self.get_batch_data(batch_start)


class TestInMemGenerator(tf.keras.utils.Sequence):

    def __init__(self,
                 supports,
                 cellids,
                 assayids,
                 transform=None,
                 batch_size=256,
                 return_dict=False):
        self.X = apply_transformation(transform, supports)
        self.cellids = cellids
        self.assayids = assayids
        self.batch_size = batch_size
        self.transform = transform
        self.return_dict = return_dict

    def __len__(self):
        return math.ceil(self.X.shape[0]/self.batch_size)

    def __getitem__(self, batch_index):
        batch_start = batch_index*self.batch_size
        batch_stop = (batch_index+1)*self.batch_size
        batch_X = self.X[batch_start: batch_stop]
        batch_size = batch_X.shape[0]

        # each (batch_size, n_tracks) same as batch_X
        batch_cellids = np.tile(self.cellids, (batch_size, 1))
        batch_assayids = np.tile(self.assayids, (batch_size, 1))

        batch_target_cellids = np.tile(self.target_cellids,
                                       (batch_size, 1))
        batch_target_assayids = np.tile(self.target_assayids,
                                       (batch_size, 1))

        inps = {'supports': batch_X,
                'support_cell_ids': batch_cellids,
                'support_assay_ids': batch_assayids,
                'target_cell_ids': batch_target_cellids,
                'target_assay_ids': batch_target_assayids}
        return format_inputs(inps, self.return_dict)


class TissueLOO_TrainInMemGenerator(tf.keras.utils.Sequence):

    """
    TODO: look at NPs data generator for inspiration - 
          they have this random target context split
          maybe that tensorflow meta learning repo too...
    """

    def __init__(self,
                 signal_values,
                 cellids,
                 assayids,
                 transform=None,
                 shuffle=True,
                 batch_size=256,
                #  n_targets=50,
                #  max_targets=None,
                #  min_targets=None,
                 return_dict=False):
        # assert max_targets or n_targets, "Must specify n_targets or max_targets" 
        self.X = apply_transformation(transform, signal_values)
        self.cellids = np.asarray(cellids)
        self.assayids = np.asarray(assayids)
        self.shuffle = shuffle
        self.batch_size = batch_size
        # self.n_targets = n_targets
        self.n_tracks = self.X.shape[1]
        self.transform = transform
        self.return_dict = return_dict
        # self.max_targets = max_targets
        # self.min_targets = min_targets
        assert self.n_tracks == len(self.cellids), \
            "Length of cell ids must match number of tracks"
        assert self.n_tracks == len(self.assayids), \
            "Length of assay ids must match number of tracks"
        if self.shuffle:
            print('Shuffling data', flush=True)
            self.perm = np.random.permutation(self.X.shape[0])
            self.X = self.X[self.perm]

    def __len__(self):
        return math.ceil(self.X.shape[0]/self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            print('Shuffling data', flush=True)
            self.perm = np.random.permutation(self.X.shape[0])
            self.X = self.X[self.perm]

    def __getitem__(self, batch_index):
        """
        The idea of this implementation vs the previous one is:
          since it's a partition, we can permute and slice.

        This also makes it quite similar to the val/test version,
            which just dont permute/partition
        """
        batch_start = batch_index*self.batch_size
        batch_stop = (batch_index+1)*self.batch_size
        batch_X = self.X[batch_start: batch_stop].copy()
        batch_size = batch_X.shape[0]

        # each (batch_size, n_tracks) same as batch_X
        # batch_cellids = np.tile(self.cellids, (batch_size, 1))
        # batch_assayids = np.tile(self.assayids, (batch_size, 1))

        # batch_n_targets = _get_n_batch_targets(self.n_targets,
        #                                        self.min_targets,
        #                                        self.max_targets)
        target_tissue = np.random.choice(self.cellids)
        # ixs = np.where(self.cellids==target_cellid)[0]
        target_mask = (self.cellids==target_tissue)
        support_mask = np.logical_not(target_mask)
        # support_mask = np.ones(len(batch_cellids), np.bool)

        support_batch_X = batch_X[:, support_mask]
        target_batch_X = batch_X[:, target_mask]

        support_cellids = self.cellids[support_mask]
        target_cellids = self.cellids[target_mask]
        support_batch_cellids = np.tile(support_cellids, (batch_size, 1))
        target_batch_cellids = np.tile(target_cellids, (batch_size, 1))

        support_assayids = self.assayids[support_mask]
        target_assayids = self.assayids[target_mask]
        support_batch_assayids = np.tile(support_assayids, (batch_size, 1))
        target_batch_assayids = np.tile(target_assayids, (batch_size, 1))

        n_support_tracks = support_mask.sum()
        n_target_tracks = target_mask.sum()

        # here a batched_gather i.e. tf.gather with batch_dims 1 would help
        for i in range(batch_size):
            supp_perm = np.random.permutation(n_support_tracks)
            support_batch_X[i,:] = support_batch_X[i, supp_perm]
            support_batch_cellids[i, :] = support_batch_cellids[i, supp_perm]
            support_batch_assayids[i, :] = support_batch_assayids[i, supp_perm]

            target_perm = np.random.permutation(n_target_tracks)
            target_batch_X[i,:] = target_batch_X[i, target_perm]
            target_batch_cellids[i, :] = target_batch_cellids[i, target_perm]
            target_batch_assayids[i, :] = target_batch_assayids[i, target_perm]

        inp = {'supports': support_batch_X,
               'support_cell_ids': support_batch_cellids,
               'support_assay_ids': support_batch_assayids,
               'target_cell_ids': target_batch_cellids,
               'target_assay_ids': target_batch_assayids}

        return format_inputs(inp, self.return_dict), target_batch_X
