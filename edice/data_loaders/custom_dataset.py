
from edice.data_loaders.metadata import TrackMappingMixin
from edice.data_loaders.hdf5_datasets import HDF5Dataset


class CustomMetadata(TrackMappingMixin):
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




class CustomDataset(HDF5Dataset, CustomMetadata):
    pass

def load_custom_dataset(**kwargs):
    config = {}#dataset_configs[dataset_name].copy()
    # dataset = config.pop('dataset_class')
    config.update(kwargs)
    print(config)
    return CustomDataset(**config)


