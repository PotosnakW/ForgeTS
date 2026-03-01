import bisect
import numpy as np
import math
import random
from torch.utils.data.dataset import Dataset, T_co
from typing import Iterable, List, Tuple
import warnings

# todo NINA: removechannel folding from the infini code for pretraining!
class MultivariateConcatDataset(Dataset):
    r"""Dataset as a concatenation of multiple datasets, with support for splitting datasets
    with more than a specified number of channels into sub-datasets of exactly that number of channels.
    
    Args:
        datasets (sequence): List of datasets to be concatenated
        n_channels (int): Number of channels for splitting datasets with more channels
    """
    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]
    
    @staticmethod
    def cumsum(sequence: List[Dataset]) -> List[int]:
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r
    

    def __init__(self, datasets: Iterable[Dataset], n_channels: int = 8) -> None:
        super().__init__()
        
        self.n_channels = n_channels
        self.datasets = self._process_datasets(datasets)
        self.number_of_channels_per_dataset= [dataset.n_channels for dataset in self.datasets]
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'
        
        self.cumulative_sizes = self.cumsum(self.datasets)

    def _process_datasets(self, datasets: Iterable[Dataset]) -> List[Dataset]:
        processed_datasets = []
        for dataset in datasets:
            # print(f"Processing dataset {dataset} with {dataset.n_channels} channels, ")
            # print(f"Dataset length: {len(dataset)}")
            # print(f"Dataset data shape: {dataset.data.shape}")

            if self._has_more_channels(dataset):
                # print(f"Splitting dataset {dataset} into sub-datasets with exactly {self.n_channels} channels.")
                processed_datasets.extend(self._split_dataset(dataset))
            else:
                processed_datasets.append(dataset)
        return processed_datasets

    def _has_more_channels(self, dataset: Dataset, strict: bool = True) -> bool:
        # Check if any item in the dataset has more channels than n_channels
        if dataset.n_channels > self.n_channels:  # Assuming data.shape[1] is the number of channels
            return True
        elif strict and dataset.n_channels > 1: 
            return True
        return False

    def _split_dataset(self, dataset: Dataset) -> List[Dataset]:
        """Split the dataset into sub-datasets with exactly n_channels channels."""
        sub_datasets = []
        channels = dataset.n_channels
        num_items = len(dataset)
        
        num_items = math.ceil(channels / self.n_channels)
        available_channels = list(range(channels))

        # print(f"Splitting dataset with {channels} channels into {num_items} sub-datasets with {self.n_channels} channels each.")
        
        for _ in range(num_items):
            try:
                selected_channels = random.sample(available_channels, k=self.n_channels)
                for channel in selected_channels:
                    available_channels.remove(channel)  
            
            except ValueError: # occurs if too little channels are left
                number_remaining_channels = len(available_channels)
                number_channels_buffer = self.n_channels - number_remaining_channels
                selected_channels = random.choices(available_channels, k = number_channels_buffer) + available_channels    
            
            # delete those choices from available channels
            sub_dataset = ChannelSubsetDataset(dataset, selected_channels)
            sub_datasets.append(sub_dataset)
        
        return sub_datasets

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]
    
    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

class ChannelSubsetDataset(Dataset):
    def __init__(self, original_dataset, selected_channels):
        super().__init__()
        
        # Infer how the dataset is structured
        # channel_dim = original_dataset.data.shape.index(self.n_channels)
    
        # Copy attributes from the original dataset
        self.__dict__.update(original_dataset.copy_self_without_data().__dict__)
        self.dataset = original_dataset.copy_self_without_data()
        self.selected_channels = selected_channels
        self.n_channels = len(selected_channels)

        # print(f"{original_dataset.data.shape=}")

        self.dataset.data = np.copy(original_dataset.data[:, self.selected_channels])
        self.dataset.n_channels = len(selected_channels)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)