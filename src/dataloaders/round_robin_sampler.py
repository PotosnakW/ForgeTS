import itertools
import logging
import numpy as np
import random

from torch.utils.data import  Sampler, BatchSampler
from torch.utils.data import ConcatDataset

class DatasetSampler(Sampler):
    """
    A sampler for datasets concatenated using ConcatDataset. It generates indices for sampling 
    from individual datasets within the concatenated dataset, with optional shuffling.

    Args:
        dataset (ConcatDataset): The concatenated dataset.
        cumulative_sizes (list): Cumulative sizes of the individual datasets in the concatenated dataset.
        shuffle (bool): Whether to shuffle the datasets and samples within the datasets. Default is True.
    """
    def __init__(self, dataset: ConcatDataset, cumulative_sizes: list, shuffle: bool = True):
        self.dataset = dataset
        self.cumulative_sizes = cumulative_sizes
        self.shuffle = shuffle
        self.num_datasets = len(cumulative_sizes)
        
        # Generate the start and end indices for each dataset
        self.dataset_indices = self._generate_dataset_indices()

    def _generate_dataset_indices(self):
        """
        Generate start and end indices for each dataset in the concatenated dataset.

        Returns:
            list: A list of tuples where each tuple contains the start and end indices of a dataset.
        """
        dataset_indices = []
        start_idx = 0
        for size in self.cumulative_sizes:
            dataset_indices.append((start_idx, size))
            start_idx = size
        return dataset_indices

    def __iter__(self):
        """
        Generate an iterator over the indices of the concatenated dataset.

        Returns:
            iterator: An iterator over the dataset indices.
        """
        indices = []
        # Determine the order of datasets (shuffled or sequential)
        if self.shuffle:
            dataset_order = np.random.permutation(self.num_datasets)
        else:
            dataset_order = range(self.num_datasets)
        
        # Generate indices for each dataset in the determined order
        for dataset_idx in dataset_order:
            start, end = self.dataset_indices[dataset_idx]
            if self.shuffle:
                # Shuffle indices within the dataset
                indices += np.random.permutation(range(start, end)).tolist()
            else:
                indices += list(range(start, end))

        return iter(indices)

    def __len__(self):
        """
        Get the total number of samples in the concatenated dataset.

        Returns:
            int: The total number of samples.
        """
        return len(self.dataset)
    

class RoundRobinBatchSampler(BatchSampler):
    """
    A batch sampler that generates batches in a round-robin fashion from datasets grouped by channel numbers.
    This sampler ensures that batches are interleaved across datasets with the same channel number.

    Args:
        dataset_sampler (DatasetSampler): The sampler for the concatenated dataset.
        batch_size (int): The number of samples per batch.
        cumulative_sizes (list[int]): Cumulative sizes of the individual datasets in the concatenated dataset.
        channels_number (list[int]): A list specifying the channel number for each dataset.
        drop_last (bool): Whether to drop the last incomplete batch. Default is False.
    """
    def __init__(self, 
                 dataset_sampler: DatasetSampler, 
                 batch_size: int, 
                 cumulative_sizes: list[tuple[int, int]], 
                 number_of_channels_per_dataset: list[int] = None,
                 drop_last: bool = False):
        
        self.sampler = dataset_sampler
        self.batch_size = batch_size
        self.cumulative_sizes = cumulative_sizes
        self.num_datasets = len(cumulative_sizes)
        
        if number_of_channels_per_dataset is None:
            logging.warning("`number_of_channels_per_dataset` is None, using default of 1 for all datasets")
            number_of_channels_per_dataset = [1 for i in range(self.num_datasets)]
        self.number_of_channels_per_dataset = number_of_channels_per_dataset
        
        self.dataset_indices = dataset_sampler.dataset_indices
        self.dataset_lengths = [end - start for start, end in self.dataset_indices]
        self.drop_last = drop_last
        
        # Group dataset indices by channel number
        self.channel_groups = self._group_by_channels()
        self.interleave = True  # Interleaving datasets is the default behavior

    def _group_by_channels(self):
        """
        Group datasets by their channel number.

        This method organizes the dataset indices into groups based on the number
        of channels associated with each dataset. Each unique channel number acts
        as a key in the resulting dictionary, and the value is a list of tuples
        representing the start and end indices of datasets with that channel number.

        Returns:
            dict: A dictionary where the keys are channel numbers (int) and the
                  values are lists of tuples. Each tuple contains the start and
                  end indices of datasets with the corresponding channel number.
        """
        channel_groups = {}
        
        for i, (start, end) in enumerate(self.dataset_indices):
            n_channels = self.number_of_channels_per_dataset[i]
            
            if n_channels not in channel_groups:
                channel_groups[n_channels] = []
            
            channel_groups[n_channels].append((start, end))
        
        return channel_groups

    def __iter__(self):
        """
        Generate an iterator that yields batches of indices in a round-robin fashion.

        The batches are created by interleaving samples from datasets grouped by channel numbers.
        If `drop_last` is False, the last incomplete batch is also yielded.

        Yields:
            list: A batch of indices.
        """
        # Create iterators for each dataset within the channel groups
        channel_iters = {
            n_channels: [iter(range(start, end)) for start, end in dataset_ranges]
            for n_channels, dataset_ranges in self.channel_groups.items()
        }  
        
        while any(channel_iters.values()):  # Continue until all iterators are exhausted
            # Get the list of available channel numbers with active iterators
            available_channel_numbers = [n for n in channel_iters if channel_iters[n]]
            
            if not available_channel_numbers:
                break
            
            # Randomly select a channel number for the current batch
            chosen_channel_number = random.choice(available_channel_numbers)
            iter_list = channel_iters[chosen_channel_number]

            # Shuffle the iterators within the chosen channel group
            random.shuffle(iter_list)
            batch = []
            
            for _ in range(self.batch_size):
                for dataset_iter in iter_list:
                    if dataset_iter:
                        try:
                            # Add the next sample from the iterator to the batch
                            batch.append(next(dataset_iter))
                        except StopIteration:
                            # Remove exhausted iterators from the list
                            iter_list.remove(dataset_iter)
                        if len(batch) == self.batch_size:
                            # Yield the batch when it reaches the desired size
                            yield batch
                            batch = []  # Reset batch after yielding
                            break
            # Yield the last incomplete batch if `drop_last` is False
            if batch and not self.drop_last:
                yield batch
                
def count_iterator_elements(iterator):
        # Create a copy of the iterator
        iterator_copy, iterator = itertools.tee(iterator)
        
        # Count the elements in the copy
        count = sum(1 for _ in iterator_copy)
        
        return count, iterator