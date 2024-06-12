import numpy as np
import pandas as pd


class Dataset:
    """ Data container. """
    def __init__(self, data: np.ndarray):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        return self.data[idx]
    

class DataLoader:
    """ Data loader. Loads samples from a dataset. """
    def __init__(self, dataset: Dataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self._iterator = 0
        self._num_batches = int(np.ceil(len(self.dataset) / self.batch_size))

    def __iter__(self):
        """ With each iteration returns a batch of samples. 
        After getting batch, increases iterator by given batch size. """
        for _ in range(self._num_batches):
            batch = self.dataset[self._iterator:self._iterator + self.batch_size]
            self._iterator += self.batch_size
            yield batch

        self._reset_state()

    def __len__(self):
        return self._num_batches
    
    def _reset_state(self):
        """ Resets loader's iterator. """
        self._iterator = 0
