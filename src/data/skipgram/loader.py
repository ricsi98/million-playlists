import os
import json
import h5py
import torch
import random
import logging
import numpy as np
from . import constants
from torch.utils.data import Dataset


class H5Dataset(Dataset):

    DEFAULT_BUFFER_SIZE = 50_000_000

    def __init__(self, path: str, alpha: float = 0.75, n_negatives: int = 1, buffer_size: int = DEFAULT_BUFFER_SIZE) -> None:
        self.path = path
        self.n_negatives = n_negatives
        self.buffer_size = buffer_size
        self._init_negative_distribution(path, alpha)
        self.__data = {
            "chunk_index": -1,
            "data": torch.zeros((buffer_size, 2), dtype=torch.long)
        }
        
    def _init_negative_distribution(self, path: str, alpha: float):
        with open(os.path.join(path, constants.FNAME_FREQUENCIES), "r") as f:
            freqs = json.load(f)
            weights = np.array(list(freqs.values())) ** alpha
            weights /= np.sum(weights)
            self.cum_weights = np.cumsum(weights)
            self.ids = np.arange(len(weights), dtype=np.int64)

    def _get(self, index):
        chunk_index = index // self.buffer_size
        if chunk_index != self.__data["chunk_index"]:
            logging.info(f"Loading chunk {chunk_index}")
            path = os.path.join(self.path, constants.FNAME_SKIPGRAMS)
            with h5py.File(path, "r") as f:
                start_ = chunk_index * self.buffer_size 
                end_ = (chunk_index + 1) * self.buffer_size
                assert start_ < len(f["data"]), "index out of bounds"
                self.__data["data"][:] = torch.tensor(f["data"][start_:end_])
                self.__data["chunk_index"] = chunk_index
        offset = index - chunk_index * self.buffer_size
        return self.__data["data"][offset]

    def __len__(self):
        if not hasattr(self, "__h5_size"):
            path = os.path.join(self.path, constants.FNAME_SKIPGRAMS)
            with h5py.File(path, "r") as f:
                self.__h5_size = len(f["data"])
        return self.__h5_size

    def _negatives(self, k):
        return np.array(random.choices(self.ids, cum_weights=self.cum_weights, k=k))

    def __getitem__(self, index):
        return self._get(index)