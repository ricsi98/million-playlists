import os
import json
import h5py
import torch
import random
import logging
import numpy as np
from . import constants
from torch.utils.data import Dataset, Sampler, DataLoader


class H5Dataset(Dataset):

    DEFAULT_BUFFER_SIZE = 5_000_000

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
        pos = self._get(index)
        neg = torch.tensor([pos[0].item(), self._negatives(self.n_negatives)[0]])
        x = torch.stack((pos, neg), dim=0)
        y = torch.tensor([1] + [0] * self.n_negatives)
        return x, y
    


class RandomOrderSampler(Sampler[int]):

    def __init__(self, from_: int, to_: int, generator=None) -> None:
        assert to_ > from_
        self.from_ = from_
        self.to_ = to_
        self.n_ = to_ - from_
        self.generator = generator

    def __len__(self):
        return self.n_
    
    def __iter__(self):
        for i in torch.randperm(self.n_, generator=self.generator):
            yield self.from_ + i
    

class CustomH5Sampler(Sampler):
    """Sample indices within loaded chunk then skip to next chunk"""

    def __init__(self, data_source, generator=None) -> None:
        super().__init__(data_source)
        assert isinstance(data_source, H5Dataset)
        self.ds = data_source
        carry = 1 if len(self.ds) % self.ds.buffer_size > 0 else 0
        self.n_chunks = len(self.ds) // self.ds.buffer_size + carry
        self.generator = generator

    def __len__(self):
        return len(self.ds)
    
    def __iter__(self):
        chunk_sampler = RandomOrderSampler(0, self.n_chunks, self.generator)
        for chunk_index in chunk_sampler:
            chunk_index = chunk_index.item()
            start_ = chunk_index * self.ds.buffer_size
            end_ = min((chunk_index + 1) * self.ds.buffer_size, len(self.ds))
            sampler = RandomOrderSampler(start_, end_, self.generator)
            for i in sampler:
                yield i

def _collate_fn(sample):
    x, y = list(zip(*sample))
    return torch.cat(x, dim=0), torch.cat(y, dim=0)

def get_data_loader(
        path: str, 
        alpha: float = 0.75, 
        n_negatives: int = 1, 
        buffer_size: int = H5Dataset.DEFAULT_BUFFER_SIZE,
        **loader_kwargs
    ):
    ds = H5Dataset(
        path=path,
        alpha=alpha,
        n_negatives=n_negatives,
        buffer_size=buffer_size
    )
    sampler = CustomH5Sampler(ds)
    return DataLoader(
        dataset=ds,
        sampler=sampler, 
        collate_fn=_collate_fn, 
        **loader_kwargs
    )