import json
import h5py
import numpy as np
from collections import Counter
from functools import reduce

from ..utils import PlaylistIterator
from . import transform as T


DATASET_NAME = "data"


class BufferedH5Writer:

    def __init__(self, file, dataset, buffer_size) -> None:
        self.file = file
        self.dataset = dataset
        self.buffer_size = buffer_size
        self.buffer = []
        self.index = 0
        
    def _len(self):
        if len(self.buffer) == 0:
            return 0
        return sum(len(xi) for xi in self.buffer)
        
    def _write(self):
        if self._len() == 0:
            return
        h, w = self.file[self.dataset].shape
        n = self._len()
        if n > h - self.index:
            self.file[self.dataset].resize((self.index + n, w))
        data = np.concatenate((self.buffer), axis=0).reshape(n, w)
        self.file[self.dataset][self.index:self.index+n,:] = data
        self.index += n
        self.file.flush()
        self.buffer = []

    def append(self, data: np.ndarray):
        self.buffer.append(data)
        if self._len() >= self.buffer_size:
            self._write()


def convert(save_path: str, playlists: PlaylistIterator, k: int = 0, dtype=np.int64):
    c = Counter()
    for pl in playlists:
        c.update(pl)
    c = dict(c)
    if k > 0:
        c = {song: freq for song,freq in c.items() if freq >= k}
    idx2song = {i:s for i,s in enumerate(c.keys())}
    song2idx = {s:i for i,s in idx2song.items()}
    
    # write index mapping
    with open(os.path.join(save_path, "idx2song.json"), "w") as f:
        json.dump(idx2song, f)

    # write frequencies file
    c_ = {song2idx[song]: freq for song,freq in c.items()}
    with open(os.path.join(save_path, "frequencies.json"), "w") as f:
        json.dump(c_, f,)

    # write h5 file
    transform = T.Compose(
        T.RemoveUnknownTracks(song2idx.keys()),
        T.TrackURI2Idx(song2idx)
    )
    sg = T.SkipGram(3)
    with h5py.File(os.path.join(save_path, "skipgrams.h5"), "w") as f:
        f.create_dataset(DATASET_NAME, (200, 2), maxshape=(None, 2))
        bw = BufferedH5Writer(f, DATASET_NAME, 50000)
        for pl in playlists:
            pl = transform(pl)
            if len(pl) == 0: continue
            sgrams = sg(pl)
            lhs = reduce(list.__add__, sgrams[0])
            rhs = reduce(list.__add__, sgrams[1])
            lhs, rhs = np.array(lhs, dtype=dtype), np.array(rhs, dtype=dtype)
            data = np.stack((lhs, rhs), axis=1)
            bw.append(data)
        bw._write()

import os
base_path = "./data/processed"
files = os.listdir(base_path)
files = [os.path.join(base_path, f) for f in files]
print(files)
pl = PlaylistIterator(files)
print(convert("./data/kcore", pl, 30))