import json
import h5py
import numpy as np
from collections import Counter
from functools import reduce

from ..utils import PlaylistIterator
from . import transform as T


class BufferedH5Writer:

    def __init__(self, file, dataset) -> None:
        self.file = file
        self.dataset = dataset

    def append(self, data: np.ndarray):
        pass


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
    with h5py.File(os.path.join(save_path, "skipgrams.h5")) as f:
        dset = f.create_dataset()
        for pl in playlists:
            pl = transform(pl)
            if len(pl) == 0: continue
            sgrams = sg(pl)
            lhs = reduce(list.__add__, sgrams[0])
            rhs = reduce(list.__add__, sgrams[1])
            lhs, rhs = np.array(lhs, dtype=dtype), np.array(rhs, dtype=dtype)
            data = np.stack((lhs, rhs), axis=1)
            print(data.shape)
            asd

import os
base_path = "./data/processed"
files = os.listdir(base_path)
files = [os.path.join(base_path, f) for f in files][:1]
print(files)
pl = PlaylistIterator(files)
print(convert("./data/kcore", pl, 1000))