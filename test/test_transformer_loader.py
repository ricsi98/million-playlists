import pytest
from src.models.transformer.loader import PlaylistDataset


def firstk(iterable, k):
    l = []
    for i in iterable:
        l.append(i)
        if len(l) >= k:
            return l


def test1():
    files = [
        "./data/processed/chunk_0.json"
    ]
    ds = PlaylistDataset(files, 50000, None)
    def iterable():
        for i in range(50000):
            yield ds[i]
    assert len(firstk(iterable(), 100)) == 100
    