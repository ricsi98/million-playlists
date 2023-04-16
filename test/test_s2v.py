import pytest
from src.models.w2v.loaders import PlaylistIterator


def test_pl_iterator():
    pi = PlaylistIterator("./data/processed", 100)
    l1 = list(iter(pi))
    l2 = list(iter(pi))
    for p1, p2 in zip(l1, l2):
        assert p1 == p2