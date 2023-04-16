import pytest
from src.models.w2v.loaders import PlaylistIterator

def get_iterator(k):
    return PlaylistIterator("./data/processed", k)


def test_pl_limit():
    k = 100
    pi = get_iterator(k)
    assert len(pi) == k, f"{len(pi)} != {k}"

def test_pi_same_lists():
    k = 100
    pi = get_iterator(k)
    l1 = list(iter(pi))
    l2 = list(iter(pi))
    assert len(l1) == k, f"{len(l1)} != {k}"
    assert len(l2) == k, f"{len(l1)} != {k}"
    for p1, p2 in zip(l1, l2):
        assert p1 == p2