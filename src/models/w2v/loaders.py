import os
import json
import logging


def _chunk_index(fname):
    return int(fname.split("_")[1].split(".")[0])


class PlaylistIterator:

    def __init__(self, path, limit=None):
        self.path = path
        self.limit = limit

    def __len__(self):
        if not hasattr(self, "__n_samples"):
            for i, _ in enumerate(iter(self)):
                pass
            self.__n_samples = i
        return self.__n_samples

    def __iter__(self):
        fnames = sorted(os.listdir(self.path), key=_chunk_index)

        n_read = 0
        for fname in fnames:
            if fname.endswith(".json"):
                fullpath = os.sep.join((self.path, fname))

                with open(fullpath) as f:
                    logging.info(f"Reading {fname}")
                    slice = json.loads(f.read())

                for plist in slice["playlists"]:
                    n_read += 1
                    yield plist
                    if self.limit is not None and n_read > self.limit:
                        return
            else:
                logging.warning(f"{fname} not expected")



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    pi = PlaylistIterator("./data/processed")
    for i in pi:
        pass
    print("DONE")