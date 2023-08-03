import json
import logging


class PlaylistIterator:

    def __init__(self, files, limit=None):
        self.files = files
        self.limit = limit

    def __len__(self):
        if self.limit is not None:
            return self.limit
        if not hasattr(self, "__n_samples"):
            for i, _ in enumerate(iter(self)):
                pass
            self.__n_samples = i + 1
        return self.__n_samples

    def __iter__(self):
        n = 0
        for file in self.files:
            if file.endswith(".json"):
                with open(file) as f:
                    logging.debug(f"Reading {file}")
                    slice = json.loads(f.read())

                for plist in slice["playlists"]:
                    if self.limit is not None and n > self.limit:
                        return
                    yield plist
                    n += 1
            else:
                logging.warning(f"{file} not expected")