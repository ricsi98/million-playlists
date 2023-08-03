import json
import logging


class PlaylistIterator:

    def __init__(self, files):
        self.files = files

    def __len__(self):
        if not hasattr(self, "__n_samples"):
            for i, _ in enumerate(iter(self)):
                pass
            self.__n_samples = i + 1
        return self.__n_samples

    def __iter__(self):
        for file in self.files:
            if file.endswith(".json"):
                with open(file) as f:
                    logging.debug(f"Reading {file}")
                    slice = json.loads(f.read())

                for plist in slice["playlists"]:
                    yield plist
            else:
                logging.warning(f"{file} not expected")