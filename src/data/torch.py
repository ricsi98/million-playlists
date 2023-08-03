import json
import logging
from torch.utils.data import Dataset


class PlaylistDataset(Dataset):
    
    def __init__(self, files, playlist_per_file, transform=None):
        self.files = files
        self.current_file_index = -1
        self.data = None
        self.ppf = playlist_per_file
        self.transform = transform
        
    def __len__(self):
        return self.ppf * len(self.files)
    
    def _load(self, path):
        with open(path, "r") as f:
            self.data = json.load(f)
    
    def __getitem__(self, index):
        file_index = index // self.ppf
        offset = index % self.ppf
        if self.current_file_index != file_index:
            logging.debug(f"Loading file {self.files[file_index]}")
            self._load(self.files[file_index])
            self.current_file_index = file_index
        tracks = self.data["playlists"][offset]
        
        if self.transform is not None:
            tracks = self.transform(tracks)
        
        return tracks
