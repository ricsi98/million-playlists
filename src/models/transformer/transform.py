import torch


class Compose:
    """Execute transforms sequentially"""
    
    def __init__(self, *tfs):
        self.tfs = tfs
        
    def __call__(self, x):
        for tf in self.tfs:
            x = tf(x)
        return x
    
    
class RemoveUnknownTracks:
    
    def __init__(self, known_tracks):
        kt = known_tracks
        if not isinstance(kt, set):
            kt = set(kt)
        self.kt = kt
        
    def __call__(self, x):
        return [xi for xi in x if xi in self.kt]
    
    
class TrackURI2Idx:
    
    def __init__(self, uri2idx, offset=0):
        self.offset = offset
        self.uri2idx = uri2idx
        
    def __call__(self, x):
        return [self.uri2idx[xi] + self.offset for xi in x]
    
    
class ToLongTensor:
    
    def __call__(self, x):
        return torch.LongTensor(x)
    

class PadOrTrim:
    
    def __init__(self, pad_token, target_length):
        self.token = pad_token
        self.t = target_length
    
    def __call__(self, x):
        if len(x) == self.t:
            return x
        if len(x) < self.t:
            return x + [self.token] * (self.t - len(x))
        return x[:self.t]
