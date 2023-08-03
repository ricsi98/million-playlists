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
    
    def __init__(self, uri2idx):
        self.uri2idx = uri2idx
        
    def __call__(self, x):
        return [self.uri2idx[xi] for xi in x]


def _skip_gram(x, i, w):
    rhs = x[max(0, i-w):i] + x[i+1:i+w+1]
    lhs = [x[i]] * len(rhs)
    return lhs, rhs

class SkipGram:
    
    def __init__(self, window):
        self.window = window
        
    def __call__(self, x):
        z = [_skip_gram(x, i, self.window) for i in range(len(x))]
        return list(zip(*z))
    

class PadOrTrim:
    
    def __init__(self, pad_token, target_length):
        self.token = pad_token
        self.t = target_length
    
    def __call__(self, x):
        if len(x) == self.t:
            return x,
        if len(x) < self.t:
            return x + [self.token] * (self.t - len(x))
        return x[:self.t]
    

class ToLongTensor:
    
    def __call__(self, x):
        return torch.LongTensor(x).view(-1)
    