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
    
def skip_gram(x, i, w):
    rhs = x[max(0, i-w):i] + x[i+1:i+w+1]
    lhs = [x[i]] * len(rhs)
    return lhs, rhs

class SkipGram:
    
    def __init__(self, window):
        self.window = window
        
    def __call__(self, x):
        z = [skip_gram(x, i, self.window) for i in range(len(x))]
        return list(zip(*z))