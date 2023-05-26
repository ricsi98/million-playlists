import torch
import torch.nn as nn
import geoopt
import logging


class ManifoldEmbedding(nn.Module):
    
    def __init__(self, manifold, num_embeddings, embedding_dim, dtype=torch.double, requires_grad=True, weights=None):
        super().__init__()
        if dtype != torch.double:
            logging.warning("Double precision is recommended for embeddings on manifold")
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self._manifold = manifold
        if weights is None:
            data = torch.zeros((num_embeddings, embedding_dim), dtype=dtype)
            self.w = geoopt.ManifoldParameter(data, requires_grad=requires_grad, manifold=self._manifold)
            self.reset_parameters()
        else:
            raise NotImplementedError()
            
    def forward(self, x):
        s0 = x.shape
        ws = self.w[x.view(-1)]
        return ws.view(*s0, self.embedding_dim)
    
    def reset_parameters(self) -> None:
        nn.init.normal_(self.w.data, std=0.25)
        self.w.data[:] = self._manifold.retr(torch.zeros(self.embedding_dim), self.w.data)
        
        
class LorentzEmbedding(ManifoldEmbedding):
    
    def __init__(self, num_embeddings, embedding_dim, k=1.0, learnable=False, **kwargs):
        manifold = geoopt.manifolds.Lorentz(k, learnable=learnable)
        super().__init__(manifold, num_embeddings, embedding_dim, **kwargs)
        
        
class ManifoldDotProduct(nn.Module):
    
    def __init__(self, manifold):
        super().__init__()
        self._manifold = manifold
        
    def forward(self, a, b):
        x0 = torch.zeros(a.shape[-1]).to(a.device)
        return self._manifold.inner(x0, a, b)
    

class ManifoldDistance(nn.Module):

    def __init__(self, manifold: geoopt.Manifold) -> None:
        super().__init__()
        self._manifold = manifold

    def forward(self, a, b):
        return self._manifold.dist(a, b)