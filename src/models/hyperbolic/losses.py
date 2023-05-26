import torch
import torch.nn as nn

    
class SGNSLoss(nn.Module):
    
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, y_, y):
        y.masked_fill_(y == 0, -1)
        loss = -torch.log(torch.sigmoid(y * y_))
        if self.reduction is None:
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.rediction == "sum":
            return loss.sum()
        raise NotImplementedError()