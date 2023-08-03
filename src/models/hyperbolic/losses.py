import torch
import torch.nn as nn

    
class SGNSLoss(nn.Module):
    
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, y_, y):
        loss = -(y * torch.log(torch.sigmoid(-y_)) + (1-y) * torch.log(torch.sigmoid(y_)))
        if self.reduction is None:
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.rediction == "sum":
            return loss.sum()
        raise NotImplementedError()