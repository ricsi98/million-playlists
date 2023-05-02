import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from lightning import LightningModule
import logging


def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class MaskedLanguageModel(LightningModule):

    def __init__(self, model, mask_token_id, mask_prob, lr=3e-5):
        super().__init__()
        self.model = model
        self.mask_token_id = mask_token_id
        self.mask_prob = mask_prob
        self.lr = lr

    def forward(self, x, **kwargs):
        return self.model(x, apply_softmax=True, **kwargs)


    def training_step(self, batch, batch_idx):
        # batch: bs * seq_len
        inputs = batch
        inputs = inputs.transpose(0,1)
        targets = inputs.contiguous()[1:]

        seq_len = inputs.shape[0]
        src_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        src_key_padding_mask = (inputs == self.mask_token_id).T.bool()

        predictions = self(inputs, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        loss = F.cross_entropy(predictions[:-1].view(-1, self.model.ntoken), targets.view(-1), ignore_index=-100)
        self.log('train_loss', loss, prog_bar=True)
        print(loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch
        inputs = inputs.transpose(0,1)
        targets = inputs.contiguous()[1:]

        seq_len = inputs.shape[0]
        src_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        src_key_padding_mask = (inputs == self.mask_token_id).T.bool()

        predictions = self(inputs, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        loss = F.cross_entropy(predictions[:-1].view(-1, self.model.ntoken), targets.view(-1), ignore_index=-100)
        self.log('val_loss', loss, prog_bar=True)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)