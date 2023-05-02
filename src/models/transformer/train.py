import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from lightning import LightningModule
import logging


def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class MaskedLanguageModel(LightningModule):

    def __init__(self, model, pad_token_id, lr=3e-5):
        super().__init__()
        self.model = model
        self.pad_token_id = pad_token_id
        self.lr = lr

    def forward(self, x, **kwargs):
        return self.model(x, apply_softmax=False, **kwargs)
    
    def _filter_out_too_shorts(self, batch):
        # bs * seq_len
        mask = (batch != self.pad_token_id).sum(dim=1) > 1
        return batch[mask]


    def training_step(self, batch, batch_idx):
        # batch: bs * seq_len
        inputs = self._filter_out_too_shorts(batch)
        assert inputs.shape[0] > 0, "Wrong batch (no sequence with more than 1 valid items) check _filter_out_too_shorts"
        print(inputs.shape[0])
        inputs = inputs.transpose(0,1)
        targets = inputs.contiguous()[1:]
        inputs = inputs[:-1]
        #print("TARGE", targets, "\nINPUT", inputs[:-1])

        seq_len = inputs.shape[0]
        src_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        #bs = batch.shape[0]
        #src_key_padding_mask = torch.zeros(bs, seq_len).bool()#(inputs == self.pad_token_id).T.bool()
        src_key_padding_mask = (inputs == self.pad_token_id).T.bool()

        #print("INPUT", inputs)
        #print("TARGET", targets)
        #print("SRC_MASK", src_mask)
        #print("PAD_TOKEN", self.pad_token_id)
        #print("PADDING_MASK", src_key_padding_mask)

        predictions = self(inputs, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        loss = F.cross_entropy(predictions.view(-1, self.model.ntoken), targets.view(-1), ignore_index=self.pad_token_id)
        #print(predictions[0,0], targets[0,0])
        self.log('train_loss', loss, prog_bar=True, on_step=True)
        #import time
        #time.sleep(.1)
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