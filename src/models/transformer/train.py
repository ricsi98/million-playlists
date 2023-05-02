import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from lightning import LightningModule
import logging


class MaskedLanguageModelDataset(Dataset):
    # Implement your dataset class to provide tokenized sequences
    ...


def mask_tokens(inputs, mask_token_id, mask_prob):
    """
    Prepare masked tokens inputs/labels for masked language modeling: randomly masks some of the tokens
    from the inputs to train the model to predict those tokens.
    """
    labels = inputs.clone()
    mask = torch.full(labels.shape, False, dtype=torch.bool)

    for i in range(inputs.size(0)):
        mask[i] = torch.bernoulli(torch.full(labels[i].shape, mask_prob)).bool()
        labels[i] = labels[i].masked_fill(mask[i], -100)
        inputs[i] = inputs[i].masked_fill(mask[i], mask_token_id)

    return inputs, labels


def add_subsequent_mask(src, mask):
    """1 in mask means we want to cover that"""
    seq_len = src.size(1)
    mask = mask.clone().masked_fill_(mask == 1, float('-inf'))
    subsequent_mask = (torch.triu(torch.ones(seq_len, seq_len)) == 0).float().masked_fill_(torch.triu(torch.ones(seq_len, seq_len)) == 0, float('-inf')).T
    logging.warning("HELO SUBSEQUENT MERET" + str(subsequent_mask.shape))
    combined_mask = subsequent_mask.unsqueeze(0) + mask.unsqueeze(1).unsqueeze(2)
    logging.warning("HELO COMBINED MERET" + str(combined_mask.shape))
    return combined_mask


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