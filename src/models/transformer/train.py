import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningModule
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


class MaskedLanguageModel(LightningModule):

    def __init__(self, model, mask_token_id, mask_prob, lr=3e-5):
        super().__init__()
        self.model = model
        self.mask_token_id = mask_token_id
        self.mask_prob = mask_prob
        self.lr = lr

    def forward(self, x, src_mask=None):
        return self.model(x, src_mask, apply_softmax=True)

    def create_combined_mask(self, src, pad_mask):
        seq_len = src.size(1)
        #subsequent_mask = (torch.triu(torch.ones(seq_len, seq_len)) == 0).float().masked_fill_(torch.triu(torch.ones(seq_len, seq_len)) == 0, float('-inf'))
        subsequent_mask = (torch.triu(torch.ones(seq_len, seq_len)) == 0).float().masked_fill_(torch.triu(torch.ones(seq_len, seq_len)) == 0, float('-inf'))
        logging.warning("HELO SUBSEQUENT MERET" + str(subsequent_mask.shape))
        combined_mask = subsequent_mask.unsqueeze(0) * pad_mask.unsqueeze(1).unsqueeze(2)
        logging.warning("HELO COMBINED MERET" + str(combined_mask.shape))
        return combined_mask

    def training_step(self, batch, batch_idx):
        inputs, pad_mask = batch
        inputs, labels = mask_tokens(inputs, self.mask_token_id, self.mask_prob)
        src_mask = self.create_combined_mask(inputs, pad_mask)
        
        logging.warning("HELO MASK MERET" + str(src_mask.shape))
        predictions = self(inputs, src_mask)
        # TODO: itt a predictions as sl * bs * ntoken mig a labels az bs * sl * ntoken
        loss = F.cross_entropy(predictions.view(-1, self.model.ntoken), labels.view(-1), ignore_index=-100)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, pad_mask = batch
        inputs, labels = mask_tokens(inputs, self.mask_token_id, self.mask_prob)
        src_mask = self.create_combined_mask(inputs, pad_mask)
        predictions = self(inputs, src_mask)
        loss = F.cross_entropy(predictions.view(-1, self.model.ntoken), labels.view(-1), ignore_index=-100)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
