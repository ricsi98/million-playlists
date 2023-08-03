import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class TransformerModel(nn.Module):

    def __init__(self, embeddings, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        ntoken, d_model = embeddings.shape
        self.ntoken = ntoken
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.decoder = nn.Linear(d_model, ntoken, bias=embeddings is None)

        self.init_weights(embeddings)

    def init_weights(self, embeddings) -> None:
        initrange = 0.1
        if embeddings is None:
            self.encoder.weight.data.uniform_(-initrange, initrange)
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)
        else:
            self.encoder.weight.data.copy_(embeddings)
            self.decoder.weight.data.copy_(embeddings)
            self.encoder.weight.requires_grad = False
            self.decoder.weight.requires_grad = False

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None, \
                apply_softmax: bool = False, **kwargs) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, batch_size]
        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model) # sl * bs * h
        src = self.pos_encoder(src) # sl * bs * h
        output = self.transformer_encoder(src, src_mask, **kwargs) # sl * bs * h
        seq_len, batch_size, d_latent = output.shape
        output = self.decoder(output.view(-1, d_latent)) # flatten sl*bs dims to apply decoder Linera model
        if apply_softmax:
            output = torch.softmax(output, dim=1)
        return output.view(seq_len, batch_size, self.ntoken)
    



class MaskedLanguageModel(LightningModule):

    def __init__(self, model, pad_token_id, lr=3e-5, device="cpu"):
        super().__init__()
        self.dev = device
        self.model = model.to(self.dev)
        self.pad_token_id = pad_token_id
        self.lr = lr

    def forward(self, x, **kwargs):
        return self.model(x, apply_softmax=False, **kwargs)
    
    def _filter_out_too_shorts(self, batch):
        # bs * seq_len
        mask = (batch != self.pad_token_id).sum(dim=1) > 1
        return batch[mask]

    def _src_mask(self, seq_len):
        if not hasattr(self, "__src_mask"):
            self.__src_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(self.dev)
        return self.__src_mask[:seq_len, :seq_len]

    def training_step(self, batch, batch_idx):
        # batch: bs * seq_len
        inputs = self._filter_out_too_shorts(batch)
        assert inputs.shape[0] > 0, "Wrong batch (no sequence with more than 1 valid items) check _filter_out_too_shorts"
        
        inputs = inputs.transpose(0,1).to(self.dev)
        targets = inputs.contiguous()[1:]
        inputs = inputs[:-1]
        
        seq_len = inputs.shape[0]
        src_mask = self._src_mask(seq_len)
        src_key_padding_mask = (inputs == self.pad_token_id).T.bool().to(self.dev)
        
        predictions = self(inputs, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        loss = F.cross_entropy(predictions.view(-1, self.model.ntoken), targets.view(-1), ignore_index=self.pad_token_id)
        self.log('train_loss', loss, prog_bar=True, on_step=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
