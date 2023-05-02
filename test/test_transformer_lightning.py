import pytest
import torch
from torch.utils.data import Dataset
from src.models.transformer.model import TransformerModel
from src.models.transformer.train import MaskedLanguageModel

# Dummy dataset for testing purposes
class DummyDataset(Dataset):
    def __init__(self, num_samples, seq_len, vocab_size):
        self.samples = torch.randint(vocab_size, (num_samples, seq_len))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

def test_masked_language_model_forward():
    seq_len, batch_size, ntoken = 10, 3, 20
    embeddings = torch.randn(ntoken, 300)
    model = TransformerModel(embeddings, nhead=4, d_hid=512, nlayers=2)
    lightning_model = MaskedLanguageModel(model, mask_token_id=2, mask_prob=0.15)

    src = torch.randint(ntoken, (seq_len, batch_size))
    out = lightning_model(src)
    assert out.shape == (seq_len, batch_size, ntoken), f"Expected output shape {(batch_size, seq_len, ntoken)}, but got {out.shape}"

def test_masked_language_model_training_step():
    seq_len, batch_size, ntoken = 10, 3, 20
    embeddings = torch.randn(ntoken, 300)
    model = TransformerModel(embeddings, nhead=4, d_hid=512, nlayers=2)
    lightning_model = MaskedLanguageModel(model, mask_token_id=2, mask_prob=0.15)

    src = torch.randint(ntoken, (batch_size, seq_len))
    batch = src
    loss = lightning_model.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor), f"Expected a tensor, but got {type(loss)}"

def test_masked_language_model_validation_step():
    seq_len, batch_size, ntoken = 10, 3, 20
    embeddings = torch.randn(ntoken, 300)
    model = TransformerModel(embeddings, nhead=4, d_hid=512, nlayers=2)
    lightning_model = MaskedLanguageModel(model, mask_token_id=2, mask_prob=0.15)

    src = torch.randint(ntoken, (batch_size, seq_len))
    batch = src
    lightning_model.validation_step(batch, 0)

def test_masked_language_model_create_combined_mask():
    seq_len, batch_size, ntoken = 10, 3, 20
    embeddings = torch.randn(ntoken, 300)
    model = TransformerModel(embeddings, nhead=4, d_hid=512, nlayers=2)
    lightning_model = MaskedLanguageModel(model, mask_token_id=2, mask_prob=0.15)

    src = torch.randint(ntoken, (batch_size, seq_len))
    pad_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    #combined_mask = lightning_model.create_combined_mask(src, pad_mask)

    #expected_shape = (batch_size, seq_len, seq_len)
    #assert combined_mask.shape == expected_shape, f"Expected combined mask shape {expected_shape}, but got {combined_mask.shape}."
