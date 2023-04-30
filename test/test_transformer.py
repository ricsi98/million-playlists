import torch
from src.models.transformer.model import PositionalEncoding, TransformerModel


def test_1():
    batch_size = 3
    seq_len = 10
    embedding_dim = 8
    pe = PositionalEncoding(embedding_dim, 0, max_len=16)
    x = torch.zeros(seq_len, batch_size, embedding_dim)
    y = pe(x)
    assert y.shape == x.shape
    for i in range(batch_size-1):
        assert torch.allclose(y[:, i, :], y[:, i+1, :])


def test_2():
    batch_size = 3
    seq_len = 10
    embedding_dim = 8
    n_items = 60
    tf = TransformerModel(torch.zeros(n_items, embedding_dim), nhead=1, d_hid=16, nlayers=1, dropout=0)
    x = torch.ones(batch_size, seq_len, dtype=torch.long)
    y = tf(x, apply_softmax=True)

    assert y.shape == (seq_len, batch_size, n_items)
    assert torch.allclose(torch.ones(batch_size * seq_len), y.view(-1, n_items).sum(dim=1))




# Test PositionalEncoding
def test_positional_encoding():
    d_model = 512
    pe = PositionalEncoding(d_model)
    x = torch.randn(10, 3, d_model)
    out = pe(x)
    assert x.shape == out.shape, f"Expected output shape {x.shape}, but got {out.shape}"

# Test TransformerModel
def test_transformer_model():
    seq_len, batch_size, ntoken = 10, 3, 20
    embeddings = torch.randn(ntoken, 300)
    model = TransformerModel(embeddings, nhead=4, d_hid=512, nlayers=2)
    
    # Forward pass test
    src = torch.randint(ntoken, (batch_size, seq_len))
    out = model(src)
    assert out.shape == (seq_len, batch_size, ntoken), f"Expected output shape {(seq_len, batch_size, ntoken)}, but got {out.shape}"

    # Test with src_mask
    src_mask = (torch.triu(torch.ones(seq_len, seq_len)) == 0).float().masked_fill_(torch.triu(torch.ones(seq_len, seq_len)) == 0, float('-inf'))
    out_with_mask = model(src, src_mask)
    assert out.shape == out_with_mask.shape, f"Expected output shape {out.shape}, but got {out_with_mask.shape}"    