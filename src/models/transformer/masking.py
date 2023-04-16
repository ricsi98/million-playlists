import torch


def _generate_attn_mask_single(seq_mask):
    """
        If a BoolTensor is provided, positions with True are not allowed 
        to attend while False values will be unchanged.
        (Softmax goes along -1 dimension)
    """
    n = seq_mask.shape[0]
    mask = torch.zeros((n,n), dtype=torch.bool)
    mask[:, seq_mask.nonzero()] = True
    return mask


def _generate_attn_mask_batch(seq_mask, n_heads):
    bs, n = seq_mask.shape
    mask = torch.zeros((bs, n, n), dtype=torch.bool)
    nz = seq_mask.nonzero()
    a, b = nz[:, 0], nz[:, 1]
    mask[a, :, b] = True
    if n_heads > 1:
        mask = mask.repeat(1, n_heads, 1)
        mask = mask.view(bs * n_heads, n, n)
    return mask


def generate_attn_mask(seq_mask, n_heads=1):
    if len(seq_mask.shape) == 1:
        return _generate_attn_mask_single(seq_mask, n_heads)
    elif len(seq_mask.shape) == 2:
        return _generate_attn_mask_batch(seq_mask, n_heads)
    else:
        assert False, f"Input should be BATCH_SIZE * SEQ_LEN matrix, got {seq_mask.shape}"