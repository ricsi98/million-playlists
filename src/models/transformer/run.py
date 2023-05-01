import os
from argparse import ArgumentParser

import lightning as pl
import torch
import numpy as np
from gensim.models import Word2Vec
from loader import PlaylistDataset
from model import TransformerModel
from train import MaskedLanguageModel
from transform import *


PAD_TOKEN = 0


def _build_dataset(args, wv):
    files = sorted([os.path.join(args.playlists, f) for f in os.listdir(args.playlists)])

    transforms = Compose(
        RemoveUnknownTracks(wv.key_to_index.keys()),
        TrackURI2Idx(wv.key_to_index, offset=1),
        PadOrTrim(PAD_TOKEN, args.seqlen),
        ToLongTensor(),
        IncludePaddingMask(PAD_TOKEN)
    )

    limit = args.limit if args.limit > 0 else None
    assert limit is None, f"Currently not supported limit={args.limit, limit}"

    return PlaylistDataset(files, playlist_per_file=args.ppf, transform=transforms)


def _build_model(args):
    wv = Word2Vec.load(args.embeddings).wv
    dim = wv.vectors.shape[1]
    embeddings = np.concatenate((np.random.normal(size=(1, dim)), wv.vectors), axis=0)
    transformer = TransformerModel(
        embeddings=torch.tensor(embeddings),
        nhead=args.heads,
        nlayers=args.layers,
        dropout=args.pdropout,
        d_hid=args.dhidden
    )
    m = MaskedLanguageModel(transformer, 100, 0)
    return m, wv




def main():
    parser = ArgumentParser()
    # I/O
    parser.add_argument("--embeddings", type=str, required=True, help="path to learned embeddings")
    parser.add_argument("--output", type=str, required=True, help="path where to save model")
    parser.add_argument("--playlists", type=str, required=True, help="path to playlist files")
    parser.add_argument("--ppf", type=int, default=50000, help="playlist per file (required for indexing)")
    parser.add_argument("--seqlen", type=int, default=50, help="playlists are padded/trimmed to seqlen length")
    parser.add_argument("--limit", type=int, default=-1, help="if specified, only use this many training examples (playlists)")
    
    # Model
    parser.add_argument("--positional", type=bool, default=False, help="whether to user positional embeddings")
    parser.add_argument("--heads", type=int, default=4, help="number of attention heads to user")
    parser.add_argument("--layers", type=int, default=2, help="number of encoder layers to use")
    parser.add_argument("--dhidden", type=int, default=64, help="size of hidden layers in transformer encoder")
    parser.add_argument("--pdropout", type=float, default=.5, help="dropout probability")

    # Optimizer
    ...

    # Training
    parser.add_argument("--batchsize", type=int, default=256)

    args = parser.parse_args()
    
    model, wv = _build_model(args)
    dataset = _build_dataset(args, wv)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=False)
    trainer = pl.Trainer()

    trainer.fit(model, loader)


if __name__ == "__main__":
    main()