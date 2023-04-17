import os
from argparse import ArgumentParser

import lightning as pl
import torch
from gensim.models import Word2Vec
from loader import PlaylistDataset
from model import TransformerModel


def _build_dataset(args):
    files = sorted([os.path.join(args.playlists, f) for f in os.listdir(args.playlists)])
    limit = args.limit if args.limit > 0 else None
    # TODO: transforms
    return PlaylistDataset(files, limit)


def _load_embeddings(path):
    wv = Word2Vec.load(path).wv
    return torch.tensor(wv)


def _build_model(args):
    return TransformerModel(
        embeddings=_load_embeddings(args.embeddings),
        nhead=args.heads,
        nlayers=args.layers,
        dropout=args.pdropout,
        d_hid=args.dhidden
    )


class ModelWrapper(pl.LightningModule):

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass


def main():
    parser = ArgumentParser()
    # I/O
    parser.add_argument("--embeddings", type=str, required=True, help="path to learned embeddings")
    parser.add_argument("--output", type=str, required=True, help="path where to save model")
    parser.add_argument("--playlists", type=str, required=True, help="path to playlist files")
    parser.add_argument("--ppf", type=int, default=50000, help="playlist per file (required for indexing)")
    parser.add_argument("--limit", type=int, default=-1, help="if specified, only use this many training examples (playlists)")
    
    # Model
    parser.add_argument("--positional", type=bool, default=False, help="whether to user positional embeddings")
    parser.add_argument("--heads", type=int, default=4, help="number of attention heads to user")
    parser.add_argument("--layers", type=int, default=2, help="number of encoder layers to use")
    parser.add_argument("--dhidden", type=int, default=64, help="size of hidden layers in transformer encoder")
    parser.add_argument("--pdropout", type=float, default=.5, help="dropout probability")

    # Optimizer
    ...

    args = parser.parse_args()
    
    dataset = _build_dataset(args)
    model = ModelWrapper(_build_model(args))
    trainer = pl.Trainer()

    trainer.fit(model, dataset)


if __name__ == "__main__":
    main()