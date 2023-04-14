# Train the model
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import json
import os
import wandb


# Set the parameters for the model
SIZE = 100
WINDOW_SIZE = 10
MIN_COUNT = 15
EPOCHS = 100
LIMIT = 99999999

model = Word2Vec(
    vector_size=SIZE,
    window=WINDOW_SIZE,
    min_count=MIN_COUNT,
    workers=4
)


loader = PlaylistIterator("./data")
print("Building vocabulary")
model.build_vocab(loader)
print("Training")
model.train(loader, epochs=EPOCHS, total_examples=1_000_000, compute_loss=True, \
    callbacks=[
        LogEpochLossCallback(),
        SaveModelCallback(
            "./checkpoints",
            "model",
            3
        )
    ]
)