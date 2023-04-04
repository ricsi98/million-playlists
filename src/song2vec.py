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

w_run = wandb.init(
    group="experiment_1",
    project="song2vec",
    config={
        "epochs": EPOCHS,
        "vector_length": SIZE,
        "min_count": MIN_COUNT,
        "context_window_size": WINDOW_SIZE
    }
)

def log(epoch, loss):
    wandb.log({"epoch": epoch, "loss": loss})


class PlaylistIterator:

    def __init__(self, path):
        self.path = path

    def __iter__(self):
        for fname in sorted(os.listdir(self.path))[:LIMIT]:
            if fname.startswith("mpd.slice.") and fname.endswith(".json"):
                fullpath = os.sep.join((self.path, fname))
                f = open(fullpath)
                js = f.read()
                f.close()
                slice = json.loads(js)
                for plist in slice["playlists"]:
                    sequence = [song["track_uri"] for song in plist["tracks"]]
                    yield sequence


class LogEpochLossCallback(CallbackAny2Vec):

    def __init__(self) -> None:
        self.epochs = 1
        self.prev_loss = 0
    
    def on_epoch_end(self, model: Word2Vec):
        if self.epochs > 1:
            loss = model.get_latest_training_loss() - self.prev_loss
        else:
            loss = model.get_latest_training_loss()
        self.prev_loss = model.get_latest_training_loss()
        print(f"Epoch {self.epochs} loss {loss:.5f}")
        log(self.epochs, loss)
        self.epochs += 1
    

class SaveModelCallback(CallbackAny2Vec):

    def __init__(self, path, prefix, modulo=1) -> None:
        self.path = path
        self.prefix = prefix
        self.modulo = modulo
        self.epoch = 0

    def on_epoch_end(self, model):
        self.epoch += 1
        if self.epoch % self.modulo != 0:
            return
        path = os.path.join(self.path, f"{self.prefix}_{self.epoch}.model")
        model.save(path)


    def on_train_end(self, model):
        path = os.path.join(self.path, f"{self.prefix}_final.model")
        model.save(path)


# Train the model
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

wandb.finish()