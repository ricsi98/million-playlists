import geoopt
from . import layers, losses
import lightning as pl


class ManifoldSkipGram(pl.LightningModule):

    def __init__(self, manifold, num_embeddings, embedding_dim, similarity="distance", detach_other=False, opt_kwargs={"lr": 1e-3}):
        super().__init__()
        assert similarity in ["distance", "dot"]
        self.opt_kwargs = opt_kwargs
        self.detach_other = detach_other
        self.encoder = layers.ManifoldEmbedding(manifold, num_embeddings, embedding_dim)
        if similarity == "dot":
            self.sim = layers.ManifoldDotProduct(manifold)
        elif similarity == "distance":
            self.sim = layers.ManifoldDistance(manifold)
        self.loss_fn = losses.SGNSLoss()

    def forward(self, a, b):
        va, vb = self.encoder(a), self.encoder(b)
        if self.detach_other:
            vb.detach()
        return self.sim(va, vb)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x1, x2 = x[:, 0], x[:, 1]
        y_ = self(x1, x2)
        loss = self.loss_fn(y_, y)
        self.log("train_loss", loss.item(), prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        if "algo" in self.opt_kwargs:
            algo = self.opt_kwargs["algo"]
            if algo == "sgd":
                opt = geoopt.optim.RiemannianSGD
            elif algo == "adam":
                opt = geoopt.optim.RiemannianAdam
            else:
                raise NotImplementedError()
            kwargs = {k:v for k,v in self.opt_kwargs.items() if k != "algo"}
        else:
            opt = geoopt.optim.RiemannianSGD
            kwargs = self.opt_kwargs
        return opt(self.parameters(), **kwargs)