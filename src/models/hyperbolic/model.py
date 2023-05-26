import geoopt
import layers
import losses
import lightning as pl


class ManifoldSkipGram(pl.LightningModule):

    def __init__(self, manifold, num_embeddings, embedding_dim, similarity="distance", opt_kwargs={"lr": 1e-3}):
        super().__init__()
        assert similarity in ["distance", "dot"]
        self.encoder = layers.ManifoldEmbedding(manifold, num_embeddings, embedding_dim)
        if similarity == "dot":
            self.sim = layers.ManifoldDotProduct(manifold)
        elif similarity == "distance":
            self.sim = layers.ManifoldDistance(manifold)
        self.opt_kwargs = opt_kwargs
        self.loss_fn = losses.SGNSLoss()

    def forward(self, a, b):
        va, vb = self.encoder(a), self.encoder(b)
        return self.sim(va, vb)

    def training_step(self, batch, batch_idx):
        x1, x2, y = batch
        y_ = self(x1, x2)
        loss = self.loss_fn(y_, y)
        self.log("train_loss", loss.item())
        return loss

    def configure_optimizers(self):
        return geoopt.optim.RiemannianSGD(self.parameters(), **self.opt_kwargs)