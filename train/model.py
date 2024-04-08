import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyper import Hyper


class Model(L.LightningModule):
    def __init__(self, id_: str, alphabet_len: int, hyper: Hyper):
        super().__init__()
        self.gru = nn.GRU(
            alphabet_len, hyper.hidden_size, hyper.layers, batch_first=True
        )
        self.output = nn.Linear(hyper.hidden_size, alphabet_len)
        self.loss = nn.CrossEntropyLoss()
        self.hyper = hyper
        self.alphabet_len = alphabet_len
        self.id = id_
        self.example_input_array = (
            torch.randn(hyper.chunk_size),
            torch.zeros(hyper.layers, hyper.hidden_size),
        )
        self.save_hyperparameters()

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], _):
        x, y = batch
        out = self.train_forward(x)
        loss = self.loss(out.transpose(1, 2), y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], _):
        x, y = batch
        out = self.train_forward(x)
        loss = self.loss(out.transpose(1, 2), y)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

    def train_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.one_hot(x.long(), self.alphabet_len)
        h = torch.zeros(
            self.hyper.layers, self.hyper.batch_size, self.hyper.hidden_size
        )
        return self.output(self.gru(x.float(), h.to(x).float())[0])

    def predict_forward(self, x, skip_first: bool = False):
        h = torch.zeros(self.hyper.layers, self.hyper.hidden_size)
        y = self.forward(x, h)
        if skip_first:
            return y[1:].multinomial(1)
        else:
            return y.multinomial(1)

    def forward(self, x, h) -> torch.Tensor:
        x = F.one_hot(x.long(), self.alphabet_len)
        y = self.output(self.gru(x.float(), h.to(x).float())[0])
        return F.softmax(y[-1], 0)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.hyper.learning_rate, momentum=self.hyper.momentum
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class ModelCallback(L.Callback):
    def on_train_epoch_start(self, trainer, model: Model):
        model.to_onnx(f"data/{model.id}.onnx", export_params=True)
        trainer.save_checkpoint(f"data/{model.id}.ckpt")
