from glob import glob

import click
import pytorch_lightning as L
from dataset import DATASETS, BaseDataset
from hyper import Hyper
from model import Model, ModelCallback
from optuna import Trial
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader


def train(
    hyper: Hyper,
    dataset: str,
    checkpoint: int | None,
    epochs: int,
    trial: Trial | None = None,
):
    train_dataset: BaseDataset = DATASETS[dataset](hyper.chunk_size)
    val_dataset: BaseDataset = DATASETS[dataset](hyper.chunk_size)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hyper.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=3,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=hyper.batch_size,
        drop_last=True,
        num_workers=3,
    )
    for x, y in train_dataloader:
        click.echo(f"Shape of x: {x.shape} {x.dtype}")
        click.echo(f"Shape of y: {y.shape} {y.dtype}")
        break

    model = Model(train_dataset.id, len(train_dataset.chars), hyper)
    if checkpoint is not None:
        ckpt = glob(f"./lightning_logs/version_{checkpoint}/checkpoints/*.ckpt")[0]
    elif trial is None:
        ckpt = "last"
    else:
        ckpt = None

    if trial is None:
        callbacks = [ModelCallback(), ModelCheckpoint(save_last="link")]
    else:

        class PatchedPruning(PyTorchLightningPruningCallback, L.Callback):
            pass

        callbacks = [PatchedPruning(trial, monitor="train_loss")]

    trainer = L.Trainer(
        limit_train_batches=1000,
        limit_val_batches=100,
        log_every_n_steps=10,
        max_epochs=epochs,
        enable_checkpointing=trial is None,
        logger=TensorBoardLogger(".", log_graph=trial is None),
        callbacks=callbacks,
    )
    trainer.logger.log_hyperparams(hyper.__dict__)
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt,
    )

    return trainer.callback_metrics["train_loss"].item()
