import difflib
from glob import glob

import click
import optuna
import torch
import training
from dataset import DATASETS, BaseDataset
from hyper import Hyper
from model import Model
from optuna import Trial


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "-d",
    "--dataset",
    type=click.Choice(list(DATASETS.keys()), case_sensitive=False),
    default="english",
)
@click.option("-c", "--checkpoint", type=int, default=None)
@click.option("-e", "--epochs", type=int, default=250)
def train(dataset: str, checkpoint: int | None, epochs: int):
    hyper = Hyper()
    training.train(hyper, dataset, checkpoint, epochs)


@cli.command()
@click.option(
    "-d",
    "--dataset",
    type=click.Choice(list(DATASETS.keys()), case_sensitive=False),
    default="english",
)
@click.option("-e", "--epochs", type=int, default=10)
@click.option("-nt", "--n-trials", type=int, default=100)
@click.option("-t", "--timeout", type=int, default=600)
@click.option("-nj", "--num-jobs", type=int, default=4)
def optimize(dataset: str, epochs: int, n_trials: int, timeout: int, n_jobs: int):
    def train_optimize(trial: Trial):
        return training.train(Hyper.trial(trial), dataset, None, epochs, trial)

    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(train_optimize, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    best_trial = study.best_trial
    print("  Value: {}".format(best_trial.value))
    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))


@cli.command()
@click.option("-c", "--checkpoint", type=int, default=None)
@click.option("-p", "--prompt", type=str, default=" ")
@click.option("-n", "--num-words", type=int, default=20)
def predict(checkpoint: int | None, prompt: str, num_words: int):
    if checkpoint is not None:
        ckpt = glob(f"./lightning_logs/version_{checkpoint}/checkpoints/*.ckpt")[0]
    else:
        ckpt = "last"

    model = Model.load_from_checkpoint(ckpt).to("cpu")
    hyper = model.hyper
    dataset: BaseDataset = DATASETS[model.id](hyper.chunk_size)
    for _ in range(num_words):
        x = prompt
        y = ""
        while y != " ":
            xi = torch.Tensor([dataset.chars.index(c) for c in x]).long()
            y = model.predict_forward(xi, x in dataset.words).item()
            y = dataset.chars[y]
            x += y
        x = x.strip()
        nearest = difflib.get_close_matches(x, dataset.words)
        ratio = (
            difflib.SequenceMatcher(None, x, nearest[0]).ratio()
            if len(nearest) != 0
            else 0
        )
        click.echo(f"{x=} {nearest=} {ratio=}")


@cli.command()
def ray():
    pass


if __name__ == "__main__":
    cli()
