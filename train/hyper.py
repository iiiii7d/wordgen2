from dataclasses import dataclass

from optuna import Trial


@dataclass
class Hyper:
    batch_size: int = 1
    layers: int = 1
    hidden_size: int = 128
    learning_rate: float = 1e-2
    chunk_size: int = 10
    momentum: float = 0.9

    @classmethod
    def trial(cls, trial: Trial):
        c = cls()
        c.layers = trial.suggest_int("layers", 1, 5)
        c.hidden_size = trial.suggest_int("hidden_size", 16, 512, log=True)
        c.learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-0, log=True)
        c.chunk_size = trial.suggest_int("chunk_size", 5, 15)
        c.momentum = trial.suggest_float("momentum", 0.0, 1.0)
        return c
