from abc import ABC, abstractmethod
from random import shuffle
from typing import Type

import requests
import torch
from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    chars: list[str]
    words: list[str]
    train_data: list[str]
    id: str = "base"

    @staticmethod
    @abstractmethod
    def get_words() -> list[str]:
        pass

    def char2vec(self, word):
        try:
            return [self.chars.index(c) for c in word]
        except Exception as e:
            raise ValueError(repr(word)) from e

    def __init__(self, chunk_size: int):
        words = self.__class__.get_words()
        self.words = words
        words = " ".join(words)
        self.train_data = []
        for i in range(len(words) - chunk_size):
            self.train_data.append(words[i : i + chunk_size])

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        word = self.train_data[idx]
        x = torch.Tensor(self.char2vec(word[:-1])).long()
        y = torch.Tensor(self.char2vec(word[1:])).long()
        return x, y


class EnglishDataset(BaseDataset):
    id = "english"
    chars = " abcdefghijklmnopqrstuvwxyz"

    @staticmethod
    def get_words() -> list[str]:
        text = requests.get(
            "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
        ).text
        text = text.strip().replace("\r", "").split("\n")
        shuffle(text)
        return text


DATASETS: dict[str, Type[BaseDataset]] = {"english": EnglishDataset}
