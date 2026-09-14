from torch.utils.data import IterableDataset
import torch
from .data_utils import SubsetAndPath
from conllu import parse_incr, TokenList
import random
from copy import deepcopy
from typing import Iterator


class PosAndMorphologyDataset(IterableDataset[dict[str, TokenList]]):

    def __init__(
        self,
        subsets_paths: list[SubsetAndPath],
        debug_fit:bool=False,
        shuffle:bool=False,
        bufsize:int=10000,
    ):
        super().__init__()

        self.subsets_paths = subsets_paths
        self.debug_fit = debug_fit
        self.shuffle = shuffle
        self.bufsize = bufsize

    def _get_subsets_paths(self) -> list[SubsetAndPath]:
        subsets_paths = deepcopy(self.subsets_paths)
        if self.shuffle:
            random.shuffle(subsets_paths)
        return subsets_paths

    def _parse_sentences(self, subsets_paths: list[SubsetAndPath]) -> Iterator[TokenList]:
        for pair in subsets_paths:

            file_path = pair["path"]

            with open(file_path, "r", encoding="utf-8") as f:
                for sentence in parse_incr(f):
                    yield sentence

    def _debug_repeat(self, sentences: Iterator[TokenList]) -> Iterator[TokenList]:
        """Infinitely repeats the first sentence for model fitting debug."""
        sentence = next(sentences)
        while True:
            yield deepcopy(sentence)

    def _buffer_shuffle(self, sentences: Iterator[TokenList]) -> Iterator[TokenList]:
        """Buffer shuffle for streaming data."""
        buffer = []

        for sentence in sentences:
            buffer.append(sentence)
            if len(buffer) >= self.bufsize:
                idx = random.randint(0, len(buffer) - 1)
                yield buffer.pop(idx)

        # Drain remaining buffer
        random.shuffle(buffer)
        while buffer:
            yield buffer.pop()

    def _prepare_model_input(self, sentence: TokenList) -> dict[str, TokenList]:
        """Prepares actual model inputs and labels"""
        return {"inpt": sentence}

    def __iter__(self) -> Iterator[dict[str, TokenList]]:

        subsets_paths = self._get_subsets_paths()
        stream = self._parse_sentences(subsets_paths)

        if self.debug_fit:
            stream = self._debug_repeat(stream)
        elif self.shuffle:
            stream = self._buffer_shuffle(stream)

        for sentence in stream:
            yield self._prepare_model_input(sentence)
