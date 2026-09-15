from torch.utils.data import IterableDataset
from .data_utils import SubsetAndPath
from conllu import parse_incr, TokenList
import random
from copy import deepcopy
from typing import Iterator, TypedDict
from transformers import PreTrainedTokenizer
from collections import UserDict
from .categories import UPOS2ID, UNDEFINED


class TaskDefinedBatch(UserDict[str, list[int]]):
    def __init__(self, task_name: str, **kwargs: list[int]):
        super().__init__(kwargs)
        self.task_name: str = task_name


class BaseConlluDataset(IterableDataset[TaskDefinedBatch]):

    def __init__(
        self,
        subsets_paths: list[SubsetAndPath],
        encoder_tokenizer: PreTrainedTokenizer,
        decoder_tokenizer: PreTrainedTokenizer,
        debug_fit: bool = False,
        shuffle: bool = False,
        bufsize: int = 10000,
    ):
        super().__init__()

        self.subsets_paths = subsets_paths
        self.debug_fit = debug_fit
        self.shuffle = shuffle
        self.bufsize = bufsize
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer

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

    def _prepare_model_input(self, sentence: TokenList) -> Iterator[TaskDefinedBatch]:
        """Prepares actual model inputs and labels"""
        raise NotImplementedError

    def _get_sentence_as_string(self, sentence: TokenList) -> str:
        return " ".join([token['form'] for token in sentence])

    def __iter__(self) -> Iterator[TaskDefinedBatch]:

        subsets_paths = self._get_subsets_paths()
        stream = self._parse_sentences(subsets_paths)

        if self.debug_fit:
            stream = self._debug_repeat(stream)
        elif self.shuffle:
            stream = self._buffer_shuffle(stream)

        for sentence in stream:
            yield from self._prepare_model_input(sentence)


class PosAndMorphologyDataset(BaseConlluDataset):

    def _prepare_model_input(self, sentence: TokenList) -> Iterator[TaskDefinedBatch]:

        sentence_str = self._get_sentence_as_string(sentence)

        yield TaskDefinedBatch(
            task_name="pos+morphology",
            labels=[
                UPOS2ID.get(
                    token['upos'],
                    UPOS2ID[UNDEFINED],
                )
                for token in sentence
            ],
            **self.encoder_tokenizer(sentence_str)
        )


class LemmatizationDataset(BaseConlluDataset):

    def _prepare_model_input(self, sentence: TokenList) -> Iterator[TaskDefinedBatch]:

        sentence_str = self._get_sentence_as_string(sentence)

        encoder_inputs = self.encoder_tokenizer(sentence_str)
        encoder_inputs = {k+"_encoder": val for k, val in encoder_inputs.items()}

        for token in sentence:

            # TODO: define which tokens in the encoder input correspond to current lemma (make a mask)

            decoder_inputs = self.decoder_tokenizer(token['form'])
            decoder_inputs = {k + "_decoder": val for k, val in decoder_inputs.items()}

            labels = self.decoder_tokenizer.encode(token['lemma'])

            yield TaskDefinedBatch(
                task_name="lemmatization",
                labels=labels,
                **encoder_inputs,
                **decoder_inputs,
            )
