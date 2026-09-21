from torch.utils.data import IterableDataset
from .data_utils import SubsetAndPath
from conllu import parse_incr, TokenList
import random
from copy import deepcopy
from typing import Iterator, Any
from transformers import PreTrainedTokenizer
from collections import UserDict
from uuid import uuid4
import torch

from .categories import (
    UNDEFINED,
    name2mapping_to_id,
    names_order,
)


class TaskDefinedBatch(UserDict[str, Any]):
    def __init__(self, task_name: str, **kwargs: Any):
        super().__init__(**kwargs)
        self.task_name = task_name

    def to(self, device: torch.device | str) -> "TaskDefinedBatch":
        for key in list(self.data.keys()):
            value = self.data[key]
            if isinstance(value, torch.Tensor):
                self.data[key] = value.to(device)
        return self


IGNORE_INDEX = -100


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
                    ## NOTE: Tokenizer-level truncation is potentially bad for syntax learning
                    # (most likely one of our future tasks).
                    # This avoids parsing explicitly large texts instead.
                    if len(sentence) > 256:
                        continue
                    yield sentence

    def _debug_repeat(self, sentences: Iterator[TokenList]) -> Iterator[TokenList]:
        """Infinitely repeats the first sentence for model fitting debug."""
        sentence = next(sentences, None)
        if sentence is None:
            return
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

    def __iter__(self) -> Iterator[TaskDefinedBatch]:

        subsets_paths = self._get_subsets_paths()
        stream = self._parse_sentences(subsets_paths)

        if self.debug_fit:
            stream = self._debug_repeat(stream)
        elif self.shuffle:
            stream = self._buffer_shuffle(stream)

        for sentence in stream:
            yield from self._prepare_model_input(sentence)


def transform_features(features: dict[str, str]) -> dict[str, str]:
    new_features = deepcopy(features)

    if features["upos"] is None:  # Change upos == None to X just in case
        new_features["upos"] = "X"

    if features.get("Case") == "Par":  # Equal in Russian, Gen2 more frequent in corpus
        new_features["Case"] = "Gen2"

    if features.get("Case") == "Nom1":  # probably markup mistake
        new_features["Case"] = "Nom"

    if features.get("Voice") == "Act,Pass":  # looked at the corpus, looks like its Mid in both cases (there are only 2)
        new_features["Voice"] = "Mid"

    if features.get("Clitic") == "Yes":  # just 1 occurrence in whole corpus, can remove
        del new_features["Clitic"]

    return new_features


def get_vector_from_features(feats: dict[str, str], ignore_this: bool = False) -> list[int]:

    if ignore_this:
        vector = [IGNORE_INDEX] * len(names_order)

    else:
        vector = []
        for name in names_order:
            mapping = name2mapping_to_id[name]
            val = feats.get(name, UNDEFINED)
            val_id = mapping[val]
            vector.append(val_id)

    return vector


class PosAndMorphologyDataset(BaseConlluDataset):

    def _prepare_model_input(self, sentence: TokenList) -> Iterator[TaskDefinedBatch]:

        words = [token['form'] for token in sentence]
        all_feats = []
        for token in sentence:

            # add upos as a regular feature for convenience, then preprocess features
            upos = token["upos"]
            feats = token['feats'] or dict()
            feats['upos'] = upos
            feats = transform_features(feats)

            all_feats.append(feats)

        encoder_inputs = self.encoder_tokenizer(
            words,
            return_tensors=None,
            is_split_into_words=True,
        )

        word_ids = encoder_inputs.word_ids()
        aligned_labels = []

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(get_vector_from_features(dict(), ignore_this=True))
            else:
                aligned_labels.append(get_vector_from_features(all_feats[word_idx], ignore_this=False))

        yield TaskDefinedBatch(
            task_name="pos+morphology",
            labels=aligned_labels,
            **encoder_inputs
        )


class LemmatizationDataset(BaseConlluDataset):

    def _prepare_model_input(self, sentence: TokenList) -> Iterator[TaskDefinedBatch]:

        words = [token['form'] for token in sentence]
        cur_uuid = uuid4()

        encoder_input = self.encoder_tokenizer(
            words,
            return_tensors=None,
            is_split_into_words=True,
        )
        encoder_word_ids = encoder_input.word_ids()
        encoder_input = {k + "_encoder": val for k, val in encoder_input.items()}

        for current_word_idx, token in enumerate(sentence):

            context_mask = [1 if wid == current_word_idx else 0 for wid in encoder_word_ids]

            decoder_input = self.decoder_tokenizer(
                token['form'],
                return_tensors=None,
            )
            decoder_input = {k + "_decoder": val for k, val in decoder_input.items()}

            labels = self.decoder_tokenizer.encode(token['lemma'])

            # TODO: Add uuid4 to avoid embedding exact same sentence len(tokens) times
            yield TaskDefinedBatch(
                task_name="lemmatization",
                sentence_uuid=cur_uuid,
                labels=labels,
                encoder_context_mask=context_mask,
                **encoder_input,  # same for every word in the sentence, pull contextualized vectors from here
                **decoder_input,
            )
