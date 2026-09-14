from torch.utils.data import IterableDataset
from .data_utils import SubsetAndPath
from conllu import parse_incr
import random
from copy import deepcopy


class PosAndMorphologyDataset(IterableDataset):

    def __init__(
        self,
        subsets_paths: list[SubsetAndPath],
        debug_fit=False,
        shuffle=False,
        bufsize=10000,
    ):
        super().__init__()

        self.subsets_paths = subsets_paths
        self.debug_fit = debug_fit
        self.shuffle = shuffle
        self.bufsize = bufsize

    def _get_subsets_paths(self):
        subsets_paths = deepcopy(self.subsets_paths)
        if self.shuffle:
            random.shuffle(subsets_paths)
        yield from subsets_paths

    def _parse_sentences(self, subsets_paths: list[SubsetAndPath]):
        for pair in subsets_paths:

            file_path = pair["path"]

            with open(file_path, "r", encoding="utf-8") as f:
                for sentence in parse_incr(f):
                    yield sentence

    def _debug_repeat(self, sentences):
        """Infinitely repeats the first sentence for model fitting debug."""
        sentence = next(sentences)
        while True:
            yield deepcopy(sentence)

    def _buffer_shuffle(self, sentences):
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

    def __iter__(self):

        subsets_paths = self._get_subsets_paths()
        stream = self._parse_sentences(subsets_paths)

        if self.debug_fit:
            stream = self._debug_repeat(stream)
        elif self.shuffle:
            stream = self._buffer_shuffle(stream)

        yield from stream
