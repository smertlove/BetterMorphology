from src.data_utils import get_train_dev_test_paths
from src.model import MorphologyClassifier
from src.inference import infer
from src.categories import names_order, name2mapping_from_id
from src.data_utils import SubsetAndPath
from argparse import ArgumentParser
from conllu import parse_incr, TokenList
from itertools import batched
from copy import deepcopy
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import torch
from tqdm import tqdm
from typing import cast
import sys
import stanza
from stanza.models.common.doc import Sentence

Sentence.add_property(
    'metadata', 
    default="template text",
    getter=lambda self: getattr(self, '_metadata', None),
    setter=lambda self, value: setattr(self, '_metadata', value)
)


def get_predictions_for_file(
    filename: Path,
    nlp,
    batch_size: int,
    device: str
) -> list[TokenList]:
    with open(filename, "r", encoding="utf-8") as file:
        it = parse_incr(file)
        all_preds = []

        for batch in batched(it, n=batch_size):
            sentences = [
                [
                    token["form"]
                    for token in token_list
                ]
                for token_list in batch
            ]

            doc = nlp(sentences)

            assert len(batch) == len(doc.sentences)
            for original_sentence, pred_sentence in zip(batch, doc.sentences):
                pred_sentence.index = original_sentence.metadata["sent_id"]  # TODO: this doesnt work actually
                pred_sentence.id = original_sentence.metadata["sent_id"]
            all_preds.extend(doc.sentences)

        return all_preds


def get_all_predictions_and_write_files(
    test_files: list[SubsetAndPath],
    target_dir: Path,
    nlp,
    batch_size: int,
    device: str,
) -> None:

    for pair in tqdm(test_files):
        all_preds = get_predictions_for_file(pair["path"], nlp, batch_size, device)

        cur_target_dir = target_dir / pair["subset"]
        cur_target_dir.mkdir(exist_ok=True)
        with open(cur_target_dir / pair["path"].name, 'w', encoding="utf-8") as f:
            f.writelines(
                [
                    "{:C}".format(pred) + "\n\n\n"
                    for pred in all_preds
                ]
            )


def main() -> None:

    parser = ArgumentParser()
    parser.add_argument("--run-name")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--description",
        required=False,
    )
    args = parser.parse_args()
    if not args.description:
        if sys.stdin.isatty():
            args.description = input("Experiment description: ").strip()
        if not args.description:
            parser.error("--description is required (or run interactively)")

    data_dir = args.data_dir
    test_files_splits = get_train_dev_test_paths(data_dir)["test"]

    target_dir = args.runs_dir / args.run_name
    assert not target_dir.exists()
    target_dir.mkdir()

    with open(target_dir / "README.md", "w", encoding="utf-8") as file:
        file.write(args.description.strip() + "\n")

    nlp = stanza.Pipeline('ru', tokenize_pretokenized=True)

    get_all_predictions_and_write_files(
        test_files_splits,
        target_dir,
        nlp,
        args.batch_size,
        args.device,
    )


if __name__ == "__main__":
    main()
