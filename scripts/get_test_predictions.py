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


def get_predictions_for_file(
    filename: Path,
    model: MorphologyClassifier,
    tokenizer: PreTrainedTokenizerFast,
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

            cur_preds = infer(
                sentences=sentences,
                model=model,
                tokenizer=tokenizer,
                device=device,
                batch_size=batch_size,
            )
            assert len(batch) == len(cur_preds)
            for original_sentence, pred_sentence in zip(batch, cur_preds):
                pred_sentence.metadata = deepcopy(original_sentence.metadata)

            all_preds.extend(cur_preds)

        return all_preds


def get_all_predictions_and_write_files(
    test_files: list[SubsetAndPath],
    target_dir: Path,
    model: MorphologyClassifier,
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int,
    device: str,
) -> None:

    for pair in tqdm(test_files):
        all_preds = get_predictions_for_file(pair["path"], model, tokenizer, batch_size, device)

        cur_target_dir = target_dir / pair["subset"]
        cur_target_dir.mkdir(exist_ok=True)
        with open(cur_target_dir / pair["path"].name, 'w', encoding="utf-8") as f:
            f.writelines([pred.serialize() + "\n" for pred in all_preds])


def main() -> None:

    parser = ArgumentParser()
    parser.add_argument("--run-name")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--state-dict-path")
    parser.add_argument("--backbone-model-id")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")

    args = parser.parse_args()

    DATA_PATH = "/mnt/data_storage/datasets/conllu/rubic_data-master"
    test_files_splits = get_train_dev_test_paths(DATA_PATH)["test"]

    CURDIR = Path(__file__).parent.resolve()

    target_dir = CURDIR / "data" / "runs" / args.run_name
    assert not target_dir.exists()
    target_dir.mkdir()

    tokenizer = cast(PreTrainedTokenizerFast, AutoTokenizer.from_pretrained(args.tokenizer_path))
    model = MorphologyClassifier(names_order, name2mapping_from_id, args.backbone_model_id)
    state_dict = torch.load(args.state_dict_path)
    model.load_state_dict(state_dict)
    model.to(args.device)

    get_all_predictions_and_write_files(
        test_files_splits,
        target_dir,
        model,
        tokenizer,
        args.batch_size,
        args.device,
    )


if __name__ == "__main__":
    main()
