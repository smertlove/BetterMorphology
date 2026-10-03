from src.data_utils import get_train_dev_test_paths
from src.categories import names_order
from itertools import zip_longest

from argparse import ArgumentParser
from conllu import parse_incr
from src.categories import names_order, UNDEFINED
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support
import json
from tqdm import tqdm


def get_all_golds_preds(gold: Path, pred: Path):
    assert pred.parent.name == gold.parent.parent.name

    all_golds: dict[str, list[str]]= defaultdict(list)
    all_preds: dict[str, list[str]]= defaultdict(list)

    with open(gold, "r", encoding="utf-8") as gold_file, open(pred, "r", encoding="utf-8") as pred_file:
        
        for gold_sent, pred_sent in zip_longest(parse_incr(gold_file), parse_incr(pred_file)):
            assert gold_sent is not None and pred_sent is not None
            assert len(gold_sent) == len(pred_sent) and gold_sent.metadata["sent_id"] == pred_sent.metadata["sent_id"]

            for gold_token, pred_token in zip(gold_sent, pred_sent):
                all_golds['upos'].append(gold_token['upos'])
                all_preds['upos'].append(pred_token['upos'])

                for name in names_order:
                    if name == 'upos': continue

                    gold_feat = (gold_token['feats'] or dict()).get(name, UNDEFINED)
                    pred_feat = (pred_token['feats'] or dict()).get(name, UNDEFINED)

                    if gold_feat == UNDEFINED and pred_feat == UNDEFINED:
                        continue
                    all_golds[name].append(gold_feat)
                    all_preds[name].append(pred_feat)

    return all_golds, all_preds


def cmp_conllus(golds: list[Path], preds: list[Path]):
    all_golds: dict[str, list[str]]= defaultdict(list)
    all_preds: dict[str, list[str]]= defaultdict(list)

    for gold, pred in tqdm(zip(golds, preds), total=len(golds)):
        cur_golds, cur_preds = get_all_golds_preds(gold, pred)
        for name in names_order:
            all_golds[name].extend(cur_golds[name])
            all_preds[name].extend(cur_preds[name])

    result = defaultdict(dict)

    for name in names_order:
        # TODO: s is always null, sth is probably wrong, needs fix
        p, r, f1, s = precision_recall_fscore_support(all_golds[name], all_preds[name], average='weighted', zero_division=0)
        result[name]["prec"] = p
        result[name]["rec"] = r
        result[name]["f1"] = f1
        result[name]["support"] = s

    return result


def main() -> None:

    parser = ArgumentParser()
    parser.add_argument("--run-name")
    parser.add_argument("--report-path")

    args = parser.parse_args()

    DATA_PATH = "/mnt/data_storage/datasets/conllu/rubic_data-master"
    test_files = [pair['path'] for pair in get_train_dev_test_paths(DATA_PATH)["test"]]

    CURDIR = Path(__file__).parent.resolve()
    run_dir = CURDIR / "data" / "runs" / args.run_name
    assert run_dir.exists()
    run_files = []
    for folder in run_dir.iterdir():
        for file in folder.iterdir():
            run_files.append(file)
    assert len(test_files) == len(run_files)

    result = cmp_conllus(
        sorted(test_files),
        sorted(run_files),
    )

    with open(args.report_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)



if __name__ == "__main__":
    main()
