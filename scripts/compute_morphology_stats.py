# type: ignore

# Автоматически собранные файлы outliers.json и heuristics.json правятся вручную.
raise Exception("Подумай трижды, прежде чем запускать это, путник...")

from src.dataset import BaseConlluDataset
from src.data_utils import get_train_dev_test_paths
from collections import defaultdict
import json
from pathlib import Path

CURDIR = Path(__file__).parent.resolve()
DATADIR = CURDIR / "data"
DATADIR.mkdir(exist_ok=True)

OUT_FILE_freqs = DATADIR / "freqs.json"
OUT_FILE_heuristics = DATADIR / "heuristics.json"
OUT_FILE_warnings = DATADIR / "warnings.json"
FREQ_THR = 30
DATA_PATH = "/mnt/data_storage/datasets/conllu/rubic_data-master"
UNDEFINED = "[UND]"


splits = get_train_dev_test_paths(DATA_PATH)


class FeaturesExtractor(BaseConlluDataset):

    def _prepare_model_input(self, sentence):
      for token in sentence:
            feats = token['feats'] or dict()
            upos = token["upos"]
            if not (
                (upos is None)
                or
                (feats.get("Typo") == "Yes")
                or
                (feats.get("Anom") == "Yes")
            ):                

                yield upos, feats


dataset = FeaturesExtractor(splits["train"], None, None)

upos2feats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
upos_freqs = defaultdict(int)
tot = 0

# Compute upos freqs and features freqs per upos
for upos, feats in iter(dataset):
    tot += 1
    upos_freqs[upos] += 1
    for feat, val in feats.items():
        upos2feats[upos][feat][val] += 1

# Compute undefined features per upos
for upos, feats in upos2feats.items():
    for feat, values in feats.items():
        sum_defined = sum(values.values())
        sum_undefined = upos_freqs[upos] - sum_defined

        values[UNDEFINED] = sum_undefined

## Define heuristics: 

heuristics = defaultdict(lambda: defaultdict(str))
warnings = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

for upos, feats in upos2feats.items():
    for feat, vals in feats.items():
        if any([count < FREQ_THR for count in vals.values()]):

            if len(vals) < 3  :
                heuristics[upos][feat] = max(vals.keys(), key=lambda k: vals[k])

            else:
                for val, count in vals.items():
                    if count < FREQ_THR:
                        warnings[upos][feat][val] = f"{count} / {sum(vals.values())}"


with open(OUT_FILE_freqs, "w", encoding="utf-8") as file:
    json.dump(upos2feats, file, ensure_ascii=False, indent=2)

with open(OUT_FILE_heuristics, "w", encoding="utf-8") as file:
    json.dump(heuristics, file, ensure_ascii=False, indent=2)

with open(OUT_FILE_warnings, "w", encoding="utf-8") as file:
    json.dump(warnings, file, ensure_ascii=False, indent=2)
