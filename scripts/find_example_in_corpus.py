# type: ignore


from src.dataset import BaseConlluDataset
from src.data_utils import get_train_dev_test_paths
from pathlib import Path


DATA_PATH = "/mnt/data_storage/datasets/conllu/rubic_data-master"

N_EXAMPLES = 8

UPOS = "VERB"
FEATURE = "Voice"
VALUE = "Mid"
# VALUE = "Act,Pass"


splits = get_train_dev_test_paths(DATA_PATH)


class FeaturesExtractor(BaseConlluDataset):


    def _prepare_model_input(self, sentence):
      for token in sentence:

            feats = token['feats'] or dict()
            upos = token["upos"]

            if upos == UPOS and feats.get(FEATURE) == VALUE:                
                yield token, sentence


dataset = FeaturesExtractor(splits["train"], None, None)




for (token, sentence), _ in zip(iter(dataset), range(N_EXAMPLES)):
    print(token)
    print(sentence)
    print()
