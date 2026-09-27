from .model import MorphologyClassifier
from .dataset import TaskDefinedBatch
from .categories import name2mapping_from_id, names_order, UNDEFINED
from transformers import PreTrainedTokenizer, AutoTokenizer
from collections import defaultdict
from itertools import batched
import torch


@torch.no_grad()
def infer(
    sentences: list[list[str]],
    model: MorphologyClassifier,
    tokenizer: PreTrainedTokenizer,
    device: torch.Device | str = "cpu",
    batch_size=64,
) -> list[dict[str, str]]:

    model.eval()
    result: list[list[dict[str, str]]] = []

    for batch in batched(sentences, n=batch_size):

        # --- Get model predictions ---
        encoder_inputs = tokenizer(
            batch,
            return_tensors="pt",
            is_split_into_words=True,
            padding=True
        ).to(device)
        all_word_ids = [encoder_inputs.word_ids(i) for i in range(len(batch))]
        tdb = TaskDefinedBatch("pos+morphology", **encoder_inputs)
        model_output = model(tdb)

        # --- Decode predictions ---
        cur_result = [
            [dict() for _ in range(len(sentence))]
            for sentence in batch
        ]
        # TODO: fix this code, it looks ugly
        for i, word_ids in enumerate(all_word_ids):
            prev_id = None
            for j, word_id in enumerate(word_ids):
                # Skip None ids
                # Get logits from first token of the word.
                # TODO: make better decoding
                if word_id is None or word_id == prev_id:
                    continue
                prev_id = word_id

                for name in names_order:
                    logits = model_output["per_category_logits"][name][i, j]
                    if logits.numel() == 1:
                        pred = int(logits.item() > 0)
                    else:
                        pred = int(logits.argmax().item())
                    decoded = name2mapping_from_id[name][pred]
                    if decoded != UNDEFINED:
                        cur_result[i][word_id][name] = decoded

        result.extend(cur_result)

    return result

if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    model = MorphologyClassifier(names_order=names_order, name2mapping_from_id=name2mapping_from_id, encoder_id="cointegrated/rubert-tiny2")
    state_dict = torch.load("./checkpoints/cpt_6/state_dict.pt")
    model.load_state_dict(state_dict)
    inputs = TaskDefinedBatch(
        "pos+morphology",

    )
    words = ["мама", "мыла", "раму",]
    sentences = [
        ["мама", "мыла", "раму",],
        ["люблю", "маму", "и", "раму"],
        "Глокая куздра штеко будланула бокра и курдячит бокрёнка".split()
    ]
    result = infer(
        sentences,
        model, tok, "cpu"
    )

    for words, cur_res in zip(sentences, result):
        for word, feats in zip(words, cur_res):
            print(word, feats)
        print()
