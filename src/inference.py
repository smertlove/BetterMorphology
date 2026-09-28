from .model import MorphologyClassifier
from .dataset import TaskDefinedBatch
from .categories import name2mapping_from_id, names_order, UNDEFINED
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from typing import cast
from itertools import batched
import torch
from conllu import TokenList, Token
from uuid import uuid4


@torch.no_grad()
def infer(
    sentences: list[list[str]],
    model: MorphologyClassifier,
    tokenizer: PreTrainedTokenizerFast,
    device: torch.device | str = "cpu",
    batch_size: int = 64,
) -> list[TokenList]:

    model.eval()
    result: list[TokenList] = []

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
        cur_feats: list[list[dict[str, str]]] = [
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
                        cur_feats[i][word_id][name] = decoded

        # --- Transform to conllu objects ---
        for words, feats_list in zip(batch, cur_feats):
            tokens: list[Token] = []
            for word, feats in zip(words, feats_list):
                token = Token(
                    form=word,
                    upos=feats["upos"],
                    feats={k: val for k, val in feats.items() if k != "upos"}, 
                )
                tokens.append(token)
            token_list = TokenList(tokens)
            result.append(token_list)

    return result


if __name__ == "__main__":

    # --- init model ---
    tok: PreTrainedTokenizerFast = cast(PreTrainedTokenizerFast, AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2"))
    model = MorphologyClassifier(names_order=names_order, name2mapping_from_id=name2mapping_from_id, encoder_id="cointegrated/rubert-tiny2")
    state_dict = torch.load("./checkpoints/cpt_6/state_dict.pt")
    model.load_state_dict(state_dict)
    inputs = TaskDefinedBatch(
        "pos+morphology",

    )

    # --- test inference ---
    sentences = [
        ["мама", "мыла", "раму",],
        ["Люблю", "маму", "и", "раму"],
        "Глокая куздра штеко будланула бокра и курдячит бокрёнка".split()
    ]
    result = infer(
        sentences,
        model, tok, "cpu", batch_size=2
    )

    for token_list in result:
        print(token_list.serialize())
