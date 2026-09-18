import torch
from typing import Any, cast
from .dataset import TaskDefinedBatch, IGNORE_INDEX


def _get_taskname_or_error(samples: list[TaskDefinedBatch]) -> str:
    distinct_tasks = {sample.task_name for sample in samples}
    assert len(distinct_tasks) == 1, f"Differrent task types in batch: {distinct_tasks}"
    task_name = distinct_tasks.pop()
    return task_name


def _pad_sequence(sequence: Any, pad_value: Any, maxlen: int, torch_dtype: torch.dtype) -> torch.Tensor:
    padded = [
        torch.tensor(
            elem + [pad_value] * (maxlen - len(elem)),
            dtype=torch_dtype
        )
        for elem in sequence
    ]

    return torch.stack(padded)


class PosAndMorphologyCollator:

    def __init__(self, pad_token_id: int, ignore_index_id: int):
        self.pad_token_id = pad_token_id
        self.ignore_index_id = ignore_index_id

    def __call__(self, samples: list[TaskDefinedBatch]) -> TaskDefinedBatch:

        task_name = _get_taskname_or_error(samples)
        labels_vector_len = len(samples[0]["labels"][0])
        maxlen = max(len(sample["input_ids"]) for sample in samples)

        ignore_labels_vector = [self.ignore_index_id] * labels_vector_len

        padded_lables = _pad_sequence([sample["labels"] for sample in samples], ignore_labels_vector, maxlen, torch.long)
        padded_input_ids = _pad_sequence([sample['input_ids'] for sample in samples], self.pad_token_id, maxlen, torch.long)
        padded_token_type_ids = _pad_sequence([sample['token_type_ids'] for sample in samples], 0, maxlen, torch.long)
        padded_attention_mask = _pad_sequence([sample['attention_mask'] for sample in samples], 0, maxlen, torch.long)

        result = {
            'labels': padded_lables,
            'input_ids': padded_input_ids,
            'token_type_ids': padded_token_type_ids,
            'attention_mask': padded_attention_mask,
        }

        return TaskDefinedBatch(
            task_name=task_name,
            **result
        )


class LemmatizationCollator:

    def __init__(
        self,
        encoder_pad_token_id: int,
        decoder_pad_token_id: int,
        ignore_index_id: int
    ):
        self.encoder_pad_token_id = encoder_pad_token_id
        self.decoder_pad_token_id = decoder_pad_token_id
        self.ignore_index_id = ignore_index_id

    def __call__(self, samples: list[TaskDefinedBatch]) -> TaskDefinedBatch:

        task_name = _get_taskname_or_error(samples)

        # We need to collapse contexts to avoid embedding same sentence multiple times
        ctx_uuid2idx = {
            uuid: idx
            for idx, uuid
            in enumerate(
                {sample['sentence_uuid'] for sample in samples}
            )
        }

        uniq_input_ids_encoder = [None] * len(ctx_uuid2idx)
        uniq_token_type_ids_encoder = [None] * len(ctx_uuid2idx)
        uniq_attention_mask_encoder = [None] * len(ctx_uuid2idx)
        sample2uniq_id = []

        for sample in samples:
            cur_uniq_idx = ctx_uuid2idx[sample["sentence_uuid"]]
            uniq_input_ids_encoder[cur_uniq_idx] = sample["input_ids_encoder"]
            uniq_token_type_ids_encoder[cur_uniq_idx] = sample["token_type_ids_encoder"]
            uniq_attention_mask_encoder[cur_uniq_idx] = sample["attention_mask_encoder"]
            sample2uniq_id.append(cur_uniq_idx)

        # Now pad
        # assert all(elem is not None for elem in uniq_input_ids_encoder)
        context_maxlen = max(
            len(cast(list[int], ids))
            for ids in uniq_input_ids_encoder
        )
        decoder_input_maxlen = max(len(sample["input_ids_decoder"]) for sample in samples)
        label_maxlen = max(len(sample["labels"]) for sample in samples)

        padded_uniq_input_ids_encoder = _pad_sequence(uniq_input_ids_encoder, self.encoder_pad_token_id, context_maxlen, torch.long)
        padded_uniq_token_type_ids_encoder = _pad_sequence(uniq_token_type_ids_encoder, 0, context_maxlen, torch.long)
        padded_uniq_attention_mask_encoder = _pad_sequence(uniq_attention_mask_encoder, 0, context_maxlen, torch.long)
        padded_encoder_context_mask = _pad_sequence([sample["encoder_context_mask"] for sample in samples], 0, context_maxlen, torch.long)

        padded_input_ids_decoder = _pad_sequence([sample["input_ids_decoder"] for sample in samples],
                                                 self.decoder_pad_token_id, decoder_input_maxlen, torch.long)
        padded_token_type_ids_decoder = _pad_sequence([sample["token_type_ids_decoder"] for sample in samples],
                                                      0, decoder_input_maxlen, torch.long)
        padded_attention_mask_decoder = _pad_sequence([sample["attention_mask_decoder"] for sample in samples], 0, decoder_input_maxlen, torch.long)

        padded_labels = _pad_sequence([sample["labels"] for sample in samples], self.ignore_index_id, label_maxlen, torch.long)

        result = {
            'labels': padded_labels,
            'encoder_context_mask': padded_encoder_context_mask,
            'sample2uniq_id': torch.tensor(sample2uniq_id, dtype=torch.long),
            'input_ids_encoder': padded_uniq_input_ids_encoder,
            'token_type_ids_encoder': padded_uniq_token_type_ids_encoder,
            'attention_mask_encoder': padded_uniq_attention_mask_encoder,
            'input_ids_decoder': padded_input_ids_decoder,
            'token_type_ids_decoder': padded_token_type_ids_decoder,
            'attention_mask_decoder': padded_attention_mask_decoder,
        }

        return TaskDefinedBatch(
            task_name=task_name,
            **result
        )


if __name__ == "__main__":
    collator = PosAndMorphologyCollator(pad_token_id=0, ignore_index_id=IGNORE_INDEX)

    sample1 = TaskDefinedBatch("some_task", **{
        'labels': [[0, 0], [0, 0], [0, 0]],
        'input_ids': [1, 2, 3, ],
        'token_type_ids': [0, 0, 0],
        'attention_mask': [1, 1, 1],
    })

    sample2 = TaskDefinedBatch("some_task", **{
        'labels': [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
        'input_ids': [1, 2, 3, 4, 5],
        'token_type_ids': [0, 0, 0, 0, 0],
        'attention_mask': [1, 1, 1, 1, 1],
    })

    print("=== PosAndMorphologyCollator ===")
    print(collator([sample1, sample2]))

    # ----- LemmatizationCollator -----
    collator2 = LemmatizationCollator(
        encoder_pad_token_id=0,
        decoder_pad_token_id=22,
        ignore_index_id=100,
    )

    # Two samples share the SAME sentence_uuid -> should collapse into one
    # unique encoder input. The third sample has its own context.
    #
    # Sample A: context len 3, decoder len 2, labels len 2
    # Sample B: context len 3 (same uuid as A), decoder len 3, labels len 3
    # Sample C: context len 5 (own uuid), decoder len 1, labels len 1
    sampleA = TaskDefinedBatch("lemma_task", **{
        'sentence_uuid': "uuid-A",
        'input_ids_encoder': [10, 11, 12],
        'token_type_ids_encoder': [0, 0, 0],
        'attention_mask_encoder': [1, 1, 1],
        'encoder_context_mask': [1, 1, 0],
        'input_ids_decoder': [20, 21],
        'token_type_ids_decoder': [0, 0],
        'attention_mask_decoder': [1, 1],
        'labels': [30, 31],
    })

    sampleB = TaskDefinedBatch("lemma_task", **{
        'sentence_uuid': "uuid-A",          # same context as A
        'input_ids_encoder': [10, 11, 12],
        'token_type_ids_encoder': [0, 0, 0],
        'attention_mask_encoder': [1, 1, 1],
        'encoder_context_mask': [0, 0, 1],
        'input_ids_decoder': [20, 21, 22],
        'token_type_ids_decoder': [0, 0, 0],
        'attention_mask_decoder': [1, 1, 1],
        'labels': [30, 31, 32],
    })

    sampleC = TaskDefinedBatch("lemma_task", **{
        'sentence_uuid': "uuid-C",          # different, longer context
        'input_ids_encoder': [40, 41, 42, 43, 44],
        'token_type_ids_encoder': [0, 0, 0, 0, 0],
        'attention_mask_encoder': [1, 1, 1, 1, 1],
        'encoder_context_mask': [0, 0, 1, 0, 0],
        'input_ids_decoder': [50],
        'token_type_ids_decoder': [0],
        'attention_mask_decoder': [1],
        'labels': [60],
    })

    batch = collator2([sampleA, sampleB, sampleC])

    print("\n=== LemmatizationCollator ===")
    for k, v in batch.items():
        if k == "task_name":
            print(f"{k}: {v}")
        else:
            print(f"{k}: shape={tuple(v.shape)}\n{v}\n")

    # ---- Sanity checks ----
    assert batch["input_ids_encoder"].shape[0] == 2, \
        "Expected 2 unique contexts (uuid-A, uuid-C)"

    # sample2uniq_id should map A,B -> 0 and C -> 1 (or the other way),
    # but the two samples sharing uuid-A must map to the SAME index.
    s2u = batch["sample2uniq_id"].tolist()
    assert s2u[0] == s2u[1], f"A and B must share a unique id, got {s2u}"
    assert s2u[2] != s2u[0], f"C must have its own id, got {s2u}"

    # Contexts padded to the longest unique context (len 5)
    assert batch["input_ids_encoder"].shape == (2, 5)
    assert batch["encoder_context_mask"].shape == (3, 5)

    # Decoder padded to the longest decoder input (len 3 from sample B)
    assert batch["input_ids_decoder"].shape == (3, 3)

    # Labels padded to the longest labels (len 3 from sample B),
    # and short ones filled with ignore_index_id=100
    assert batch["labels"].shape == (3, 3)
    assert batch["labels"][2, 1].item() == 100.0, \
        f"Expected ignore index padding, got {batch['labels'][2]}"

    print("All LemmatizationCollator checks passed.")
