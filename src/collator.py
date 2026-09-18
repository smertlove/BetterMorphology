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

        padded_lables = _pad_sequence([sample["labels"] for sample in samples], ignore_labels_vector, maxlen, torch.float)
        padded_input_ids = _pad_sequence([sample['input_ids'] for sample in samples], self.pad_token_id, maxlen, torch.long)
        padded_token_type_ids = _pad_sequence([sample['token_type_ids'] for sample in samples], self.pad_token_id, maxlen, torch.long)
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
        padded_uniq_token_type_ids_encoder = _pad_sequence(uniq_token_type_ids_encoder, self.encoder_pad_token_id, context_maxlen, torch.long)
        padded_uniq_attention_mask_encoder = _pad_sequence(uniq_attention_mask_encoder, 0, context_maxlen, torch.long)
        padded_encoder_context_mask = _pad_sequence([sample["encoder_context_mask"] for sample in samples], 0, context_maxlen, torch.long)

        padded_input_ids_decoder = _pad_sequence([sample["input_ids_decoder"] for sample in samples], self.decoder_pad_token_id, decoder_input_maxlen, torch.long)
        padded_token_type_ids_decoder = _pad_sequence([sample["token_type_ids_decoder"] for sample in samples], self.decoder_pad_token_id, decoder_input_maxlen, torch.long)
        padded_attention_mask_decoder = _pad_sequence([sample["attention_mask_decoder"] for sample in samples], 0, decoder_input_maxlen, torch.long)

        padded_labels = _pad_sequence([sample["labels"] for sample in samples], self.ignore_index_id, label_maxlen, torch.float)

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
        'input_ids': [1, 2, 3,],
        'token_type_ids': [0, 0, 0],
        'attention_mask': [1, 1, 1],
    })

    sample2 = TaskDefinedBatch("some_task", **{
        'labels': [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
        'input_ids': [1, 2, 3, 4, 5],
        'token_type_ids': [0, 0, 0, 0, 0],
        'attention_mask': [1, 1, 1, 1, 1],
    })

    print(
        collator(
            [sample1, sample2]
        )
    )
