import torch
from .dataset import TaskDefinedBatch, IGNORE_INDEX


class PosAndMorphologyDataCollator:

    def __init__(self, pad_token_id, ignore_index_id):
        self.pad_token_id = pad_token_id
        self.ignore_index_id = ignore_index_id

    def __call__(self, samples: list[TaskDefinedBatch]):

        distinct_tasks = {sample.task_name for sample in samples}
        assert len(distinct_tasks) == 1, f"Differrent task types in batch: {distinct_tasks}"

        task_name = distinct_tasks.pop()
        labels_vector_len = len(samples[0]["labels"][0])
        maxlen = max(len(sample["input_ids"]) for sample in samples)

        ignore_labels_vector = [self.ignore_index_id] * labels_vector_len

        padded_lables = [
            torch.tensor(
                sample["labels"] + [ignore_labels_vector] * (maxlen - len(sample["labels"]))
            )
            for sample in samples
        ]

        padded_input_ids = [
            torch.tensor(
                sample['input_ids'] + [self.pad_token_id] * (maxlen - len(sample["input_ids"])),
                dtype=torch.long
            )
            for sample in samples
        ]

        padded_token_type_ids = [
            torch.tensor(
                sample['token_type_ids'] + [self.pad_token_id] * (maxlen - len(sample["token_type_ids"])),
                dtype=torch.long
            )
            for sample in samples
        ]

        padded_attention_mask = [
            torch.tensor(
                sample['attention_mask'] + [0] * (maxlen - len(sample["attention_mask"])),
                dtype=torch.long
            )
            for sample in samples
        ]

        result = {
            'labels': torch.stack(padded_lables),
            'input_ids': torch.stack(padded_input_ids),
            'token_type_ids': torch.stack(padded_token_type_ids),
            'attention_mask': torch.stack(padded_attention_mask),
        }

        return TaskDefinedBatch(
            task_name=task_name,
            **result
        )


if __name__ == "__main__":
    collator = PosAndMorphologyDataCollator(pad_token_id=0, ignore_index_id=IGNORE_INDEX)

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