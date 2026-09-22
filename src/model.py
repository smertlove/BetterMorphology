from .dataset import TaskDefinedBatch
from torch import nn
import torch
from transformers import AutoModel
from typing import Any


class MorphologyClassifier(nn.Module):

    def __init__(self, names_order: tuple[str, ...], name2mapping_from_id: dict[str, dict[int, str]], encoder_id: str, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.names_order = names_order

        self.encoder = AutoModel.from_pretrained(encoder_id)
        self.encoder_dim = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)

        self.heads = nn.ModuleDict()
        for name in names_order:
            out_dim = len(name2mapping_from_id[name])
            if out_dim == 2:
                out_dim = 1
            self.heads[name] = nn.Linear(self.encoder_dim, out_dim)

    def _forward_morphology(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        x = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        x = self.dropout(x)

        per_category_logits = {
            name: head(x)
            for name, head in self.heads.items()
        }

        per_category_losses = dict()
        if labels is not None:
            for i, name in enumerate(self.names_order):

                cur_labels = labels[:, :, i]  # [bs, seqlen]
                cur_logits = per_category_logits[name]  # [bs, seqlen, nclasses]

                _, _, nclasses = cur_logits.shape
                logits_flat = cur_logits.view(-1, nclasses)  # [bs*seqlen, nclasses]
                labels_flat = cur_labels.view(-1)  # [bs*seqlen]

                task_is_binary = logits_flat.shape[-1] == 1
                if task_is_binary:
                    # Filter out invalid positions (where label == -100)
                    valid_mask = labels_flat != -100
                    logits_valid = logits_flat[valid_mask].squeeze(-1)
                    labels_valid = labels_flat[valid_mask].float()

                    criterion: nn.BCEWithLogitsLoss | nn.CrossEntropyLoss = nn.BCEWithLogitsLoss()
                    cur_loss = criterion(logits_valid, labels_valid)
                else:
                    criterion = nn.CrossEntropyLoss(ignore_index=-100)
                    cur_loss = criterion(logits_flat, labels_flat)

                per_category_losses[name] = cur_loss

        result = {
            "per_category_logits": per_category_logits,
        }
        if per_category_losses:
            result["per_category_losses"] = per_category_losses

        return result

    def _forward_lemmatization(self, **kwargs: Any) -> dict[str, dict[str, torch.Tensor]]:
        raise NotImplementedError

    def forward(
        self,
        task_defined_batch: TaskDefinedBatch,
    ) -> dict[str, dict[str, torch.Tensor]]:
        if task_defined_batch.task_name == "pos+morphology":
            return self._forward_morphology(**task_defined_batch)
        elif task_defined_batch.task_name == "lemmatization":
            return self._forward_lemmatization(**task_defined_batch)
        else:
            raise ValueError(f"Unknown task {task_defined_batch.task_name}")

    def train_backbone(self, flg: bool):
        for param in self.encoder.parameters():
            param.requires_grad = flg

    def save(self, path):
        torch.save(self.state_dict(), path)
