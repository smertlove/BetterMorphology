from torch import nn
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, GPT2LMHeadModel
from typing import Any
from pathlib import Path
from .dataset import TaskDefinedBatch


class SumOfLosses(nn.Module):
    def __init__(self, weights: list[float]):
        super().__init__()
        self.weights: torch.Tensor
        self.register_buffer("weights", torch.tensor(weights))

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        w_losses = torch.stack(losses) * self.weights
        return w_losses.sum()


class MorphologyClassifier(nn.Module):

    def __init__(
        self,
        names_order: tuple[str, ...],
        name2mapping_from_id: dict[str, dict[int, str]],
        encoder_id: str,
        decoder_id: str,
        *args: Any, **kwargs: Any):
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

        n_tasks = len(names_order)

        self.multitask_loss = SumOfLosses([1/n_tasks for _ in range(n_tasks)])


    def _forward_morphology(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
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

                cur_labels = labels[:, :, i]  # [B, seqlen]
                cur_logits = per_category_logits[name]  # [B, seqlen, C]

                _, _, C = cur_logits.shape
                logits_flat = cur_logits.reshape(-1, C)  # [B*seqlen, C]
                labels_flat = cur_labels.reshape(-1)  # [B*seqlen]

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
            result["loss"] = self.multitask_loss([per_category_losses[name] for name in self.names_order])

        return result

    def _embed_decoder_tokens(self, input_ids: torch.Tensor, start: int = 0) -> torch.Tensor:
        word_embeds: torch.Tensor = self.decoder.transformer.wte(input_ids)       # [1, seq_len, 768]

        position_ids = torch.arange(start, word_embeds.shape[1] + start, dtype=torch.long, device=self.decoder.device).unsqueeze(0)
        pos_embeds: torch.Tensor = self.decoder.transformer.wpe(position_ids)     # [1, seq_len, 768]

        return word_embeds + pos_embeds

    def forward(
        self,
        task_defined_batch: TaskDefinedBatch,
    ) -> dict[str, dict[str, torch.Tensor]] | dict[str, torch.Tensor]:
        if task_defined_batch.task_name == "pos+morphology":
            return self._forward_morphology(**task_defined_batch)
        else:
            raise ValueError(f"Unknown task {task_defined_batch.task_name}")

    def train_backbone(self, flg: bool) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = flg

    def save(self, path: str | Path) -> None:
        torch.save(self, path)
