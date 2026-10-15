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


# from transformers.models.gemma4.modeling_gemma4
class Gemma4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale

        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim), requires_grad=True)

    def _norm(self, hidden_states: torch.Tensor):
        mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        # Use torch.pow() (over torch.sqrt() or torch.rsqrt()) to address compiler differences between Torch and JAX
        return hidden_states * torch.pow(mean_squared, -0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed_output = self._norm(hidden_states.float())
        if self.with_scale:
            normed_output = normed_output * self.weight.float()
        return normed_output.type_as(hidden_states)


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

        self.decoder = GPT2LMHeadModel.from_pretrained(decoder_id)
        self.norm = Gemma4RMSNorm(self.encoder.config.hidden_size, with_scale=False)
        self.proj = torch.nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size, bias=False)

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

    def _embed_decoder_tokens(self, input_ids, start=0):
        word_embeds = self.decoder.transformer.wte(input_ids)       # [1, seq_len, 768]

        position_ids = torch.arange(start, word_embeds.shape[1] + start, dtype=torch.long, device=self.decoder.device).unsqueeze(0)
        pos_embeds = self.decoder.transformer.wpe(position_ids)     # [1, seq_len, 768]

        return word_embeds + pos_embeds

    def _forward_lemmatization(
        self,

        encoder_context_mask,
        sample2uniq_id,
        input_ids_encoder,
        token_type_ids_encoder,
        attention_mask_encoder,

        # dont use these yet, we configure input first
        input_ids_decoder,
        attention_mask_decoder,

        labels,
    ) -> dict[str, dict[str, torch.Tensor]]:

        # TODO: make this hyperparameter or hardcode 4
        K = 3  # max num of ctx vectors, 1 for debug, should be 4

        # embed sentences first
        ctx_embeddings = self.encoder(
            input_ids=input_ids_encoder,
            token_type_ids=token_type_ids_encoder,
            attention_mask=attention_mask_encoder
        ).last_hidden_state

        ctx_embeddings = self.proj(self.norm(ctx_embeddings))

        B, _ = encoder_context_mask.shape
        D = ctx_embeddings.shape[-1]

        # TODO: get rid of loop
        # sample2uniq_id has ids of batch we need to take atm, e.g. e.g. tensor([0, 1, 1])
        # encoder_context_mask has int 1/0 masks which show embedding ids we need to take
        # # e.g. tensor([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         #[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         #[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        pulled_ctx_embeddings = torch.zeros((B, K, D), device=self.decoder.device)
        ctx_masks      = torch.zeros((B, K), dtype=torch.long, device=ctx_embeddings.device)

        for i, (ctx_id, mask) in enumerate(zip(sample2uniq_id, encoder_context_mask)):

            cur_ctx    = ctx_embeddings[ctx_id,:]                         # [seqlen, D]
            # TODO: use some sort of pooling; do not truncate when m_i > K
            cur_tokens = cur_ctx[mask.bool()][:K]                       # [min(m_i, K), D]
            m = cur_tokens.shape[0]
            pulled_ctx_embeddings[i, :m] = cur_tokens
            ctx_masks[i, :m]       = 1

        # TODO: the following does not work :(
        # probably need to also pass attn mask for labels

        prompt_embeddings = self._embed_decoder_tokens(input_ids_decoder, start=K)

        full_decoder_input = torch.cat(
            [
                pulled_ctx_embeddings,
                prompt_embeddings
            ],
            dim=1
        )  # "ctx_embeds word_form_embeds labels_embeds"

        full_decoder_attn_mask = torch.cat(
            [
                ctx_masks,
                attention_mask_decoder,
            ],
            dim=1
        )

        full_labels = torch.cat(
            [
                torch.full((B, K), -100, device=self.decoder.device),  # -100 labels for ctx embeds 
                labels,
            ],
            dim=1
        )

        out = self.decoder(
            inputs_embeds=full_decoder_input,
            attention_mask=full_decoder_attn_mask,
            labels=full_labels,
        )
        return out

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

    def train_backbone(self, flg: bool) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = flg

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)
