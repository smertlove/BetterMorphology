from torch import nn
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, GPT2LMHeadModel
from typing import Any
from pathlib import Path
from .dataset import TaskDefinedBatch


# from transformers.models.gemma4.modeling_gemma4
class Gemma4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale

        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim), requires_grad=True)

    def _norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        # Use torch.pow() (over torch.sqrt() or torch.rsqrt()) to address compiler differences between Torch and JAX
        return hidden_states * torch.pow(mean_squared, -0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed_output = self._norm(hidden_states.float())
        if self.with_scale:
            normed_output = normed_output * self.weight.float()
        return normed_output.type_as(hidden_states)


class MorphologyAndLemmatization(nn.Module):

    def __init__(
        self,
        morphology_classifier_path: str,
        decoder_id: str,
        *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.morphology_classifier = torch.load(morphology_classifier_path, weights_only=False)

        self.decoder = GPT2LMHeadModel.from_pretrained(decoder_id)
        self.norm = Gemma4RMSNorm(self.encoder.config.hidden_size, with_scale=False)
        self.proj = torch.nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size, bias=False)


    def _embed_decoder_tokens(self, input_ids: torch.Tensor, start: int = 0) -> torch.Tensor:
        word_embeds: torch.Tensor = self.decoder.transformer.wte(input_ids)       # [1, seq_len, 768]

        position_ids = torch.arange(start, word_embeds.shape[1] + start, dtype=torch.long, device=self.decoder.device).unsqueeze(0)
        pos_embeds: torch.Tensor = self.decoder.transformer.wpe(position_ids)     # [1, seq_len, 768]

        return word_embeds + pos_embeds

    def _forward_lemmatization(
        self,

        encoder_context_mask: torch.Tensor,
        sample2uniq_id: torch.Tensor,
        input_ids_encoder: torch.Tensor,
        token_type_ids_encoder: torch.Tensor,
        attention_mask_encoder: torch.Tensor,

        # dont use these yet, we configure input first
        input_ids_decoder: torch.Tensor,
        attention_mask_decoder: torch.Tensor,

        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        # # TODO: make this hyperparameter or hardcode 4
        K = 4  # max num of ctx vectors, 1 for debug, should be 4

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

        prompt_embeddings = self._embed_decoder_tokens(input_ids_decoder)
        prompt_embeddings[:,:K] = pulled_ctx_embeddings  # add embeddings to slots 

        attention_mask_decoder[:,:K] = ctx_masks  # mask empty slots

        out:  dict[str, torch.Tensor] = self.decoder(
            inputs_embeds=prompt_embeddings,
            attention_mask=attention_mask_decoder,
            labels=labels,
        )
        return out

    def forward(
        self,
        task_defined_batch: TaskDefinedBatch,
    ) -> dict[str, dict[str, torch.Tensor]] | dict[str, torch.Tensor]:
        if task_defined_batch.task_name == "pos+morphology":
            return self.morphology_classifier._forward_morphology(task_defined_batch)
        elif task_defined_batch.task_name == "lemmatization":
            return self._forward_lemmatization(**task_defined_batch)
        else:
            raise ValueError(f"Unknown task {task_defined_batch.task_name}")

    def train_backbone(self, flg: bool) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = flg

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)


