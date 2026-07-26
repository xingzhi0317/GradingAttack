import functools
from contextlib import contextmanager
from typing import List, Union

import torch
import torch.nn.functional as F

from .base import BaseDefense


class HijackingSuppression(BaseDefense):
    """Suppress suffix-dominant attention during defended inference.

    The suffix length is set per sample before hooks are installed, which lets
    the same HS defense work for RolePlay prompts and optimized GCG suffixes.
    """

    def __init__(
        self,
        beta: float = 0.1,
        top_fraction: float = 0.01,
        layers: Union[str, List[int]] = "all",
    ):
        super().__init__()
        if not (0.0 < beta <= 1.0):
            raise ValueError(f"beta must be in (0, 1], got {beta}")
        if not (0.0 < top_fraction <= 1.0):
            raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}")
        self.beta = beta
        self.top_fraction = top_fraction
        self.layers = layers
        self._suffix_token_len = 0

    def requires_model_hooks(self) -> bool:
        return True

    def requires_inference_context(self) -> bool:
        return True

    def set_inference_context(
        self,
        tokenizer,
        prompt_content: str,
        attack_suffix: str = "",
    ) -> None:
        del prompt_content
        if not attack_suffix:
            self._suffix_token_len = 0
            return
        self._suffix_token_len = len(
            tokenizer.encode(attack_suffix, add_special_tokens=False)
        )

    def install_model_hooks(self, model) -> list:
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise ValueError("HijackingSuppression expects a causal LM with model.layers")

        state = _SuppressionState(
            suffix_token_len=self._suffix_token_len,
            beta=self.beta,
            top_fraction=self.top_fraction,
        )
        removers = []
        for layer_idx in self._resolve_layer_indices(model):
            attn = model.model.layers[layer_idx].self_attn
            original_forward = attn.forward
            attn.forward = _make_suppression_forward(original_forward, state)
            removers.append(lambda attn=attn, orig=original_forward: setattr(attn, "forward", orig))
        return removers

    def _resolve_layer_indices(self, model) -> List[int]:
        num_layers = len(model.model.layers)
        if self.layers == "all":
            return list(range(num_layers))
        if isinstance(self.layers, list):
            return [i for i in self.layers if 0 <= i < num_layers]
        raise ValueError(f"Unsupported layers spec: {self.layers!r}")


class _SuppressionState:
    def __init__(self, suffix_token_len: int, beta: float, top_fraction: float):
        self.suffix_token_len = suffix_token_len
        self.beta = beta
        self.top_fraction = top_fraction


def _apply_hijacking_suppression(
    attn_weights: torch.Tensor,
    suffix_token_len: int,
    beta: float,
    top_fraction: float,
) -> torch.Tensor:
    if attn_weights.dim() != 4:
        return attn_weights

    weights = attn_weights
    if suffix_token_len > 0:
        key_len = weights.size(-1)
        suffix_start = max(0, key_len - suffix_token_len)
        weights = weights.clone()
        weights[..., suffix_start:] = weights[..., suffix_start:] * beta

    if top_fraction > 0 and weights.size(-2) > 0:
        weights = weights.clone() if weights is attn_weights else weights
        last_q = weights[..., -1, :]
        flat = last_q.reshape(-1)
        n_top = max(1, int(flat.numel() * top_fraction))
        threshold = torch.topk(flat, n_top).values.min()
        hijack_mask = last_q >= threshold
        weights[..., -1, :] = torch.where(
            hijack_mask,
            weights[..., -1, :] * beta,
            weights[..., -1, :],
        )

    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return weights.to(attn_weights.dtype)


@contextmanager
def _suppression_softmax_context(state: _SuppressionState):
    orig_f_softmax = F.softmax
    orig_torch_softmax = torch.softmax

    def _patch(input, dim=None, dtype=None, **kw):
        weights = orig_f_softmax(input, dim=dim, dtype=dtype, **kw)
        if isinstance(weights, torch.Tensor) and weights.dim() == 4:
            if state.suffix_token_len > 0 or state.top_fraction > 0:
                weights = _apply_hijacking_suppression(
                    weights,
                    state.suffix_token_len,
                    state.beta,
                    state.top_fraction,
                )
        return weights

    def _patch_torch(input, dim=None, dtype=None, **kw):
        weights = orig_torch_softmax(input, dim=dim, dtype=dtype, **kw)
        if isinstance(weights, torch.Tensor) and weights.dim() == 4:
            if state.suffix_token_len > 0 or state.top_fraction > 0:
                weights = _apply_hijacking_suppression(
                    weights,
                    state.suffix_token_len,
                    state.beta,
                    state.top_fraction,
                )
        return weights

    F.softmax = _patch
    torch.softmax = _patch_torch
    try:
        yield
    finally:
        F.softmax = orig_f_softmax
        torch.softmax = orig_torch_softmax


def _make_suppression_forward(original_forward, state: _SuppressionState):
    @functools.wraps(original_forward)
    def forward(*args, **kwargs):
        with _suppression_softmax_context(state):
            return original_forward(*args, **kwargs)

    return forward
