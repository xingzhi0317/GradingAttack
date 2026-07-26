import functools
from contextlib import contextmanager
from typing import List, Union

import torch
import torch.nn.functional as F

from .base import BaseDefense


def _get_decoder_layers(model):
    candidates = [
        ("model", "layers"),
        ("language_model", "model", "layers"),
        ("model", "language_model", "layers"),
        ("model", "language_model", "model", "layers"),
    ]
    for path in candidates:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise ValueError("AttentionSharpening expects a model with decoder layers")


class AttentionSharpening(BaseDefense):
    """Sharpen attention distributions during defended inference.

    This migrates the RolePlay AS defense to the shared defense interface so it
    can also be used by GCG pipeline configs.
    """

    def __init__(
        self,
        temperature: float = 0.5,
        layers: Union[str, List[int]] = "all",
    ):
        super().__init__()
        if temperature <= 0 or temperature >= 1:
            raise ValueError(
                f"attention sharpening temperature must be in (0, 1), got {temperature}"
            )
        self.temperature = temperature
        self.layers = layers

    def requires_model_hooks(self) -> bool:
        return True

    def install_model_hooks(self, model) -> list:
        layers = _get_decoder_layers(model)

        removers = []
        for layer_idx in self._resolve_layer_indices(model):
            attn = layers[layer_idx].self_attn
            original_forward = attn.forward
            attn.forward = _make_sharpened_forward(original_forward, self.temperature)
            removers.append(lambda attn=attn, orig=original_forward: setattr(attn, "forward", orig))
        return removers

    def _resolve_layer_indices(self, model) -> List[int]:
        num_layers = len(_get_decoder_layers(model))
        if self.layers == "all":
            return list(range(num_layers))
        if isinstance(self.layers, list):
            return [i for i in self.layers if 0 <= i < num_layers]
        raise ValueError(f"Unsupported layers spec: {self.layers!r}")


@contextmanager
def _sharpened_softmax_context(temperature: float):
    orig_f_softmax = F.softmax
    orig_torch_softmax = torch.softmax

    def _scale(input, dim=None, dtype=None, **kw):
        if isinstance(input, torch.Tensor) and input.dim() == 4:
            input = input / temperature
        return orig_f_softmax(input, dim=dim, dtype=dtype, **kw)

    def _scale_torch(input, dim=None, dtype=None, **kw):
        if isinstance(input, torch.Tensor) and input.dim() == 4:
            input = input / temperature
        return orig_torch_softmax(input, dim=dim, dtype=dtype, **kw)

    F.softmax = _scale
    torch.softmax = _scale_torch
    try:
        yield
    finally:
        F.softmax = orig_f_softmax
        torch.softmax = orig_torch_softmax


def _make_sharpened_forward(original_forward, temperature: float):
    @functools.wraps(original_forward)
    def forward(*args, **kwargs):
        with _sharpened_softmax_context(temperature):
            return original_forward(*args, **kwargs)

    return forward
