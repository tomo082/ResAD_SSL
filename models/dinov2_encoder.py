import math
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn


DINOV2_EMBED_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


def default_dinov2_out_indices(model_name: str) -> List[int]:
    if model_name in ("dinov2_vitl14", "dinov2_vitg14"):
        return [5, 11, 17, 23]
    return [2, 5, 8, 11]


class DINOv2IBStyleEncoder(nn.Module):
    """Frozen DINOv2 image encoder with an ImageBind-style token API.

    out_indices are zero-indexed transformer block indices. The default
    dinov2_vits14 setting [2, 5, 8, 11] corresponds to 4 intermediate levels.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        out_indices: Sequence[int] = None,
        hub_repo: str = "facebookresearch/dinov2",
        hub_source: str = "github",
        freeze: bool = True,
    ):
        super().__init__()
        if out_indices is None:
            out_indices = default_dinov2_out_indices(model_name)
        self.model_name = model_name
        self.out_indices = list(out_indices)
        self.model = torch.hub.load(hub_repo, model_name, source=hub_source)
        self.embed_dim = getattr(self.model, "embed_dim", DINOV2_EMBED_DIMS.get(model_name))
        if self.embed_dim is None:
            raise ValueError(f"Cannot infer DINOv2 embed_dim for {model_name}.")

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    def _to_patch_tokens(self, output: torch.Tensor) -> torch.Tensor:
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.dim() != 3:
            raise ValueError(f"Expected DINOv2 token output [B, N, C], got {tuple(output.shape)}.")

        n_tokens = output.shape[1]
        side = int(math.sqrt(n_tokens))
        if side * side == n_tokens:
            return output

        side_without_cls = int(math.sqrt(n_tokens - 1))
        if side_without_cls * side_without_cls == n_tokens - 1:
            return output[:, 1:, :]

        raise ValueError(
            f"DINOv2 patch token count must be square, got {n_tokens} tokens."
        )

    @torch.no_grad()
    def encode_image_from_tensors(self, images: torch.Tensor) -> List[torch.Tensor]:
        self.model.eval()
        outputs = self.model.get_intermediate_layers(
            images,
            n=self.out_indices,
            reshape=False,
            return_class_token=False,
            norm=True,
        )
        return [self._to_patch_tokens(output).contiguous() for output in outputs]

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        features = self.encode_image_from_tensors(images)
        maps = []
        for feature in features:
            b, n_tokens, channels = feature.shape
            side = int(math.sqrt(n_tokens))
            if side * side != n_tokens:
                raise ValueError(f"Cannot reshape {n_tokens} DINOv2 tokens to a square grid.")
            maps.append(feature.permute(0, 2, 1).reshape(b, channels, side, side).contiguous())
        return maps


def print_dinov2_ibstyle_config(encoder: DINOv2IBStyleEncoder):
    print("[DINOv2-IBStyle] model:", encoder.model_name)
    print("[DINOv2-IBStyle] out_indices:", encoder.out_indices)
    print("[DINOv2-IBStyle] embed_dim:", encoder.embed_dim)
    print("[DINOv2-IBStyle] feature_levels:", len(encoder.out_indices))
