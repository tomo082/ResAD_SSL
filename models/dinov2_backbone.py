import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


DINOV2_BACKBONES = ("dinov2_vits14", "dinov2_vitb14")
_DINOV2_EMBED_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
}


class _FeatureInfo:
    def __init__(self, channels):
        self._channels = list(channels)

    def channels(self):
        return self._channels


class DINOv2BackboneWrapper(nn.Module):
    def __init__(
        self,
        model_name="dinov2_vits14",
        out_dims=(40, 72, 200),
        out_sizes=(56, 28, 14),
        freeze=True,
    ):
        super().__init__()
        if model_name not in DINOV2_BACKBONES:
            raise ValueError(f"Unsupported DINOv2 model_name: {model_name}")
        if len(out_dims) != 3 or len(out_sizes) != 3:
            raise ValueError("DINOv2BackboneWrapper expects exactly 3 output levels.")

        self.model_name = model_name
        self.out_dims = tuple(out_dims)
        self.out_sizes = tuple(out_sizes)
        self.freeze = freeze
        self.dino = torch.hub.load("facebookresearch/dinov2", model_name)
        self.dino.eval()

        if self.freeze:
            for param in self.dino.parameters():
                param.requires_grad = False

        embed_dim = self._infer_embed_dim()
        self.proj0 = nn.Conv2d(embed_dim, self.out_dims[0], kernel_size=1)
        self.proj1 = nn.Conv2d(embed_dim, self.out_dims[1], kernel_size=1)
        self.proj2 = nn.Conv2d(embed_dim, self.out_dims[2], kernel_size=1)
        self._init_projections()
        self.feature_info = _FeatureInfo(self.out_dims)

    def _infer_embed_dim(self):
        for attr in ("embed_dim", "num_features"):
            value = getattr(self.dino, attr, None)
            if isinstance(value, int):
                return value
        if self.model_name in _DINOV2_EMBED_DIMS:
            return _DINOV2_EMBED_DIMS[self.model_name]
        raise ValueError(f"Could not infer DINOv2 embed dim for {self.model_name}")

    def _init_projections(self):
        # Keep reference-feature extraction and training runs in the same projected feature space.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            for projection in (self.proj0, self.proj1, self.proj2):
                nn.init.kaiming_uniform_(projection.weight, a=math.sqrt(5))
                if projection.bias is not None:
                    nn.init.zeros_(projection.bias)

    def train(self, mode=True):
        super().train(mode)
        self.dino.eval()
        return self

    def _forward_dino(self, images):
        if self.freeze:
            with torch.no_grad():
                return self.dino.forward_features(images)
        return self.dino.forward_features(images)

    def _extract_patch_tokens(self, outputs):
        if isinstance(outputs, dict):
            if "x_norm_patchtokens" in outputs:
                patch_tokens = outputs["x_norm_patchtokens"]
            elif "x_prenorm" in outputs:
                patch_tokens = outputs["x_prenorm"]
            else:
                raise ValueError("DINOv2 forward_features output has no patch-token tensor.")
        else:
            patch_tokens = outputs

        if patch_tokens.dim() != 3:
            raise ValueError(f"Expected patch tokens with shape [B, N, C], got {tuple(patch_tokens.shape)}")

        num_tokens = patch_tokens.shape[1]
        grid_size = int(math.sqrt(num_tokens))
        if grid_size * grid_size != num_tokens:
            cls_removed_tokens = num_tokens - 1
            cls_removed_grid = int(math.sqrt(cls_removed_tokens))
            if cls_removed_grid * cls_removed_grid != cls_removed_tokens:
                raise ValueError(f"DINOv2 patch token count must be square, got N={num_tokens}")
            patch_tokens = patch_tokens[:, 1:, :]

        return patch_tokens

    def forward(self, images):
        outputs = self._forward_dino(images)
        patch_tokens = self._extract_patch_tokens(outputs)
        batch_size, num_tokens, channels = patch_tokens.shape
        grid_size = int(math.sqrt(num_tokens))
        if grid_size * grid_size != num_tokens:
            raise ValueError(f"DINOv2 patch token count must be square, got N={num_tokens}")

        patch_map = patch_tokens.transpose(1, 2).reshape(batch_size, channels, grid_size, grid_size)
        feat0 = F.interpolate(self.proj0(patch_map), size=(self.out_sizes[0], self.out_sizes[0]), mode="bilinear", align_corners=False)
        feat1 = F.interpolate(self.proj1(patch_map), size=(self.out_sizes[1], self.out_sizes[1]), mode="bilinear", align_corners=False)
        feat2 = F.interpolate(self.proj2(patch_map), size=(self.out_sizes[2], self.out_sizes[2]), mode="bilinear", align_corners=False)
        return [feat0, feat1, feat2]


def dinov2_shape_test(model_name="dinov2_vits14", device="cpu"):
    encoder = DINOv2BackboneWrapper(model_name=model_name).to(device).eval()
    images = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        features = encoder(images)
    expected_shapes = [(2, 40, 56, 56), (2, 72, 28, 28), (2, 200, 14, 14)]
    for feature, expected_shape in zip(features, expected_shapes):
        if tuple(feature.shape) != expected_shape:
            raise AssertionError(f"Expected {expected_shape}, got {tuple(feature.shape)}")
    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="dinov2_vits14", choices=DINOV2_BACKBONES)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    dinov2_shape_test(model_name=args.model_name, device=args.device)
    print("DINOv2 shape test passed.")
