import math

import torch
import torch.nn as nn


class _FeatureInfo:
    def __init__(self, channels):
        self._channels = list(channels)

    def channels(self):
        return self._channels


class CLIPRawFeatureExtractor(nn.Module):
    """
    Frozen CLIP ViT raw patch-token extractor.

    clip_layers are user-facing 1-indexed transformer block numbers. For example,
    --clip_layers 6 12 24 captures resblocks[5], resblocks[11], and resblocks[23].
    """

    def __init__(
        self,
        model_name="ViT-L-14-336",
        pretrained="openai",
        layers=(6, 12, 24),
        image_size=518,
        freeze=True,
    ):
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "open_clip_torch is required for --feature_backbone clip_raw. "
                "Install it with `pip install open_clip_torch`."
            ) from exc

        self.model_name = model_name
        self.pretrained = pretrained
        self.layers = tuple(int(layer) for layer in layers)
        self.layer_indices = tuple(layer - 1 for layer in self.layers)
        self.image_size = int(image_size)
        self.freeze = freeze

        try:
            self.model, _, _ = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                force_image_size=self.image_size,
            )
        except TypeError:
            self.model, _, _ = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
            )
        self.model.eval()
        self.visual = self.model.visual
        self.resblocks = self._get_resblocks()
        self._validate_layers()

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.embed_dim = self._infer_embed_dim()
        self.patch_size = self._infer_patch_size()
        self.feature_info = _FeatureInfo([self.embed_dim] * len(self.layers))
        self._debug_printed = False

        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("clip_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("clip_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1), persistent=False)

    def _get_resblocks(self):
        transformer = getattr(self.visual, "transformer", None)
        resblocks = getattr(transformer, "resblocks", None)
        if resblocks is None:
            raise ValueError("CLIP visual transformer has no resblocks; only ViT CLIP models are supported.")
        return resblocks

    def _validate_layers(self):
        if not self.layers:
            raise ValueError("clip_layers must contain at least one layer.")
        num_blocks = len(self.resblocks)
        for layer in self.layers:
            if layer < 1 or layer > num_blocks:
                raise ValueError(f"clip layer {layer} is out of range for {num_blocks} transformer blocks.")

    def _infer_embed_dim(self):
        conv1 = getattr(self.visual, "conv1", None)
        out_channels = getattr(conv1, "out_channels", None)
        if isinstance(out_channels, int):
            return out_channels
        width = getattr(getattr(self.visual, "transformer", None), "width", None)
        if isinstance(width, int):
            return width
        raise ValueError("Could not infer CLIP visual embed dimension.")

    def _infer_patch_size(self):
        conv1 = getattr(self.visual, "conv1", None)
        kernel_size = getattr(conv1, "kernel_size", None)
        if isinstance(kernel_size, tuple):
            return kernel_size[0]
        if isinstance(kernel_size, int):
            return kernel_size
        return 14

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    def _normalize_for_clip(self, images):
        images = images * self.imagenet_std + self.imagenet_mean
        return (images - self.clip_mean) / self.clip_std

    def _to_bnc(self, tokens, batch_size):
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[0]
        if tokens.dim() != 3:
            raise ValueError(f"Expected CLIP block output [B,N,C] or [N,B,C], got {tuple(tokens.shape)}")
        if tokens.shape[0] == batch_size and tokens.shape[1] != batch_size:
            return tokens
        if tokens.shape[1] == batch_size:
            return tokens.permute(1, 0, 2).contiguous()
        raise ValueError(f"Cannot infer CLIP token layout from shape {tuple(tokens.shape)} and batch={batch_size}")

    def _tokens_to_map(self, tokens):
        num_tokens = tokens.shape[1]
        grid_size = int(math.sqrt(num_tokens))
        if grid_size * grid_size != num_tokens:
            patch_tokens = num_tokens - 1
            patch_grid = int(math.sqrt(patch_tokens))
            if patch_grid * patch_grid != patch_tokens:
                raise ValueError(f"CLIP patch token count must be square after class-token removal, got N={num_tokens}")
            tokens = tokens[:, 1:, :]
            grid_size = patch_grid
        b, n, c = tokens.shape
        if grid_size * grid_size != n:
            raise ValueError(f"CLIP patch token count must be square, got N={n}")
        return tokens.transpose(1, 2).reshape(b, c, grid_size, grid_size)

    def _debug_shapes(self, features):
        if self._debug_printed:
            return
        print("feature_backbone = clip_raw")
        print(f"clip_model = {self.model_name}")
        print(f"clip_layers = {list(self.layers)}")
        for layer, feature in zip(self.layers, features):
            print(f"layer{layer} shape = {tuple(feature.shape)}")
        self._debug_printed = True

    def forward(self, images):
        batch_size = images.shape[0]
        captured = {}
        hooks = []

        def make_hook(layer):
            def hook(_module, _inputs, output):
                captured[layer] = self._to_bnc(output, batch_size).detach()
            return hook

        for layer, layer_idx in zip(self.layers, self.layer_indices):
            hooks.append(self.resblocks[layer_idx].register_forward_hook(make_hook(layer)))

        try:
            images = self._normalize_for_clip(images)
            if self.freeze:
                with torch.no_grad():
                    _ = self.visual(images)
            else:
                _ = self.visual(images)
        finally:
            for hook in hooks:
                hook.remove()

        features = []
        for layer in self.layers:
            if layer not in captured:
                raise RuntimeError(f"CLIP layer {layer} was not captured.")
            features.append(self._tokens_to_map(captured[layer]))
        self._debug_shapes(features)
        return features
