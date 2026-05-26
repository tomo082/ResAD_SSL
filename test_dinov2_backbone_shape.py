from unittest.mock import patch

import torch
import torch.nn as nn

from models.dinov2_backbone import DINOv2BackboneWrapper


class _FakeDINOv2(nn.Module):
    embed_dim = 384

    def forward_features(self, images):
        batch_size = images.shape[0]
        patch_tokens = torch.randn(batch_size, 16 * 16, self.embed_dim, device=images.device)
        return {"x_norm_patchtokens": patch_tokens}


def test_dinov2_backbone_shape():
    with patch("torch.hub.load", return_value=_FakeDINOv2()):
        encoder = DINOv2BackboneWrapper(model_name="dinov2_vits14").eval()
    with torch.no_grad():
        features = encoder(torch.randn(2, 3, 224, 224))
    expected_shapes = [(2, 40, 56, 56), (2, 72, 28, 28), (2, 200, 14, 14)]
    assert [tuple(feature.shape) for feature in features] == expected_shapes
    assert encoder.feature_info.channels() == [40, 72, 200]


if __name__ == "__main__":
    test_dinov2_backbone_shape()
    print("DINOv2 backbone shape test passed.")
