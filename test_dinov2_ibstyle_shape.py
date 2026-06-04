import argparse

import torch

from models.dinov2_encoder import DINOv2IBStyleEncoder, default_dinov2_out_indices


def main(args):
    out_indices = args.dinov2_out_indices or default_dinov2_out_indices(args.backbone)
    encoder = DINOv2IBStyleEncoder(
        model_name=args.backbone,
        out_indices=out_indices,
        hub_repo=args.dinov2_hub_repo,
        hub_source=args.dinov2_hub_source,
        freeze=True,
    ).to(args.device)
    encoder.eval()

    x = torch.zeros(args.batch_size, 3, 224, 224, device=args.device)
    tokens = encoder.encode_image_from_tensors(x)
    maps = encoder(x)

    expected_tokens = 16 * 16
    for level, (token_feature, map_feature) in enumerate(zip(tokens, maps)):
        expected_token_shape = (args.batch_size, expected_tokens, encoder.embed_dim)
        expected_map_shape = (args.batch_size, encoder.embed_dim, 16, 16)
        assert tuple(token_feature.shape) == expected_token_shape, (
            level,
            tuple(token_feature.shape),
            expected_token_shape,
        )
        assert tuple(map_feature.shape) == expected_map_shape, (
            level,
            tuple(map_feature.shape),
            expected_map_shape,
        )

    print("[DINOv2-IBStyle] tokens:", [tuple(feature.shape) for feature in tokens])
    print("[DINOv2-IBStyle] maps:", [tuple(feature.shape) for feature in maps])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="dinov2_vits14")
    parser.add_argument("--dinov2_out_indices", type=int, nargs="+", default=None)
    parser.add_argument("--dinov2_hub_repo", type=str, default="facebookresearch/dinov2")
    parser.add_argument("--dinov2_hub_source", type=str, default="github", choices=["github", "local"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    main(args)
