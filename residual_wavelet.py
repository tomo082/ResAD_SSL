from typing import List

import torch
from torch import Tensor
import torch.nn.functional as F


def apply_residual_wavelet_filter(rfeatures: List[Tensor], wave: str = "haar",
                                  hf_weight: float = 1.0) -> List[Tensor]:
    """
    Apply a 1-level Haar DWT filter in residual space.

    Each input/output feature keeps shape (B, C, H, W). The filtered residual is
    LL_upsampled + hf_weight * sqrt(LH^2 + HL^2 + HH^2)_upsampled.
    """
    if wave != "haar":
        raise ValueError(f"Only Haar wavelet is supported, but got {wave}.")
    return [_apply_haar_residual_filter(rfeature, hf_weight=hf_weight) for rfeature in rfeatures]


def _apply_haar_residual_filter(x: Tensor, hf_weight: float = 1.0) -> Tensor:
    if x.dim() != 4:
        raise ValueError(f"Residual feature must be 4D (B, C, H, W), but got {tuple(x.shape)}.")

    B, C, H, W = x.shape
    pad_h = H % 2
    pad_w = W % 2
    if pad_h or pad_w:
        x_dwt = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    else:
        x_dwt = x

    kernels = _haar_kernels(device=x.device, dtype=x.dtype).repeat(C, 1, 1, 1)
    coeffs = F.conv2d(x_dwt, kernels, stride=2, groups=C)
    h_dwt, w_dwt = coeffs.shape[-2:]
    coeffs = coeffs.view(B, C, 4, h_dwt, w_dwt).permute(0, 2, 1, 3, 4)
    ll, lh, hl, hh = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2], coeffs[:, 3]

    hf_energy = torch.sqrt(lh.pow(2) + hl.pow(2) + hh.pow(2) + 1e-12)
    ll = F.interpolate(ll, size=(H, W), mode="bilinear", align_corners=False)
    hf_energy = F.interpolate(hf_energy, size=(H, W), mode="bilinear", align_corners=False)
    return ll + hf_weight * hf_energy


def _haar_kernels(device, dtype) -> Tensor:
    return torch.tensor(
        [
            [[0.5, 0.5], [0.5, 0.5]],      # LL
            [[-0.5, 0.5], [-0.5, 0.5]],    # LH
            [[-0.5, -0.5], [0.5, 0.5]],    # HL
            [[0.5, -0.5], [-0.5, 0.5]],    # HH
        ],
        device=device,
        dtype=dtype,
    ).view(4, 1, 2, 2)


def residual_wavelet_shape_test(device: str = "cpu") -> None:
    rfeatures = [
        torch.randn(2, 8, 56, 56, device=device),
        torch.randn(2, 16, 28, 28, device=device),
        torch.randn(2, 32, 15, 17, device=device),
    ]
    filtered = apply_residual_wavelet_filter(rfeatures, wave="haar", hf_weight=1.0)
    for before, after in zip(rfeatures, filtered):
        if before.shape != after.shape:
            raise AssertionError(f"Shape changed from {tuple(before.shape)} to {tuple(after.shape)}.")
