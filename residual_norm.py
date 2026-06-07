import torch


RESIDUAL_NORM_MODES = ("none", "layer_rms", "layer_std", "channel_std")


def residual_norm_enabled(args):
    return getattr(args, "residual_norm", "none") != "none"


def validate_residual_norm_args(args):
    mode = getattr(args, "residual_norm", "none")
    if mode not in RESIDUAL_NORM_MODES:
        raise ValueError(f"Unsupported residual_norm: {mode}")
    if mode != "none" and getattr(args, "residual_stats_batches", 50) <= 0:
        raise ValueError("residual_stats_batches must be > 0 when residual_norm is enabled.")
    if getattr(args, "residual_norm_eps", 1e-6) <= 0:
        raise ValueError("residual_norm_eps must be > 0.")
    if getattr(args, "residual_norm_clip", 0.0) < 0:
        raise ValueError("residual_norm_clip must be >= 0.")


def create_residual_norm_accumulator(mode):
    if mode not in RESIDUAL_NORM_MODES:
        raise ValueError(f"Unsupported residual_norm: {mode}")
    return {"mode": mode, "levels": []}


def _ensure_level(accumulator, level, residual):
    while len(accumulator["levels"]) <= level:
        accumulator["levels"].append(None)
    if accumulator["levels"][level] is not None:
        return accumulator["levels"][level]

    _, channels, _, _ = residual.shape
    device = residual.device
    stats = {
        "sum": torch.zeros((), device=device),
        "sum_sq": torch.zeros((), device=device),
        "count": 0,
        "channel_sum": torch.zeros(channels, device=device),
        "channel_sum_sq": torch.zeros(channels, device=device),
        "channel_count": 0,
    }
    accumulator["levels"][level] = stats
    return stats


def update_residual_norm_accumulator(accumulator, residual_features):
    for level, residual in enumerate(residual_features):
        residual = residual.detach().float()
        stats = _ensure_level(accumulator, level, residual)
        stats["sum"] += residual.sum()
        stats["sum_sq"] += residual.pow(2).sum()
        stats["count"] += residual.numel()
        stats["channel_sum"] += residual.sum(dim=(0, 2, 3))
        stats["channel_sum_sq"] += residual.pow(2).sum(dim=(0, 2, 3))
        stats["channel_count"] += residual.shape[0] * residual.shape[2] * residual.shape[3]


def _safe_std(sum_value, sum_sq_value, count):
    count_t = torch.as_tensor(float(count), device=sum_value.device, dtype=sum_value.dtype)
    mean = sum_value / count_t
    var = sum_sq_value / count_t - mean.pow(2)
    return mean, torch.sqrt(torch.clamp(var, min=0.0))


def finalize_residual_norm_stats(accumulator):
    mode = accumulator["mode"]
    levels = []
    for stats in accumulator["levels"]:
        if stats is None or stats["count"] == 0:
            raise ValueError("No residual samples were collected for residual normalization stats.")

        raw_mean, raw_std = _safe_std(stats["sum"], stats["sum_sq"], stats["count"])
        count_t = torch.as_tensor(float(stats["count"]), device=stats["sum"].device, dtype=stats["sum"].dtype)
        raw_rms = torch.sqrt(torch.clamp(stats["sum_sq"] / count_t, min=0.0))
        level_stats = {
            "raw_mean": raw_mean.detach().cpu(),
            "raw_std": raw_std.detach().cpu(),
            "raw_rms": raw_rms.detach().cpu(),
        }

        if mode == "layer_rms":
            level_stats["rms"] = raw_rms.detach().cpu()
        elif mode == "layer_std":
            level_stats["mean"] = raw_mean.detach().cpu()
            level_stats["std"] = raw_std.detach().cpu()
        elif mode == "channel_std":
            channel_mean, channel_std = _safe_std(
                stats["channel_sum"], stats["channel_sum_sq"], stats["channel_count"]
            )
            level_stats["mean"] = channel_mean.detach().cpu().view(1, -1, 1, 1)
            level_stats["std"] = channel_std.detach().cpu().view(1, -1, 1, 1)

        levels.append(level_stats)
    return {"mode": mode, "levels": levels}


def _to_feature_device(value, feature):
    return value.to(device=feature.device, dtype=torch.float32)


def apply_residual_norm(residual_features, stats, mode=None, eps=1e-6, clip=0.0):
    if mode is None:
        mode = stats.get("mode", "none") if stats is not None else "none"
    if mode == "none":
        return residual_features
    if stats is None:
        raise ValueError("residual_norm is enabled, but residual normalization stats are missing.")

    normalized = []
    for level, residual in enumerate(residual_features):
        residual = residual.float()
        level_stats = stats["levels"][level]
        if mode == "layer_rms":
            residual = residual / (_to_feature_device(level_stats["rms"], residual) + eps)
        elif mode == "layer_std":
            residual = (residual - _to_feature_device(level_stats["mean"], residual)) / (
                _to_feature_device(level_stats["std"], residual) + eps
            )
        elif mode == "channel_std":
            residual = (residual - _to_feature_device(level_stats["mean"], residual)) / (
                _to_feature_device(level_stats["std"], residual) + eps
            )
        else:
            raise ValueError(f"Unsupported residual_norm: {mode}")

        if clip and clip > 0:
            residual = torch.clamp(residual, min=-clip, max=clip)
        normalized.append(residual)
    return normalized


def apply_residual_norm_from_args(args, residual_features):
    mode = getattr(args, "residual_norm", "none")
    if mode == "none":
        return residual_features
    return apply_residual_norm(
        residual_features,
        getattr(args, "residual_norm_stats", None),
        mode=mode,
        eps=getattr(args, "residual_norm_eps", 1e-6),
        clip=getattr(args, "residual_norm_clip", 0.0),
    )


def pack_residual_norm_state(args):
    mode = getattr(args, "residual_norm", "none")
    if mode == "none":
        return None
    return {
        "mode": mode,
        "eps": getattr(args, "residual_norm_eps", 1e-6),
        "clip": getattr(args, "residual_norm_clip", 0.0),
        "stats": getattr(args, "residual_norm_stats", None),
    }


def load_residual_norm_state_into_args(args, checkpoint):
    state = checkpoint.get("residual_norm")
    if state is None:
        if getattr(args, "residual_norm", "none") != "none":
            raise ValueError("Checkpoint does not contain residual_norm stats.")
        args.residual_norm_stats = None
        return

    checkpoint_mode = state.get("mode", "none")
    requested_mode = getattr(args, "residual_norm", "none")
    if requested_mode == "none":
        args.residual_norm = checkpoint_mode
    elif requested_mode != checkpoint_mode:
        raise ValueError(
            f"Requested residual_norm={requested_mode}, but checkpoint was saved with {checkpoint_mode}."
        )

    args.residual_norm_eps = state.get("eps", getattr(args, "residual_norm_eps", 1e-6))
    args.residual_norm_clip = state.get("clip", getattr(args, "residual_norm_clip", 0.0))
    args.residual_norm_stats = state.get("stats")


def print_residual_norm_stats(stats, prefix="[ResidualNorm]"):
    if stats is None or stats.get("mode", "none") == "none":
        print(f"{prefix} mode: none")
        return

    mode = stats["mode"]
    print(f"{prefix} mode: {mode}")
    for level, level_stats in enumerate(stats["levels"]):
        raw_mean = float(level_stats["raw_mean"])
        raw_std = float(level_stats["raw_std"])
        raw_rms = float(level_stats["raw_rms"])
        if mode == "layer_rms":
            scale = max(float(level_stats["rms"]), 1e-12)
            norm_mean = raw_mean / scale
            norm_std = raw_std / scale
            norm_rms = raw_rms / scale
        elif mode in ("layer_std", "channel_std"):
            norm_mean = 0.0
            norm_std = 1.0
            norm_rms = 1.0
        else:
            norm_mean = raw_mean
            norm_std = raw_std
            norm_rms = raw_rms

        print(
            f"{prefix} level {level}: raw mean/std/rms="
            f"{raw_mean:.6g}/{raw_std:.6g}/{raw_rms:.6g}, normalized mean/std/rms="
            f"{norm_mean:.6g}/{norm_std:.6g}/{norm_rms:.6g}"
        )
