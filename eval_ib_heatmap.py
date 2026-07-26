import argparse
import csv
import gc
import importlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


IMAGEBIND_MEAN = np.asarray(
    (0.48145466, 0.4578275, 0.40821073), dtype=np.float32
)
IMAGEBIND_STD = np.asarray(
    (0.26862954, 0.26130258, 0.27577711), dtype=np.float32
)
IMAGEBIND_FEATURE_DIMS = [1280, 1280, 1280, 1280]
DEFAULT_IMAGEBIND_CHECKPOINT = (
    "/home/ueno/pretrained_weights/imagebind/imagebind_huge.pth"
)
METRIC_NAMES = (
    "image_auc",
    "image_ap",
    "image_f1",
    "pixel_auc",
    "pixel_ap",
    "pixel_f1",
    "aupro",
)
SCORE_KEYS = {
    "Logps": "scores1",
    "BScore": "scores2",
    "Merged": "scores",
}
MAP_KEYS = {
    "Logps": "score_maps_logps",
    "BScore": "score_maps_bscores",
    "Merged": "score_maps",
}

DATASET_ALIASES = {
    "mvtec_loco": "mvtecloco",
    "mvtec-loco": "mvtecloco",
    "mvtec_3d": "mvtec3d",
    "mvtec-3d": "mvtec3d",
}

# Imports are resolved lazily so --help works without loading ImageBind.
DATASET_REGISTRY = {
    "mvtec": ("datasets.mvtec", "MVTEC"),
    "visa": ("datasets.visa", "VISA"),
    "btad": ("datasets.btad", "BTAD"),
    "mvtec3d": ("datasets.mvtec_3d", "MVTEC3D"),
    "mpdd": ("datasets.mpdd", "MPDD"),
    "mvtecloco": ("datasets.mvtec_loco", "MVTECLOCO"),
    "brats": ("datasets.brats", "BRATS"),
}


def normalize_dataset_name(name):
    normalized = name.strip().lower()
    return DATASET_ALIASES.get(normalized, normalized)


def resolve_dataset_registry():
    registry = {}
    for name, (module_name, class_name) in DATASET_REGISTRY.items():
        module = importlib.import_module(module_name)
        dataset_class = getattr(module, class_name)
        registry[name] = {
            "dataset_class": dataset_class,
            "class_names": list(dataset_class.CLASS_NAMES),
        }
    return registry


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a saved ImageBind ResAD checkpoint and save "
            "Input/GT/Logps/BScore/Merged heatmaps."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_dataset_dir", required=True)
    parser.add_argument("--test_ref_feature_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--class_name", default=None)
    parser.add_argument("--num_ref_shot", type=int, default=4)
    parser.add_argument("--total_ref_shot", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--backbone", default="imagebind")
    parser.add_argument("--flow_arch", default="flow_model")
    parser.add_argument("--feature_levels", type=int, default=4)
    parser.add_argument("--coupling_layers", type=int, default=4)
    parser.add_argument("--clamp_alpha", type=float, default=1.9)
    parser.add_argument("--pos_embed_dim", type=int, default=256)
    parser.add_argument("--pos_beta", type=float, default=0.05)
    parser.add_argument("--margin_tau", type=float, default=0.1)
    parser.add_argument("--bgspp_lambda", type=float, default=1.0)
    parser.add_argument("--fdm_alpha", type=float, default=0.4)
    parser.add_argument("--num_embeddings", type=int, default=1536)
    parser.add_argument(
        "--imagebind_checkpoint",
        default=DEFAULT_IMAGEBIND_CHECKPOINT,
        help="Path to imagebind_huge.pth.",
    )
    return parser.parse_args()


def resolve_device(device_name):
    try:
        device = torch.device(device_name)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid --device value: {device_name}") from exc

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                f"--device {device_name} requests CUDA, but CUDA is unavailable."
            )
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(
                f"--device {device_name} does not exist. "
                f"Available CUDA device count: {torch.cuda.device_count()}."
            )
    return device


def resolve_class_names(args, dataset_name, registry):
    available = registry[dataset_name]["class_names"]
    if args.class_name in (None, "", "all"):
        return available
    if args.class_name not in available:
        raise ValueError(
            f"Class '{args.class_name}' is not part of dataset '{dataset_name}'.\n"
            f"Available classes:\n{', '.join(available)}"
        )
    return [args.class_name]


def validate_cli_args(args, dataset_name, class_names):
    if dataset_name not in DATASET_REGISTRY:
        supported = ", ".join(DATASET_REGISTRY)
        raise ValueError(
            f"Unsupported dataset '{args.dataset}'. Supported datasets: {supported}."
        )
    if args.backbone != "imagebind":
        raise ValueError("ImageBind evaluation requires --backbone imagebind.")
    if args.feature_levels != 4:
        raise ValueError("ImageBind evaluation requires --feature_levels 4.")
    if args.total_ref_shot < 1:
        raise ValueError("--total_ref_shot must be at least 1.")
    if not 1 <= args.num_ref_shot <= args.total_ref_shot:
        raise ValueError(
            "--num_ref_shot must satisfy "
            "1 <= num_ref_shot <= total_ref_shot."
        )
    if args.max_images < -1:
        raise ValueError(
            "--max_images must be -1, 0, or a positive integer."
        )
    if args.num_workers < 0:
        raise ValueError("--num_workers must be zero or greater.")

    file_paths = {
        "checkpoint": Path(args.checkpoint).expanduser(),
        "ImageBind checkpoint": Path(args.imagebind_checkpoint).expanduser(),
    }
    directory_paths = {
        "test dataset root": Path(args.test_dataset_dir).expanduser(),
        "reference feature root": Path(args.test_ref_feature_dir).expanduser(),
    }
    for label, path in file_paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {label} file.\n"
                f"dataset: {dataset_name}\n"
                f"path: {path}"
            )
    for label, path in directory_paths.items():
        if not path.is_dir():
            raise FileNotFoundError(
                f"Missing {label} directory.\n"
                f"dataset: {dataset_name}\n"
                f"path: {path}"
            )

    output_root = Path(args.output_dir).expanduser()
    try:
        (output_root / dataset_name).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(
            f"Cannot create output directory for dataset '{dataset_name}': "
            f"{output_root}"
        ) from exc

    validate_reference_files(args, dataset_name, class_names)


def validate_reference_files(args, dataset_name, class_names):
    root = Path(args.test_ref_feature_dir).expanduser()
    for class_name in class_names:
        for level in range(1, 5):
            path = root / class_name / f"layer{level}.npy"
            if not path.is_file():
                raise FileNotFoundError(
                    "Missing ImageBind reference feature.\n"
                    f"dataset: {dataset_name}\n"
                    f"class: {class_name}\n"
                    f"layer: {level}\n"
                    f"path: {path}"
                )
            features = np.load(path, mmap_mode="r")
            if features.ndim != 2:
                raise ValueError(
                    f"Reference feature must be 2D, got {features.shape}.\n"
                    f"dataset: {dataset_name}\nclass: {class_name}\npath: {path}"
                )
            if features.shape[0] == 0:
                raise ValueError(
                    "Reference feature contains no patches.\n"
                    f"dataset: {dataset_name}\nclass: {class_name}\npath: {path}"
                )
            if features.shape[1] != 1280:
                raise ValueError(
                    f"Reference feature dimension must be 1280, "
                    f"got {features.shape[1]}.\n"
                    f"dataset: {dataset_name}\nclass: {class_name}\npath: {path}"
                )
            if features.shape[0] % args.total_ref_shot != 0:
                raise ValueError(
                    f"Reference rows ({features.shape[0]}) are not divisible by "
                    f"--total_ref_shot {args.total_ref_shot}.\n"
                    f"dataset: {dataset_name}\nclass: {class_name}\npath: {path}"
                )


def load_reference_features(args, class_name, device):
    class_root = Path(args.test_ref_feature_dir).expanduser() / class_name
    reference_features = []
    for level in range(1, 5):
        features = np.load(class_root / f"layer{level}.npy")
        num_features = (
            features.shape[0] // args.total_ref_shot
        ) * args.num_ref_shot
        tensor = torch.from_numpy(features[:num_features]).to(
            device=device, dtype=torch.float32
        )
        reference_features.append(tensor)
    return tuple(reference_features)


def build_test_dataset(dataset_name, dataset_root, class_name, registry):
    dataset_class = registry[dataset_name]["dataset_class"]
    return dataset_class(
        str(dataset_root),
        class_name=class_name,
        train=False,
        normalize="imagebind",
        img_size=224,
        crp_size=224,
        msk_size=224,
        msk_crp_size=224,
    )


def _checkpoint_context(args):
    return (
        f"checkpoint: {Path(args.checkpoint).expanduser()}\n"
        f"coupling_layers: {args.coupling_layers}\n"
        f"flow_arch: {args.flow_arch}\n"
        f"num_embeddings: {args.num_embeddings}\n"
        f"feature_levels: {args.feature_levels}"
    )


def _load_state(module, state, module_name, args):
    try:
        module.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load '{module_name}' with strict=True.\n"
            f"{_checkpoint_context(args)}\n"
            f"Original error: {exc}"
        ) from exc


def build_and_load_models(args, device):
    from models.fc_flow import load_flow_model
    from models.imagebind import ImageBindModel
    from models.modules import MultiScaleOrthogonalProjector
    from models.vq import MultiScaleVQ4

    checkpoint_path = Path(args.checkpoint).expanduser()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Checkpoint must be a dict, got {type(checkpoint).__name__}: "
            f"{checkpoint_path}"
        )
    if "constraintor" not in checkpoint or "estimators" not in checkpoint:
        raise KeyError(
            "Checkpoint must contain 'constraintor' and 'estimators'.\n"
            f"path: {checkpoint_path}\nkeys: {sorted(checkpoint.keys())}"
        )
    if not isinstance(checkpoint["estimators"], (list, tuple)):
        raise TypeError(
            f"Checkpoint 'estimators' must be a list or tuple: {checkpoint_path}"
        )
    if len(checkpoint["estimators"]) != args.feature_levels:
        raise ValueError(
            f"Checkpoint has {len(checkpoint['estimators'])} estimators, "
            f"but --feature_levels is {args.feature_levels}.\n"
            f"{_checkpoint_context(args)}"
        )

    encoder = ImageBindModel(
        device=args.device,
        checkpoint_path=str(Path(args.imagebind_checkpoint).expanduser()),
    ).to(device)
    constraintor = MultiScaleOrthogonalProjector(
        IMAGEBIND_FEATURE_DIMS
    ).to(device)
    estimators = [
        load_flow_model(args, feature_dim).to(device)
        for feature_dim in IMAGEBIND_FEATURE_DIMS
    ]
    vq_ops = None
    if "vq_ops" in checkpoint:
        vq_ops = MultiScaleVQ4(
            num_embeddings=args.num_embeddings,
            channels=IMAGEBIND_FEATURE_DIMS,
        ).to(device)

    _load_state(constraintor, checkpoint["constraintor"], "constraintor", args)
    print("Loaded constraintor")
    for index, (estimator, state) in enumerate(
        zip(estimators, checkpoint["estimators"]), start=1
    ):
        _load_state(estimator, state, f"estimator {index}", args)
        print(f"Loaded estimator {index}/{args.feature_levels}")
    if vq_ops is not None:
        _load_state(vq_ops, checkpoint["vq_ops"], "vq_ops", args)
        print("Loaded vq_ops")
    else:
        print("Checkpoint has no vq_ops; evaluating without VQ/EFDM")
    del checkpoint

    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    constraintor.eval()
    if vq_ops is not None:
        vq_ops.eval()
    for estimator in estimators:
        estimator.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    return encoder, vq_ops, constraintor, estimators


def denormalize_image(image):
    image = np.asarray(image, dtype=np.float32).transpose(1, 2, 0)
    image = image * IMAGEBIND_STD + IMAGEBIND_MEAN
    return np.clip(image, 0.0, 1.0)


def compute_display_range(score_maps):
    values = np.asarray(score_maps)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, (1, 99))
    if vmax <= vmin:
        delta = max(abs(float(vmin)) * 1e-6, 1e-6)
        vmin -= delta
        vmax += delta
    return float(vmin), float(vmax)


def num_images_to_save(total_images, max_images):
    if max_images == -1:
        return total_images
    if max_images == 0:
        return 0
    if max_images > 0:
        return min(total_images, max_images)
    raise ValueError("--max_images must be -1, 0, or a positive integer.")


def _safe_stem(path):
    stem = Path(path).stem if path else "image"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def _dataset_item_metadata(dataset, index):
    image_paths = getattr(dataset, "image_paths", None)
    image_path = image_paths[index] if image_paths is not None else ""
    anomaly_type = ""
    for attribute in ("anomaly_types", "img_types"):
        values = getattr(dataset, attribute, None)
        if values is not None and index < len(values):
            anomaly_type = str(values[index])
            break
    if not anomaly_type and image_path:
        anomaly_type = Path(image_path).parent.name
    return str(image_path), anomaly_type


def save_class_heatmaps(
    args,
    dataset_name,
    class_name,
    dataset,
    metrics,
):
    class_dir = Path(args.output_dir).expanduser() / dataset_name / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    total_images = len(metrics["test_imgs"])
    save_count = num_images_to_save(total_images, args.max_images)
    ranges = {
        name: compute_display_range(metrics[key])
        for name, key in MAP_KEYS.items()
    }
    skipped = 0

    for index in range(save_count):
        image_path, anomaly_type = _dataset_item_metadata(dataset, index)
        output_path = class_dir / (
            f"{index:04d}_{_safe_stem(image_path)}.png"
        )
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        image = denormalize_image(metrics["test_imgs"][index])
        gt_mask = np.asarray(metrics["gt_masks"][index], dtype=np.float32)
        label = bool(metrics["labels"][index])
        label_text = "anomaly" if label else "normal"
        metadata = (
            f"{dataset_name} / {class_name} / {index:04d} / {label_text}"
        )
        if anomaly_type:
            metadata += f" / {anomaly_type}"
        if image_path:
            metadata += f" / {Path(image_path).name}"

        figure, axes = plt.subplots(1, 5, figsize=(22, 4.8))
        axes[0].imshow(image)
        axes[0].set_title("Input")
        axes[1].imshow(gt_mask, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1].set_title("Ground Truth")
        for axis, score_name in zip(axes[2:], ("Logps", "BScore", "Merged")):
            score_map = np.asarray(metrics[MAP_KEYS[score_name]][index])
            vmin, vmax = ranges[score_name]
            axis.imshow(image)
            overlay = axis.imshow(
                score_map,
                cmap="jet",
                alpha=0.5,
                vmin=vmin,
                vmax=vmax,
            )
            axis.set_title(score_name)
            figure.colorbar(overlay, ax=axis, fraction=0.046, pad=0.04)
        for axis in axes:
            axis.axis("off")
        figure.suptitle(metadata)
        figure.tight_layout()
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
    return save_count - skipped, skipped


def metrics_to_dict(values):
    return {
        name: float(value)
        for name, value in zip(METRIC_NAMES, values)
    }


def print_metric_block(header, score_metrics):
    print(f"\n{header}")
    labels = (
        ("Image AUC", "image_auc"),
        ("Image AP", "image_ap"),
        ("Image F1", "image_f1"),
        ("Pixel AUC", "pixel_auc"),
        ("Pixel AP", "pixel_ap"),
        ("Pixel F1", "pixel_f1"),
        ("AUPRO", "aupro"),
    )
    for score_name in ("Logps", "BScore", "Merged"):
        print(f"\n{score_name}")
        for label, key in labels:
            print(f"  {label:<10}: {score_metrics[score_name][key]:.5f}")


def average_metrics(class_results):
    averaged = {}
    for score_name in SCORE_KEYS:
        averaged[score_name] = {}
        for metric_name in METRIC_NAMES:
            values = np.asarray(
                [
                    result[score_name][metric_name]
                    for result in class_results.values()
                ],
                dtype=np.float64,
            )
            finite = values[np.isfinite(values)]
            averaged[score_name][metric_name] = (
                float(finite.mean()) if finite.size else float("nan")
            )
    return averaged


def save_metric_files(args, dataset_name, class_results, average):
    output_root = Path(args.output_dir).expanduser() / dataset_name
    json_path = output_root / "metrics.json"
    csv_path = output_root / "metrics.csv"
    payload = _json_safe(
        {
            "dataset": dataset_name,
            "checkpoint": str(Path(args.checkpoint).expanduser()),
            "classes": class_results,
            "average": average,
        }
    )
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)

    fieldnames = ["class_name", "score_type", *METRIC_NAMES]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for class_name, score_results in class_results.items():
            for score_name, metric_values in score_results.items():
                writer.writerow(
                    {
                        "class_name": class_name,
                        "score_type": score_name,
                        **metric_values,
                    }
                )
        for score_name, metric_values in average.items():
            writer.writerow(
                {
                    "class_name": "Average",
                    "score_type": score_name,
                    **metric_values,
                }
            )
    print(f"\nSaved metrics: {json_path}")
    print(f"Saved metrics: {csv_path}")


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    args = parse_args()
    dataset_name = normalize_dataset_name(args.dataset)
    if dataset_name not in DATASET_REGISTRY:
        supported = ", ".join(DATASET_REGISTRY)
        raise ValueError(
            f"Unsupported dataset '{args.dataset}'. Supported datasets: {supported}."
        )

    registry = resolve_dataset_registry()
    class_names = resolve_class_names(args, dataset_name, registry)
    validate_cli_args(args, dataset_name, class_names)
    device = resolve_device(args.device)

    print(f"[Evaluation] dataset: {dataset_name}")
    print(f"[Evaluation] classes: {class_names}")
    print(f"[Evaluation] checkpoint: {Path(args.checkpoint).expanduser()}")
    print(f"[Evaluation] device: {device}")
    print(f"[Evaluation] max_images per class: {args.max_images}")

    encoder, vq_ops, constraintor, estimators = build_and_load_models(
        args, device
    )
    from validate import validate

    class_results = {}
    total_saved = 0
    total_skipped = 0
    for class_name in class_names:
        dataset = build_test_dataset(
            dataset_name,
            Path(args.test_dataset_dir).expanduser(),
            class_name,
            registry,
        )
        if len(dataset) == 0:
            raise ValueError(
                f"No test images found.\ndataset: {dataset_name}\n"
                f"class: {class_name}\npath: {args.test_dataset_dir}"
            )
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=device.type == "cuda",
        )
        reference_features = load_reference_features(
            args, class_name, device
        )
        metrics = validate(
            args,
            encoder,
            vq_ops,
            constraintor,
            estimators,
            loader,
            reference_features,
            device,
            class_name,
            return_maps=True,
        )
        score_results = {
            score_name: metrics_to_dict(metrics[metric_key])
            for score_name, metric_key in SCORE_KEYS.items()
        }
        class_results[class_name] = score_results
        print_metric_block(
            f"[{dataset_name} / {class_name}]", score_results
        )
        saved, skipped = save_class_heatmaps(
            args,
            dataset_name,
            class_name,
            dataset,
            metrics,
        )
        total_saved += saved
        total_skipped += skipped
        print(
            f"[Heatmaps] {class_name}: saved={saved}, skipped={skipped}, "
            f"evaluated={len(dataset)}"
        )

        del metrics, reference_features, loader, dataset
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    average = average_metrics(class_results)
    print_metric_block(f"[Average / {dataset_name}]", average)
    save_metric_files(args, dataset_name, class_results, average)
    print(
        f"\nHeatmap summary: saved={total_saved}, "
        f"skipped_existing={total_skipped}"
    )


if __name__ == "__main__":
    main()
