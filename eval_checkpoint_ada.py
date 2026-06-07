import argparse
import csv
import glob
import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from main_ada import (
    SETTINGS,
    build_feature_encoder,
    build_test_dataset,
    get_feature_image_size,
    load_mc_reference_features,
    str2bool,
)
from models.fc_flow import load_flow_model
from models.modules import MultiScaleOrthogonalProjector
from models.vq import MultiScaleVQ4
from residual_norm import (
    RESIDUAL_NORM_MODES,
    load_residual_norm_state_into_args,
    print_residual_norm_stats,
    validate_residual_norm_args,
)
from validate_ada import validate


warnings.filterwarnings("ignore")

SCORE_TYPES = (
    ("scores1", "Logps"),
    ("scores2", "BScores"),
    ("scores", "Merged"),
)
HEATMAP_SCORE_TYPES = (
    ("Logps", "logps"),
    ("BScores", "bscores"),
    ("Merged", "merged"),
)
CSV_COLUMNS = (
    "class_name",
    "score_type",
    "image_auc",
    "image_ap",
    "image_f1",
    "pixel_auc",
    "pixel_ap",
    "pixel_f1",
    "aupro",
)
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def resolve_checkpoint_file(args):
    if args.checkpoint_file:
        if not os.path.isfile(args.checkpoint_file):
            raise FileNotFoundError(f"checkpoint_file not found: {args.checkpoint_file}")
        return args.checkpoint_file

    if not args.checkpoint_path:
        raise ValueError("Set --checkpoint_file or --checkpoint_path.")
    checkpoint_files = sorted(glob.glob(os.path.join(args.checkpoint_path, "*.pth")))
    if not checkpoint_files:
        raise FileNotFoundError(f"No .pth files found in checkpoint_path: {args.checkpoint_path}")
    return checkpoint_files[-1]


def resolve_eval_epoch(args, checkpoint_file):
    if args.eval_epoch is not None:
        return args.eval_epoch
    match = re.search(r"epoch_(\d+)", os.path.basename(checkpoint_file))
    if match:
        return int(match.group(1))
    return None


def build_modules(args, feat_dims):
    vq_ops = MultiScaleVQ4(num_embeddings=args.num_embeddings, channels=feat_dims).to(args.device)
    constraintor = MultiScaleOrthogonalProjector(feat_dims).to(args.device)
    estimators = [load_flow_model(args, feat_dim).to(args.device) for feat_dim in feat_dims]
    return vq_ops, constraintor, estimators


def load_checkpoint_states(args, checkpoint_file, vq_ops, constraintor, estimators):
    checkpoint = torch.load(checkpoint_file, map_location=args.device)
    if "constraintor" not in checkpoint:
        raise KeyError("checkpoint does not contain 'constraintor'.")
    if "estimators" not in checkpoint:
        raise KeyError("checkpoint does not contain 'estimators'.")

    constraintor.load_state_dict(checkpoint["constraintor"])
    if len(checkpoint["estimators"]) != len(estimators):
        raise ValueError(
            f"checkpoint has {len(checkpoint['estimators'])} estimators, "
            f"but --feature_levels={len(estimators)}."
        )
    for estimator, state_dict in zip(estimators, checkpoint["estimators"]):
        estimator.load_state_dict(state_dict)

    load_residual_norm_state_into_args(args, checkpoint)

    if getattr(args, "disable_vqops", False):
        print("[VQOps] disabled by --disable_vqops; evaluating without VQOps/EFDM.")
        return None

    if "vq_ops" in checkpoint:
        vq_ops.load_state_dict(checkpoint["vq_ops"])
        return vq_ops

    print("[VQOps] checkpoint has no vq_ops state; evaluating without VQOps/EFDM.")
    return None


def metric_row(class_name, score_type, values):
    image_auc, image_ap, image_f1, pixel_auc, pixel_ap, pixel_f1, aupro = values
    return {
        "class_name": class_name,
        "score_type": score_type,
        "image_auc": image_auc,
        "image_ap": image_ap,
        "image_f1": image_f1,
        "pixel_auc": pixel_auc,
        "pixel_ap": pixel_ap,
        "pixel_f1": pixel_f1,
        "aupro": aupro,
    }


def print_metric_block(title, values):
    image_auc, image_ap, image_f1, pixel_auc, pixel_ap, pixel_f1, aupro = values
    print(f"[{title}]")
    print(f"Image AUC | AP | F1: {image_auc:.3f} | {image_ap:.3f} | {image_f1:.3f}")
    print(f"Pixel AUC | AP | F1 | AUPRO: {pixel_auc:.3f} | {pixel_ap:.3f} | {pixel_f1:.3f} | {aupro:.3f}")


def print_average_line(label, values):
    image_auc, image_ap, image_f1, pixel_auc, pixel_ap, pixel_f1, aupro = values
    print(
        f"({label}) Average Image AUC | AP | F1: {image_auc:.3f} | {image_ap:.3f} | {image_f1:.3f}, "
        f"Average Pixel AUC | AP | F1 | AUPRO: {pixel_auc:.3f} | {pixel_ap:.3f} | "
        f"{pixel_f1:.3f} | {aupro:.3f}"
    )


def save_csv(path, rows):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] saved: {path}")


def resolve_eval_class_names(args, classes):
    requested = []
    if args.class_names:
        requested.extend(args.class_names)
    if args.class_name:
        requested.append(args.class_name)
    if not requested:
        return list(classes["unseen"])

    deduped = []
    for class_name in requested:
        if class_name not in deduped:
            deduped.append(class_name)
        if class_name not in classes["unseen"]:
            print(
                f"[Eval-Ada-IBStyle] class_name '{class_name}' is not in "
                f"the unseen list for setting '{args.setting}'; evaluating it anyway."
            )
    return deduped


def _ensure_batch_array(array):
    array = np.asarray(array)
    if array.ndim == 2:
        return array[None, ...]
    return array


def denormalization(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"image must have shape [C,H,W], got {x.shape}")
    x = x.transpose(1, 2, 0)
    x = (x * IMAGENET_STD) + IMAGENET_MEAN
    x = (x * 255.0).clip(0, 255).astype(np.uint8)
    return x


def save_score_comparison(path, class_name, index, image, gt_mask, score_maps):
    num_cols = 2 + len(HEATMAP_SCORE_TYPES)
    fig, axs = plt.subplots(1, num_cols, figsize=(4 * num_cols, 5))
    fig.suptitle(f"{class_name} - Sample {index:03d}", fontsize=16)

    axs[0].imshow(image)
    axs[0].set_title("Input Image")
    axs[0].axis("off")

    axs[1].imshow(gt_mask, cmap="gray")
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")

    for offset, (score_name, _) in enumerate(HEATMAP_SCORE_TYPES):
        ax = axs[2 + offset]
        im = ax.imshow(score_maps[score_name], cmap="jet", interpolation="nearest")
        ax.imshow(image, alpha=0.3, interpolation="none")
        ax.set_title(score_name)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_visual_outputs(output_root, class_name, metrics):
    if "score_maps" not in metrics:
        print("[Heatmap] validate() did not return score_maps; skipped visual output.")
        return

    comparison_dir = os.path.join(output_root, class_name, "score_visuals")
    os.makedirs(comparison_dir, exist_ok=True)

    loaded_images = np.asarray(metrics["input_images"])
    loaded_masks = _ensure_batch_array(metrics["gt_masks"]).astype(np.float32)
    score_maps = {name: _ensure_batch_array(values) for name, values in metrics["score_maps"].items()}
    total_samples = min([len(loaded_images), len(loaded_masks)] + [values.shape[0] for values in score_maps.values()])

    print(f"\n--- Saving score heatmaps ({class_name}) ---")
    plt.ioff()
    for index in range(total_samples):
        image = denormalization(loaded_images[index])
        gt_mask = loaded_masks[index]
        sample_scores = {score_name: score_maps[score_name][index] for score_name, _ in HEATMAP_SCORE_TYPES}

        save_score_comparison(
            os.path.join(comparison_dir, f"sample_{index:03d}_scores.png"),
            class_name,
            index,
            image,
            gt_mask,
            sample_scores,
        )

        if (index + 1) % 10 == 0:
            print(f"Processed {index + 1}/{total_samples} images...")

    print(f"[Heatmap] saved {total_samples} comparison figures to {comparison_dir}")


def main(args):
    validate_residual_norm_args(args)
    if args.setting not in SETTINGS:
        raise ValueError(f"Dataset setting must be in {SETTINGS.keys()}, but got {args.setting}.")
    if args.feature_levels != len(args.clip_layers) and args.feature_backbone in ("clip_raw", "adaclip_prompted"):
        raise ValueError(
            f"feature_levels={args.feature_levels} must match len(clip_layers)={len(args.clip_layers)} "
            "for CLIP/AdaCLIP feature backbones."
        )

    classes = SETTINGS[args.setting]
    eval_class_names = resolve_eval_class_names(args, classes)
    checkpoint_file = resolve_checkpoint_file(args)
    eval_epoch = resolve_eval_epoch(args, checkpoint_file)
    image_size = get_feature_image_size(args)

    print("[Eval-Ada-IBStyle] checkpoint_file:", checkpoint_file)
    print("[Eval-Ada-IBStyle] eval_epoch:", eval_epoch)
    print("[Eval-Ada-IBStyle] setting:", args.setting)
    print("[Eval-Ada-IBStyle] classes:", eval_class_names)
    print("[Eval-Ada-IBStyle] feature_backbone:", args.feature_backbone)
    print("[Eval-Ada-IBStyle] clip_layers:", args.clip_layers)
    print("[Eval-Ada-IBStyle] clip_image_size:", args.clip_image_size)
    print("[Eval-Ada-IBStyle] feature_levels:", args.feature_levels)
    print("[Eval-Ada-IBStyle] num_ref_shot:", args.num_ref_shot)
    print("[Eval-Ada-IBStyle] test_ref_feature_dir:", args.test_ref_feature_dir)
    print("[Eval-Ada-IBStyle] residual_mode: sq")
    print("[Eval-Ada-IBStyle] adaclip_feature_l2norm:", args.feature_backbone == "adaclip_prompted" and args.adaclip_feature_l2norm)
    print("[Eval-Ada-IBStyle] requested residual_norm:", args.residual_norm)
    if args.save_heatmap_dir:
        print("[Heatmap] save_heatmap_dir:", args.save_heatmap_dir)
        print("[Heatmap] output: Input / GT / Logps / BScores / Merged comparison figures")
        print("[Heatmap] colorbar range: automatic")

    encoder, feat_dims = build_feature_encoder(args)
    print("[Eval-Ada-IBStyle] feat_dims:", feat_dims)
    vq_ops, constraintor, estimators = build_modules(args, feat_dims)
    vq_ops = load_checkpoint_states(args, checkpoint_file, vq_ops, constraintor, estimators)
    print("[Eval-Ada-IBStyle] residual_norm:", args.residual_norm)
    print("[Eval-Ada-IBStyle] residual_norm_eps:", args.residual_norm_eps)
    print("[Eval-Ada-IBStyle] residual_norm_clip:", args.residual_norm_clip)
    print_residual_norm_stats(getattr(args, "residual_norm_stats", None))

    encoder.eval()
    if vq_ops is not None:
        vq_ops.eval()
    constraintor.eval()
    for estimator in estimators:
        estimator.eval()

    test_ref_features = load_mc_reference_features(
        args.test_ref_feature_dir,
        eval_class_names,
        args.device,
        args.num_ref_shot,
        feature_levels=args.feature_levels,
    )

    results_by_type = {label: [] for _, label in SCORE_TYPES}
    csv_rows = []
    for class_name in eval_class_names:
        test_dataset = build_test_dataset(args, class_name, image_size)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )
        metrics = validate(
            args,
            encoder,
            vq_ops,
            constraintor,
            estimators,
            test_loader,
            test_ref_features[class_name],
            args.device,
            class_name,
        )

        if args.save_heatmap_dir:
            save_visual_outputs(args.save_heatmap_dir, class_name, metrics)

        print(f"\nClass: {class_name}")
        for key, label in SCORE_TYPES:
            values = metrics[key]
            print_metric_block(label, values)
            results_by_type[label].append(values)
            csv_rows.append(metric_row(class_name, label, values))

    print("\nAverages")
    for _, label in SCORE_TYPES:
        values = np.mean(np.asarray(results_by_type[label]), axis=0)
        print_average_line(label, values)
        csv_rows.append(metric_row("Average", label, values))

    if args.save_csv:
        save_csv(args.save_csv, csv_rows)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=str, required=True)
    parser.add_argument("--classes", type=str, default="none")
    parser.add_argument("--class_name", type=str, default="")
    parser.add_argument("--class_names", type=str, nargs="+", default=None)
    parser.add_argument("--test_dataset_dir", type=str, required=True)
    parser.add_argument("--test_ref_feature_dir", type=str, required=True)
    parser.add_argument("--checkpoint_file", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2")
    parser.add_argument("--feature_backbone", type=str, default="adaclip_prompted", choices=["adaclip_prompted", "clip_raw", "original"])
    parser.add_argument("--num_ref_shot", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_csv", type=str, default="")
    parser.add_argument("--save_heatmap_dir", type=str, default="")
    parser.add_argument("--eval_epoch", type=int, default=None)

    parser.add_argument("--clip_model", type=str, default="ViT-L-14-336")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument("--clip_weight_source", type=str, default="open_clip", choices=["open_clip", "openai_local"])
    parser.add_argument("--clip_checkpoint", type=str, default="")
    parser.add_argument("--clip_layers", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--clip_image_size", type=int, default=336)
    parser.add_argument("--adaclip_repo_url", type=str, default="https://github.com/tomo082/AdaCLIP_res")
    parser.add_argument("--adaclip_repo_path", type=str, default="")
    parser.add_argument("--adaclip_checkpoint", type=str, default="")
    parser.add_argument("--adaclip_checkpoint_url", type=str, default="")
    parser.add_argument("--adaclip_cache_dir", type=str, default="~/.cache/adaclip_res")
    parser.add_argument("--adaclip_model", type=str, default="ViT-L-14-336")
    parser.add_argument("--adaclip_return_projected", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--adaclip_feature_l2norm", action="store_true")

    parser.add_argument("--flow_arch", type=str, default="flow_model")
    parser.add_argument("--feature_levels", default=4, type=int)
    parser.add_argument("--coupling_layers", type=int, default=4)
    parser.add_argument("--clamp_alpha", type=float, default=1.9)
    parser.add_argument("--pos_embed_dim", type=int, default=256)
    parser.add_argument("--fdm_alpha", type=float, default=0.4)
    parser.add_argument("--num_embeddings", type=int, default=1536)
    parser.add_argument("--disable_vqops", action="store_true")
    parser.add_argument("--residual_norm", type=str, default="none", choices=RESIDUAL_NORM_MODES)
    parser.add_argument("--residual_stats_batches", type=int, default=50)
    parser.add_argument("--residual_norm_eps", type=float, default=1e-6)
    parser.add_argument("--residual_norm_clip", type=float, default=0.0)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
