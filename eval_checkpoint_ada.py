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
    if not args.class_name:
        return list(classes["unseen"])
    if args.class_name not in classes["unseen"]:
        print(
            f"[Eval-Ada-IBStyle] class_name '{args.class_name}' is not in "
            f"the unseen list for setting '{args.setting}'; evaluating it anyway."
        )
    return [args.class_name]


def _ensure_batch_array(array):
    array = np.asarray(array)
    if array.ndim == 2:
        return array[None, ...]
    return array


def denormalize_w50_images(images):
    images = np.asarray(images, dtype=np.float32)
    if images.ndim != 4:
        raise ValueError(f"input_images must have shape [N,3,H,W], got {images.shape}")
    images = images.transpose(0, 2, 3, 1)
    images = images * IMAGENET_STD.reshape(1, 1, 1, 3) + IMAGENET_MEAN.reshape(1, 1, 1, 3)
    return np.clip(images, 0.0, 1.0)


def save_heatmap_png(path, score_map):
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=150)
    im = ax.imshow(score_map, cmap="jet", vmin=0.0, vmax=2.0)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0.0, 0.5, 1.0, 1.5, 2.0])
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def save_visual_outputs(output_root, class_name, metrics):
    if "score_maps" not in metrics:
        print("[Heatmap] validate() did not return score_maps; skipped visual output.")
        return

    class_dir = os.path.join(output_root, class_name)
    os.makedirs(class_dir, exist_ok=True)

    images = denormalize_w50_images(metrics["input_images"])
    gt_masks = _ensure_batch_array(metrics["gt_masks"]).astype(np.float32)
    score_maps = {name: _ensure_batch_array(values) for name, values in metrics["score_maps"].items()}
    n_images = min([images.shape[0], gt_masks.shape[0]] + [values.shape[0] for values in score_maps.values()])

    for index in range(n_images):
        prefix = f"{index:04d}"
        plt.imsave(os.path.join(class_dir, f"{prefix}_input.png"), images[index])
        plt.imsave(os.path.join(class_dir, f"{prefix}_gt_mask.png"), gt_masks[index], cmap="gray", vmin=0.0, vmax=1.0)
        for score_name, slug in HEATMAP_SCORE_TYPES:
            save_heatmap_png(
                os.path.join(class_dir, f"{prefix}_{slug}_heatmap.png"),
                score_maps[score_name][index],
            )
    print(f"[Heatmap] saved {n_images} samples with Logps/BScores/Merged heatmaps to {class_dir}")


def main(args):
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
    if args.save_heatmap_dir:
        print("[Heatmap] save_heatmap_dir:", args.save_heatmap_dir)
        print("[Heatmap] score_types: Logps, BScores, Merged")
        print("[Heatmap] colorbar range: 0.0 - 2.0")

    encoder, feat_dims = build_feature_encoder(args)
    print("[Eval-Ada-IBStyle] feat_dims:", feat_dims)
    vq_ops, constraintor, estimators = build_modules(args, feat_dims)
    vq_ops = load_checkpoint_states(args, checkpoint_file, vq_ops, constraintor, estimators)

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

    parser.add_argument("--flow_arch", type=str, default="flow_model")
    parser.add_argument("--feature_levels", default=4, type=int)
    parser.add_argument("--coupling_layers", type=int, default=4)
    parser.add_argument("--clamp_alpha", type=float, default=1.9)
    parser.add_argument("--pos_embed_dim", type=int, default=256)
    parser.add_argument("--fdm_alpha", type=float, default=0.4)
    parser.add_argument("--num_embeddings", type=int, default=1536)
    parser.add_argument("--disable_vqops", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
