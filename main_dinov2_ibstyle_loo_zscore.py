"""Evaluate DINOv2 IB-style checkpoints with 4-shot LOO score calibration.

This script keeps the trained main_dinov2_ibstyle.py model path intact and only
adds an evaluation-time calibration pass:

1. Split each class reference memory bank into original 4-shot chunks.
2. Hold one shot out, match it against the remaining reference shots, and score it
   with the same ResAD downstream path used for test images.
3. Estimate per-pixel mu_LOO and sigma_LOO from those normal LOO score maps.
4. Evaluate test images with both the baseline score maps and LOO z-score maps.

Example:
python main_dinov2_ibstyle_loo_zscore.py \
  --setting visa_to_mvtec \
  --test_dataset_dir /data/home/ueno/mvtec-data \
  --test_ref_feature_dir /data/home/ueno/ref_features/dinov2/mvtec_4shot_ib \
  --checkpoint_file /data/home/ueno/checkpoint/visa_to_mvtec_dinov2_ibstyle/visa_to_mvtec_epoch_5_checkpoints.pth \
  --backbone dinov2_vits14 \
  --device cuda:0 \
  --num_ref_shot 4
"""

import csv
import os
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from eval_checkpoint_dinov2_ibstyle import (
    SCORE_TYPES,
    SETTINGS,
    build_encoder,
    build_modules,
    build_parser as build_eval_parser,
    build_test_dataset,
    dinov2_tokens_to_maps,
    load_checkpoint_states,
    resolve_checkpoint_file,
    resolve_eval_epoch,
)
from losses.utils import get_logp_a
from models.modules import get_position_encoding
from models.utils import get_logp
from utils import applying_EFDM, calculate_metrics, get_matched_ref_features, get_residual_features
from validate import aggregate_anomaly_scores, convert_to_anomaly_scores


warnings.filterwarnings("ignore")

TOTAL_SHOT = 4
CSV_COLUMNS = (
    "class_name",
    "calibration",
    "score_type",
    "image_auc",
    "image_ap",
    "image_f1",
    "pixel_auc",
    "pixel_ap",
    "pixel_f1",
    "aupro",
)


def as_score_batch(scores):
    if scores.ndim == 2:
        return scores[None, ...]
    return scores


def flat_ref_to_feature_map(flat_ref):
    patches, channels = flat_ref.shape
    side = int(patches ** 0.5)
    if side * side != patches:
        raise ValueError(f"Expected square per-shot reference grid, got {patches} patches.")
    return flat_ref.reshape(1, side, side, channels).permute(0, 3, 1, 2).contiguous()


def load_reference_feature_groups(root_dir, class_name, device, num_shot=4, feature_levels=4, total_shot=TOTAL_SHOT):
    if num_shot < 2:
        raise ValueError("LOO calibration needs at least 2 reference shots.")
    if num_shot > total_shot:
        raise ValueError(f"num_ref_shot={num_shot} exceeds expected total_shot={total_shot}.")

    chunks_by_level = []
    patches_per_level = []
    for level in range(feature_levels):
        path = os.path.join(root_dir, class_name, f"layer{level + 1}.npy")
        layer_refs = torch.from_numpy(np.load(path)).to(device=device, dtype=torch.float32)
        if layer_refs.shape[0] % total_shot != 0:
            raise ValueError(
                f"{path} has {layer_refs.shape[0]} rows, which is not divisible by total_shot={total_shot}. "
                "This first LOO implementation expects original 4-shot reference features without augmentation."
            )
        patches_per_shot = layer_refs.shape[0] // total_shot
        chunks = [
            layer_refs[shot * patches_per_shot:(shot + 1) * patches_per_shot]
            for shot in range(num_shot)
        ]
        chunks_by_level.append(chunks)
        patches_per_level.append(patches_per_shot)

    full_refs = tuple(torch.cat(level_chunks, dim=0).contiguous() for level_chunks in chunks_by_level)
    return {
        "chunks_by_level": chunks_by_level,
        "full_refs": full_refs,
        "patches_per_level": patches_per_level,
    }


def build_loo_query_and_refs(reference_groups, held_out_idx):
    query_features = []
    loo_refs = []
    for level_chunks in reference_groups["chunks_by_level"]:
        query_features.append(flat_ref_to_feature_map(level_chunks[held_out_idx]))
        ref_chunks = [chunk for idx, chunk in enumerate(level_chunks) if idx != held_out_idx]
        loo_refs.append(torch.cat(ref_chunks, dim=0).contiguous())
    return query_features, tuple(loo_refs)


def append_logps_for_feature_maps(args, feature_maps, ref_features, vq_ops, constraintor, estimators, logps1_list, logps2_list):
    mfeatures = get_matched_ref_features(feature_maps, ref_features)
    rfeatures = get_residual_features(feature_maps, mfeatures, pos_flag=True)
    if not args.residual:
        rfeatures = feature_maps

    if vq_ops is not None:
        fdm_features = vq_ops(rfeatures, train=False)
        rfeatures = applying_EFDM(rfeatures, fdm_features, alpha=args.fdm_alpha)
    rfeatures = constraintor(*rfeatures)

    for level in range(args.feature_levels):
        e = rfeatures[level]
        bs, dim, h, w = e.size()
        e = e.permute(0, 2, 3, 1).reshape(-1, dim)

        pos_embed = get_position_encoding(args.pos_embed_dim, h, w).to(args.device).unsqueeze(0).repeat(bs, 1, 1, 1)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, args.pos_embed_dim)
        estimator = estimators[level]

        if args.flow_arch == "flow_model":
            z, log_jac_det = estimator(e)
        else:
            z, log_jac_det = estimator(e, [pos_embed])

        logps = get_logp(dim, z, log_jac_det)
        logps = logps / dim
        logps1_list[level].append(logps.reshape(bs, h, w))

        logps_a = get_logp_a(dim, z, log_jac_det)
        logits = torch.stack([logps, logps_a], dim=-1)
        sa = torch.softmax(logits, dim=-1)[:, 1]
        logps2_list[level].append(sa.reshape(bs, h, w))


def score_maps_from_logps(logps1_list, logps2_list, args, class_name, size):
    scores1 = convert_to_anomaly_scores(logps1_list, feature_levels=args.feature_levels, class_name=class_name, size=size)
    scores2 = aggregate_anomaly_scores(logps2_list, feature_levels=args.feature_levels, class_name=class_name, size=size)
    scores1 = as_score_batch(scores1)
    scores2 = as_score_batch(scores2)
    scores = (scores1 + scores2) / 2
    return {
        "scores1": scores1,
        "scores2": scores2,
        "scores": scores,
    }


def compute_loo_scores(args, reference_groups, vq_ops, constraintor, estimators, class_name):
    logps1_list = [list() for _ in range(args.feature_levels)]
    logps2_list = [list() for _ in range(args.feature_levels)]
    for held_out_idx in range(args.num_ref_shot):
        query_features, loo_refs = build_loo_query_and_refs(reference_groups, held_out_idx)
        with torch.no_grad():
            append_logps_for_feature_maps(
                args,
                query_features,
                loo_refs,
                vq_ops,
                constraintor,
                estimators,
                logps1_list,
                logps2_list,
            )
    return score_maps_from_logps(logps1_list, logps2_list, args, class_name, args.image_size)


def collect_test_scores(args, encoder, vq_ops, constraintor, estimators, test_loader, ref_features, device, class_name):
    label_list, gt_mask_list = [], []
    logps1_list = [list() for _ in range(args.feature_levels)]
    logps2_list = [list() for _ in range(args.feature_levels)]
    size = args.image_size

    progress_bar = tqdm(total=len(test_loader))
    progress_bar.set_description("Evaluating")
    for batch in test_loader:
        progress_bar.update(1)
        image, label, mask = batch[0], batch[1], batch[2]
        gt_mask_list.append(mask.squeeze(1).cpu().numpy().astype(bool))
        label_list.append(label.cpu().numpy().astype(bool).ravel())

        image = image.to(device)
        size = image.shape[-1]
        with torch.no_grad():
            features = dinov2_tokens_to_maps(encoder.encode_image_from_tensors(image))
            append_logps_for_feature_maps(
                args,
                features,
                ref_features,
                vq_ops,
                constraintor,
                estimators,
                logps1_list,
                logps2_list,
            )
    progress_bar.close()

    labels = np.concatenate(label_list)
    gt_masks = np.concatenate(gt_mask_list, axis=0)
    scores = score_maps_from_logps(logps1_list, logps2_list, args, class_name, size)
    return scores, labels, gt_masks


def apply_loo_zscore(test_scores, loo_scores, sigma_eps):
    calibrated = {}
    stats = {}
    for key in test_scores:
        mu = loo_scores[key].mean(axis=0, keepdims=True)
        sigma = loo_scores[key].std(axis=0, keepdims=True)
        calibrated[key] = (test_scores[key] - mu) / (sigma + sigma_eps)
        stats[key] = {"mu": mu, "sigma": sigma}
    return calibrated, stats


def evaluate_score_maps(score_maps, labels, gt_masks):
    return {
        key: calculate_metrics(scores, labels, gt_masks, pro=False, only_max_value=True)
        for key, scores in score_maps.items()
    }


def metric_row(class_name, calibration, score_type, values):
    image_auc, image_ap, image_f1, pixel_auc, pixel_ap, pixel_f1, aupro = values
    return {
        "class_name": class_name,
        "calibration": calibration,
        "score_type": score_type,
        "image_auc": image_auc,
        "image_ap": image_ap,
        "image_f1": image_f1,
        "pixel_auc": pixel_auc,
        "pixel_ap": pixel_ap,
        "pixel_f1": pixel_f1,
        "aupro": aupro,
    }


def print_metric_line(calibration, score_type, values):
    image_auc, image_ap, image_f1, pixel_auc, pixel_ap, pixel_f1, aupro = values
    print(
        f"({calibration}/{score_type}) Image AUC | AP | F1: "
        f"{image_auc:.3f} | {image_ap:.3f} | {image_f1:.3f}, "
        f"Pixel AUC | AP | F1 | AUPRO: "
        f"{pixel_auc:.3f} | {pixel_ap:.3f} | {pixel_f1:.3f} | {aupro:.3f}"
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


def main(args):
    if args.setting not in SETTINGS:
        raise ValueError(f"Dataset setting must be in {SETTINGS.keys()}, but got {args.setting}.")
    if args.num_ref_shot != TOTAL_SHOT:
        print(
            f"[LOO] warning: this first implementation is intended for original {TOTAL_SHOT}-shot references; "
            f"got num_ref_shot={args.num_ref_shot}."
        )

    classes = SETTINGS[args.setting]
    checkpoint_file = resolve_checkpoint_file(args)
    eval_epoch = resolve_eval_epoch(args, checkpoint_file)

    print("[LOO-ZScore] checkpoint_file:", checkpoint_file)
    print("[LOO-ZScore] eval_epoch:", eval_epoch)
    print("[LOO-ZScore] setting:", args.setting)
    print("[LOO-ZScore] classes:", classes["unseen"])
    print("[LOO-ZScore] backbone:", args.backbone)
    print("[LOO-ZScore] feature_levels:", args.feature_levels)
    print("[LOO-ZScore] num_ref_shot:", args.num_ref_shot)
    print("[LOO-ZScore] sigma_eps:", args.loo_sigma_eps)
    print("[LOO-ZScore] residual_mode: sq")

    encoder, feat_dims = build_encoder(args)
    vq_ops, constraintor, estimators = build_modules(args, feat_dims)
    vq_ops = load_checkpoint_states(args, checkpoint_file, vq_ops, constraintor, estimators)

    encoder.eval()
    if vq_ops is not None:
        vq_ops.eval()
    constraintor.eval()
    for estimator in estimators:
        estimator.eval()

    results_by_kind = {}
    csv_rows = []
    for class_name in classes["unseen"]:
        print(f"\nClass: {class_name}")
        reference_groups = load_reference_feature_groups(
            args.test_ref_feature_dir,
            class_name,
            args.device,
            num_shot=args.num_ref_shot,
            feature_levels=args.feature_levels,
            total_shot=TOTAL_SHOT,
        )
        print("[LOO] patches_per_level:", reference_groups["patches_per_level"])

        loo_scores = compute_loo_scores(args, reference_groups, vq_ops, constraintor, estimators, class_name)
        for key, label in SCORE_TYPES:
            print(f"[LOO] {label} normal score maps: {tuple(loo_scores[key].shape)}")

        test_dataset = build_test_dataset(args, class_name)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )
        test_scores, labels, gt_masks = collect_test_scores(
            args,
            encoder,
            vq_ops,
            constraintor,
            estimators,
            test_loader,
            reference_groups["full_refs"],
            args.device,
            class_name,
        )
        loo_z_scores, _ = apply_loo_zscore(test_scores, loo_scores, args.loo_sigma_eps)

        metrics_by_calibration = {
            "baseline": evaluate_score_maps(test_scores, labels, gt_masks),
            "loo_zscore": evaluate_score_maps(loo_z_scores, labels, gt_masks),
        }
        for calibration, metrics in metrics_by_calibration.items():
            for key, label in SCORE_TYPES:
                values = metrics[key]
                print_metric_line(calibration, label, values)
                results_by_kind.setdefault((calibration, label), []).append(values)
                csv_rows.append(metric_row(class_name, calibration, label, values))

    print("\nAverages")
    for calibration in ("baseline", "loo_zscore"):
        for _, label in SCORE_TYPES:
            values = np.mean(np.asarray(results_by_kind[(calibration, label)]), axis=0)
            print_metric_line(calibration, label, values)
            csv_rows.append(metric_row("Average", calibration, label, values))

    if args.save_csv:
        save_csv(args.save_csv, csv_rows)


def build_parser():
    parser = build_eval_parser()
    parser.description = "Evaluate DINOv2 IB-style checkpoints with 4-shot Leave-One-Out z-score calibration."
    parser.add_argument("--loo_sigma_eps", type=float, default=1e-6)
    parser.add_argument("--image_size", type=int, default=224)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
