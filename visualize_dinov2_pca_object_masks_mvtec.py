import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

from models.dinov2_encoder import DINOv2IBStyleEncoder, default_dinov2_out_indices, print_dinov2_ibstyle_config


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PIL_BICUBIC = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
PIL_NEAREST = getattr(getattr(Image, "Resampling", Image), "NEAREST")
CSV_COLUMNS = [
    "idx",
    "class",
    "defect_type",
    "image_path",
    "save_path",
    "has_mask",
    "mask_area_ratio",
    "bbox_area_ratio",
    "gt_area_ratio",
    "gt_covered_by_mask",
    "mask_covered_by_gt",
    "pca_threshold",
    "foreground_direction",
    "patch_grid_h",
    "patch_grid_w",
]


def build_transform(image_size):
    return T.Compose(
        [
            T.Resize(image_size, T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_encoder(args):
    out_indices = args.dinov2_out_indices or default_dinov2_out_indices(args.backbone)
    encoder = DINOv2IBStyleEncoder(
        model_name=args.backbone,
        out_indices=out_indices,
        hub_repo=args.dinov2_hub_repo,
        hub_source=args.dinov2_hub_source,
        freeze=True,
    ).to(args.device)
    encoder.eval()
    print_dinov2_ibstyle_config(encoder)
    return encoder


def iter_mvtec_test_images(mvtec_root, class_name):
    test_root = mvtec_root / class_name / "test"
    if not test_root.is_dir():
        raise FileNotFoundError(f"MVTec test directory not found: {test_root}")
    for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
        for image_path in sorted(defect_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield defect_dir.name, image_path


def load_gt_mask(mvtec_root, class_name, defect_type, image_path, image_size):
    if defect_type == "good":
        return np.zeros((image_size[1], image_size[0]), dtype=bool)

    mask_path = mvtec_root / class_name / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
    if not mask_path.is_file():
        print(f"[WARN] missing GT mask, using zeros: {mask_path}")
        return np.zeros((image_size[1], image_size[0]), dtype=bool)

    mask = Image.open(mask_path).convert("L")
    if mask.size != image_size:
        mask = mask.resize(image_size, PIL_NEAREST)
    return np.asarray(mask) > 0


def extract_patch_tokens(encoder, transform, image, args):
    tensor = transform(image).unsqueeze(0).to(args.device)
    with torch.no_grad():
        features = encoder.encode_image_from_tensors(tensor)
    layer_index = args.pca_layer
    if layer_index < 0:
        layer_index = len(features) + layer_index
    if layer_index < 0 or layer_index >= len(features):
        raise ValueError(f"pca_layer={args.pca_layer} is out of range for {len(features)} DINOv2 levels.")
    tokens = features[layer_index][0].detach().float().cpu().numpy()
    return tokens


def pca_first_component(tokens):
    if tokens.ndim != 2:
        raise ValueError(f"Expected patch tokens [N,C], got {tokens.shape}.")
    centered = tokens - tokens.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ vt[0]
    side = int(math.sqrt(scores.shape[0]))
    if side * side != scores.shape[0]:
        raise ValueError(f"Expected square DINOv2 patch grid, got {scores.shape[0]} tokens.")
    return scores.reshape(side, side)


def border_mask_for_grid(height, width):
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border


def auto_oriented_patch_mask(pc1, foreground_percentile):
    high_threshold = float(np.percentile(pc1, foreground_percentile))
    low_threshold = float(np.percentile(pc1, 100.0 - foreground_percentile))
    high_mask = pc1 >= high_threshold
    low_mask = pc1 <= low_threshold

    border = border_mask_for_grid(*pc1.shape)
    high_border = float(high_mask[border].mean())
    low_border = float(low_mask[border].mean())

    if high_border <= low_border:
        return high_mask, high_threshold, "high"
    return low_mask, low_threshold, "low"


def resize_mask_nearest(mask, image_size):
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_image = mask_image.resize(image_size, PIL_NEAREST)
    return np.asarray(mask_image) > 0


def require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for mask post-processing and overlays.") from exc
    return cv2


def remove_small_regions(mask, min_area_ratio):
    if min_area_ratio <= 0:
        return mask
    cv2 = require_cv2()
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    min_area = max(1, int(round(mask.size * min_area_ratio)))
    cleaned = np.zeros_like(mask_u8)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 1
    return cleaned.astype(bool)


def fill_holes(mask):
    cv2 = require_cv2()
    mask_u8 = (mask.astype(np.uint8) * 255)
    flood = mask_u8.copy()
    h, w = flood.shape
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask_u8, holes)
    return filled > 0


def dilate_mask(mask, kernel_size, iterations):
    cv2 = require_cv2()
    kernel_size = max(1, int(kernel_size))
    iterations = max(1, int(iterations))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations)
    return dilated > 0


def postprocess_mask(mask, args):
    mask = remove_small_regions(mask, args.min_area_ratio)
    mask = fill_holes(mask)
    mask = dilate_mask(mask, args.dilation_kernel_size, args.dilation_iterations)
    return mask


def padded_bbox_area_ratio(mask, padding_ratio):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0.0
    h, w = mask.shape
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    bw = x1 - x0
    bh = y1 - y0
    pad_x = int(round(bw * padding_ratio))
    pad_y = int(round(bh * padding_ratio))
    x0 = max(0, x0 - pad_x)
    x1 = min(w, x1 + pad_x)
    y0 = max(0, y0 - pad_y)
    y1 = min(h, y1 + pad_y)
    return float((x1 - x0) * (y1 - y0) / mask.size)


def coverage_metrics(mask, gt_mask):
    mask_area = int(mask.sum())
    gt_area = int(gt_mask.sum())
    intersection = int(np.logical_and(mask, gt_mask).sum())
    gt_covered_by_mask = np.nan if gt_area == 0 else float(intersection / gt_area)
    mask_covered_by_gt = np.nan if mask_area == 0 else float(intersection / mask_area)
    return gt_covered_by_mask, mask_covered_by_gt


def mask_to_rgb(mask, color=(255, 255, 255)):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask] = np.asarray(color, dtype=np.uint8)
    return out


def add_label(image, label):
    image = Image.fromarray(image)
    label_height = 28
    labeled = Image.new("RGB", (image.width, image.height + label_height), color=(255, 255, 255))
    labeled.paste(image, (0, label_height))
    draw = ImageDraw.Draw(labeled)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    draw.text((8, 7), label, fill=(0, 0, 0), font=font)
    return np.asarray(labeled)


def save_triplet(path, image_rgb, gt_mask, pred_mask):
    panels = [
        add_label(image_rgb, "Input image"),
        add_label(mask_to_rgb(gt_mask), "GT mask"),
        add_label(mask_to_rgb(pred_mask), "DINOv2 PCA foreground mask"),
    ]
    triplet = np.concatenate(panels, axis=1)
    Image.fromarray(triplet).save(path)


def draw_mask_contour(image_rgb, mask, color=(255, 220, 0), thickness=2):
    cv2 = require_cv2()
    overlay = image_rgb.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, thickness=thickness)
    return overlay


def blend_mask(image_rgb, mask, color, alpha=0.35):
    out = image_rgb.astype(np.float32).copy()
    color_arr = np.asarray(color, dtype=np.float32)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def save_overlays(overlay_dir, stem, image_rgb, gt_mask, pred_mask):
    overlay_dir.mkdir(parents=True, exist_ok=True)
    pred_contour = draw_mask_contour(image_rgb, pred_mask)
    Image.fromarray(pred_contour).save(overlay_dir / f"{stem}_dinov2_contour.png")

    combined = blend_mask(image_rgb, gt_mask, color=(255, 0, 0), alpha=0.35)
    combined = blend_mask(combined, pred_mask, color=(0, 255, 0), alpha=0.35)
    combined = draw_mask_contour(combined, gt_mask, color=(255, 0, 0), thickness=2)
    combined = draw_mask_contour(combined, pred_mask, color=(0, 255, 0), thickness=2)
    Image.fromarray(combined).save(overlay_dir / f"{stem}_gt_dinov2_overlay.png")


def save_summary_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def safe_nanmean(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmean(values))


def safe_nanmedian(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmedian(values))


def print_class_summary(class_name, rows):
    mask_detected = sum(1 for row in rows if row["has_mask"])
    defect_gt_cover = [
        row["gt_covered_by_mask"]
        for row in rows
        if row["defect_type"] != "good" and not np.isnan(row["gt_covered_by_mask"])
    ]
    print(f"\n[{class_name}] summary")
    print("total images:", len(rows))
    print("mask detected:", mask_detected)
    print("no mask detected:", len(rows) - mask_detected)
    print("mean mask_area_ratio:", safe_nanmean([row["mask_area_ratio"] for row in rows]))
    print("mean bbox_area_ratio:", safe_nanmean([row["bbox_area_ratio"] for row in rows]))
    print("mean gt_covered_by_mask for defect images:", safe_nanmean(defect_gt_cover))
    print("median gt_covered_by_mask for defect images:", safe_nanmedian(defect_gt_cover))


def process_image(encoder, transform, mvtec_root, save_root, class_name, defect_type, image_path, idx, args):
    image = Image.open(image_path).convert("RGB")
    image_size = image.size
    image_rgb = np.asarray(image)
    gt_mask = load_gt_mask(mvtec_root, class_name, defect_type, image_path, image_size)

    tokens = extract_patch_tokens(encoder, transform, image, args)
    pc1 = pca_first_component(tokens)
    patch_mask, threshold, direction = auto_oriented_patch_mask(pc1, args.foreground_percentile)
    resized_mask = resize_mask_nearest(patch_mask, image_size)
    pred_mask = postprocess_mask(resized_mask, args)

    triplet_dir = save_root / "triplets" / class_name / defect_type
    overlay_dir = save_root / "overlays" / class_name / defect_type
    triplet_dir.mkdir(parents=True, exist_ok=True)
    save_path = triplet_dir / f"{idx:05d}_{image_path.stem}_triplet.png"
    save_triplet(save_path, image_rgb, gt_mask, pred_mask)
    save_overlays(overlay_dir, f"{idx:05d}_{image_path.stem}", image_rgb, gt_mask, pred_mask)

    mask_area_ratio = float(pred_mask.sum() / pred_mask.size)
    bbox_area_ratio = padded_bbox_area_ratio(pred_mask, args.bbox_padding_ratio)
    gt_area_ratio = float(gt_mask.sum() / gt_mask.size)
    gt_covered_by_mask, mask_covered_by_gt = coverage_metrics(pred_mask, gt_mask)
    has_mask = bool(pred_mask.any())

    print(
        f"{class_name} | {idx:05d} | {defect_type} | "
        f"{'mask' if has_mask else 'no_mask'} | "
        f"mask_area={mask_area_ratio:.4f} | bbox_area={bbox_area_ratio:.4f} | "
        f"gt_cover={gt_covered_by_mask if not np.isnan(gt_covered_by_mask) else 'nan'}"
    )

    return {
        "idx": idx,
        "class": class_name,
        "defect_type": defect_type,
        "image_path": str(image_path),
        "save_path": str(save_path),
        "has_mask": has_mask,
        "mask_area_ratio": mask_area_ratio,
        "bbox_area_ratio": bbox_area_ratio,
        "gt_area_ratio": gt_area_ratio,
        "gt_covered_by_mask": gt_covered_by_mask,
        "mask_covered_by_gt": mask_covered_by_gt,
        "pca_threshold": threshold,
        "foreground_direction": direction,
        "patch_grid_h": pc1.shape[0],
        "patch_grid_w": pc1.shape[1],
    }


def main(args):
    mvtec_root = Path(args.mvtec_root)
    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    transform = build_transform(args.image_size)
    encoder = build_encoder(args)

    print("[DINOv2-PCA] mvtec_root:", mvtec_root)
    print("[DINOv2-PCA] target_classes:", args.target_classes)
    print("[DINOv2-PCA] save_root:", save_root)
    print("[DINOv2-PCA] image_size:", args.image_size)
    print("[DINOv2-PCA] pca_layer:", args.pca_layer)
    print("[DINOv2-PCA] foreground_percentile:", args.foreground_percentile)
    print("[DINOv2-PCA] min_area_ratio:", args.min_area_ratio)
    print("[DINOv2-PCA] dilation:", args.dilation_kernel_size, "x", args.dilation_iterations)
    print("[DINOv2-PCA] bbox_padding_ratio:", args.bbox_padding_ratio)

    all_rows = []
    global_idx = 0
    for class_name in args.target_classes:
        class_rows = []
        for defect_type, image_path in iter_mvtec_test_images(mvtec_root, class_name):
            if args.max_images > 0 and global_idx >= args.max_images:
                break
            row = process_image(
                encoder,
                transform,
                mvtec_root,
                save_root,
                class_name,
                defect_type,
                image_path,
                global_idx,
                args,
            )
            class_rows.append(row)
            all_rows.append(row)
            global_idx += 1
        print_class_summary(class_name, class_rows)

    summary_path = save_root / "summary.csv"
    save_summary_csv(summary_path, all_rows)
    print("\n[DINOv2-PCA] summary saved:", summary_path)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvtec_root", type=str, default="/data/home/ueno/mvtec-data")
    parser.add_argument("--target_classes", type=str, nargs="+", default=["screw", "capsule"])
    parser.add_argument("--save_root", type=str, default="./mvtec_test_dinov2_pca_masks")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--backbone", type=str, default="dinov2_vits14")
    parser.add_argument("--dinov2_out_indices", type=int, nargs="+", default=None)
    parser.add_argument("--dinov2_hub_repo", type=str, default="facebookresearch/dinov2")
    parser.add_argument("--dinov2_hub_source", type=str, default="github", choices=["github", "local"])
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--pca_layer", type=int, default=-1)
    parser.add_argument("--foreground_percentile", type=float, default=70.0)
    parser.add_argument("--min_area_ratio", type=float, default=0.005)
    parser.add_argument("--dilation_kernel_size", type=int, default=5)
    parser.add_argument("--dilation_iterations", type=int, default=2)
    parser.add_argument("--bbox_padding_ratio", type=float, default=0.15)
    parser.add_argument("--max_images", type=int, default=0)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
