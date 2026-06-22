import argparse
import csv
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


DEFAULT_CLASS_PROMPTS = {
    "screw": "a metal screw",
    "capsule": "a capsule",
}


def load_grounding_dino_api():
    try:
        from groundingdino.util.inference import load_image, load_model, predict
    except ImportError as exc:
        raise ImportError(
            "Grounding DINO is required. Install IDEA-Research/GroundingDINO "
            "and make the groundingdino package importable."
        ) from exc
    return load_image, load_model, predict


def load_rgb_image(path: Path) -> Image.Image:
    with path.open("rb") as f:
        return Image.open(f).convert("RGB")


def load_gt_mask(
    mvtec_root: Path,
    class_name: str,
    defect_type: str,
    image_path: Path,
    image_size: Tuple[int, int],
) -> np.ndarray:
    width, height = image_size
    if defect_type == "good":
        return np.zeros((height, width), dtype=np.uint8)

    mask_path = mvtec_root / class_name / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
    if not mask_path.exists():
        warnings.warn(f"GT mask not found, using an all-zero mask: {mask_path}")
        return np.zeros((height, width), dtype=np.uint8)

    with mask_path.open("rb") as f:
        mask = Image.open(f).convert("L")
        if mask.size != image_size:
            mask = mask.resize(image_size, Image.Resampling.NEAREST)
        return (np.asarray(mask) > 0).astype(np.uint8)


def iter_mvtec_test_images(
    mvtec_root: Path,
    target_classes: Iterable[str],
) -> Iterable[Tuple[str, str, Path]]:
    for class_name in target_classes:
        test_dir = mvtec_root / class_name / "test"
        if not test_dir.exists():
            warnings.warn(f"Missing test directory: {test_dir}")
            continue
        for defect_dir in sorted(path for path in test_dir.iterdir() if path.is_dir()):
            for image_path in sorted(defect_dir.glob("*.png")):
                yield class_name, defect_dir.name, image_path


def parse_class_prompts(values: Sequence[str]) -> Dict[str, str]:
    prompts = dict(DEFAULT_CLASS_PROMPTS)
    for value in values:
        if "=" not in value:
            raise ValueError(f"Prompt override must use CLASS=PROMPT format: {value}")
        class_name, prompt = value.split("=", 1)
        class_name = class_name.strip()
        prompt = prompt.strip()
        if not class_name or not prompt:
            raise ValueError(f"Invalid prompt override: {value}")
        prompts[class_name] = prompt
    return prompts


def normalized_cxcywh_to_padded_xyxy(
    box: np.ndarray,
    image_size: Tuple[int, int],
    padding_ratio: float,
) -> Tuple[int, int, int, int]:
    width, height = image_size
    cx, cy, box_w, box_h = [float(value) for value in box]
    x0 = (cx - box_w / 2.0) * width
    y0 = (cy - box_h / 2.0) * height
    x1 = (cx + box_w / 2.0) * width
    y1 = (cy + box_h / 2.0) * height
    pad_x = (x1 - x0) * padding_ratio
    pad_y = (y1 - y0) * padding_ratio
    x0 = max(0, int(math.floor(x0 - pad_x)))
    y0 = max(0, int(math.floor(y0 - pad_y)))
    x1 = min(width - 1, int(math.ceil(x1 + pad_x)))
    y1 = min(height - 1, int(math.ceil(y1 + pad_y)))
    return x0, y0, x1, y1


def select_detections(
    boxes: torch.Tensor,
    logits: torch.Tensor,
    phrases: Sequence[str],
    max_detections: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if boxes.numel() == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), []
    order = torch.argsort(logits, descending=True)
    if max_detections > 0:
        order = order[:max_detections]
    indices = order.cpu().tolist()
    return (
        boxes[order].detach().cpu().numpy().astype(np.float32),
        logits[order].detach().cpu().numpy().astype(np.float32),
        [str(phrases[index]) for index in indices],
    )


def boxes_to_mask(
    boxes: np.ndarray,
    image_size: Tuple[int, int],
    padding_ratio: float,
    dilation_kernel_size: int,
    dilation_iterations: int,
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    xyxy_boxes = []
    for box in boxes:
        x0, y0, x1, y1 = normalized_cxcywh_to_padded_xyxy(box, image_size, padding_ratio)
        if x1 < x0 or y1 < y0:
            continue
        mask[y0 : y1 + 1, x0 : x1 + 1] = 1
        xyxy_boxes.append((x0, y0, x1, y1))

    if mask.max() > 0:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("OpenCV is required for mask dilation. Install opencv-python.") from exc
        kernel_size = max(1, int(dilation_kernel_size))
        iterations = max(1, int(dilation_iterations))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=iterations)
    return (mask > 0).astype(np.uint8), xyxy_boxes


def bbox_area_ratio(mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0
    height, width = mask.shape
    area = (int(xs.max()) - int(xs.min()) + 1) * (int(ys.max()) - int(ys.min()) + 1)
    return float(area / (height * width))


def compute_metrics(gt_mask: np.ndarray, object_mask: np.ndarray) -> Dict[str, float]:
    image_pixels = float(gt_mask.size)
    gt_pixels = float((gt_mask > 0).sum())
    mask_pixels = float((object_mask > 0).sum())
    intersection = float(np.logical_and(gt_mask > 0, object_mask > 0).sum())
    return {
        "has_mask": mask_pixels > 0,
        "mask_area_ratio": mask_pixels / image_pixels,
        "bbox_area_ratio": bbox_area_ratio(object_mask),
        "gt_area_ratio": gt_pixels / image_pixels,
        "gt_covered_by_mask": intersection / gt_pixels if gt_pixels > 0 else np.nan,
        "mask_covered_by_gt": intersection / mask_pixels if mask_pixels > 0 else np.nan,
    }


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return np.repeat((mask[:, :, None] * 255).astype(np.uint8), 3, axis=2)


def save_triplet(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    object_mask: np.ndarray,
    save_path: Path,
) -> None:
    triplet = np.concatenate([image_rgb, mask_to_rgb(gt_mask), mask_to_rgb(object_mask)], axis=1)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(triplet).save(save_path)


def save_detection_overlay(
    image_rgb: np.ndarray,
    xyxy_boxes: Sequence[Tuple[int, int, int, int]],
    confidences: np.ndarray,
    phrases: Sequence[str],
    save_path: Path,
) -> None:
    image = Image.fromarray(image_rgb).copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for box, confidence, phrase in zip(xyxy_boxes, confidences, phrases):
        x0, y0, x1, y1 = box
        draw.rectangle(box, outline=(0, 255, 255), width=3)
        label = f"{phrase} {float(confidence):.3f}"
        text_bbox = draw.textbbox((x0, y0), label, font=font)
        text_height = text_bbox[3] - text_bbox[1]
        label_y = max(0, y0 - text_height - 4)
        draw.rectangle((x0, label_y, min(image.width - 1, text_bbox[2] + 4), y0), fill=(0, 0, 0))
        draw.text((x0 + 2, label_y + 1), label, fill=(0, 255, 255), font=font)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)


def save_gt_detection_overlay(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    object_mask: np.ndarray,
    save_path: Path,
) -> None:
    overlay = image_rgb.astype(np.float32).copy()
    gt = gt_mask.astype(bool)
    detected = object_mask.astype(bool)
    both = np.logical_and(gt, detected)
    gt_only = np.logical_and(gt, ~detected)
    detected_only = np.logical_and(detected, ~gt)
    alpha = 0.45
    overlay[gt_only] = (1.0 - alpha) * overlay[gt_only] + alpha * np.array([255, 0, 0])
    overlay[detected_only] = (1.0 - alpha) * overlay[detected_only] + alpha * np.array([0, 255, 255])
    overlay[both] = (1.0 - alpha) * overlay[both] + alpha * np.array([255, 255, 0])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(save_path)


def write_summary_csv(rows: List[Dict[str, object]], save_root: Path) -> Path:
    fieldnames = [
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
        "num_detections",
        "max_confidence",
        "text_prompt",
        "box_threshold",
        "text_threshold",
    ]
    save_root.mkdir(parents=True, exist_ok=True)
    csv_path = save_root / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in rows)
    return csv_path


def summarize_class(rows: List[Dict[str, object]], class_name: str) -> None:
    class_rows = [row for row in rows if row["class"] == class_name]
    if not class_rows:
        return
    detected = sum(bool(row["has_mask"]) for row in class_rows)
    defect_cover = np.asarray(
        [
            float(row["gt_covered_by_mask"])
            for row in class_rows
            if row["defect_type"] != "good" and not np.isnan(float(row["gt_covered_by_mask"]))
        ],
        dtype=np.float32,
    )
    print("")
    print(f"[summary] class={class_name}")
    print(f"  total images: {len(class_rows)}")
    print(f"  mask detected: {detected}")
    print(f"  no mask detected: {len(class_rows) - detected}")
    print(f"  mean mask_area_ratio: {np.mean([float(row['mask_area_ratio']) for row in class_rows]):.6f}")
    print(f"  mean bbox_area_ratio: {np.mean([float(row['bbox_area_ratio']) for row in class_rows]):.6f}")
    if defect_cover.size > 0:
        print(f"  mean gt_covered_by_mask for defect images: {float(np.mean(defect_cover)):.6f}")
        print(f"  median gt_covered_by_mask for defect images: {float(np.median(defect_cover)):.6f}")
    else:
        print("  mean gt_covered_by_mask for defect images: nan")
        print("  median gt_covered_by_mask for defect images: nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvtec_root", type=Path, default=Path("/data/home/ueno/mvtec-data"))
    parser.add_argument("--target_classes", nargs="+", default=["screw", "capsule"])
    parser.add_argument("--save_root", type=Path, default=Path("./mvtec_test_grounding_dino_masks"))
    parser.add_argument(
        "--config_path",
        type=Path,
        default=Path("groundingdino/config/GroundingDINO_SwinT_OGC.py"),
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("weights/groundingdino_swint_ogc.pth"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.20)
    parser.add_argument("--class_prompt", action="append", default=[], metavar="CLASS=PROMPT")
    parser.add_argument("--max_detections", type=int, default=1)
    parser.add_argument("--padding_ratio", type=float, default=0.15)
    parser.add_argument("--dilation_kernel_size", type=int, default=9)
    parser.add_argument("--dilation_iterations", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_prompts = parse_class_prompts(args.class_prompt)
    load_image, load_model, predict = load_grounding_dino_api()
    model = load_model(str(args.config_path), str(args.checkpoint_path), device=args.device)

    rows: List[Dict[str, object]] = []
    image_items = list(iter_mvtec_test_images(args.mvtec_root, args.target_classes))
    if not image_items:
        warnings.warn("No MVTec test images were found.")

    for idx, (class_name, defect_type, image_path) in enumerate(image_items):
        pil_image = load_rgb_image(image_path)
        image_rgb = np.asarray(pil_image)
        gt_mask = load_gt_mask(args.mvtec_root, class_name, defect_type, image_path, pil_image.size)
        prompt = class_prompts.get(class_name, class_name.replace("_", " "))

        _, transformed_image = load_image(str(image_path))
        boxes, logits, phrases = predict(
            model=model,
            image=transformed_image,
            caption=prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=args.device,
        )
        selected_boxes, confidences, selected_phrases = select_detections(
            boxes,
            logits,
            phrases,
            max_detections=args.max_detections,
        )
        object_mask, xyxy_boxes = boxes_to_mask(
            selected_boxes,
            pil_image.size,
            padding_ratio=args.padding_ratio,
            dilation_kernel_size=args.dilation_kernel_size,
            dilation_iterations=args.dilation_iterations,
        )

        save_dir = args.save_root / class_name / defect_type
        triplet_path = save_dir / f"{image_path.stem}_triplet.png"
        detection_overlay_path = save_dir / f"{image_path.stem}_grounding_dino_overlay.png"
        gt_overlay_path = save_dir / f"{image_path.stem}_gt_grounding_dino_overlay.png"
        save_triplet(image_rgb, gt_mask, object_mask, triplet_path)
        save_detection_overlay(image_rgb, xyxy_boxes, confidences, selected_phrases, detection_overlay_path)
        save_gt_detection_overlay(image_rgb, gt_mask, object_mask, gt_overlay_path)

        metrics = compute_metrics(gt_mask, object_mask)
        max_confidence = float(confidences.max()) if confidences.size > 0 else np.nan
        row = {
            "idx": idx,
            "class": class_name,
            "defect_type": defect_type,
            "image_path": str(image_path),
            "save_path": str(triplet_path),
            **metrics,
            "num_detections": len(xyxy_boxes),
            "max_confidence": max_confidence,
            "text_prompt": prompt,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
        }
        rows.append(row)

        mask_label = "mask" if metrics["has_mask"] else "no_mask"
        gt_cover = metrics["gt_covered_by_mask"]
        gt_cover_text = "nan" if np.isnan(gt_cover) else f"{gt_cover:.6f}"
        print(
            f"{class_name} | {idx} | {defect_type} | {mask_label} | "
            f"mask_area={metrics['mask_area_ratio']:.6f} | "
            f"bbox_area={metrics['bbox_area_ratio']:.6f} | gt_cover={gt_cover_text}"
        )

    csv_path = write_summary_csv(rows, args.save_root)
    for class_name in args.target_classes:
        summarize_class(rows, class_name)
    print("")
    print(f"Saved summary CSV: {csv_path}")


if __name__ == "__main__":
    main()

