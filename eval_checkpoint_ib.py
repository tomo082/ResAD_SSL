import argparse
import csv
import glob
import os
import re
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

from classes import (
    MVTEC_TO_BRATS,
    MVTEC_TO_BTAD,
    MVTEC_TO_MPDD,
    MVTEC_TO_MVTEC,
    MVTEC_TO_MVTEC3D,
    MVTEC_TO_MVTECLOCO,
    MVTEC_TO_VISA,
    VISA_TO_MVTEC,
)
from datasets.brats import BRATS
from datasets.btad import BTAD
from datasets.mpdd import MPDD
from datasets.mvtec import MVTEC
from datasets.mvtec_3d import MVTEC3D
from datasets.mvtec_loco import MVTECLOCO
from datasets.visa import VISA
from models.fc_flow import load_flow_model
from models.imagebind import ImageBindModel
from models.modules import MultiScaleOrthogonalProjector
from models.vq import MultiScaleVQ4
from validate import validate


warnings.filterwarnings("ignore")

TOTAL_SHOT = 4
SETTINGS = {
    "visa_to_mvtec": VISA_TO_MVTEC,
    "mvtec_to_visa": MVTEC_TO_VISA,
    "mvtec_to_btad": MVTEC_TO_BTAD,
    "mvtec_to_mvtec3d": MVTEC_TO_MVTEC3D,
    "mvtec_to_mpdd": MVTEC_TO_MPDD,
    "mvtec_to_mvtecloco": MVTEC_TO_MVTECLOCO,
    "mvtec_to_brats": MVTEC_TO_BRATS,
    "mvtec_to_mvtec": MVTEC_TO_MVTEC,
}
SCORE_TYPES = (
    ("scores1", "Logps"),
    ("scores2", "BScores"),
    ("scores", "Merged"),
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


def build_encoder(args):
    if args.backbone != "imagebind":
        raise ValueError("eval_checkpoint_ib.py is for main_ib.py checkpoints; use --backbone imagebind.")
    encoder = ImageBindModel(device=args.device)
    encoder = encoder.to(args.device)
    encoder.eval()
    return encoder, [1280] * args.feature_levels


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

    if "vq_ops" in checkpoint:
        vq_ops.load_state_dict(checkpoint["vq_ops"])
        return vq_ops

    print("[VQOps] checkpoint has no vq_ops state; evaluating without VQOps/EFDM.")
    return None


def load_mc_reference_features(root_dir, class_names, device, num_shot=4, feature_levels=4):
    refs = {}
    for class_name in class_names:
        layers = []
        for level in range(feature_levels):
            path = os.path.join(root_dir, class_name, f"layer{level + 1}.npy")
            layer_refs = torch.from_numpy(np.load(path)).to(device=device, dtype=torch.float32)
            keep = (layer_refs.shape[0] // TOTAL_SHOT) * num_shot
            layers.append(layer_refs[:keep, :])
        refs[class_name] = tuple(layers)
    return refs


def build_test_dataset(args, class_name):
    dataset_kwargs = dict(
        class_name=class_name,
        train=False,
        normalize="imagebind",
        img_size=224,
        crp_size=224,
        msk_size=224,
        msk_crp_size=224,
    )
    if class_name in MVTEC.CLASS_NAMES:
        return MVTEC(args.test_dataset_dir, **dataset_kwargs)
    if class_name in VISA.CLASS_NAMES:
        return VISA(args.test_dataset_dir, **dataset_kwargs)
    if class_name in BTAD.CLASS_NAMES:
        return BTAD(args.test_dataset_dir, **dataset_kwargs)
    if class_name in MVTEC3D.CLASS_NAMES:
        return MVTEC3D(args.test_dataset_dir, **dataset_kwargs)
    if class_name in MPDD.CLASS_NAMES:
        return MPDD(args.test_dataset_dir, **dataset_kwargs)
    if class_name in MVTECLOCO.CLASS_NAMES:
        return MVTECLOCO(args.test_dataset_dir, **dataset_kwargs)
    if class_name in BRATS.CLASS_NAMES:
        return BRATS(args.test_dataset_dir, **dataset_kwargs)
    raise ValueError(f"Unrecognized class name: {class_name}")


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


def main(args):
    if args.setting not in SETTINGS:
        raise ValueError(f"Dataset setting must be in {SETTINGS.keys()}, but got {args.setting}.")

    classes = SETTINGS[args.setting]
    checkpoint_file = resolve_checkpoint_file(args)
    eval_epoch = resolve_eval_epoch(args, checkpoint_file)

    print("[Eval-IB] checkpoint_file:", checkpoint_file)
    print("[Eval-IB] eval_epoch:", eval_epoch)
    print("[Eval-IB] setting:", args.setting)
    print("[Eval-IB] classes:", classes["unseen"])
    print("[Eval-IB] backbone:", args.backbone)
    print("[Eval-IB] feature_levels:", args.feature_levels)
    print("[Eval-IB] num_ref_shot:", args.num_ref_shot)
    print("[Eval-IB] test_ref_feature_dir:", args.test_ref_feature_dir)
    print("[VQOps] enabled: True")

    encoder, feat_dims = build_encoder(args)
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
        classes["unseen"],
        args.device,
        args.num_ref_shot,
        feature_levels=args.feature_levels,
    )

    results_by_type = {label: [] for _, label in SCORE_TYPES}
    csv_rows = []
    for class_name in classes["unseen"]:
        test_dataset = build_test_dataset(args, class_name)
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
    parser.add_argument("--test_dataset_dir", type=str, required=True)
    parser.add_argument("--test_ref_feature_dir", type=str, required=True)
    parser.add_argument("--checkpoint_file", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--backbone", type=str, default="imagebind")
    parser.add_argument("--num_ref_shot", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_csv", type=str, default="")
    parser.add_argument("--eval_epoch", type=int, default=None)

    parser.add_argument("--flow_arch", type=str, default="flow_model")
    parser.add_argument("--feature_levels", default=4, type=int)
    parser.add_argument("--coupling_layers", type=int, default=4)
    parser.add_argument("--clamp_alpha", type=float, default=1.9)
    parser.add_argument("--pos_embed_dim", type=int, default=256)
    parser.add_argument("--fdm_alpha", type=float, default=0.4)
    parser.add_argument("--num_embeddings", type=int, default=1536)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
