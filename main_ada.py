import argparse
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader
from tqdm import tqdm

from classes import (
    CAPSULES_TO_CAPSULES,
    MVTEC_TO_BRATS,
    MVTEC_TO_BTAD,
    MVTEC_TO_MPDD,
    MVTEC_TO_MVTEC,
    MVTEC_TO_MVTEC3D,
    MVTEC_TO_MVTECLOCO,
    MVTEC_TO_VISA,
    VISA_TO_MVTEC,
    VISA_TO_VISA,
)
from datasets.brats import BRATS
from datasets.btad import BTAD
from datasets.capsules import CAPSULES, CAPSULESANO
from datasets.mpdd import MPDD
from datasets.mvtec import MVTEC, MVTECANO
from datasets.mvtec_3d import MVTEC3D
from datasets.mvtec_loco import MVTECLOCO
from datasets.visa import VISA, VISAANO
from losses.loss import calculate_log_barrier_bi_occ_loss, calculate_orthogonal_regularizer
from models.adaclip_feature_extractor import AdaCLIPPromptedFeatureExtractor
from models.clip_feature_extractor import CLIPRawFeatureExtractor
from models.fc_flow import load_flow_model
from models.modules import MultiScaleOrthogonalProjector
from models.vq import MultiScaleVQ4
from residual_norm import (
    RESIDUAL_NORM_MODES,
    apply_residual_norm_from_args,
    create_residual_norm_accumulator,
    finalize_residual_norm_stats,
    pack_residual_norm_state,
    print_residual_norm_stats,
    residual_norm_enabled,
    update_residual_norm_accumulator,
    validate_residual_norm_args,
)
from train import train2
from utils import BoundaryAverager, get_mc_matched_ref_features, get_mc_reference_features, get_residual_features, init_seeds
from validate_ada import validate


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
    "visa_to_visa": VISA_TO_VISA,
    "capsules_to_capsules": CAPSULES_TO_CAPSULES,
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1"):
        return True
    if value in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def to_float(value):
    if torch.is_tensor(value):
        return value.detach().item()
    return float(value)


def get_feature_image_size(args):
    if args.feature_backbone in ("clip_raw", "adaclip_prompted"):
        return args.clip_image_size
    return 224


def _should_l2_normalize_adaclip(args):
    return args.feature_backbone == "adaclip_prompted" and getattr(args, "adaclip_feature_l2norm", False)


def normalize_adaclip_feature_maps_if_enabled(args, features):
    if not _should_l2_normalize_adaclip(args):
        return features
    return [F.normalize(feature.float(), p=2, dim=1) for feature in features]


def normalize_adaclip_reference_features_if_enabled(args, ref_features):
    if not _should_l2_normalize_adaclip(args):
        return ref_features
    normalized = {}
    for class_name, features in ref_features.items():
        normalized[class_name] = tuple(F.normalize(feature.float(), p=2, dim=1) for feature in features)
    return normalized


def build_feature_encoder(args):
    if args.feature_backbone in ("clip_raw", "adaclip_prompted") and len(args.clip_layers) != args.feature_levels:
        raise ValueError(
            f"{args.feature_backbone} expects len(--clip_layers) == --feature_levels. "
            f"Got {len(args.clip_layers)} layers and feature_levels={args.feature_levels}."
        )

    if args.feature_backbone == "clip_raw":
        encoder = CLIPRawFeatureExtractor(
            model_name=args.clip_model,
            pretrained=args.clip_pretrained,
            layers=args.clip_layers,
            image_size=args.clip_image_size,
            freeze=True,
            weight_source=args.clip_weight_source,
            checkpoint=args.clip_checkpoint,
        ).to(args.device)
        encoder.eval()
        return encoder, encoder.feature_info.channels()

    if args.feature_backbone == "adaclip_prompted":
        encoder = AdaCLIPPromptedFeatureExtractor(
            adaclip_repo_url=args.adaclip_repo_url,
            adaclip_repo_path=args.adaclip_repo_path,
            checkpoint=args.adaclip_checkpoint,
            checkpoint_url=args.adaclip_checkpoint_url,
            cache_dir=args.adaclip_cache_dir,
            model_name=args.adaclip_model,
            layers=args.clip_layers,
            image_size=args.clip_image_size,
            return_projected=args.adaclip_return_projected,
            freeze=True,
            device=args.device,
        ).to(args.device)
        encoder.eval()
        return encoder, encoder.feature_info.channels()

    if args.feature_backbone != "original":
        raise ValueError(f"Unsupported feature_backbone: {args.feature_backbone}")
    if args.backbone == "wide_resnet50_2":
        encoder = timm.create_model("wide_resnet50_2", features_only=True, out_indices=(1, 2, 3), pretrained=True).eval()
        return encoder.to(args.device), encoder.feature_info.channels()
    if args.backbone == "tf_efficientnet_b6":
        encoder = timm.create_model("tf_efficientnet_b6", features_only=True, out_indices=(1, 2, 3), pretrained=True).eval()
        return encoder.to(args.device), encoder.feature_info.channels()
    raise ValueError(f"Unsupported backbone: {args.backbone}")


def build_train_loaders(args, classes, image_size):
    common = dict(train=True, normalize="w50", img_size=image_size, crp_size=image_size, msk_size=image_size, msk_crp_size=image_size)
    if args.classes == "capsules":
        train_dataset1 = CAPSULES(args.train_dataset_dir, class_name=classes["seen"], **common)
        train_dataset2 = CAPSULESANO(args.train_dataset_dir, class_name=classes["seen"], **common)
    elif classes["seen"][0] in MVTEC.CLASS_NAMES:
        train_dataset1 = MVTEC(args.train_dataset_dir, class_name=classes["seen"], **common)
        train_dataset2 = MVTECANO(args.train_dataset_dir, class_name=classes["seen"], **common)
    else:
        train_dataset1 = VISA(args.train_dataset_dir, class_name=classes["seen"], **common)
        train_dataset2 = VISAANO(args.train_dataset_dir, class_name=classes["seen"], **common)

    train_loader1 = DataLoader(train_dataset1, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    train_loader2 = DataLoader(train_dataset2, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    return train_loader1, train_loader2


def build_test_dataset(args, class_name, image_size):
    common = dict(train=False, normalize="w50", img_size=image_size, crp_size=image_size, msk_size=image_size, msk_crp_size=image_size)
    if args.classes == "capsules":
        return CAPSULES(args.test_dataset_dir, class_name=class_name, **common)
    if class_name in MVTEC.CLASS_NAMES:
        return MVTEC(args.test_dataset_dir, class_name=class_name, **common)
    if class_name in VISA.CLASS_NAMES:
        return VISA(args.test_dataset_dir, class_name=class_name, **common)
    if class_name in BTAD.CLASS_NAMES:
        return BTAD(args.test_dataset_dir, class_name=class_name, **common)
    if class_name in MVTEC3D.CLASS_NAMES:
        return MVTEC3D(args.test_dataset_dir, class_name=class_name, **common)
    if class_name in MPDD.CLASS_NAMES:
        return MPDD(args.test_dataset_dir, class_name=class_name, **common)
    if class_name in MVTECLOCO.CLASS_NAMES:
        return MVTECLOCO(args.test_dataset_dir, class_name=class_name, **common)
    if class_name in BRATS.CLASS_NAMES:
        return BRATS(args.test_dataset_dir, class_name=class_name, **common)
    raise ValueError(f"Unrecognized class name: {class_name}")


def load_mc_reference_features(root_dir, class_names, device, num_shot=4, feature_levels=4):
    refs = {}
    for class_name in class_names:
        layers = []
        for level in range(feature_levels):
            layer_refs = np.load(os.path.join(root_dir, class_name, f"layer{level + 1}.npy"))
            layer_refs = torch.from_numpy(layer_refs).to(device=device, dtype=torch.float32)
            keep = (layer_refs.shape[0] // TOTAL_SHOT) * num_shot
            layers.append(layer_refs[:keep, :])
        refs[class_name] = tuple(layers)
    return refs


def compute_residual_norm_stats(args, encoder, train_loader, image_size):
    if not residual_norm_enabled(args):
        args.residual_norm_stats = None
        return None

    print(
        f"[ResidualNorm] collecting stats from up to {args.residual_stats_batches} "
        "training normal batch(es)."
    )
    accumulator = create_residual_norm_accumulator(args.residual_norm)
    encoder.eval()
    collected_batches = 0
    with torch.no_grad():
        for batch in train_loader:
            if collected_batches >= args.residual_stats_batches:
                break
            images, _, _, class_names = batch
            images = images.to(args.device)
            features = encoder(images)
            features = normalize_adaclip_feature_maps_if_enabled(args, features)
            ref_features = get_mc_reference_features(
                encoder,
                args.train_dataset_dir,
                class_names,
                images.device,
                args.train_ref_shot,
                img_size=image_size,
            )
            ref_features = normalize_adaclip_reference_features_if_enabled(args, ref_features)
            mfeatures = get_mc_matched_ref_features(features, class_names, ref_features)
            rfeatures = get_residual_features(features, mfeatures, pos_flag=True)
            update_residual_norm_accumulator(accumulator, rfeatures)
            collected_batches += 1

    stats = finalize_residual_norm_stats(accumulator)
    args.residual_norm_stats = stats
    print(f"[ResidualNorm] collected batches: {collected_batches}")
    print_residual_norm_stats(stats)
    return stats


def main(args):
    first_stage_epoch = args.first_epoch
    validate_residual_norm_args(args)
    if args.setting in SETTINGS:
        classes = SETTINGS[args.setting]
    else:
        raise ValueError(f"Dataset setting must be in {SETTINGS.keys()}, but got {args.setting}.")
    if args.feature_levels != 4:
        raise ValueError("main_ada.py now follows main_ib.py and expects --feature_levels 4.")

    image_size = get_feature_image_size(args)
    train_loader1, train_loader2 = build_train_loaders(args, classes, image_size)
    encoder, feat_dims = build_feature_encoder(args)
    if len(feat_dims) != args.feature_levels:
        raise ValueError(f"feature_levels={args.feature_levels} does not match encoder outputs {len(feat_dims)}.")

    print("[Ada-IBStyle] feature_backbone:", args.feature_backbone)
    print("[Ada-IBStyle] clip_layers:", args.clip_layers)
    print("[Ada-IBStyle] feature_levels:", args.feature_levels)
    print("[Ada-IBStyle] feat_dims:", feat_dims)
    print("[Ada-IBStyle] first_epoch:", first_stage_epoch)
    print("[Ada-IBStyle] adaclip_feature_l2norm:", _should_l2_normalize_adaclip(args))
    print("[ResidualNorm] residual_norm:", args.residual_norm)
    print("[ResidualNorm] residual_stats_batches:", args.residual_stats_batches)
    print("[ResidualNorm] residual_norm_eps:", args.residual_norm_eps)
    print("[ResidualNorm] residual_norm_clip:", args.residual_norm_clip)
    compute_residual_norm_stats(args, encoder, train_loader1, image_size)

    boundary_ops = BoundaryAverager(num_levels=args.feature_levels)
    use_vqops = not args.disable_vqops
    print("[VQOps] use_vqops:", use_vqops)
    vq_ops = None
    optimizer_vq = None
    scheduler_vq = None
    if use_vqops:
        vq_ops = MultiScaleVQ4(num_embeddings=args.num_embeddings, channels=feat_dims).to(args.device)
        optimizer_vq = torch.optim.Adam(vq_ops.parameters(), lr=args.lr, weight_decay=0.0005)
        scheduler_vq = torch.optim.lr_scheduler.MultiStepLR(optimizer_vq, milestones=[70, 90], gamma=0.1)

    constraintor = MultiScaleOrthogonalProjector(feat_dims).to(args.device)
    optimizer0 = torch.optim.Adam(constraintor.parameters(), lr=args.lr, weight_decay=0.0005)
    scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer0, milestones=[70, 90], gamma=0.1)

    estimators = [load_flow_model(args, feat_dim).to(args.device) for feat_dim in feat_dims]
    params = []
    for estimator in estimators:
        params += list(estimator.parameters())
    optimizer1 = torch.optim.Adam(params, lr=args.lr, weight_decay=0.0005)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[70, 90], gamma=0.1)

    best_img_auc = 0
    n_batch = 16 * 16 * 32
    for epoch in range(args.epochs):
        if vq_ops is not None:
            vq_ops.train()
        constraintor.train()
        for estimator in estimators:
            estimator.train()

        train_loader = train_loader1 if epoch < first_stage_epoch else train_loader2
        loss_totals = {name: [0.0, 0] for name in ["vq", "occ", "occn", "occa", "ort", "flow"]}
        progress_bar = tqdm(total=len(train_loader))
        progress_bar.set_description(f"Epoch[{epoch}/{args.epochs}]")
        for batch in train_loader:
            progress_bar.update(1)
            images, _, masks, class_names = batch
            images = images.to(args.device)
            masks = masks.to(args.device)

            with torch.no_grad():
                features = encoder(images)
                features = normalize_adaclip_feature_maps_if_enabled(args, features)

            ref_features = get_mc_reference_features(
                encoder,
                args.train_dataset_dir,
                class_names,
                images.device,
                args.train_ref_shot,
                img_size=image_size,
            )
            ref_features = normalize_adaclip_reference_features_if_enabled(args, ref_features)
            mfeatures = get_mc_matched_ref_features(features, class_names, ref_features)
            rfeatures = get_residual_features(features, mfeatures, pos_flag=True)
            rfeatures = apply_residual_norm_from_args(args, rfeatures)

            lvl_masks = []
            for level in range(args.feature_levels):
                _, _, h, w = rfeatures[level].size()
                mask = F.interpolate(masks, size=(h, w), mode="bilinear").squeeze(1)
                lvl_masks.append((mask > 0.3).to(torch.float32))
            rfeatures_t = [rfeature.detach().clone() for rfeature in rfeatures]

            if vq_ops is not None:
                loss_vq = vq_ops(rfeatures, lvl_masks, train=True)
                loss_totals["vq"][0] += loss_vq.item()
                loss_totals["vq"][1] += 1
                optimizer_vq.zero_grad()
                loss_vq.backward()
                optimizer_vq.step()

            rfeatures = constraintor(*rfeatures)
            loss = 0
            for level in range(args.feature_levels):
                e = rfeatures[level]
                t = rfeatures_t[level]
                _, dim, _, _ = e.size()
                e = e.permute(0, 2, 3, 1).reshape(-1, dim)
                t = t.permute(0, 2, 3, 1).reshape(-1, dim)
                mask = lvl_masks[level].reshape(-1)

                loss_occ, loss_occn, loss_occa = calculate_log_barrier_bi_occ_loss(e, mask, t)
                loss_ort = calculate_orthogonal_regularizer(e, mask)
                loss = loss + loss_occ + loss_ort

                loss_totals["occ"][0] += to_float(loss_occ)
                loss_totals["occ"][1] += 1
                loss_totals["occn"][0] += to_float(loss_occn)
                loss_totals["occn"][1] += 1
                loss_totals["occa"][0] += to_float(loss_occa)
                loss_totals["occa"][1] += 1
                loss_totals["ort"][0] += to_float(loss_ort)
                loss_totals["ort"][1] += 1

            optimizer0.zero_grad()
            loss.backward()
            optimizer0.step()

            rfeatures = [rfeature.detach().clone() for rfeature in rfeatures]
            flow_loss, flow_num = train2(
                args,
                rfeatures,
                estimators,
                optimizer1,
                lvl_masks,
                boundary_ops,
                epoch,
                N_batch=n_batch,
                FIRST_STAGE_EPOCH=first_stage_epoch,
            )
            loss_totals["flow"][0] += flow_loss
            loss_totals["flow"][1] += flow_num

        if scheduler_vq is not None:
            scheduler_vq.step()
        scheduler0.step()
        scheduler1.step()
        progress_bar.close()

        def avg(name):
            total, count = loss_totals[name]
            return total / count if count > 0 else 0

        print(
            f"Epoch[{epoch}/{args.epochs}]: VQ loss: {avg('vq')}, "
            f"OCC loss: {avg('occ')} (n: {avg('occn')}, a: {avg('occa')}), "
            f"Ort loss: {avg('ort')}, Flow loss: {avg('flow')}"
        )

        if (epoch + 1) % args.eval_freq != 0:
            continue

        s1_res, s2_res, s_res = [], [], []
        test_ref_features = load_mc_reference_features(
            args.test_ref_feature_dir,
            classes["unseen"],
            args.device,
            args.num_ref_shot,
            feature_levels=args.feature_levels,
        )
        for class_name in classes["unseen"]:
            test_dataset = build_test_dataset(args, class_name, image_size)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)
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
            img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = metrics["scores"]
            print(
                "Epoch: {}, Class Name: {}, Image AUC | AP | F1_Score: {} | {} | {}, "
                "Pixel AUC | AP | F1_Score | AUPRO: {} | {} | {} | {}".format(
                    epoch,
                    class_name,
                    img_auc,
                    img_ap,
                    img_f1_score,
                    pix_auc,
                    pix_ap,
                    pix_f1_score,
                    pix_aupro,
                )
            )
            s1_res.append(metrics["scores1"])
            s2_res.append(metrics["scores2"])
            s_res.append(metrics["scores"])

        s1_res = np.asarray(s1_res)
        s2_res = np.asarray(s2_res)
        s_res = np.asarray(s_res)
        img_auc1, img_ap1, img_f1_score1, pix_auc1, pix_ap1, pix_f1_score1, pix_aupro1 = np.mean(s1_res, axis=0)
        img_auc2, img_ap2, img_f1_score2, pix_auc2, pix_ap2, pix_f1_score2, pix_aupro2 = np.mean(s2_res, axis=0)
        img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = np.mean(s_res, axis=0)
        print("(Logps) Average Image AUC | AP | F1_Score: {:.3f} | {:.3f} | {:.3f}, Average Pixel AUC | AP | F1_Score | AUPRO: {:.3f} | {:.3f} | {:.3f} | {:.3f}".format(img_auc1, img_ap1, img_f1_score1, pix_auc1, pix_ap1, pix_f1_score1, pix_aupro1))
        print("(BScores) Average Image AUC | AP | F1_Score: {:.3f} | {:.3f} | {:.3f}, Average Pixel AUC | AP | F1_Score | AUPRO: {:.3f} | {:.3f} | {:.3f} | {:.3f}".format(img_auc2, img_ap2, img_f1_score2, pix_auc2, pix_ap2, pix_f1_score2, pix_aupro2))
        print("(Merged) Average Image AUC | AP | F1_Score: {:.3f} | {:.3f} | {:.3f}, Average Pixel AUC | AP | F1_Score | AUPRO: {:.3f} | {:.3f} | {:.3f} | {:.3f}".format(img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro))

        if img_auc > best_img_auc:
            os.makedirs(args.checkpoint_path, exist_ok=True)
            best_img_auc = img_auc
            state_dict = {
                "constraintor": constraintor.state_dict(),
                "estimators": [estimator.state_dict() for estimator in estimators],
            }
            if vq_ops is not None:
                state_dict["vq_ops"] = vq_ops.state_dict()
            residual_norm_state = pack_residual_norm_state(args)
            if residual_norm_state is not None:
                state_dict["residual_norm"] = residual_norm_state
            torch.save(state_dict, os.path.join(args.checkpoint_path, f"{args.setting}_epoch_{epoch}_checkpoints.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=str, default="visa_to_mvtec")
    parser.add_argument("--classes", type=str, default="none")
    parser.add_argument("--train_dataset_dir", type=str, default="")
    parser.add_argument("--test_dataset_dir", type=str, default="")
    parser.add_argument("--test_ref_feature_dir", type=str, default="./ref_features/ada_ibstyle/mvtec_4shot")
    parser.add_argument("--bgadweight_dir", type=str, default="none")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/")
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2")
    parser.add_argument("--feature_backbone", type=str, default="adaclip_prompted", choices=["adaclip_prompted", "clip_raw", "original"])
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
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--first_epoch", type=int, default=1)

    parser.add_argument("--flow_arch", type=str, default="flow_model")
    parser.add_argument("--feature_levels", default=4, type=int)
    parser.add_argument("--coupling_layers", type=int, default=4)
    parser.add_argument("--clamp_alpha", type=float, default=1.9)
    parser.add_argument("--pos_embed_dim", type=int, default=256)
    parser.add_argument("--pos_beta", type=float, default=0.05)
    parser.add_argument("--margin_tau", type=float, default=0.1)
    parser.add_argument("--bgspp_lambda", type=float, default=1)

    parser.add_argument("--fdm_alpha", type=float, default=0.4)
    parser.add_argument("--num_embeddings", type=int, default=1536)
    parser.add_argument("--disable_vqops", action="store_true")
    parser.add_argument("--train_ref_shot", type=int, default=4)
    parser.add_argument("--num_ref_shot", type=int, default=4)
    parser.add_argument("--residual_norm", type=str, default="none", choices=RESIDUAL_NORM_MODES)
    parser.add_argument("--residual_stats_batches", type=int, default=50)
    parser.add_argument("--residual_norm_eps", type=float, default=1e-6)
    parser.add_argument("--residual_norm_clip", type=float, default=0.0)

    args = parser.parse_args()
    init_seeds(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main(args)
