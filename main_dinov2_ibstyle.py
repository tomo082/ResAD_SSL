import argparse
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

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
from datasets.mvtec import MVTEC, MVTECANO
from datasets.mvtec_3d import MVTEC3D
from datasets.mvtec_loco import MVTECLOCO
from datasets.visa import VISA, VISAANO
from losses.loss import calculate_log_barrier_bi_occ_loss, calculate_orthogonal_regularizer
from models.dinov2_encoder import DINOv2IBStyleEncoder, default_dinov2_out_indices, print_dinov2_ibstyle_config
from models.fc_flow import load_flow_model
from models.modules import MultiScaleOrthogonalProjector
from models.vq import MultiScaleVQ4
from train import train2
from utils import BoundaryAverager, get_mc_matched_ref_features, get_random_normal_images, get_residual_features, init_seeds
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


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def unpack_train_batch(batch):
    images = batch[0]
    masks = batch[2]
    class_names = batch[3]
    return images, masks, class_names


def dinov2_tokens_to_maps(features):
    maps = []
    for feature in features:
        b, tokens, channels = feature.shape
        side = int(tokens ** 0.5)
        if side * side != tokens:
            raise ValueError(f"Expected square DINOv2 token grid, got {tokens} tokens.")
        maps.append(feature.permute(0, 2, 1).reshape(b, channels, side, side).contiguous())
    return maps


def print_feature_shapes_once(features, prefix, printed):
    if printed:
        return True
    for level, feature in enumerate(features):
        print(f"[{prefix}] level {level}: {tuple(feature.shape)}")
    return True


def build_train_loaders(args, classes):
    common = dict(train=True, normalize="w50", img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
    if classes["seen"][0] in MVTEC.CLASS_NAMES:
        train_dataset1 = MVTEC(args.train_dataset_dir, class_name=classes["seen"], **common)
        train_dataset2 = MVTECANO(args.train_dataset_dir, class_name=classes["seen"], **common)
    else:
        train_dataset1 = VISA(args.train_dataset_dir, class_name=classes["seen"], **common)
        train_dataset2 = VISAANO(args.train_dataset_dir, class_name=classes["seen"], **common)
    train_loader1 = DataLoader(train_dataset1, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    train_loader2 = DataLoader(train_dataset2, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    return train_loader1, train_loader2


def build_test_dataset(args, class_name):
    common = dict(train=False, normalize="w50", img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
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


def load_and_transform_vision_data(image_paths, device):
    transform = T.Compose([
        T.Resize(224, T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    images = []
    for image_path in image_paths:
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")
        images.append(transform(image).to(device))
    return torch.stack(images, dim=0)


def get_mc_reference_features_dinov2(encoder, root, class_names, device, num_shot=4):
    reference_features = {}
    for class_name in np.unique(class_names):
        normal_paths = get_random_normal_images(root, class_name, num_shot)
        images = load_and_transform_vision_data(normal_paths, device)
        with torch.no_grad():
            features = encoder.encode_image_from_tensors(images)
            reference_features[class_name] = [feature.reshape(-1, feature.shape[-1]).contiguous() for feature in features]
    return reference_features


def load_mc_reference_features(root_dir, class_names, device, num_shot=4, feature_levels=4):
    refs = {}
    for class_name in class_names:
        layers = []
        for level in range(feature_levels):
            layer_refs = np.load(os.path.join(root_dir, class_name, f"layer{level + 1}.npy"))
            layer_refs = torch.from_numpy(layer_refs).to(device)
            keep = (layer_refs.shape[0] // TOTAL_SHOT) * num_shot
            layers.append(layer_refs[:keep, :])
        refs[class_name] = tuple(layers)
    return refs


def build_encoder(args):
    out_indices = default_dinov2_out_indices(args.backbone) if args.dinov2_out_indices is None else args.dinov2_out_indices
    if len(out_indices) != args.feature_levels:
        raise ValueError(f"DINOv2 IB-style expects {args.feature_levels} out_indices, got {len(out_indices)}.")
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


def run_shape_test(args, encoder):
    dummy = torch.zeros(2, 3, 224, 224, device=args.device)
    tokens = encoder.encode_image_from_tensors(dummy)
    maps = dinov2_tokens_to_maps(tokens)
    expected_channels = encoder.embed_dim
    for level, feature in enumerate(tokens):
        assert feature.shape == (2, 256, expected_channels), tuple(feature.shape)
        assert maps[level].shape == (2, expected_channels, 16, 16), tuple(maps[level].shape)
    print("[DINOv2-IBStyle] shape test tokens:", [tuple(feature.shape) for feature in tokens])
    print("[DINOv2-IBStyle] shape test maps:", [tuple(feature.shape) for feature in maps])


def main(args):
    first_stage_epoch = args.first_epoch
    if args.setting not in SETTINGS:
        raise ValueError(f"Dataset setting must be in {SETTINGS.keys()}, but got {args.setting}.")
    classes = SETTINGS[args.setting]

    encoder = build_encoder(args)
    if args.dino_shape_test:
        run_shape_test(args, encoder)
        return

    feat_dims = [encoder.embed_dim] * args.feature_levels
    train_loader1, train_loader2 = build_train_loaders(args, classes)

    boundary_ops = BoundaryAverager(num_levels=args.feature_levels)
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
    printed_train_shapes = False

    for epoch in range(args.epochs):
        vq_ops.train()
        constraintor.train()
        for estimator in estimators:
            estimator.train()

        train_loader = train_loader1 if epoch < FIRST_STAGE_EPOCH else train_loader2
        loss_totals = {name: [0, 0] for name in ["vq", "occ", "occn", "occa", "ort", "flow"]}
        progress_bar = tqdm(total=len(train_loader))
        progress_bar.set_description(f"Epoch[{epoch}/{args.epochs}]")
        for batch in train_loader:
            progress_bar.update(1)
            images, masks, class_names = unpack_train_batch(batch)
            images = images.to(args.device)
            masks = masks.to(args.device)

            with torch.no_grad():
                features = dinov2_tokens_to_maps(encoder.encode_image_from_tensors(images))
            printed_train_shapes = print_feature_shapes_once(features, "DINOv2-IBStyle train features", printed_train_shapes)

            ref_features = get_mc_reference_features_dinov2(encoder, args.train_dataset_dir, class_names, images.device, args.train_ref_shot)
            mfeatures = get_mc_matched_ref_features(features, class_names, ref_features)
            rfeatures = get_residual_features(features, mfeatures)
            if not args.residual:
                rfeatures = features

            lvl_masks = []
            for level in range(args.feature_levels):
                _, _, h, w = rfeatures[level].size()
                mask = F.interpolate(masks, size=(h, w), mode="bilinear").squeeze(1)
                lvl_masks.append((mask > 0.3).to(torch.float32))
            rfeatures_t = [rfeature.detach().clone() for rfeature in rfeatures]

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
                loss += loss_occ + loss_ort

                loss_totals["occ"][0] += loss_occ.item()
                loss_totals["occ"][1] += 1
                loss_totals["occn"][0] += loss_occn
                loss_totals["occn"][1] += 1
                loss_totals["occa"][0] += loss_occa
                loss_totals["occa"][1] += 1
                loss_totals["ort"][0] += loss_ort.item()
                loss_totals["ort"][1] += 1

            optimizer0.zero_grad()
            loss.backward()
            optimizer0.step()

            rfeatures = [rfeature.detach().clone() for rfeature in rfeatures]
            flow_loss, flow_num = train2(args, rfeatures, estimators, optimizer1, lvl_masks, boundary_ops, epoch, N_batch=n_batch, FIRST_STAGE_EPOCH=FIRST_STAGE_EPOCH)
            loss_totals["flow"][0] += flow_loss
            loss_totals["flow"][1] += flow_num

        scheduler_vq.step()
        scheduler0.step()
        scheduler1.step()
        progress_bar.close()

        def avg(name):
            total, count = loss_totals[name]
            return total / count if count > 0 else 0

        print(f"Epoch[{epoch}/{args.epochs}]: VQ loss: {avg('vq')}, OCC loss: {avg('occ')} (n: {avg('occn')}, a: {avg('occa')}), Ort loss: {avg('ort')}, Flow loss: {avg('flow')}")

        if (epoch + 1) % args.eval_freq != 0:
            continue

        s1_res, s2_res, s_res = [], [], []
        test_proto_features = load_mc_reference_features(args.test_ref_feature_dir, classes["unseen"], args.device, args.num_ref_shot, feature_levels=args.feature_levels)
        for class_name_eval in classes["unseen"]:
            test_dataset = build_test_dataset(args, class_name_eval)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)
            metrics = validate(args, encoder, vq_ops, constraintor, estimators, test_loader, test_proto_features[class_name_eval], args.device, class_name_eval)
            img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = metrics["scores"]
            print("Epoch: {}, Class Name: {}, Image AUC | AP | F1_Score: {} | {} | {}, Pixel AUC | AP | F1_Score | AUPRO: {} | {} | {} | {}".format(epoch, class_name_eval, img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro))
            s1_res.append(metrics["scores1"])
            s2_res.append(metrics["scores2"])
            s_res.append(metrics["scores"])

        s1_res = np.array(s1_res)
        s2_res = np.array(s2_res)
        s_res = np.array(s_res)
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
                "vq_ops": vq_ops.state_dict(),
                "constraintor": constraintor.state_dict(),
                "estimators": [estimator.state_dict() for estimator in estimators],
            }
            torch.save(state_dict, os.path.join(args.checkpoint_path, f"{args.setting}_epoch_{epoch}_checkpoints.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", type=str, default="visa")
    parser.add_argument("--setting", type=str, default="visa_to_mvtec")
    parser.add_argument("--train_dataset_dir", type=str, default="")
    parser.add_argument("--test_dataset_dir", type=str, default="")
    parser.add_argument("--test_ref_feature_dir", type=str, default="/data/home/ueno/ref_features/dinov2/mvtec_tf_8shot_select")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint_path", type=str, default="/data/home/ueno/checkpoint/visa_to_mvtec_dinov2_ibstyle")
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--backbone", type=str, default="dinov2_vits14")
    parser.add_argument("--dinov2_out_indices", type=int, nargs="+", default=None)
    parser.add_argument("--dinov2_hub_repo", type=str, default="facebookresearch/dinov2")
    parser.add_argument("--dinov2_hub_source", type=str, default="github", choices=["github", "local"])
    parser.add_argument("--dino_shape_test", action="store_true")
    parser.add_argument("--residual", type=str2bool, nargs="?", const=True, default=True)
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
    parser.add_argument("--train_ref_shot", type=int, default=4)
    parser.add_argument("--num_ref_shot", type=int, default=4)
    parser.add_argument("--first_epoch", type=int, default=1)
    args = parser.parse_args()
    init_seeds(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main(args)
