import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import timm
import torchvision.transforms as T
import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from datasets.brats import BRATS
from datasets.btad import BTAD
from datasets.mpdd import MPDD
from datasets.mvtec import MVTEC
from datasets.mvtec_3d import MVTEC3D
from datasets.mvtec_loco import MVTECLOCO
from datasets.visa import VISA
from models.adaclip_feature_extractor import AdaCLIPPromptedFeatureExtractor
from models.clip_feature_extractor import CLIPRawFeatureExtractor
from models.dinov2_backbone import DINOv2BackboneWrapper, DINOV2_BACKBONES, DINOV2_FEATURE_MODES, print_dinov2_config
from models.fc_flow import load_flow_model
from residual_wavelet import apply_feature_wavelet_filter
from utils import load_weights


SETTINGS = {
    "mvtec": MVTEC.CLASS_NAMES,
    "visa": VISA.CLASS_NAMES,
    "btad": BTAD.CLASS_NAMES,
    "mvtec3d": MVTEC3D.CLASS_NAMES,
    "mpdd": MPDD.CLASS_NAMES,
    "mvtecloco": MVTECLOCO.CLASS_NAMES,
    "brats": BRATS.CLASS_NAMES,
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


def rotate_rgb_image(img, angle, fill_mode="reflect"):
    if angle % 360 == 0:
        return img
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("cv2 is required for --ref_aug rotate.") from exc

    img_np = np.array(img.convert("RGB"))
    h, w = img_np.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    if fill_mode == "reflect":
        border_mode = cv2.BORDER_REFLECT_101
        border_value = 0
    elif fill_mode == "constant":
        border_mode = cv2.BORDER_CONSTANT
        border_value = (0, 0, 0)
    else:
        raise ValueError(f"Unsupported ref_aug_fill: {fill_mode}")
    rotated = cv2.warpAffine(
        img_np,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode,
        borderValue=border_value,
    )
    return Image.fromarray(rotated)


class FEWSHOTDATA(Dataset):
    def __init__(self, root, class_name="bottle", train=True, **kwargs):
        self.root = root
        self.class_name = class_name
        self.train = train
        self.mask_size = [kwargs.get("msk_crp_size"), kwargs.get("msk_crp_size")]
        self.ref_aug = kwargs.get("ref_aug", "none")
        self.ref_aug_angles = list(kwargs.get("ref_aug_angles", [0]))
        self.ref_aug_fill = kwargs.get("ref_aug_fill", "reflect")
        if self.ref_aug not in ("none", "rotate"):
            raise ValueError(f"Unsupported ref_aug: {self.ref_aug}")
        if not self.ref_aug_angles:
            raise ValueError("ref_aug_angles must contain at least one angle.")

        self.image_paths, self.labels, self.mask_paths, self.class_names = self._load_data(self.class_name)
        self.transform = T.Compose([
            T.Resize(kwargs.get("img_size", 224), T.InterpolationMode.BICUBIC),
            T.CenterCrop(kwargs.get("crp_size", 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.target_transform = T.Compose([
            T.Resize(kwargs.get("msk_size", 224), T.InterpolationMode.NEAREST),
            T.CenterCrop(kwargs.get("msk_crp_size", 224)),
            T.ToTensor(),
        ])

    def __len__(self):
        if self.ref_aug == "rotate":
            return len(self.image_paths) * len(self.ref_aug_angles)
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.ref_aug == "rotate":
            image_idx = idx // len(self.ref_aug_angles)
            angle = self.ref_aug_angles[idx % len(self.ref_aug_angles)]
        else:
            image_idx = idx
            angle = None

        image_path = self.image_paths[image_idx]
        label = self.labels[image_idx]
        mask_path = self.mask_paths[image_idx]
        class_name = self.class_names[image_idx]
        image, label, mask = self._load_image_and_mask(image_path, label, mask_path, angle=angle)
        return image, label, mask, class_name

    def _load_image_and_mask(self, image_path, label, mask_path, angle=None):
        image = Image.open(image_path).convert("RGB")
        if angle is not None:
            image = rotate_rgb_image(image, angle, fill_mode=self.ref_aug_fill)
        image = self.transform(image)

        if label == 0:
            mask = torch.zeros([1, self.mask_size[0], self.mask_size[1]])
        else:
            mask = Image.open(mask_path)
            mask = self.target_transform(mask)
        return image, label, mask

    def _load_data(self, class_name):
        image_paths, labels, mask_paths = [], [], []
        phase = "train" if self.train else "test"
        image_dir = os.path.join(self.root, class_name, phase)
        mask_dir = os.path.join(self.root, class_name, "ground_truth")

        for image_type in sorted(os.listdir(image_dir)):
            image_type_dir = os.path.join(image_dir, image_type)
            if not os.path.isdir(image_type_dir):
                continue
            image_files = sorted(os.path.join(image_type_dir, file_name) for file_name in os.listdir(image_type_dir))
            image_paths.extend(image_files)
            if image_type == "good":
                labels.extend([0] * len(image_files))
                mask_paths.extend([None] * len(image_files))
            else:
                labels.extend([1] * len(image_files))
                gt_type_dir = os.path.join(mask_dir, image_type)
                image_stems = [os.path.splitext(os.path.basename(file_name))[0] for file_name in image_files]
                mask_paths.extend(os.path.join(gt_type_dir, stem + "_mask.png") for stem in image_stems)
        class_names = [class_name] * len(image_paths)
        return image_paths, labels, mask_paths, class_names


def apply_feature_wavelet_from_args(args, features):
    return apply_feature_wavelet_filter(
        features,
        wave=args.wave,
        feature_wav_mode=args.feature_wav_mode,
        hf_weight=args.hf_weight,
        ll_skip_alpha=args.ll_skip_alpha,
        hf_skip_alpha=args.hf_skip_alpha,
        wav_hf_normalize=args.wav_hf_normalize,
    )


def normalize_adaclip_feature_maps_if_enabled(args, features):
    if args.feature_backbone != "adaclip_prompted" or not args.adaclip_feature_l2norm:
        return features
    return [F.normalize(feature.float(), p=2, dim=1) for feature in features]


def print_wavelet_config(args):
    if not args.use_wav:
        return
    print("[Wavelet] wav_on:", args.wav_on)
    print("[Wavelet] feature_wav_mode:", args.feature_wav_mode)
    print("[Wavelet] wav_hf_normalize:", args.wav_hf_normalize)


def get_feature_image_size(args):
    if args.feature_backbone in ("clip_raw", "adaclip_prompted"):
        return args.clip_image_size
    return 224


def build_feature_encoder(args, device):
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
        ).to(device)
        encoder.eval()
        return encoder

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
            device=device,
        ).to(device)
        encoder.eval()
        return encoder

    if args.feature_backbone != "original":
        raise ValueError(f"Unsupported feature_backbone: {args.feature_backbone}")
    if args.backbone == "wide_resnet50_2":
        encoder = timm.create_model("wide_resnet50_2", features_only=True, out_indices=(1, 2, 3), pretrained=True).eval()
        return encoder.to(device)
    if args.backbone == "tf_efficientnet_b6":
        encoder = timm.create_model("tf_efficientnet_b6", features_only=True, out_indices=(1, 2, 3), pretrained=True).eval()
        return encoder.to(device)
    if args.backbone in DINOV2_BACKBONES:
        encoder = DINOv2BackboneWrapper(
            model_name=args.backbone,
            out_dims=(40, 72, 200),
            out_sizes=(56, 28, 14),
            freeze=True,
            feature_mode=args.dinov2_feature_mode,
            layers=args.dinov2_layers,
            proj_dim=args.dinov2_proj_dim,
        ).to(device)
        encoder.eval()
        print_dinov2_config(encoder, image_size=get_feature_image_size(args))
        return encoder
    raise ValueError(f"Unsupported backbone: {args.backbone}")


def main(args):
    image_size = get_feature_image_size(args)
    device = args.device
    root_dir = args.dataset_dir or args.few_shot_dir
    save_dir = args.output_dir or args.save_dir
    if args.ref_aug == "rotate":
        print("[RefAug] mode:", args.ref_aug)
        print("[RefAug] angles:", args.ref_aug_angles)
        print("[RefAug] fill:", args.ref_aug_fill)
        print("[RefAug] num_ref_shot=4 is recommended when using rotation augmentation.")
    print_wavelet_config(args)

    encoder = build_feature_encoder(args, device)
    feat_dims = encoder.feature_info.channels()
    if len(feat_dims) != args.feature_levels:
        raise ValueError(f"feature_levels={args.feature_levels} does not match encoder outputs {len(feat_dims)}.")
    print("[RefExtract-Ada] feature_backbone:", args.feature_backbone)
    print("[RefExtract-Ada] clip_layers:", args.clip_layers)
    print("[RefExtract-Ada] feature_levels:", args.feature_levels)
    print("[RefExtract-Ada] feat_dims:", feat_dims)
    print("[RefExtract-Ada] adaclip_feature_l2norm:", args.feature_backbone == "adaclip_prompted" and args.adaclip_feature_l2norm)

    if args.bgadweight_dir:
        decoders = [load_flow_model(args, feat_dim).to(args.device) for feat_dim in feat_dims]
        load_weights(encoder, decoders, args.bgadweight_dir)

    if args.class_name:
        class_names = [args.class_name]
    elif args.dataset in SETTINGS:
        class_names = SETTINGS[args.dataset]
    else:
        raise ValueError(f"Dataset setting must be in {SETTINGS.keys()}, but got {args.dataset}.")

    for class_name in class_names:
        train_dataset = FEWSHOTDATA(
            root_dir,
            class_name=class_name,
            train=True,
            img_size=image_size,
            crp_size=image_size,
            msk_size=image_size,
            msk_crp_size=image_size,
            ref_aug=args.ref_aug,
            ref_aug_angles=args.ref_aug_angles,
            ref_aug_fill=args.ref_aug_fill,
        )
        if args.ref_aug == "rotate":
            print(
                "[RefAug] effective reference images per class:",
                f"{len(train_dataset.image_paths)} * {len(args.ref_aug_angles)} = {len(train_dataset)}",
            )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)
        layer_features = None

        for batch in tqdm.tqdm(train_loader):
            images = batch[0].to(device)
            with torch.no_grad():
                features = encoder(images)
                features = normalize_adaclip_feature_maps_if_enabled(args, features)
                if args.use_wav and args.wav_on == "feature":
                    features = apply_feature_wavelet_from_args(args, features)
            if layer_features is None:
                layer_features = [[] for _ in range(len(features))]
            for layer_id, feature in enumerate(features):
                layer_features[layer_id].append(feature)

        layer_features = [torch.cat(features, dim=0) for features in layer_features]
        flattened_features = []
        for layer_id, features in enumerate(layer_features):
            print(f"{class_name} layer{layer_id + 1}: {tuple(features.shape)}")
            channels = features.shape[1]
            flattened = features.permute(0, 2, 3, 1).reshape(-1, channels).contiguous()
            flattened_features.append(flattened)

        class_dir = os.path.join(save_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for layer_id, features in enumerate(flattened_features):
            np.save(os.path.join(class_dir, f"layer{layer_id + 1}.npy"), features.cpu().numpy())
        print(f"Successfully saved {len(flattened_features)} layer file(s) for {class_name} to {class_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--class_name", type=str, default="")
    parser.add_argument("--dataset_dir", type=str, default="")
    parser.add_argument("--few_shot_dir", type=str, default="./4shot/mvtec")
    parser.add_argument("--flow_arch", type=str, default="flow_model")
    parser.add_argument("--bgadweight_dir", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="./ref_features/ada_ibstyle/mvtec_4shot")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=8)
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
    parser.add_argument("--feature_levels", default=4, type=int)
    parser.add_argument("--dinov2_feature_mode", type=str, default="final_projected", choices=DINOV2_FEATURE_MODES)
    parser.add_argument("--dinov2_layers", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--dinov2_proj_dim", type=int, default=256)
    parser.add_argument("--use_wav", action="store_true")
    parser.add_argument("--wav_on", type=str, default="residual", choices=["residual", "feature"])
    parser.add_argument("--wave", type=str, default="haar", choices=["haar"])
    parser.add_argument("--hf_weight", type=float, default=1.0)
    parser.add_argument("--feature_wav_mode", type=str, default="ll_only", choices=["ll_only", "hf_only", "ll_hf", "skip_ll", "skip_hf"])
    parser.add_argument("--ll_skip_alpha", type=float, default=0.5)
    parser.add_argument("--hf_skip_alpha", type=float, default=0.75)
    parser.add_argument("--wav_hf_normalize", action="store_true")
    parser.add_argument("--ref_aug", type=str, default="none", choices=["none", "rotate"])
    parser.add_argument("--ref_aug_angles", type=int, nargs="+", default=[0, 45, 90, 135, 180, 225, 270, 315])
    parser.add_argument("--ref_aug_fill", type=str, default="reflect", choices=["reflect", "constant"])
    parser.add_argument("--coupling_layers", type=int, default=4)
    parser.add_argument("--clamp_alpha", type=float, default=1.9)
    parser.add_argument("--pos_embed_dim", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    main(args)
