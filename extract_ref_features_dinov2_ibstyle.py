import argparse
import os

import numpy as np
import torch
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
from models.dinov2_encoder import DINOv2IBStyleEncoder, default_dinov2_out_indices, print_dinov2_ibstyle_config


SETTINGS = {
    "mvtec": MVTEC.CLASS_NAMES,
    "visa": VISA.CLASS_NAMES,
    "btad": BTAD.CLASS_NAMES,
    "mvtec3d": MVTEC3D.CLASS_NAMES,
    "mpdd": MPDD.CLASS_NAMES,
    "mvtecloco": MVTECLOCO.CLASS_NAMES,
    "brats": BRATS.CLASS_NAMES,
}


class FEWSHOTDATA(Dataset):
    def __init__(self, root, class_name="bottle", train=True, **kwargs):
        self.root = root
        self.class_name = class_name
        self.train = train
        self.mask_size = [kwargs.get("msk_crp_size"), kwargs.get("msk_crp_size")]
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
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        mask_path = self.mask_paths[idx]
        class_name = self.class_names[idx]
        image, label, mask = self._load_image_and_mask(image_path, label, mask_path)
        return image, label, mask, class_name

    def _load_image_and_mask(self, image_path, label, mask_path):
        image = Image.open(image_path).convert("RGB")
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


def build_encoder(args):
    out_indices = args.dinov2_out_indices or default_dinov2_out_indices(args.backbone)
    if len(out_indices) != 4:
        raise ValueError(f"DINOv2 IB-style reference extraction expects 4 out_indices, got {len(out_indices)}.")
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


def main(args):
    root_dir = args.dataset_dir or args.few_shot_dir
    save_dir = args.output_dir or args.save_dir
    if args.class_name:
        class_names = [args.class_name]
    elif args.dataset in SETTINGS:
        class_names = SETTINGS[args.dataset]
    else:
        raise ValueError(f"Dataset must be one of {SETTINGS.keys()}, got {args.dataset}.")

    encoder = build_encoder(args)
    if args.dino_shape_test:
        dummy = torch.zeros(2, 3, 224, 224, device=args.device)
        with torch.no_grad():
            features = encoder.encode_image_from_tensors(dummy)
        print("[DINOv2-IBStyle] shape test:", [tuple(feature.shape) for feature in features])
        return

    for class_name in class_names:
        dataset = FEWSHOTDATA(root_dir, class_name=class_name, train=True, img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)
        layer_features = [[] for _ in range(4)]
        for batch in tqdm.tqdm(loader):
            images = batch[0].to(args.device)
            with torch.no_grad():
                features = encoder.encode_image_from_tensors(images)
            for level, feature in enumerate(features):
                layer_features[level].append(feature)

        os.makedirs(os.path.join(save_dir, class_name), exist_ok=True)
        for level, features in enumerate(layer_features):
            features = torch.cat(features, dim=0)
            print(f"{class_name} layer{level + 1}: {tuple(features.shape)}")
            flattened = features.reshape(-1, features.shape[-1]).contiguous()
            np.save(os.path.join(save_dir, class_name, f"layer{level + 1}.npy"), flattened.cpu().numpy())
        print(f"Saved DINOv2 IB-style reference features for {class_name} to {os.path.join(save_dir, class_name)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--class_name", type=str, default="")
    parser.add_argument("--dataset_dir", type=str, default="")
    parser.add_argument("--few_shot_dir", type=str, default="./4shot/mvtec")
    parser.add_argument("--save_dir", type=str, default="/data/home/ueno/ref_features/dinov2/mvtec_tf_8shot_select")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--backbone", type=str, default="dinov2_vits14")
    parser.add_argument("--dinov2_out_indices", type=int, nargs="+", default=None)
    parser.add_argument("--dinov2_hub_repo", type=str, default="facebookresearch/dinov2")
    parser.add_argument("--dinov2_hub_source", type=str, default="github", choices=["github", "local"])
    parser.add_argument("--dino_shape_test", action="store_true")
    args = parser.parse_args()
    main(args)
