import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader

from datasets.mvtec import MVTEC
from datasets.visa import VISA
from datasets.btad import BTAD
from datasets.mvtec_3d import MVTEC3D
from datasets.mpdd import MPDD
from datasets.mvtec_loco import MVTECLOCO
from datasets.brats import BRATS
from datasets.capsules import CAPSULES

from classes import VISA_TO_MVTEC, MVTEC_TO_VISA, MVTEC_TO_BTAD, MVTEC_TO_MVTEC3D
from classes import MVTEC_TO_MPDD, MVTEC_TO_MVTECLOCO, MVTEC_TO_BRATS
from classes import MVTEC_TO_MVTEC, VISA_TO_VISA, CAPSULES_TO_CAPSULES
from utils import init_seeds

SETTINGS = {
    'visa_to_mvtec': VISA_TO_MVTEC, 'mvtec_to_visa': MVTEC_TO_VISA,
    'mvtec_to_btad': MVTEC_TO_BTAD, 'mvtec_to_mvtec3d': MVTEC_TO_MVTEC3D,
    'mvtec_to_mpdd': MVTEC_TO_MPDD, 'mvtec_to_mvtecloco': MVTEC_TO_MVTECLOCO,
    'mvtec_to_brats': MVTEC_TO_BRATS, 'mvtec_to_mvtec': MVTEC_TO_MVTEC, 
    'visa_to_visa': VISA_TO_VISA, 'capsules_to_capsules': CAPSULES_TO_CAPSULES
}


class HaarWaveletFilter(nn.Module):
    def __init__(self, low_freq_weight=0.1, high_freq_weight=1.2):
        super().__init__()
        self.lf_w = low_freq_weight
        self.hf_w = high_freq_weight
        
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        
        self.register_buffer('k_ll', ll.view(1, 1, 2, 2))
        self.register_buffer('k_hl', hl.view(1, 1, 2, 2))
        self.register_buffer('k_lh', lh.view(1, 1, 2, 2))
        self.register_buffer('k_hh', hh.view(1, 1, 2, 2))

    def forward(self, x):
        B, C, H, W = x.shape
        
        ll = F.conv2d(x, self.k_ll.expand(C, 1, 2, 2), stride=2, groups=C)
        hl = F.conv2d(x, self.k_hl.expand(C, 1, 2, 2), stride=2, groups=C)
        lh = F.conv2d(x, self.k_lh.expand(C, 1, 2, 2), stride=2, groups=C)
        hh = F.conv2d(x, self.k_hh.expand(C, 1, 2, 2), stride=2, groups=C)
        
        ll = ll * self.lf_w
        hl = hl * self.hf_w
        lh = lh * self.hf_w
        hh = hh * self.hf_w
        
        out = F.conv_transpose2d(ll, self.k_ll.expand(C, 1, 2, 2), stride=2, groups=C) + \
              F.conv_transpose2d(hl, self.k_hl.expand(C, 1, 2, 2), stride=2, groups=C) + \
              F.conv_transpose2d(lh, self.k_lh.expand(C, 1, 2, 2), stride=2, groups=C) + \
              F.conv_transpose2d(hh, self.k_hh.expand(C, 1, 2, 2), stride=2, groups=C)
        return out


def main(args):
    init_seeds(42)
    device = torch.device(args.device)

    if args.setting in SETTINGS.keys():
        CLASSES = SETTINGS[args.setting]
    else:
        raise ValueError(f"Dataset setting must be in {SETTINGS.keys()}, but got {args.setting}.")

    if args.backbone == 'wide_resnet50_2':
        encoder = timm.create_model('wide_resnet50_2', features_only=True, out_indices=(1, 2, 3), pretrained=True).eval().to(device)
    elif args.backbone == 'tf_efficientnet_b6':
        encoder = timm.create_model('tf_efficientnet_b6', features_only=True, out_indices=(1, 2, 3), pretrained=True).eval().to(device)
    else:
        raise ValueError("Unsupported backbone.")

    wav_filter = HaarWaveletFilter(low_freq_weight=args.lf_weight, high_freq_weight=args.hf_weight).to(device)
    wav_filter.eval()

    all_classes = CLASSES['seen'] + CLASSES['unseen']
    
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Saving wavelet-filtered reference features to: {args.save_dir}")
    print(f"Filter settings -> Low Freq (LL): {args.lf_weight}, High Freq (HL,LH,HH): {args.hf_weight}")

    for class_name in all_classes:
        class_save_dir = os.path.join(args.save_dir, class_name)
        os.makedirs(class_save_dir, exist_ok=True)
        
        if os.path.exists(os.path.join(class_save_dir, 'layer1.npy')):
            print(f"Features for {class_name} already exist. Skipping.")
            continue

        if args.classes == 'capsules' or class_name in CAPSULES.CLASS_NAMES:
            dataset = CAPSULES(args.dataset_dir, class_name=class_name, train=True, normalize='w50')
        elif class_name in MVTEC.CLASS_NAMES:
            dataset = MVTEC(args.dataset_dir, class_name=class_name, train=True, normalize='w50')
        elif class_name in VISA.CLASS_NAMES:
            dataset = VISA(args.dataset_dir, class_name=class_name, train=True, normalize='w50')
        elif class_name in BTAD.CLASS_NAMES:
            dataset = BTAD(args.dataset_dir, class_name=class_name, train=True, normalize='w50')
        elif class_name in MVTEC3D.CLASS_NAMES:
            dataset = MVTEC3D(args.dataset_dir, class_name=class_name, train=True, normalize='w50')
        elif class_name in MPDD.CLASS_NAMES:
            dataset = MPDD(args.dataset_dir, class_name=class_name, train=True, normalize='w50')
        elif class_name in MVTECLOCO.CLASS_NAMES:
            dataset = MVTECLOCO(args.dataset_dir, class_name=class_name, train=True, normalize='w50')
        elif class_name in BRATS.CLASS_NAMES:
            dataset = BRATS(args.dataset_dir, class_name=class_name, train=True, normalize='w50')
        else:
            raise ValueError(f"Unknown class {class_name}")

        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

        extracted_features = {0: [], 1: [], 2: []}
        count = 0

        progress_bar = tqdm(loader, desc=f"Extracting {class_name}")
        for batch in progress_bar:
            if count >= args.train_ref_shot:
                break
                
            images = batch[0].to(device)

            with torch.no_grad():
                features = encoder(images)
                
                features_wav = [wav_filter(f) for f in features]

                for i, feat in enumerate(features_wav):
                    flat_feat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1]).cpu().numpy()
                    extracted_features[i].append(flat_feat)
            
            count += 1

        for i in range(3):
            layer_feats = np.concatenate(extracted_features[i], axis=0)
            save_path = os.path.join(class_save_dir, f'layer{i+1}.npy')
            np.save(save_path, layer_feats)
        
        print(f"Saved {count} shots for {class_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=str, default="mvtec_to_mvtec")
    parser.add_argument('--classes', type=str, default="none")
    parser.add_argument('--dataset_dir', type=str, required=True, help="Path to the dataset directory")
    parser.add_argument('--save_dir', type=str, default="./ref_features/w50/mvtec_4shot_wav", help="Directory to save filtered features")
    parser.add_argument('--backbone', type=str, default="wide_resnet50_2")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument("--train_ref_shot", type=int, default=4, help="Number of reference images to extract")
    
    parser.add_argument("--lf_weight", type=float, default=0.1, help="Weight for low frequency (LL) components")
    parser.add_argument("--hf_weight", type=float, default=1.2, help="Weight for high frequency (LH, HL, HH) components")
    
    args = parser.parse_args()
    main(args)
