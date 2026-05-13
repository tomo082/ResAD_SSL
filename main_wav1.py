import os
import warnings
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from torch.utils.data import DataLoader

from validate_wav1 import validate
from datasets.mvtec import MVTEC, MVTECANO
from datasets.visa import VISA, VISAANO
from datasets.btad import BTAD
from datasets.mvtec_3d import MVTEC3D
from datasets.mpdd import MPDD
from datasets.mvtec_loco import MVTECLOCO
from datasets.brats import BRATS
from datasets.capsules import CAPSULES, CAPSULESANO

from models.fc_flow import load_flow_model
from models.modules import MultiScaleConv
from models.vq import VectorQuantizer # 正しいクラス名に変更
from utils import init_seeds, get_residual_features, get_mc_matched_ref_features, get_mc_reference_features_wav
from utils import BoundaryAverager
from losses.loss import calculate_log_barrier_bi_occ_loss
from classes import VISA_TO_MVTEC, MVTEC_TO_VISA, MVTEC_TO_BTAD, MVTEC_TO_MVTEC3D
from classes import MVTEC_TO_MPDD, MVTEC_TO_MVTECLOCO, MVTEC_TO_BRATS
from classes import MVTEC_TO_MVTEC, VISA_TO_VISA
from classes import CAPSULES_TO_CAPSULES

warnings.filterwarnings('ignore')

TOTAL_SHOT = 4  
FIRST_STAGE_EPOCH = 10
SETTINGS = {'visa_to_mvtec': VISA_TO_MVTEC, 'mvtec_to_visa': MVTEC_TO_VISA,
            'mvtec_to_btad': MVTEC_TO_BTAD, 'mvtec_to_mvtec3d': MVTEC_TO_MVTEC3D,
            'mvtec_to_mpdd': MVTEC_TO_MPDD, 'mvtec_to_mvtecloco': MVTEC_TO_MVTECLOCO,
            'mvtec_to_brats': MVTEC_TO_BRATS,'mvtec_to_mvtec':MVTEC_TO_MVTEC, 'visa_to_visa':VISA_TO_VISA, 'capsules_to_capsules': CAPSULES_TO_CAPSULES}

# ==========================================
# Haar Wavelet Filter
# ==========================================
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
    if args.setting in SETTINGS.keys():
        CLASSES = SETTINGS[args.setting]
    else:
        raise ValueError(f"Dataset setting must be in {SETTINGS.keys()}, but got {args.setting}.")
                
    if args.classes == 'capsules':  
        train_dataset1 = CAPSULES(args.train_dataset_dir, class_name=CLASSES['seen'], train=True, normalize="w50", img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
        train_loader1 = DataLoader(train_dataset1, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
        train_dataset2 = CAPSULESANO(args.train_dataset_dir, class_name=CLASSES['seen'], train=True, normalize='w50', img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
        train_loader2 = DataLoader(train_dataset2, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)    
    elif CLASSES['seen'][0] in MVTEC.CLASS_NAMES:  
        train_dataset1 = MVTEC(args.train_dataset_dir, class_name=CLASSES['seen'], train=True, normalize="w50", img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
        train_loader1 = DataLoader(train_dataset1, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
        train_dataset2 = MVTECANO(args.train_dataset_dir, class_name=CLASSES['seen'], train=True, normalize='w50', img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
        train_loader2 = DataLoader(train_dataset2, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    else:  
        train_dataset1 = VISA(args.train_dataset_dir, class_name=CLASSES['seen'], train=True, normalize="w50", img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
        train_loader1 = DataLoader(train_dataset1, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
        train_dataset2 = VISAANO(args.train_dataset_dir, class_name=CLASSES['seen'], train=True, normalize="w50", img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
        train_loader2 = DataLoader(train_dataset2, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
        
    if args.backbone == 'wide_resnet50_2':
        encoder = timm.create_model('wide_resnet50_2', features_only=True, out_indices=(1, 2, 3), pretrained=True).eval().to(args.device)
    elif args.backbone == 'tf_efficientnet_b6':
        encoder = timm.create_model('tf_efficientnet_b6', features_only=True, out_indices=(1, 2, 3), pretrained=True).eval().to(args.device)
    feat_dims = encoder.feature_info.channels()
        
    boundary_ops = BoundaryAverager(num_levels=args.feature_levels)
    
    wav_filter = HaarWaveletFilter(low_freq_weight=args.lf_weight, high_freq_weight=args.hf_weight).to(args.device)
    wav_filter.eval()
    
    # VQモデルの初期化 (引数を VectorQuantizer の仕様に合わせる)
    vqs = [VectorQuantizer(n_e=args.num_embeddings, vq_embed_dim=feat_dim, beta=0.25).to(args.device) for feat_dim in feat_dims]
    params_vq = [p for vq in vqs for p in vq.parameters()]
    optimizer2 = torch.optim.Adam(params_vq, lr=args.lr, weight_decay=0.005)
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, milestones=[30, 50], gamma=0.1)

    constraintor = MultiScaleConv(feat_dims).to(args.device)
    optimizer0 = torch.optim.Adam(constraintor.parameters(), lr=args.lr, weight_decay=0.005)
    scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer0, milestones=[30, 50], gamma=0.1)
    
    estimators = [load_flow_model(args, feat_dim).to(args.device) for feat_dim in feat_dims]
    params = list(estimators[0].parameters())
    for l in range(1, args.feature_levels):
        params += list(estimators[l].parameters())
    optimizer1 = torch.optim.Adam(params, lr=args.lr, weight_decay=0.005)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[30, 50], gamma=0.1)
    
    from train import train
    best_img_auc = 0
    N_batch = 8192
    
    for epoch in range(args.epochs):
        constraintor.train()
        for vq in vqs:
            vq.train()
        for estimator in estimators:
            estimator.train()
            
        train_loader = train_loader1 if epoch < FIRST_STAGE_EPOCH else train_loader2
        train_loss_total, total_num = 0, 0
        
        progress_bar = tqdm(total=len(train_loader), desc=f"Epoch[{epoch}/{args.epochs}]")
        
        for step, batch in enumerate(train_loader):
            progress_bar.update(1)
            images, _, masks, class_names = batch
            images, masks = images.to(args.device), masks.to(args.device)
            
            with torch.no_grad():
                features = encoder(images)
                features = [wav_filter(f) for f in features]
                
                ref_features = get_mc_reference_features_wav(encoder, args.train_dataset_dir, class_names, images.device, args.train_ref_shot, wav_filter=wav_filter)
                
                mfeatures = get_mc_matched_ref_features(features, class_names, ref_features)
                rfeatures = get_residual_features(features, mfeatures, pos_flag=True)
            
            # 先にマスクのリサイズ処理を行ってVQに渡せるようにする
            lvl_masks = []
            for l in range(args.feature_levels):
                _, _, h, w = rfeatures[l].size()
                lvl_masks.append(F.interpolate(masks, size=(h, w), mode='nearest').squeeze(1))
            
            # VQの適用 (マスク情報を渡す)
            vq_loss_total = 0
            for l in range(args.feature_levels):
                z_q, vq_loss, _ = vqs[l](rfeatures[l], lvl_masks[l])
                rfeatures[l] = z_q
                vq_loss_total += vq_loss

            # 制約対象となる量子化後の特徴量を保存
            rfeatures_t = [rfeature.detach().clone() for rfeature in rfeatures]
            
            # Constraintorで空間補正
            rfeatures = constraintor(*rfeatures)
            
            noise_std = 0.01
            rfeatures_noisy = [rf + torch.randn_like(rf) * noise_std for rf in rfeatures]
            
            loss = 0
            for l in range(args.feature_levels):  
                e = rfeatures_noisy[l]  
                t = rfeatures_t[l]
                bs, dim, h, w = e.size()
                e, t = e.permute(0, 2, 3, 1).reshape(-1, dim), t.permute(0, 2, 3, 1).reshape(-1, dim)
                m = lvl_masks[l].reshape(-1)
                loss_i, _, _ = calculate_log_barrier_bi_occ_loss(e, m, t)
                loss += loss_i
                
            loss += vq_loss_total
            optimizer0.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer0.step()
            optimizer2.step()
            
            train_loss_total += loss.item()
            total_num += 1
            
            rfeatures = [rfeature.detach().clone() for rfeature in rfeatures]
            loss, num = train(args, rfeatures, estimators, optimizer1, masks, boundary_ops, epoch, N_batch=N_batch, FIRST_STAGE_EPOCH=FIRST_STAGE_EPOCH)
            train_loss_total += loss
            total_num += num
        
        scheduler0.step()
        scheduler1.step()
        scheduler2.step()
        progress_bar.close()
        print(f"Epoch[{epoch}/{args.epochs}]: train_loss: {train_loss_total / total_num}")
        
        if (epoch + 1) % args.eval_freq == 0:
            s1_res, s2_res, s_res = [], [], []
            
            test_ref_features = load_mc_reference_features(args.test_ref_feature_dir, CLASSES['unseen'], args.device, args.num_ref_shot)
            
            for class_name in CLASSES['unseen']:
                if args.classes == 'capsules':
                    test_dataset = CAPSULES(args.test_dataset_dir, class_name=class_name, train=False, normalize='w50', img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)                            
                elif class_name in MVTEC.CLASS_NAMES:
                    test_dataset = MVTEC(args.test_dataset_dir, class_name=class_name, train=False, normalize='w50', img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                elif class_name in VISA.CLASS_NAMES:
                    test_dataset = VISA(args.test_dataset_dir, class_name=class_name, train=False, normalize='w50', img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                elif class_name in BTAD.CLASS_NAMES:
                    test_dataset = BTAD(args.test_dataset_dir, class_name=class_name, train=False, normalize='w50', img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                elif class_name in MVTEC3D.CLASS_NAMES:
                    test_dataset = MVTEC3D(args.test_dataset_dir, class_name=class_name, train=False, normalize='w50', img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                elif class_name in MPDD.CLASS_NAMES:
                    test_dataset = MPDD(args.test_dataset_dir, class_name=class_name, train=False, normalize='w50', img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                elif class_name in MVTECLOCO.CLASS_NAMES:
                    test_dataset = MVTECLOCO(args.test_dataset_dir, class_name=class_name, train=False, normalize='w50', img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                elif class_name in BRATS.CLASS_NAMES:
                    test_dataset = BRATS(args.test_dataset_dir, class_name=class_name, train=False, normalize='w50', img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                else:
                    raise ValueError('Unrecognized class name: {}'.format(class_name))
                
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)
                
                metrics = validate(args, encoder, constraintor, vqs, wav_filter, estimators, test_loader, test_ref_features[class_name], args.device, class_name)
                
                img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = metrics['scores']
                print("Epoch: {}, Class Name: {}, Image AUC: {:.3f} | Pixel AUC: {:.3f} | AUPRO: {:.3f}".format(
                    epoch, class_name, img_auc, pix_auc, pix_aupro))
                s1_res.append(metrics['scores1'])
                s2_res.append(metrics['scores2'])
                s_res.append(metrics['scores'])
            
            img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = np.mean(np.array(s_res), axis=0)
            print('(Merged) Average Image AUC | AP | F1_Score: {:.3f} | {:.3f} | {:.3f}, Average Pixel AUC | AP | F1_Score | AUPRO: {:.3f} | {:.3f} | {:.3f} | {:.3f}'.format(
                img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro))
            
            if img_auc > best_img_auc:
                os.makedirs(args.checkpoint_path, exist_ok=True)
                best_img_auc = img_auc
                state_dict = {'constraintor': constraintor.state_dict(),
                              'vqs': [vq.state_dict() for vq in vqs],
                              'estimators': [estimator.state_dict() for estimator in estimators]}
                torch.save(state_dict, os.path.join(args.checkpoint_path, f'{args.setting}_epoch_{epoch}_checkpoints.pth'))

def load_mc_reference_features(root_dir: str, class_names, device: torch.device, num_shot=4):
    refs = {}
    for class_name in class_names:
        layer1_refs = torch.from_numpy(np.load(os.path.join(root_dir, class_name, 'layer1.npy'))).to(device)
        layer2_refs = torch.from_numpy(np.load(os.path.join(root_dir, class_name, 'layer2.npy'))).to(device)
        layer3_refs = torch.from_numpy(np.load(os.path.join(root_dir, class_name, 'layer3.npy'))).to(device)
        K1 = (layer1_refs.shape[0] // TOTAL_SHOT) * num_shot
        K2 = (layer2_refs.shape[0] // TOTAL_SHOT) * num_shot
        K3 = (layer3_refs.shape[0] // TOTAL_SHOT) * num_shot
        refs[class_name] = (layer1_refs[:K1, :], layer2_refs[:K2, :], layer3_refs[:K3, :])
    return refs
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=str, default="visa_to_mvtec")
    parser.add_argument('--classes', type=str, default="none")
    parser.add_argument('--train_dataset_dir', type=str, default="")
    parser.add_argument('--test_dataset_dir', type=str, default="")
    parser.add_argument('--test_ref_feature_dir', type=str, default="./ref_features/w50/mvtec_4shot_wav")
    parser.add_argument('--bgadweight_dir', type=str, default="none")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--checkpoint_path', type=str, default="./checkpoints/")
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--backbone', type=str, default="wide_resnet50_2")
    
    parser.add_argument('--flow_arch', type=str, default='conditional_flow_model')
    parser.add_argument('--feature_levels', default=3, type=int)
    parser.add_argument('--pos_embed_dim', type=int, default=256)
    parser.add_argument("--train_ref_shot", type=int, default=4)
    parser.add_argument("--num_ref_shot", type=int, default=4)
    parser.add_argument('--pos_beta', type=float, default=0.05)            
    parser.add_argument('--coupling_layers', type=int, default=10)
    parser.add_argument('--clamp_alpha', type=float, default=1.9)    
    parser.add_argument('--margin_tau', type=float, default=0.1)
    parser.add_argument('--bgspp_lambda', type=float, default=1)
    parser.add_argument('--fdm_alpha', type=float, default=0.4)
    parser.add_argument('--num_embeddings', type=int, default=1536)
            
    parser.add_argument("--lf_weight", type=float, default=0.1)
    parser.add_argument("--hf_weight", type=float, default=1.2)
    
    args = parser.parse_args()
    init_seeds(42)
    main(args)
