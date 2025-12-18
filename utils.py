import os
import math
import random
from typing import List, Dict
from PIL import Image
import numpy as np
from skimage import measure
from sklearn.metrics import auc, roc_auc_score, average_precision_score, precision_recall_curve
import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms as T
from datasets.mvtec import MVTEC
from datasets.visa import VISA


class BoundaryAverager:
    def __init__(self, num_levels=3):
        self.boundaries = [0 for _ in range(num_levels)]
    
    def update_boundary(self, boundary, level, momentum=0.9):
        lvl_boundary = self.boundaries[level]
        lvl_boundary = lvl_boundary * momentum + (1 - momentum) * boundary
        self.boundaries[level] = lvl_boundary
        
    def get_boundary(self, level):
        return self.boundaries[level]
    
    
def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def get_matched_ref_features(features: List[Tensor], ref_features: List[Tensor]) -> List[Tensor]:
    """
    Get matched reference features for one class.
    """
    matched_ref_features = []
    for layer_id in range(len(features)):
        feature = features[layer_id]
        B, C, H, W = feature.shape
        feature = feature.permute(0, 2, 3, 1).reshape(-1, C).contiguous()  # (N1, C)
        feature_n = F.normalize(feature, p=2, dim=1)
        coreset = ref_features[layer_id]  # (N2, C)
        coreset_n = F.normalize(coreset, p=2, dim=1)
        dist = feature_n @ coreset_n.T
        cidx = torch.argmax(dist, dim=1)
        index_feats = coreset[cidx]
        index_feats = index_feats.reshape(B, H, W, C).permute(0, 3, 1, 2)
        matched_ref_features.append(index_feats)
    
    return matched_ref_features


def get_residual_features(features: List[Tensor], ref_features: List[Tensor], pos_flag: bool = False) -> List[Tensor]:
    residual_features = []
    for layer_id in range(len(features)):
        fi = features[layer_id]  # (B, dim, h, w)
        pi = ref_features[layer_id]  # (B, dim, h, w)
        
        if not pos_flag:
            ri = fi - pi
        else:
            ri = F.mse_loss(fi, pi, reduction='none')
        residual_features.append(ri)
    
    return residual_features
        

def load_reference_features(root_dir: str, class_name: str, device: torch.device) -> List[Tensor]:
    """
    Load reference features for one class.
    """
    layer1_refs = np.load(os.path.join(root_dir, class_name, 'layer1.npy'))
    layer2_refs = np.load(os.path.join(root_dir, class_name, 'layer2.npy'))
    layer3_refs = np.load(os.path.join(root_dir, class_name, 'layer3.npy'))
    
    layer1_refs = torch.from_numpy(layer1_refs).to(device)
    layer2_refs = torch.from_numpy(layer2_refs).to(device)
    layer3_refs = torch.from_numpy(layer3_refs).to(device)
    
    return layer1_refs, layer2_refs, layer3_refs


def get_random_normal_images(root, class_name, num_shot=4):
    if class_name in MVTEC.CLASS_NAMES:
        root_dir = os.path.join(root, class_name, 'train', 'good')
    elif class_name in VISA.CLASS_NAMES:
        root_dir = os.path.join(root, class_name, 'Data', 'Images', 'Normal')
    else:
        raise ValueError('Unrecognized class_name!')
    filenames = os.listdir(root_dir)
    n_idxs = np.random.randint(len(filenames), size=num_shot)
    n_idxs = n_idxs.tolist()
    normal_paths = []
    for n_idx in n_idxs:
        normal_paths.append(os.path.join(root_dir, filenames[n_idx]))
    
    return normal_paths


def get_mc_reference_features(encoder, root, class_names, device, num_shot=4):
    """
    Get reference features for multiple classes.
    """
    reference_features = {}
    class_names = np.unique(class_names)
    for class_name in class_names:
        normal_paths = get_random_normal_images(root, class_name, num_shot)
        images = load_and_transform_vision_data(normal_paths, device)
        with torch.no_grad():
            features = encoder(images)
            for l in range(len(features)):
                bs, c, h, w = features[l].shape
                features[l] = features[l].permute(0, 2, 3, 1).reshape(-1, c)
            reference_features[class_name] = features
    return reference_features


def load_and_transform_vision_data(image_paths, device):
    if image_paths is None:
        return None

    image_ouputs = []
    for image_path in image_paths:
        data_transform = T.Compose([
                T.Resize(224, T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Compose([T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])])
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        image = data_transform(image).to(device)
        image_ouputs.append(image)
    return torch.stack(image_ouputs, dim=0)


def get_mc_matched_ref_features(features: List[Tensor], class_names: List[str],
                                ref_features: Dict[str, List[Tensor]]) -> List[Tensor]:
    """
    Get matched reference features for multiple classes.
    """
    matched_ref_features = [[] for _ in range(len(features))]
    for idx, c in enumerate(class_names):  # for each image
        ref_features_c = ref_features[c]
        
        for layer_id in range(len(features)):  # for all layers of one image
            feature = features[layer_id][idx:idx+1]
            _, C, H, W = feature.shape
            
            feature = feature.permute(0, 2, 3, 1).reshape(-1, C).contiguous()  # (N1, C)
            feature_n = F.normalize(feature, p=2, dim=1)
            coreset = ref_features_c[layer_id]  # (N2, C)
            coreset_n = F.normalize(coreset, p=2, dim=1)
            dist = feature_n @ coreset_n.T  # (N1, N2)
            cidx = torch.argmax(dist, dim=1)
            index_feats = coreset[cidx]
            index_feats = index_feats.permute(1, 0).reshape(C, H, W)
            matched_ref_features[layer_id].append(index_feats)
            
    matched_ref_features = [torch.stack(item, dim=0) for item in matched_ref_features]
    
    return matched_ref_features


def calculate_metrics(scores, labels, gt_masks, pro=True, only_max_value=True):
    """
    Args:
        scores (np.ndarray): shape (N, H, W).
        labels (np.ndarray): shape (N, ), 0 for normal, 1 for abnormal.
        gt_masks (np.ndarray): shape (N, H, W).
    """
    # average precision
    pix_ap = round(average_precision_score(gt_masks.flatten(), scores.flatten()), 5)
    # f1 score, f1 score is to balance the precision and recall
    # f1 score is high means the precision and recall are both high
    precisions, recalls, _ = precision_recall_curve(gt_masks.flatten(), scores.flatten())
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    pix_f1_score = round(np.max(f1_scores[np.isfinite(f1_scores)]), 5)
    # roc auc
    pix_auc = round(roc_auc_score(gt_masks.flatten(), scores.flatten()), 5)
    
    _, h, w = scores.shape
    size = h * w
    if only_max_value:
        topks = [1]
    else:
        topks = [int(size*p) for p in np.arange(0.01, 0.41, 0.01)]
        topks = [1, 100] + topks
    img_aps, img_aucs, img_f1_scores = [], [], []
    for topk in topks:
        img_scores = get_image_scores(scores, topk)
        img_ap = round(average_precision_score(labels, img_scores), 5)
        precisions, recalls, _ = precision_recall_curve(labels, img_scores)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        img_f1_score = round(np.max(f1_scores[np.isfinite(f1_scores)]), 5)
        img_auc = round(roc_auc_score(labels, img_scores), 5)
        img_aps.append(img_ap)
        img_aucs.append(img_auc)
        img_f1_scores.append(img_f1_score)
    img_ap, img_auc, img_f1_score = np.max(img_aps), np.max(img_aucs), np.max(img_f1_scores)
        
    if pro:
        pix_aupro = calculate_aupro(gt_masks, scores)
    else:
        pix_aupro = -1
    
    return img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro


def get_image_scores(scores, topk=1):
    scores_ = torch.from_numpy(scores)
    img_scores = torch.topk(scores_.reshape(scores_.shape[0], -1), topk, dim=1)[0]
    img_scores = torch.mean(img_scores, dim=1)
    img_scores = img_scores.cpu().numpy()
        
    return img_scores


def calculate_aupro(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    if fprs.shape[0] <= 2:
        return 0.5
    else:
        fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
        pro_auc = auc(fprs, pros[idxes])
        return pro_auc


def applying_EFDM(input_features_list, ref_features_list, alpha=0.5):
    """
    Args:
        input_features (Tensor): shape of (B, C, H, W).
        ref_features (Tensor): normal reference features, (B, C, H, W).
    """
    alpha = 1 - alpha
    aligned_features_list = []
    for l in range(len(input_features_list)):
        input_features, ref_features = input_features_list[l], ref_features_list[l]
        B, C, W, H = input_features.shape

        input_features_r = input_features.reshape(B, C, -1)
        ref_features_r = ref_features.reshape(B, C, -1)

        sorted_input_features, inds = torch.sort(input_features_r)
        sorted_ref_features, _ = torch.sort(ref_features_r)
        aligned_features = sorted_input_features + (sorted_ref_features - sorted_input_features) * alpha
        inv_inds = inds.argsort(-1)
        aligned_features = aligned_features.gather(-1, inv_inds)
        aligned_features = aligned_features.view(B, C, W, H)
        aligned_features_list.append(aligned_features)

    return aligned_features_list
    
def load_weights(encoder, decoders, filename):#12/16追加
    #path = os.path.join(WEIGHT_DIR, filename)
    state = torch.load(filename)
    encoder.load_state_dict(state['encoder_state_dict'], strict=False)
    decoders = [decoder.load_state_dict(state, strict=False) for decoder, state in zip(decoders, state['decoder_state_dict'])] #12/16変更
    print('Loading weights from {}'.format(filename))
