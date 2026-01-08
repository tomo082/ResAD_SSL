import warnings
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import numpy as np
import torch
import torch.nn.functional as F

from models.modules import get_position_encoding
from models.utils import get_logp
from utils import get_residual_features, get_matched_ref_features
from utils import calculate_metrics, applying_EFDM
from losses.utils import get_logp_a

warnings.filterwarnings('ignore')


def validate(args, encoder, vq_ops, constraintor, estimators, test_loader, ref_features, device, class_name):
    vq_ops.eval()
    constraintor.eval()
    for estimator in estimators:  
        estimator.eval()
        
    # UMAP可視化のために追加するリスト
    all_features_to_return = []
    all_anomaly_types_to_return = []
    all_gts_to_return = [] # 0/1の画像レベルのラベル    
    # 可視化のために追加
    all_images_raw = [] # 生の画像データ
    all_scores_map = [] # スコアマップ

    label_list, gt_mask_list = [], []
    logps1_list = [list() for _ in range(args.feature_levels)]
    logps2_list = [list() for _ in range(args.feature_levels)]
    progress_bar = tqdm(total=len(test_loader))
    progress_bar.set_description(f"Evaluating")
    for idx, batch in enumerate(test_loader):
        progress_bar.update(1)
        # データセットから返される値を修正したMVTEC/MVTECANOクラスの__getitem__を想定
        # image: 画像テンソル
        # label: 画像レベルのGTラベル (0:正常, 1:異常)
        # mask: ピクセルレベルのGTマスク
        # class_name_batch: (元のコードの_に対応) その画像のクラス名 (str) - バッチ内の全画像で同じはず
        # anomaly_type_batch: (追加) その画像の異常タイプ名 (str, 例: 'scratch', 'hole', 'good')
        image, label, mask, class_name_batch, anomaly_type_batch = batch # ここを変更        
        #image, label, mask, _ = batch    
        all_images_raw.append(image.cpu().numpy())       
        gt_mask_list.append(mask.squeeze(1).cpu().numpy().astype(bool))
        label_list.append(label.cpu().numpy().astype(bool).ravel())
        
        image = image.to(device)
        size = image.shape[-1]
        
        with torch.no_grad():
            if args.backbone == 'wide_resnet50_2':
                features = encoder(image)
                mfeatures = get_matched_ref_features(features, ref_features)
                rfeatures = get_residual_features(features, mfeatures, pos_flag=True)
            elif args.backbone == 'tf_efficientnet_b6':#10/26追加
                features = encoder(image)
                mfeatures = get_matched_ref_features(features, ref_features)
                rfeatures = get_residual_features(features, mfeatures, pos_flag=True)
            else:
                features = encoder.encode_image_from_tensors(image)
                for i in range(len(features)):
                    b, l, c = features[i].shape
                    features[i] = features[i].permute(0, 2, 1).reshape(b, c, 16, 16)
                mfeatures = get_matched_ref_features(features, ref_features)
                rfeatures = get_residual_features(features, mfeatures)

            if args.residual=='False':
                rfeatures = features
            else:
                rfeatures = rfeatures
            
            
             # --- UMAP可視化のために特徴量と異常タイプ名を収集 ---
            # ここで、UMAPに渡す特徴量を決定します。
            # 通常、最も深い層（最後の要素）の特徴量をフラットにして使います。
            fdm_features = vq_ops(rfeatures, train=False)
            rfeatures = applying_EFDM(rfeatures, fdm_features, alpha=args.fdm_alpha)
            rfeatures = constraintor(*rfeatures) 
            
            current_features_flat = rfeatures[-1].cpu().numpy().reshape(image.shape[0], -1)
            all_features_to_return.append(current_features_flat)
            all_anomaly_types_to_return.extend(anomaly_type_batch) # リストのままextend
            all_gts_to_return.extend(label.cpu().numpy()) # labelは0/1のGTラベル

            
            for l in range(args.feature_levels):
                e = rfeatures[l]  # BxCxHxW
                bs, dim, h, w = e.size()
                e = e.permute(0, 2, 3, 1).reshape(-1, dim)
                
                # (bs, 128, h, w)
                pos_embed = get_position_encoding(args.pos_embed_dim, h, w).to(args.device).unsqueeze(0).repeat(bs, 1, 1, 1)
                pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, args.pos_embed_dim)
                estimator = estimators[l]

                if args.flow_arch == 'flow_model':
                    z, log_jac_det = estimator(e)  
                else:
                    z, log_jac_det = estimator(e, [pos_embed, ])

                logps = get_logp(dim, z, log_jac_det)  
                logps = logps / dim  
                logps1_list[l].append(logps.reshape(bs, h, w))
                
                logps_a = get_logp_a(dim, z, log_jac_det)  # logps corresponding to abnormal distribution
                logits = torch.stack([logps, logps_a], dim=-1)  # (N, 2)
                sa = torch.softmax(logits, dim=-1)[:, 1]
                logps2_list[l].append(sa.reshape(bs, h, w))
    
    progress_bar.close()
    
    labels = np.concatenate(label_list)
    gt_masks = np.concatenate(gt_mask_list, axis=0)
    scores1 = convert_to_anomaly_scores(logps1_list, feature_levels=args.feature_levels, class_name=class_name, size=size)
    scores2 = aggregate_anomaly_scores(logps2_list, feature_levels=args.feature_levels, class_name=class_name, size=size)
    
    img_auc1, img_ap1, img_f1_score1, pix_auc1, pix_ap1, pix_f1_score1, pix_aupro1 = calculate_metrics(scores1, labels, gt_masks, pro=False, only_max_value=True)
    img_auc2, img_ap2, img_f1_score2, pix_auc2, pix_ap2, pix_f1_score2, pix_aupro2 = calculate_metrics(scores2, labels, gt_masks, pro=False, only_max_value=True)
    
    scores = (scores1 + scores2) / 2
    img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = calculate_metrics(scores, labels, gt_masks, pro=False, only_max_value=True)
    #visualizerを使えるようにするためにtest_imgsを返す
    metrics = {}
    metrics['scores1'] = [img_auc1, img_ap1, img_f1_score1, pix_auc1, pix_ap1, pix_f1_score1, pix_aupro1]
    metrics['scores2'] = [img_auc2, img_ap2, img_f1_score2, pix_auc2, pix_ap2, pix_f1_score2, pix_aupro2]
    metrics['scores'] = [img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro]
    # UMAP可視化のために追加した戻り値
    metrics['features'] = np.concatenate(all_features_to_return, axis=0)
    metrics['anomaly_types'] = np.array(all_anomaly_types_to_return, dtype=object) # 文字列を含むのでobject型
    metrics['gts_labels'] = np.array(all_gts_to_return) # 0/1のGTラベル
    # 可視化のために追加
    metrics['images_raw'] = np.concatenate(all_images_raw, axis=0)
    metrics['scores_map'] = scores
    metrics['gt_masks_raw'] = gt_masks    
    return metrics


def convert_to_anomaly_scores(logps_list, feature_levels=3, class_name=None, size=224):
    normal_map = [list() for _ in range(feature_levels)]
    for l in range(feature_levels):
        logps = torch.cat(logps_list[l], dim=0)  
        logps-= torch.max(logps) # normalize log-likelihoods to (-Inf:0] by subtracting a constant
        probs = torch.exp(logps) # convert to probs in range [0:1]
        # upsample
        normal_map[l] = F.interpolate(probs.unsqueeze(1),
            size=size, mode='bilinear', align_corners=True).squeeze().cpu().numpy()
    
    # score aggregation
    scores = np.zeros_like(normal_map[0])
    for l in range(feature_levels):
        scores += normal_map[l]

    # normality score to anomaly score
    scores = scores.max() - scores 
    
    #if class_name in ['pill', 'cable', 'capsule', 'screw']:
    for i in range(scores.shape[0]):
        scores[i] = gaussian_filter(scores[i], sigma=4)

    return scores


def aggregate_anomaly_scores(logps_list, feature_levels=3, class_name=None, size=224):
    abnormal_map = [list() for _ in range(feature_levels)]
    for l in range(feature_levels):
        probs = torch.cat(logps_list[l], dim=0)  
        # upsample
        abnormal_map[l] = F.interpolate(probs.unsqueeze(1),
            size=size, mode='bilinear', align_corners=True).squeeze().cpu().numpy()
    
    # score aggregation
    scores = np.zeros_like(abnormal_map[0])
    for l in range(feature_levels):
        scores += abnormal_map[l]
    scores /= feature_levels
    
    for i in range(scores.shape[0]):
        scores[i] = gaussian_filter(scores[i], sigma=4)

    return scores
