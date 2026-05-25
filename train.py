import math
import warnings
import torch
import torch.nn.functional as F

from models.modules import get_position_encoding
from models.utils import get_logp, log_theta
from losses.focal_loss import FocalLoss
from losses.loss import calculate_log_barrier_bg_spp_loss, get_flow_loss, get_flow_loss_with_boundary
from losses.utils import get_logp_a, get_normal_boundary
from models.soft_codebook import apply_soft_codebook_flat_if_enabled

warnings.filterwarnings('ignore')
logp_wrapper = log_theta


def _skip_nonfinite_nf_loss(args, loss, level, epoch):
    if not getattr(args, "use_soft_codebook", False):
        return False
    if torch.isfinite(loss.detach()).item():
        return False
    print(f"[WARN] NF loss is non-finite at epoch={epoch}, level={level}; skipping optimizer step.")
    return True


def train(args, rfeatures, decoders, optimizer, masks, boundary_ops, epoch, N_batch=4096, FIRST_STAGE_EPOCH=10, soft_codebook=None):
    train_loss_total, total_num = 0, 0
    for l in range(args.feature_levels):
        e = rfeatures[l]  
        bs, dim, h, w = e.size()
        e = e.permute(0, 2, 3, 1).reshape(-1, dim)
        masks_ = F.interpolate(masks, size=(h, w), mode='nearest').squeeze(1)
        masks_ = masks_.reshape(-1)
        
        # (bs, 128, h, w)
        pos_embed = get_position_encoding(args.pos_embed_dim, h, w).to(args.device).unsqueeze(0).repeat(bs, 1, 1, 1)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, args.pos_embed_dim)
        decoder = decoders[l]
            
        perm = torch.randperm(bs*h*w, device=args.device)
        num_N_batches = bs*h*w // N_batch
        for i in range(num_N_batches):
            idx = torch.arange(i*N_batch, (i+1)*N_batch)
            p_b = pos_embed[perm[idx]]  
            e_b = e[perm[idx]]  
            m_b = masks_[perm[idx]]  
            e_b = apply_soft_codebook_flat_if_enabled(
                args,
                soft_codebook,
                l,
                e_b,
                epoch=epoch,
                prefix="train_soft_codebook",
            )
            
            if args.flow_arch == 'flow_model':
                z, log_jac_det = decoder(e_b)  
            else:
                z, log_jac_det = decoder(e_b, [p_b, ])
                    
            # first 10 epochs only training normal samples
            if epoch < FIRST_STAGE_EPOCH:
                logps = get_logp(dim, z, log_jac_det) 
                logps = logps / dim
                loss = -logp_wrapper(logps).mean()

                if _skip_nonfinite_nf_loss(args, loss, l, epoch):
                    continue
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                b_n = get_normal_boundary(logps.detach(), m_b, pos_beta=args.pos_beta)
                boundary_ops.update_boundary(b_n, l)
                
                train_loss_total += loss.item()
                total_num += 1
            else:
                if m_b.sum() == 0:  # only normal ml loss
                    logps = get_logp(dim, z, log_jac_det)  
                    logps = logps / dim
                    loss = -logp_wrapper(logps).mean()
                    
                    b_n = get_normal_boundary(logps.detach(), m_b, pos_beta=args.pos_beta)
                    boundary_ops.update_boundary(b_n, l)
                if m_b.sum() > 0:  # normal ml loss and bg_spp loss
                    logps = get_logp(dim, z, log_jac_det)  
                    logps = logps / dim 
                    b_n = get_normal_boundary(logps.detach(), m_b, pos_beta=args.pos_beta)
                    boundary_ops.update_boundary(b_n, l)

                    logps_a = get_logp_a(dim, z, log_jac_det)
                    
                    loss_ml = -logp_wrapper(logps[m_b == 0])
                    loss_ml = torch.mean(loss_ml)
                    loss_ml_a = -logp_wrapper(logps_a[m_b == 1])
                    loss_ml_a = torch.mean(loss_ml_a)
                    
                    logits = torch.stack([logps, logps_a], dim=-1)  # (N, 2)
                    s = torch.softmax(logits, dim=-1)
                    loss_func = FocalLoss()
                    loss_focal = loss_func(s, m_b.unsqueeze(-1))
        
                    b_n = boundary_ops.get_boundary(l)
                    b_a = b_n - args.margin_tau
                    loss_n_con, loss_a_con = calculate_log_barrier_bg_spp_loss(logps, m_b, (b_n, b_a))
                
                    loss = loss_ml + loss_ml_a + loss_focal + args.bgspp_lambda * (loss_n_con + loss_a_con)
                
                optimizer.zero_grad()
                if _skip_nonfinite_nf_loss(args, loss, l, epoch):
                    continue
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                if not math.isnan(loss_item):
                    train_loss_total += loss_item
                    total_num += 1
    return train_loss_total, total_num


def train2(args, rfeatures, decoders, optimizer, lvl_masks, boundary_ops, epoch, N_batch=4096, FIRST_STAGE_EPOCH=10):
    train_loss_total, total_num = 0, 0
    for l in range(args.feature_levels):
        e = rfeatures[l]  
        bs, dim, h, w = e.size()
        e = e.permute(0, 2, 3, 1).reshape(-1, dim)
        m = lvl_masks[l].reshape(-1)
        
        # (bs, 128, h, w)
        pos_embed = get_position_encoding(args.pos_embed_dim, h, w).to(args.device).unsqueeze(0).repeat(bs, 1, 1, 1)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, args.pos_embed_dim)
        decoder = decoders[l]
            
        perm = torch.randperm(bs*h*w, device=args.device)
        num_N_batches = bs*h*w // N_batch
        for i in range(num_N_batches):
            idx = torch.arange(i*N_batch, (i+1)*N_batch)
            p_b = pos_embed[perm[idx]]  
            e_b = e[perm[idx]]  
            m_b = m[perm[idx]]  
            
            if args.flow_arch == 'flow_model':
                z, log_jac_det = decoder(e_b)  
            else:
                z, log_jac_det = decoder(e_b, [p_b, ])
            
            # moving average the boundary
            logps = get_logp(dim, z, log_jac_det)
            logps = logps / dim
            b_n = get_normal_boundary(logps.detach(), m_b, pos_beta=args.pos_beta)
            boundary_ops.update_boundary(b_n, l)
             
            # first 10 epochs only training normal samples
            if epoch < FIRST_STAGE_EPOCH:
                loss = get_flow_loss(dim, z, m_b, log_jac_det)
            else:
                b_n = boundary_ops.get_boundary(l)
                b_a = b_n - args.margin_tau
                loss = get_flow_loss_with_boundary(dim, z, m_b, log_jac_det, (b_n, b_a))
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_item = loss.item()
            train_loss_total += loss_item
            total_num += 1
    return train_loss_total, total_num
