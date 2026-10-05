"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm/lớp có `raise NotImplementedError`.
Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).

Giao diện bạn phải giữ:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls" (label smoothing), "focal", "ce_weighted"."""
    if kind == "ce":
        return nn.CrossEntropyLoss()
    elif kind == "ls":
        smoothing = kw.get("smoothing", 0.1)
        return LabelSmoothingCE(smoothing=smoothing)
    elif kind == "focal":
        gamma = kw.get("gamma", 2.0)
        alpha = kw.get("alpha", None)
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif kind == "ce_weighted":
        weight = kw.get("weight", None)
        return nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"Unknown criterion kind: {kind}")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K  (slide trang 56)."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        # Dùng hàm có sẵn của PyTorch cho nhanh và chuẩn
        self.criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)

    def forward(self, logits, targets):
        return self.criterion(logits, targets)


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (slide trang 57)."""

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)
        
        log_p_t = log_p.gather(1, targets.view(-1, 1)).squeeze(1)
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
        
        loss = -1 * ((1 - p_t) ** self.gamma) * log_p_t
        
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = loss * alpha_t
            
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN."""
    counts = torch.tensor(counts, dtype=torch.float32)
    C = len(counts)
    
    if beta == 0.0:
        weights = 1.0 / counts
    else:
        weights = (1.0 - beta) / (1.0 - beta ** counts)
        
    weights = weights / weights.sum() * C
    return weights


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
        
    batch_size = x.size(0)
    perm = torch.randperm(batch_size).to(x.device)
    
    if mode == "mixup":
        x_mix = lam * x + (1 - lam) * x[perm]
        y_a, y_b = y, y[perm]
        return x_mix, (y_a, y_b, lam)
        
    elif mode == "cutmix":
        W, H = x.size(3), x.size(2)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        x_mix = x.clone()
        x_mix[:, :, bby1:bby2, bbx1:bbx2] = x[perm, :, bby1:bby2, bbx1:bbx2]
        
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        y_a, y_b = y, y[perm]
        return x_mix, (y_a, y_b, lam)
        
    else:
        raise ValueError(f"Unknown mix mode {mode}")


def mixed_loss(criterion, logits, targets):
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
