"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def predict_logits(model, loader, device, view=None):
    """Chạy model trên loader và gom logit theo đúng thứ tự file."""
    model.eval()
    all_filenames = []
    all_y_true = []
    all_logits = []
    
    with torch.inference_mode():
        for x, y, filenames in loader:
            x = x.to(device)
            if view is not None:
                x = view(x)
            logits = model(x)
            
            all_filenames.extend(filenames)
            all_y_true.append(y.numpy())
            all_logits.append(logits.cpu().numpy())
            
    all_y_true = np.concatenate(all_y_true)
    all_logits = np.concatenate(all_logits, axis=0)
    return all_filenames, all_y_true, all_logits


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W)."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x, crop: int):
    """5 crop (4 góc + giữa) kích thước `crop`, và tuỳ chọn thêm bản lật. Trả về list các batch."""
    # Assuming x is (N, C, H, W)
    H, W = x.shape[2:]
    views = []
    
    # top-left
    views.append(x[:, :, :crop, :crop])
    # top-right
    views.append(x[:, :, :crop, W - crop:])
    # bottom-left
    views.append(x[:, :, H - crop:, :crop])
    # bottom-right
    views.append(x[:, :, H - crop:, W - crop:])
    # center
    cy, cx = H // 2, W // 2
    ch, cw = crop // 2, crop // 2
    views.append(x[:, :, cy - ch : cy - ch + crop, cx - cw : cx - cw + crop])
    
    return views


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes`, trả về list các batch."""
    views = []
    for s in sizes:
        views.append(F.interpolate(x, size=(s, s), mode='bilinear', align_corners=False))
    return views


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62)."""
    # logits_per_view is a list of numpy arrays, shape (N, 9)
    logits_stack = np.stack(logits_per_view, axis=0) # (K, N, 9)
    if space == "prob":
        # exp(x) / sum(exp(x))
        max_val = np.max(logits_stack, axis=2, keepdims=True)
        exp_val = np.exp(logits_stack - max_val)
        probs_stack = exp_val / np.sum(exp_val, axis=2, keepdims=True)
        probs = np.mean(probs_stack, axis=0)
        return probs
    elif space == "logit":
        mean_logits = np.mean(logits_stack, axis=0)
        max_val = np.max(mean_logits, axis=1, keepdims=True)
        exp_val = np.exp(mean_logits - max_val)
        probs = exp_val / np.sum(exp_val, axis=1, keepdims=True)
        return probs
    else:
        raise ValueError(f"Unknown space: {space}")


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed)."""
    probs_stack = np.stack(list_of_probs, axis=0) # (M, N, 9)
    return np.mean(probs_stack, axis=0)


def fit_temperature(val_logits, val_labels) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL."""
    import torch.optim as optim
    logits_t = torch.tensor(val_logits, dtype=torch.float32)
    labels_t = torch.tensor(val_labels, dtype=torch.long)
    
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)
    
    def eval():
        optimizer.zero_grad()
        loss = criterion(logits_t / temperature, labels_t)
        loss.backward()
        return loss
        
    optimizer.step(eval)
    return float(temperature.item())


def apply_temperature(logits, T: float):
    """Trả về softmax(logits / T)."""
    scaled = logits / T
    max_val = np.max(scaled, axis=1, keepdims=True)
    exp_val = np.exp(scaled - max_val)
    return exp_val / np.sum(exp_val, axis=1, keepdims=True)


def fuse_conv_bn(model):
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75)."""
    model.eval()
    
    from torch.nn.utils import fuse_conv_bn_eval
    
    def fuse_module(m):
        if isinstance(m, nn.Sequential):
            for i in range(len(m) - 1):
                if isinstance(m[i], nn.Conv2d) and isinstance(m[i+1], nn.BatchNorm2d):
                    # Replace with fused conv
                    fused = fuse_conv_bn_eval(m[i], m[i+1])
                    m[i] = fused
                    m[i+1] = nn.Identity()
                    
        for name, child in m.named_children():
            fuse_module(child)
            
    fuse_module(model)
    return model
