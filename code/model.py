"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Giao diện bạn phải giữ:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

import torch
import torch.nn as nn
import timm

# Gợi ý backbone (GUIDE.md mục 2.1). Tag trọng số của timm có thể đổi theo phiên bản:
# dùng timm.list_pretrained("resnet50*") để xem, và GHI LẠI tag bạn dùng trong results.xlsx.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Tạo model phân loại 9 lớp."""
    
    use_pretrained = False if init == "scratch" else pretrained
    
    model = timm.create_model(
        name,
        pretrained=use_pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate
    )
    
    if init == "frozen":
        freeze_backbone(model)
        
    print(f"Loaded model {name} with init: {init}")
    if hasattr(model, 'default_cfg'):
        print(f"Pretrained cfg: {model.default_cfg}")
        
    # Thêm một thuộc tính để nhận biết mô hình bị freeze
    model._is_frozen = (init == "frozen")
    return model


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head."""
    # Đóng băng tất cả
    for param in model.parameters():
        param.requires_grad = False
        
    # Mở khóa cho classifier (head)
    classifier = model.get_classifier()
    if isinstance(classifier, nn.Module):
        for param in classifier.parameters():
            param.requires_grad = True
    elif isinstance(classifier, (list, tuple)):
        for cls in classifier:
            for param in cls.parameters():
                param.requires_grad = True

    # Lưu ý: Trong loop train, cần custom lại việc gọi model.train() sao cho BN vẫn eval
    # Bạn có thể làm việc này bằng cách ghi đè hàm train() hoặc viết hàm set_bn_eval


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52."""
    head_params = set()
    classifier = model.get_classifier()
    if isinstance(classifier, nn.Module):
        head_params.update(classifier.parameters())
    elif isinstance(classifier, (list, tuple)):
        for cls in classifier:
            head_params.update(cls.parameters())
            
    group_backbone_decay = []
    group_backbone_nodecay = []
    group_head = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if param in head_params:
            group_head.append(param)
        else:
            if param.ndim <= 1 or name.endswith(".bias"):
                group_backbone_nodecay.append(param)
            else:
                group_backbone_decay.append(param)
                
    groups = []
    if group_backbone_decay:
        groups.append({"params": group_backbone_decay, "lr": lr_backbone, "weight_decay": weight_decay})
    if group_backbone_nodecay:
        groups.append({"params": group_backbone_nodecay, "lr": lr_backbone, "weight_decay": 0.0})
    if group_head:
        groups.append({"params": group_head, "lr": lr_head, "weight_decay": weight_decay})
        
    return groups


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size (slide tính MAC, không phải FLOPs 2x)."""
    try:
        from thop import profile
    except ImportError:
        print("Please install thop: pip install thop")
        return 0.0
        
    dummy_input = torch.randn(1, 3, img_size, img_size, device=next(model.parameters()).device)
    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
    
    return macs / 1e9
