"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

PSEUDO-CODE: chỉ có khung (cấu hình và quy ước đặt tên file); bạn tự hoàn thiện mọi hàm có
`raise NotImplementedError` và các bước TODO trong `run()`. Dùng MỘT hàm `run(cfg)` cho mọi cấu hình
(RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số dùng để chọn checkpoint (macro-F1 val) phải tính bằng eval.compute_metrics của repo gốc,
để cùng định nghĩa với lúc chấm:
    sys.path.insert(0, "<thư mục chứa eval.py>");  from eval import compute_metrics
"""
from __future__ import annotations

import sys
import os
import time
import json
import random
import argparse
from dataclasses import dataclass, fields, asdict
from pathlib import Path
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

from dataset import load_split, check_split, build_transforms, make_loader
from model import build_model, param_groups, count_params, count_gmacs, freeze_backbone
from losses import build_criterion, class_weights, mix_batch, mixed_loss

# Thêm đường dẫn tới eval.py ở ngoài (thư mục mẹ)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import save_predictions, compute_metrics


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug ...
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg: Config):
    """AdamW với 3 nhóm tham số (xem model.param_groups)."""
    groups = param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return optim.AdamW(groups)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0 (slide trang 55)."""
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)
    
    if warmup_steps > 0:
        warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps))
        scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    return scheduler


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56)."""

    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {}
        for name, param in model.state_dict().items():
            if param.dtype.is_floating_point:
                self.shadow[name] = param.clone().detach()
            else:
                self.shadow[name] = param.clone()

    def update(self, model) -> None:
        with torch.no_grad():
            for name, param in model.state_dict().items():
                if name in self.shadow:
                    if param.dtype.is_floating_point:
                        self.shadow[name].copy_(self.decay * self.shadow[name] + (1.0 - self.decay) * param.detach())
                    else:
                        self.shadow[name].copy_(param.detach())
                        
    def copy_to(self, model) -> None:
        model.load_state_dict(self.shadow)


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện. Trả về dict, ví dụ {"train_loss": ..., "lr": ...}."""
    model.train()
    if cfg.init == "frozen":
        model.apply(set_bn_eval)
        
    total_loss = 0.0
    
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        mixed = False
        if cfg.mix is not None:
            x, y = mix_batch(x, y, alpha=cfg.mix_alpha, mode=cfg.mix)
            mixed = True
            
        with autocast(enabled=cfg.amp):
            logits = model(x)
            if mixed:
                loss = mixed_loss(criterion, logits, y)
            else:
                loss = criterion(logits, y)
                
        scaler.scale(loss).backward()
        # Optional: gradient clipping
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        if ema is not None:
            ema.update(model)
            
        total_loss += loss.item() * x.size(0)
        
    avg_loss = total_loss / len(loader.dataset)
    current_lr = scheduler.get_last_lr()[0]
    return {"train_loss": avg_loss, "lr": current_lr}


def evaluate(model, loader, criterion, device):
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient."""
    model.eval()
    all_filenames = []
    all_y_true = []
    all_logits = []
    total_loss = 0.0
    
    with torch.inference_mode():
        for x, y, filenames in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            
            total_loss += loss.item() * x.size(0)
            
            all_filenames.extend(filenames)
            all_y_true.append(y.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            
    avg_loss = total_loss / len(loader.dataset)
    all_y_true = np.concatenate(all_y_true)
    all_logits = np.concatenate(all_logits, axis=0)
    
    return all_filenames, all_y_true, all_logits, avg_loss


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ đường cong training của một thí nghiệm -> curves/<exp_id>_<mota>.png (GUIDE.md mục 6.2)."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_f1 = [h["val_macro_f1"] for h in history]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(epochs, train_loss, color='tab:red', linestyle='-', label='Train Loss')
    ax1.plot(epochs, val_loss, color='tab:orange', linestyle='--', label='Val Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Macro-F1', color='tab:blue')
    ax2.plot(epochs, val_f1, color='tab:blue', linestyle='-', label='Val Macro-F1')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    fig.tight_layout()
    plt.title(title)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
    
    plt.savefig(path, dpi=150)
    plt.close()


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt."""
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    out_path = run_dir(cfg)
    out_path.mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)
    Path("curves").mkdir(exist_ok=True)
    
    with open(out_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=4)
        
    train_df, val_df, test_df = load_split(cfg.labels_dir, cfg.fold)
    check_split(train_df, val_df, test_df, cfg.images_dir)
    
    train_tf = build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    val_tf = build_transforms(train=False, img_size=cfg.img_size, aug=cfg.aug)
    
    train_loader = make_loader(train_df, cfg.images_dir, train_tf, cfg.batch_size, train=True, sampler=cfg.sampler, num_workers=cfg.num_workers)
    val_loader = make_loader(val_df, cfg.images_dir, val_tf, cfg.batch_size, train=False, num_workers=cfg.num_workers)
    
    if cfg.save_test_predictions:
        test_loader = make_loader(test_df, cfg.images_dir, val_tf, cfg.batch_size, train=False, num_workers=cfg.num_workers)
        
    model = build_model(cfg.backbone, pretrained=True, drop_rate=cfg.drop_rate, init=cfg.init)
    model = model.to(device)
    
    n_params = count_params(model)
    gmacs = count_gmacs(model, cfg.img_size)
    print(f"Params: {n_params:.2f}M | GMACs: {gmacs:.2f}G")
    
    weight_t = None
    if cfg.class_weight_beta is not None:
        counts = train_df['Label'].value_counts().sort_index().values
        weight_t = class_weights(counts, beta=cfg.class_weight_beta).to(device)
        
    criterion = build_criterion(cfg.loss, smoothing=cfg.label_smoothing, gamma=cfg.focal_gamma, weight=weight_t)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = GradScaler(enabled=cfg.amp)
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay is not None else None
    
    history = []
    best_f1 = -1.0
    best_epoch = 0
    best_model_state = None
    
    start_time = time.time()
    
    for epoch in range(1, cfg.epochs + 1):
        ep_start = time.time()
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        
        eval_model = model
        if ema is not None:
            eval_model = copy.deepcopy(model)
            ema.copy_to(eval_model)
            
        filenames, y_true, logits, val_loss = evaluate(eval_model, val_loader, criterion, device)
        y_pred = logits.argmax(axis=1)
        
        metrics = compute_metrics(y_true, y_pred, logits)
        val_f1 = metrics["macro_f1"]
        
        ep_time = time.time() - ep_start
        
        hist_dict = {
            "epoch": epoch,
            "train_loss": train_stats["train_loss"],
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
            "val_top1": metrics["top1"],
            "lr": train_stats["lr"],
            "time": ep_time
        }
        history.append(hist_dict)
        print(f"Epoch {epoch}/{cfg.epochs} | Train Loss: {train_stats['train_loss']:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Time: {ep_time:.1f}s")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            if ema is not None:
                best_model_state = copy.deepcopy(eval_model.state_dict())
            else:
                best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, out_path / "best_model.pth")
            
    total_time = time.time() - start_time
    avg_ep_time = total_time / cfg.epochs
    
    pd.DataFrame(history).to_csv(out_path / "history.csv", index=False)
    plot_curves(history, f"curves/{cfg.exp_id}_{cfg.backbone}.png", f"Exp: {cfg.exp_id} | {cfg.backbone}")
    
    # Reload best model for final evaluation
    model.load_state_dict(torch.load(out_path / "best_model.pth"))
    
    import torch.nn.functional as F
    
    # Save val predictions
    val_filenames, val_y_true, val_logits, _ = evaluate(model, val_loader, criterion, device)
    val_probs = F.softmax(torch.tensor(val_logits), dim=1).numpy()
    save_predictions(pred_path(cfg, "val"), val_filenames, val_y_true, val_probs)
    
    if cfg.save_test_predictions:
        test_filenames, test_y_true, test_logits, _ = evaluate(model, test_loader, criterion, device)
        test_probs = F.softmax(torch.tensor(test_logits), dim=1).numpy()
        save_predictions(pred_path(cfg, "test"), test_filenames, test_y_true, test_probs)
        
    return {
        "best_epoch": best_epoch,
        "macro_f1_val": best_f1,
        "avg_epoch_time": avg_ep_time,
        "params_M": n_params,
        "gmacs": gmacs
    }


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict."""
    import ast
    d = {}
    valid_fields = {f.name: f.type for f in fields(Config)}
    
    for p in pairs:
        k, v = p.split("=", 1)
        if k not in valid_fields:
            raise ValueError(f"Unknown config key: {k}")
        
        # simple parsing
        if v.lower() == "none":
            d[k] = None
        elif v.lower() == "true":
            d[k] = True
        elif v.lower() == "false":
            d[k] = False
        else:
            try:
                # ast.literal_eval handles int/float correctly
                d[k] = ast.literal_eval(v)
            except:
                d[k] = v
    return d


def main() -> None:
    """Điểm vào dòng lệnh: `python train.py --set exp_id=B01 backbone=resnet50 seed=0`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", nargs="+", help="Override config, e.g., --set seed=1 loss=focal", default=[])
    args = parser.parse_args()
    
    overrides = parse_overrides(args.set)
    cfg = Config(**overrides)
    
    print(f"Running config: {cfg}")
    res = run(cfg)
    print("Run completed. Summary:")
    print(res)


if __name__ == "__main__":
    main()
