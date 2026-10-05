"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1)."""
from __future__ import annotations

import time
import torch
import numpy as np


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây."""
    for _ in range(warmup):
        fn()
        if sync:
            sync()
            
    times = []
    for _ in range(iters):
        if sync:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        times.append(time.perf_counter() - t0)
        
    times = np.array(times) * 1000 # ms
    
    return {
        "p50": float(np.percentile(times, 50)),
        "p95": float(np.percentile(times, 95)),
        "p99": float(np.percentile(times, 99)),
        "mean": float(np.mean(times)),
        "n": iters
    }


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên."""
    model.eval()
    dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(device)
    
    if dtype == "fp16":
        model = model.half()
        dummy_input = dummy_input.half()
        
    sync = torch.cuda.synchronize if device == "cuda" else None
    
    @torch.inference_mode()
    def forward_fn():
        if dtype == "amp":
            with torch.autocast(device_type=device):
                model(dummy_input)
        else:
            model(dummy_input)
            
    stats = bench(forward_fn, warmup=warmup, iters=iters, sync=sync)
    
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    
    return {
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "p50": stats["p50"],
        "p95": stats["p95"],
        "p99": stats["p99"],
        "images_per_s": batch_size / (stats["p50"] / 1000),
        "torch": torch.__version__
    }


def tta_latency(model, k_views: int, **kw) -> dict:
    """Độ trễ của TTA K view: xấp xỉ K lần một lượt chạy (slide trang 63)."""
    base_stats = latency_report(model, batch_size=1, **kw)
    
    device = kw.get("device", "cuda")
    img_size = kw.get("img_size", 224)
    dtype = kw.get("dtype", "fp32")
    sync = torch.cuda.synchronize if device == "cuda" else None
    
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    if dtype == "fp16":
        model = model.half()
        dummy_input = dummy_input.half()
        
    @torch.inference_mode()
    def tta_fn():
        for _ in range(k_views):
            if dtype == "amp":
                with torch.autocast(device_type=device):
                    model(dummy_input)
            else:
                model(dummy_input)
                
    tta_stats = bench(tta_fn, warmup=10, iters=50, sync=sync)
    
    return {
        "k_views": k_views,
        "tta_p50": tta_stats["p50"],
        "estimated_p50": base_stats["p50"] * k_views
    }
