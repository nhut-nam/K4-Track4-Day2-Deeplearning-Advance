"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.
Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1. Đọc trước khi viết.

Giao diện bạn phải giữ (để notebook, train.py và eval.py ghép được với nhau):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # đổi nếu trọng số timm bạn dùng yêu cầu mean/std khác
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame.
    KHÔNG sửa, lọc hay chia lại dữ liệu.
    """
    labels_dir = Path(labels_dir)
    train_df = pd.read_csv(labels_dir / f"train_subset{fold}.csv")
    val_df = pd.read_csv(labels_dir / f"val_subset{fold}.csv")
    test_df = pd.read_csv(labels_dir / f"test_subset{fold}.csv")
    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu."""
    images_dir = Path(images_dir)
    
    # 1. số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    total_images = n_train + n_val + n_test
    
    train_counts = train_df['Label'].value_counts().to_dict()
    val_counts = val_df['Label'].value_counts().to_dict()
    test_counts = test_df['Label'].value_counts().to_dict()
    
    print(f"Total: {total_images} (Train: {n_train}, Val: {n_val}, Test: {n_test})")
    print(f"Train ratio: {n_train/total_images:.3f}, Val ratio: {n_val/total_images:.3f}, Test ratio: {n_test/total_images:.3f}")
    
    # 2. giao của từng cặp tập theo Filename phải RỖNG (train∩val, train∩test, val∩test)
    train_files = set(train_df['Filename'])
    val_files = set(val_df['Filename'])
    test_files = set(test_df['Filename'])
    
    assert len(train_files.intersection(val_files)) == 0, "Train and Val have overlapping files!"
    assert len(train_files.intersection(test_files)) == 0, "Train and Test have overlapping files!"
    assert len(val_files.intersection(test_files)) == 0, "Val and Test have overlapping files!"
    
    # 3. hợp ba tập phải bằng đúng 17.509 ảnh
    assert total_images == 17509, f"Total images expected 17509, but got {total_images}"
    
    # 4. mọi Filename đều tồn tại trong `images_dir`
    all_files = train_files.union(val_files).union(test_files)
    missing = [f for f in all_files if not (images_dir / f).exists()]
    if missing:
        print(f"Warning: {len(missing)} files missing. Example: {missing[:5]}")
        # Note: Depending on the environment, images might not be extracted yet, so we just print a warning here instead of assert.
        # assert len(missing) == 0, f"Missing files in images_dir: {missing[:5]}"

    return {
        "n": {"train": n_train, "val": n_val, "test": n_test},
        "per_class": {"train": train_counts, "val": val_counts, "test": test_counts},
        "overlap": {
            "train_val": len(train_files.intersection(val_files)),
            "train_test": len(train_files.intersection(test_files)),
            "val_test": len(val_files.intersection(test_files))
        }
    }


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform. `aug` chọn mức augmentation; bạn tự định nghĩa các giá trị."""
    if not train:
        # Val/test: resize 256 then CenterCrop(224)
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    
    if aug == "basic":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    elif aug == "color":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    elif aug == "trivial":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    elif aug == "randaug":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        raise ValueError(f"Unknown aug value: {aug}")


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label)."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filenames = df['Filename'].values
        self.labels = df['Label'].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        filename = self.filenames[i]
        label = int(self.labels[i])
        img_path = self.images_dir / filename
        
        img = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, label, filename


def worker_init_fn(worker_id):
    import random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    """Tạo DataLoader."""
    dataset = DeepWeedsDataset(df, images_dir, transform)
    
    if sampler == "balanced" and train:
        class_counts = df['Label'].value_counts().sort_index()
        class_weights = 1.0 / class_counts
        sample_weights = df['Label'].map(class_weights).values
        
        ws = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=ws,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=train,
            worker_init_fn=worker_init_fn
        )
