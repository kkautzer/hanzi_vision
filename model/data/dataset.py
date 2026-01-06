from datetime import datetime
from pathlib import Path
import csv
import hashlib
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as transforms


# -------------------------
# Dataset
# -------------------------

class CASIATrainValDataset(Dataset):
    """
    Deterministic train/val split derived from CASIA TRAIN set only,
    using casia_splits.csv as the authoritative source.
    """

    def __init__(
        self,
        root,
        splits_csv,
        split,                 # "train" or "val"
        class_to_idx,
        transform=None,
        val_ratio=0.1
    ):
        assert split in {"train", "val"}
        self.samples = []
        self.transform = transform

        root = Path(root)
        splits_csv = Path(splits_csv)

        # Load CASIA splits
        casia_split = {}
        with open(splits_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader: ## match sample IDs to splits
                sid = row["sample_id"].split("-")[0]
                casia_split[sid] = row["split"]

        for cls, cls_idx in class_to_idx.items():
            class_dir = root / cls
            if not class_dir.exists():
                continue

            for img_path in sorted(class_dir.glob("*.png")):
                sample_id = img_path.stem  # must match your saved naming scheme
                if casia_split.get(sample_id) != "train":
                    continue

                h = int(hashlib.md5(sample_id.encode()).hexdigest(), 16)
                is_val = (h % 100) < int(val_ratio * 100)

                if split == "val" and is_val:
                    self.samples.append((img_path, cls_idx))
                elif split == "train" and not is_val:
                    self.samples.append((img_path, cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, label


# -------------------------
# Dataloaders
# -------------------------

def get_dataloaders(
    data_dir,
    batch_size=64,
    img_size=64,
    val_ratio=0.1,
    num_workers=0
):
    data_dir = Path(data_dir)
    splits_csv = Path("model/data/casia_splits.csv")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=[0.8, 1.1], contrast=[0.85, 1.05])
        ], p=0.5),
        transforms.RandomApply([
            transforms.RandomAffine(degrees=5, scale=(0.90, 1.05),
                                    translate=(0.035, 0.035))
        ], p=0.65),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    eval_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print(f"[{datetime.now()}] Loading TRAIN/VAL from CASIA train split...")

    # Use ImageFolder only to define class structure
    base_train = datasets.ImageFolder(root=data_dir / "train")
    class_names = base_train.classes
    class_to_idx = base_train.class_to_idx

    train_set = CASIATrainValDataset(
        root=data_dir / "train",
        splits_csv=splits_csv,
        split="train",
        class_to_idx=class_to_idx,
        transform=train_transform,
        val_ratio=val_ratio
    )

    val_set = CASIATrainValDataset(
        root=data_dir / "train",
        splits_csv=splits_csv,
        split="val",
        class_to_idx=class_to_idx,
        transform=eval_transform,
        val_ratio=val_ratio
    )

    print(f"[{datetime.now()}] Loading TEST set...")

    test_set = datasets.ImageFolder(
        root=data_dir / "test",
        transform=eval_transform
    )
    test_set.class_to_idx = class_to_idx
    test_set.classes = class_names

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, test_loader, class_names
