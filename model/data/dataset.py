from datetime import datetime
import csv
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as transforms


# ============================================================
# Dataset that obeys casia_splits.csv (NO implicit re-splitting)
# ============================================================

class CASIASplitDataset(Dataset):
    """
    Dataset that loads samples strictly according to casia_splits.csv.
    """

    def __init__(self, data_dir, split, class_to_idx, transform=None):
        assert split in {"train", "val", "test"}
        self.samples = []
        self.transform = transform

        data_dir = Path(data_dir)
        processed_dir = data_dir / split
        splits_csv = "model/data/casia_splits.csv"

        if not Path(splits_csv).exists():
            raise FileNotFoundError(f"Missing split file: {splits_csv}")

        # Load split assignments
        with open(splits_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                if row["split"] != split:
                    continue

                sample_id = row["sample_id"]
                filename, char, idx = sample_id.split(":")
                filenum = int(filename.split("-")[0])  # use gnt filename without extension
                img_path = data_dir / split / char / f"{filenum:04d}.png"

                if not img_path.exists():
                    continue

                self.samples.append((img_path, class_to_idx[char]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")

        if self.transform:
            img = self.transform(img)

        return img, label


# ============================================================
# DataLoader factory (keeps your API unchanged)
# ============================================================

def get_dataloaders(data_dir, batch_size=64, img_size=64):
    """
    Returns PyTorch DataLoader objects for training, validation, and test sets.
    """

    data_dir = Path(data_dir)

    # --------------------------------------------------------
    # Transforms (UNCHANGED from your original version)
    # --------------------------------------------------------

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=[0.8, 1.1],
                contrast=[0.85, 1.05],
            ),
        ], p=0.5),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=5,
                scale=(0.90, 1.05),
                translate=(0.035, 0.035),
            ),
        ], p=0.65),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # --------------------------------------------------------
    # Class index construction (authoritative)
    # --------------------------------------------------------

    print(f"[{datetime.now()}] Reading class list from processed/train...")

    train_root = data_dir / "train"
    class_names = sorted(d.name for d in train_root.iterdir() if d.is_dir())
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    # --------------------------------------------------------
    # Dataset construction (CSV-driven)
    # --------------------------------------------------------

    print(f"[{datetime.now()}] Loading TRAIN dataset...")

    train_set = CASIASplitDataset(
        data_dir=data_dir,
        split="train",
        class_to_idx=class_to_idx,
        transform=train_transform
    )

    print(f"[{datetime.now()}] Loading VAL dataset...")
    val_set = CASIASplitDataset(
        data_dir=data_dir,
        split="val",
        class_to_idx=class_to_idx,
        transform=eval_transform
    )
    print(f"[{datetime.now()}] Loading TEST dataset...")

    test_set = CASIASplitDataset(
        data_dir=data_dir,
        split="test",
        class_to_idx=class_to_idx,
        transform=eval_transform
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    print(f"[{datetime.now()}] Wrapping DataLoader...")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"[{datetime.now()}] Successfully loaded all datasets.")
    print(f"  Train samples: {len(train_set)}")
    print(f"  Val samples  : {len(val_set)}")
    print(f"  Test samples : {len(test_set)}")

    return train_loader, val_loader, test_loader, class_names
