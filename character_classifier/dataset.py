from datetime import datetime
import os
import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader

# Function to create data loaders for train, validation, and test sets
def get_dataloaders(data_dir, batch_size=64, img_size=64):
    """
    Returns PyTorch DataLoader objects for training, validation, and test sets.

    Parameters:
        data_dir (str): Path to the root data directory containing 'train', 'val', 'test' folders.
        batch_size (int): Number of samples per batch to load.
        img_size (int): Size to which each image will be resized (img_size x img_size).

    Returns:
        train_loader, val_loader, test_loader (DataLoader): Dataloaders for training, validation, and testing.
        class_names (list): List of class (character) names.
    """

    # Transformations for TRAINING SET ONLY (includes brightness / contrast augmentation)
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure single channel (grayscale)
        transforms.Resize((img_size, img_size)),      # Resize to a consistent size
        transforms.RandomApply([ # only apply brightness / contrast on half the time
            transforms.ColorJitter( # Brightness / contrast manipulations
                brightness=[0.8,1.1], contrast=[0.85,1.05], saturation=None, hue=None
            ),
        ], p=0.5),
        transforms.RandomApply([ # only apply these about 65% of samples
            transforms.RandomAffine(
                degrees=5, scale=(0.90, 1.05), shear=0, translate=(0.035, 0.035)
            ), # Translation manipulations
        ], p=0.65),
        transforms.ToTensor(),                        # Convert image to PyTorch tensor
        transforms.Normalize((0.5,), (0.5,))          # Normalize pixel values to mean=0.5, std=0.5
    ])
    
    # Standard transformations for all images (no augmentations)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # Ensure single grayscale channel
        transforms.Resize((img_size, img_size)), # Resize to consistent size
        transforms.ToTensor(), # Convert to Tensor
        transforms.Normalize((0.5,), (0.5,)) # Normalize pixel values
    ])
    

    print(f"[{datetime.now()}] Loading TRAIN set...")
    # Load the training dataset using ImageFolder
    train_set = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transform)
    print(f"[{datetime.now()}] Successfully loaded TRAIN set. Now loading EVAL set...")
    val_set = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)
    print(f"[{datetime.now()}] Successfully loaded EVAL set. Now loading TEST set...")
    test_set = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)
    print(f"[{datetime.now()}] Successfully loaded TEST set")
    
    print(f"[{datetime.now()}] Wrapping loaders in DataLoader objects...")
    # Wrap datasets in DataLoader objects for batch processing and shuffling
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    print(f"[{datetime.now()}] Successfully wrapped all sets.")
    
    # Extract the list of class names from the training dataset
    class_names = train_set.classes
    
    return train_loader, val_loader, test_loader, class_names
