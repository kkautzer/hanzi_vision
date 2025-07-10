import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import ChineseCharacterCNN
import os
from datetime import datetime

data_dir = "data/filtered"  # Adjust based on location

batch_size = 64
img_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[{datetime.now()}] Using device: {device}")
train_loader, val_loader, test_loader, class_names = get_dataloaders(data_dir, batch_size, img_size)
num_classes = len(class_names)
model = ChineseCharacterCNN(num_classes=num_classes).to(device)

def test_model():
    print(f"[{datetime.now()}] Beginning Testing...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"[{datetime.now()}] Test Accuracy: {test_accuracy:.2f}%")


def main():
    # Initialize model
    model.load_state_dict(torch.load('checkpoints/best/model-GoogLeNet-500_best.pth', map_location=device))
    print(f"[{datetime.now()}] Finished model initialization")
    test_model()

if __name__ == "__main__":
    main()