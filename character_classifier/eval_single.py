import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from dataset import get_dataloaders
from model import ChineseCharacterCNN
import os
from datetime import datetime

data_dir = "data/filtered"  # Adjust based on script location
## if we want to use only top character data, use this, otherwise, use 'data/processed' for full dataset

batch_size = 64
img_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[{datetime.now()}] Using device: {device}")
train_loader, val_loader, test_loader, class_names = get_dataloaders(data_dir, batch_size, img_size)
num_classes = len(class_names)

model = ChineseCharacterCNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load('checkpoints/best/model-GoogLeNet-500_best.pth', map_location=device))
print(f"[{datetime.now()}] Finished model initialization")

# Series of transformations to apply to normalize each input image
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),  # Ensure single channel (grayscale)
    transforms.Resize((img_size, img_size)),      # Resize to a consistent size
    transforms.ToTensor(),                        # Convert image to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))          # Normalize pixel values to mean=0.5, std=0.5
])

def evaluate(image):
    """
    Evaluates a single input image based on the trained model, and returns the closest matching
    Hanzi character.

    Args:
        image (NumPy Array): a single image, as a NumPy array of the shape (H, W)

    Returns:
        tuple( int, char ): a tuple of the predicted character index, and the character itself
    """
    model.eval()
    with torch.no_grad():
        output = model(transform(image).unsqueeze(0)) ## unsqueeze to add batch dimension (=1)
        predicted = torch.argmax(output, 1).item()
    return (predicted, class_names[predicted])


if __name__ == "__main__":
    
    # replace with path to any image file
    image = cv2.imread('./character_classifier/custom_test_images/IMG_2009.jpg')
        
    print(evaluate(image))