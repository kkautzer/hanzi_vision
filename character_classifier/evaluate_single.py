import torch
from torchvision import transforms
from model import ChineseCharacterCNN
import datetime

img_size = 64
device = ''
model = ChineseCharacterCNN()
model.load_state_dict(torch.load('checkpoints/chinese_char_cnn.pth', map_location=device))
model.eval()

def evaluate(image):
    # Define a series of image transformations to apply to each image
    transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),  # Ensure single channel (grayscale)
        transforms.Resize((img_size, img_size)),      # Resize to a consistent size
        transforms.ToTensor(),                        # Convert image to PyTorch tensor
        transforms.Normalize((0.5,), (0.5,))          # Normalize pixel values to mean=0.5, std=0.5
    ])
    
    ## evaluate frame based on custom model
    with torch.no_grad():
        tensor_img = transform(input) ## transform and add batch dimension = 1
        output = model(tensor_img.to(device))    
        predicted = torch.argmax(output, dim=1).item()
        print(f"[{datetime.now()}] Predicted Label: {ID_TO_CLASS[str(predicted)]}")
    