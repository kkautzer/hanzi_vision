import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from scripts.create_filtered_set import create_filtered_set
from scripts.generate_whitelist import generate_whitelist
from model import ChineseCharacterCNN
import os
from datetime import datetime
import argparse
import json

# Configuration Parameters
with open("character_classifier/default_train_params.json", 'r') as file:
    defaults = json.load(file)

print(defaults)
parser = argparse.ArgumentParser(description="Parameters for Training a Model on Chinese Hanzi Characters")
parser.add_argument('--nchars', type=int, default=defaults['num_characters'], help="Number of characters to include in the training for this model.")
parser.add_argument('--lr', type=float, default=defaults['learning_rate'], help="Learning rate to use throughout the training process.")
parser.add_argument('--epochs', type=int, default=defaults['num_epochs'], help='The epoch number at which this training will end at [inclusive]')
parser.add_argument('--name', type=str, default=defaults['model_name'], help='The name under which data revolving around this model will be stored.')
parser.add_argument('--initial', type=int, default=defaults['initial_epoch'], help="The epoch value with which to start this model with [inclusive]")
parser.add_argument('--pretrained', type=str, default=defaults['saved_pretrained_model_path'], help='The path to a `.pth` file containing the weights of a compatible pretrained model.')

args = parser.parse_args()

data_dir = defaults['data_dir']
batch_size = defaults['batch_size']
img_size = defaults['img_size']

num_characters = args.nchars
learning_rate = args.lr
num_epochs = args.epochs

model_name = args.name
initial_epoch = args.initial
saved_pretrained_model_path = args.pretrained

# # # # # if (initial_epoch > 1): ## start from a specified epoch
# # # # #     saved_pretrained_model_path = f"checkpoints/training/{model_name}/tr_epoch{initial_epoch-1}-{model_name}.pth"
# # # # # elif (initial_epoch == -1): ## start from the best model
# # # # #     saved_pretrained_model_path = f"checkpoints/{model_name}.pth"
# # # # # else: ## do not load a pretrained model
# # # # #     saved_pretrained_model_path = ""

# End of Configuration Parameters

 
# Generate the whitelist for the n most common characters
print(f"[{datetime.now()}] Initializing whitelist for the {num_characters} most common characters")
generate_whitelist(num_characters)
print(f"[{datetime.now()}] Successfully initialized whitelist")

 
# Initialize filtered directories
print(f"[{datetime.now()}] Initializing filtered directories...")
create_filtered_set('data/whitelist.txt')
print(f"[{datetime.now()}] Successfully initialized filtered directories")

# Detect device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[{datetime.now()}] Using device: {device}")

# Load data loaders
print(f"[{datetime.now()}] Loading data loaders...")
train_loader, val_loader, test_loader, class_names = get_dataloaders(data_dir, batch_size, img_size)
num_classes = len(class_names)

# Initialize model
print(f"[{datetime.now()}] Initializing model...")
model = ChineseCharacterCNN(num_classes=num_classes).to(device)
if len(saved_pretrained_model_path) > 0:
    model.load_state_dict(torch.load(saved_pretrained_model_path, map_location=device))
print(f"[{datetime.now()}] Finished model initialization")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Track the highest validation accuracy for storing weights
highest_val_accuracy = -1

if initial_epoch <= 1:
    print(f"[{datetime.now()}] Beginning training...")
else:
    print(f"[{datetime.now()}] Resuming training from epoch {initial_epoch}...")
# Training loop
for epoch in range(initial_epoch, num_epochs+1):
    print(f"[{datetime.now()}] -- Beginning epoch {epoch} of {num_epochs} --")
    model.train()  # Set model to training mode
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()        # Clear gradients
        outputs = model(images).logits      # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()             # Backpropagation
        optimizer.step()            # Update model weights

        running_loss += loss.item()
        
    avg_loss = running_loss / len(train_loader)
    print(f"[{datetime.now()}] Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Validation phase
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    
    print(f"[{datetime.now()}] Validation Accuracy: {val_accuracy:.2f}%")
    
    # Save training data after each epoch model checkpoint
    os.makedirs(f"./checkpoints/training/{model_name}", exist_ok=True)
    torch.save(model.state_dict(), f"./checkpoints/training/{model_name}/tr_epoch{epoch}.pth")
    print(f"[{datetime.now()}] Model saved to ./checkpoints/training/{model_name}/train_epoch_{epoch}.pth")
    if (val_accuracy > highest_val_accuracy):
        highest_val_accuracy = val_accuracy
        os.makedirs(f"./checkpoints/best", exist_ok=True)
        torch.save(model.state_dict(), f"./checkpoints/best/{model_name}_best.pth")
        print(f"[{datetime.now()}] Model saved to ./checkpoints/best/{model_name}_best.pth")

# Save trained model checkpoint
# os.makedirs("./checkpoints", exist_ok=True)
# torch.save(model.state_dict(), f"./checkpoints/{model_name}_.pth")
# print(f"[{datetime.now()}] Model saved to ./checkpoints/{model_name}.pth")


### TODO Add main method, allowing to specify model name, number of epochs, initial epoch & path to existing weights to continue from
### TODO Put this in a separate main.py file, and call each component from there (whitelist generation & change detection, this training script, ...)