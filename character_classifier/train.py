import torch
import torch.nn as nn
import torch.optim as optim
from character_classifier.dataset import get_dataloaders
from character_classifier.scripts.create_filtered_set import create_filtered_set
from character_classifier.scripts.generate_whitelist import generate_whitelist
from character_classifier.model import ChineseCharacterCNN
import os
from datetime import datetime
import argparse
import json

def printLogAndConsole(content):
    print(content)
    try:
        with open(f"./character_classifier/logs/{log_file_name}.txt", 'a') as f:
            print(content, file=f)
    except Exception: ## if the `log_file_name` has not been defined, we cannot write to file
        print(f" {'\033[31m'} Failed to write previous statement to the log! {'\033[0m'}")
    
    
# Configuration Parameters
print("Loading Configuration Data...")

parser = argparse.ArgumentParser(description="Parameters for Training a Model on Chinese Hanzi Characters")

parser.add_argument('--name', type=str, default=f"model", help='[Required for all] The name under which data revolving around this model will be stored.')
parser.add_argument('--epochs', type=int, default=10, help='[Required for all] The number of epochs to perform')
parser.add_argument('--lr', type=float, default=0.0125, help="[Required for all] Learning rate to use throughout the training process.")

parser.add_argument('--nchars', type=int, default=5, help="[Required for new models] Number of characters to include in the training for this model.")
# parser.add_argument('--thresholded', type=bool, default=True, help='[Required for new models] Whether or not to use edge detected and thresholded images, instead of the standard training images. Defaults to True')

parser.add_argument('--resume', type=bool, default=False, help='[Optional] Whether this training will be a brand new model (set to False), or based on another model (set to True). Defaults to False')
parser.add_argument('--resepoch', type=int, default=0, help='[Optional] The epoch number to resume training from. Set to -1 to resume from the epoch with the highest validation accuracy; set to 0 to resume from the most recent epoch (default).') # epoch resuming from (default to the most recent, NOT best (use 0 to resume from best))
parser.add_argument('--resname', type=str, default="", help='[Optional] The name of the model to base training on - defaults to the model specified under "--name", but can be another model if desired.')

args = parser.parse_args()

model_name = args.name
num_epochs = args.epochs
learning_rate = args.lr

if not args.resume: # starting with a completely untrained model
    num_characters = args.nchars
    thresholded = False#args.thresholded
    
    saved_pretrained_model_path = '' ## bypass NameError errors for undefined field

else: # resuming from a pretrained weights
    # optionally, scaffold weights from one model to use with another
    if len(args.resname) > 0:
        load_model_name = args.resname
    else:
        load_model_name = model_name
    
    # try-except to check if metadata file actually exists
    try:
        with open(f'./character_classifier/models/metadata/{load_model_name}-metadata.json', 'r', encoding='utf-8') as f:
            # get nchars & thresholded from metadata
            metadata = json.load(f)
            num_characters = metadata['nchars']
            thresholded = metadata['threshold']
            pass
    except Exception as e:
        print(e)
        print("Error -- Cannot resume training from a model that does not exist / has no completed epochs!")
        quit()   
        
    if args.resepoch < 0:
        # best
        initial_epoch = metadata['max_val_epoch']
        saved_pretrained_model_path = f"./character_classifier/models/checkpoints/best/{load_model_name}_best.pth"
    elif args.resepoch == 0 or args.resepoch > metadata['epochs']:
        # most recent
        initial_epoch = metadata['epochs']
        saved_pretrained_model_path = f"./character_classifier/models/checkpoints/training/{load_model_name}/tr_epoch{metadata['epochs']}.pth"
    else:
        # specified epoch
        initial_epoch = args.resepoch
        saved_pretrained_model_path = f'./character_classifier/models/checkpoints/training/{load_model_name}/tr_epoch{args.resepoch}.pth'


data_dir = f"character_classifier/data/filtered/top-{num_characters}"
batch_size = 64
img_size = 64


## anything printed before this line will not be logged on file ##
## use the standard `print()` function, else an indicator message will be printed to the console ##

## if not resuming, or if scaffolding from a previous model, check that the new model name is not in use
if (not args.resume or (args.resume and args.resname != model_name)):
    initial_epoch = 1
    if os.path.isfile(f'./character_classifier/models/metadata/{model_name}-metadata.json'):
        print("Cannot use a model name that is already in use - please choose another using the `--name` parameter and try again!")
        quit()

log_file_name = f"log-{model_name}" # save log files by name of the model
printLogAndConsole("------------------")
printLogAndConsole("Successfully loaded configuration data!")
# printLogAndConsole(f"Defaults: {defaults}")
printLogAndConsole(f"Model Configuration: { {
    "batch_size": batch_size,
    "img_size": img_size,
    "data_dir": data_dir,
    "model_name": model_name,
    "num_characters": num_characters,
    "learning_rate": learning_rate,
    "num_epochs": num_epochs,
    "initial_epoch": initial_epoch,
    "saved_pretrained_model_path": saved_pretrained_model_path,
    "thresholded": thresholded
} }")



max_val_accuracy = 0
max_val_epoch = 0
try: # load the max validation accuracy from metadata file if it exists
    with open(f'./character_classifier/models/metadata/{model_name}-metadata.json', 'r', encoding='utf-8') as f:
        initial_metadata = json.load(f)
        max_val_accuracy = initial_metadata['max_val_accuracy']
        max_val_epoch = initial_metadata['max_val_epoch']
except FileNotFoundError as e:
    pass

# End of Configuration Parameters

 
# Generate the whitelist for the n most common characters
printLogAndConsole(f"[{datetime.now()}] Initializing whitelist for the {num_characters} most common characters...")
generate_whitelist(num_characters)
printLogAndConsole(f"[{datetime.now()}] Successfully initialized whitelist")

 
# Initialize filtered directories
printLogAndConsole(f"[{datetime.now()}] Initializing filtered directories...")
create_filtered_set('./character_classifier/data/whitelist.txt')
printLogAndConsole(f"[{datetime.now()}] Successfully initialized filtered directories")

# Detect device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
printLogAndConsole(f"[{datetime.now()}] Using device: {device}")

# Load data loaders
printLogAndConsole(f"[{datetime.now()}] Loading data loaders...")
train_loader, val_loader, test_loader, class_names = get_dataloaders(data_dir, batch_size, img_size)
num_classes = len(class_names)
printLogAndConsole(f"[{datetime.now()}] Finished loading data loaders")

# Initialize model
printLogAndConsole(f"[{datetime.now()}] Initializing model...")
model = ChineseCharacterCNN(num_classes=num_classes).to(device)
if saved_pretrained_model_path:
    model.load_state_dict(torch.load(saved_pretrained_model_path, map_location=device))
printLogAndConsole(f"[{datetime.now()}] Finished model initialization")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if initial_epoch <= 1:
    printLogAndConsole(f"[{datetime.now()}] Beginning training...")
else:
    printLogAndConsole(f"[{datetime.now()}] Resuming training from epoch {initial_epoch}...")

# Training loop
epoch_data_export = [] # {model_name, nchars, LR, epoch, train_loss, val_acc, thresholded}
for epoch in range(initial_epoch, initial_epoch+num_epochs):
    epoch_data_export = [
        f"\"{str(model_name)}\"", # [0] model name
        str(num_characters), # [1] nchars
        str(learning_rate), # [2] learning rate
        str(epoch), # [3] epoch
        None, # [4] training loss
        None, # [5] validation accuracy
        str(thresholded) # [6] thresholded images
    ]

    printLogAndConsole(f"[{datetime.now()}] -- Beginning epoch {epoch} of {num_epochs} --")
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
    printLogAndConsole(f"[{datetime.now()}] Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}")
    epoch_data_export[4] = str(avg_loss)

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
    
    printLogAndConsole(f"[{datetime.now()}] Validation Accuracy: {val_accuracy:.2f}%")
    epoch_data_export[5] = str(val_accuracy)
    
    # Save training data after each epoch model checkpoint
    with open("./character_classifier/exports/training_data.csv", "a") as f:
        f.write(f"{(",").join(epoch_data_export)}\n")
        
    printLogAndConsole(f"[{datetime.now()}] Logged epoch info to ./character_classifier/exports/training_data.csv")
    os.makedirs(f"./character_classifier/models/checkpoints/training/{model_name}", exist_ok=True)
    torch.save(model.state_dict(), f"./character_classifier/models/checkpoints/training/{model_name}/tr_epoch{epoch}.pth")
    printLogAndConsole(f"[{datetime.now()}] Model saved to ./character_classifier/models/checkpoints/training/{model_name}/train_epoch_{epoch}.pth")
    
    if (val_accuracy > max_val_accuracy):
        max_val_accuracy = val_accuracy
        max_val_epoch = epoch
        os.makedirs(f"./character_classifier/models/checkpoints/best", exist_ok=True)
        torch.save(model.state_dict(), f"./character_classifier/models/checkpoints/best/{model_name}_best.pth")
        printLogAndConsole(f"[{datetime.now()}] Model saved to ./character_classifier/models/checkpoints/best/{model_name}_best.pth")

    # update metadata (name, completed epochs, highest validation accuracy, highest val epoch, thresholded images [T/F] )
    with open(f'./character_classifier/models/metadata/{model_name}-metadata.json', 'w', encoding='utf-8') as f:
        metadata_json = {
            "model_name": model_name,
            "nchars": num_characters,
            "epochs": epoch,
            "max_val_accuracy": max_val_accuracy,
            "max_val_epoch": max_val_epoch,
            "threshold": thresholded,
        }
        
        json.dump(metadata_json, f, indent=4)
    printLogAndConsole(f"[{datetime.now()}] Metadata updated at ./character_classifier/models/metadata/{model_name}-metadata.json")

print(f"[{datetime.now()}] Training Completed!")