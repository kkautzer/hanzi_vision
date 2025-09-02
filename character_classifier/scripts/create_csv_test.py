from character_classifier.dataset import get_dataloaders
from character_classifier.model import ChineseCharacterCNN
import torch
from datetime import datetime
import os
import argparse
import json

# initialization - get test images, define model class, etc.
def initialize(nchars):
    
    data_dir = f'./character_classifier/data/filtered/top-{nchars}'
    batch_size = 64
    img_size = 64
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{datetime.now()}] Using device: {device}")
    _, _, test_loader, _ = get_dataloaders(data_dir, batch_size, img_size)
    model = ChineseCharacterCNN(num_classes=nchars).to(device)
    
    return device, model, test_loader

# get all paths to specified model training epochs
def get_model_paths(model_name):
    data_dir = f"./character_classifier/models/checkpoints/training/{model_name}"
    files = os.listdir(data_dir)
    return [f"{data_dir}/{file}" for file in files]

# for each epoch, calculate the test accuracy (test.py)
def test_model(model, model_weight_path, loader, device):
    
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total

    return test_accuracy

    # format as model_name, epoch, test_accuracy
    
def record_to_csv(model_name):
    try:
        with open(f'./character_classifier/models/metadata/{model_name}-metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        nchars = metadata['nchars']
        if metadata['epochs'] == 0:
            print("Model has no epochs to evaluate!")
            return
    except Exception as e:
        print("Failed to fetch model data!")
        print(e)
        return
        
    print(f"[{datetime.now()}] Initializing model and test data...")
    device, model, loader = initialize(nchars)
    print(f"[{datetime.now()}] Initialization successful!")
    print(f"[{datetime.now()}] Retrieving model epoch weights...")
    paths = get_model_paths(model_name)
    print(f"[{datetime.now()}] Retrieved paths!")
    print(f"[{datetime.now()}] Beginning model testing...")
    data = []
    for path in paths:
        # get epoch from the path
        # expected form of `./character_classifier/models/checkpointstraining/<model_name>/tr_epoch<epoch>.pth`
        epoch = path.split('/tr_epoch')[-1].split('.pth')[0]
        print(f"[{datetime.now()}] -- Testing epoch {epoch} of {len(paths)} --")
        accuracy = test_model(model, path, loader, device)
        print(f"[{datetime.now()}] Epoch [{epoch}/{len(paths)}], Test Accuracy: {accuracy}")
        data.append([f"\"{str(model_name)}\"", str(epoch), str(accuracy)])
        
    print(f"[{datetime.now()}] Recording data to CSV...")
    # write test accuracy to csv (`./character_classifier/exports/test/<model_name>-test.csv`)
    with open(f"./character_classifier/exports/test/{model_name}-test.csv", 'w', encoding='utf-8') as f:
        lines = ["model_name,epoch,test_accuracy\n"]
        for entry in data:
            lines.append(f"{",".join(entry)}\n")
        f.writelines(lines)
    print(f"[{datetime.now()}] Successfully recorded data to ./character_classifier/exports/test/{model_name}-test.csv")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Parameters to create a CSV file of test accuracies")
    parser.add_argument('--name', type=str, default="model", help="Name of model to create test CSV file for")

    model_name = parser.parse_args().name
    
    record_to_csv(model_name)
