import torch
from model.data.dataset import get_dataloaders
from model.model import ChineseCharacterCNN
from datetime import datetime
import json
import argparse

def test_model(model_name, epoch=-1):
    
    # get nchars based on model_name and stored metadata
    try:
        with open(f'./model/exports/metadata/{model_name}-metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        n_chars = metadata['nchars']
        max_epoch = metadata['epochs']
        architecture = metadata['architecture']
        epoch = max_epoch if epoch > max_epoch else epoch ## no out of bounds epoch values
        if max_epoch == 0:
            print("Model has no epochs to evaluate!")
            return
    except Exception as e:
        print("Model does not exist / Failed to fetch model data!")
        print(e)
        return
    
    if epoch < 1:
        model_path = f'./model/exports/checkpoints/{model_name}_best.pth'
    else:
        model_path = f'./model/checkpoints/{model_name}/tr_epoch{epoch}.pth'
    data_dir = f"./model/data/filtered/top-{n_chars}"  # Adjust based on location
    batch_size = 64
    img_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{datetime.now()}] Using device: {device}")
    _, _, test_loader, class_names = get_dataloaders(data_dir, batch_size, img_size)
    num_classes = len(class_names)
    model = ChineseCharacterCNN(architecture=architecture, num_classes=num_classes).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"[{datetime.now()}] Finished model initialization")

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


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Arguments for Calculating Test Accuracy on a Model")
    parser.add_argument("--name", type=str, help="Name of the Model to run Test Data Through")
    parser.add_argument("--epoch", type=int, default=-1, help="[Optional] Epoch Number to Run Test Data Through. If none provided, will use the epoch with the highest validation accuracy")
    
    args = parser.parse_args()
    name = args.name
    epoch = args.epoch
    
    if not name:
        print("Must provide a model name using the `--name` flag. Program terminated.")
    else:
       test_model(name, epoch)