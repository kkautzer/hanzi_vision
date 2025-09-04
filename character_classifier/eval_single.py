import cv2
import torch
from torchvision import transforms
from character_classifier.model import ChineseCharacterCNN
from character_classifier.scripts.crop_image import crop_image
import json
import numpy as np
import argparse
    
def evaluate(image, model_name):
    """
    Evaluates a single input image based on the trained model, and returns the closest matching
    Hanzi character.

    Args:
        image (NumPy Array): a single image, as a NumPy array of the shape (H, W)
        model_name (str): the name of the model that `image` will be evaluated on 
        epoch_num (int): the epoch number to evaluate this model on, or the best available if not provided

    Returns:
        tuple( int, char ): a tuple of the predicted character index, and the character itself
    """
    
    try:
        metadata_location = f'./character_classifier/{"models/metadata" if __name__=="__main__" else "/exports/metadata_public"}/{model_name}-metadata.json'
        with open(metadata_location, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        n_chars = metadata['nchars']
        if metadata['epochs'] == 0:
            print("Model has no epochs to evaluate!")
            return
    except Exception as e:
        print("Model does not exist / Failed to fetch model data!")
        print(e)
        return
    
    img_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(f"./character_classifier/classes/top-{n_chars}-classes.txt", 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]
        
    num_classes = len(class_names)

    model = ChineseCharacterCNN(num_classes=num_classes).to(device)
    path_to_model = f"./character_classifier/models/checkpoints/best/{model_name}_best.pth"
    model.load_state_dict(torch.load(path_to_model, map_location=device))
            
    # Series of transformations to apply to normalize each input image
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),  # Ensure single channel (grayscale)
        transforms.Resize((img_size, img_size)),      # Resize to a consistent size
        transforms.ToTensor(),                        # Convert image to PyTorch tensor
        transforms.Normalize((0.5,), (0.5,))          # Normalize pixel values to mean=0.5, std=0.5
    ])
            
    model.eval()
    with torch.no_grad():
        # apply the newly defined transformations in addition to originals
        output = model(transform(crop_image(image)).unsqueeze(0))
            
        predicted = torch.argmax(output, 1).item()
        
    # print(f"[{datetime.now()}] Finished Evaluation!")
    return (predicted, class_names[predicted])


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Parameters for Evaluating a Model on Chinese Hanzi Characters")
    
    parser.add_argument("--name", type=str, help="Name of the model to use for evaluating images")
    
    model_name = parser.parse_args().name
    
    # replace with path to any image file
    images = [ # assuming file run from monorepo (using VS code, usually the case for name=main)
        # typed fonts, black text & white background
        cv2.imread('./character_classifier/custom_test_images/IMG_1949.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_1949-2.jpg'),

        # computer drawn, black text & white background
        cv2.imread('./character_classifier/custom_test_images/IMG_2000.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2001.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2001-2.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2002.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2003.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2004.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2005.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2006.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2007.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2008.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2009.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2010.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2011.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2012.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2013.jpg'), 
        cv2.imread('./character_classifier/custom_test_images/IMG_2014.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2015.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2016.jpg'),
        
        # computer drawn, non-BW background / foreground colors
        cv2.imread('./character_classifier/custom_test_images/IMG_2100.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2101.jpg'),
        
        # hand drawn
        cv2.imread('./character_classifier/custom_test_images/IMG_1975.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_1976.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_1977.jpg'),
        
        # actual training samples
        cv2.imdecode(np.fromfile('./character_classifier/data/filtered/top-500/train/十/0001.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./character_classifier/data/filtered/top-500/train/只/0002.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./character_classifier/data/filtered/top-500/train/回/0003.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./character_classifier/data/filtered/top-500/train/教/0004.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./character_classifier/data/filtered/top-500/train/书/0005.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./character_classifier/data/filtered/top-500/train/书/0006.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./character_classifier/data/filtered/top-500/train/书/0007.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./character_classifier/data/filtered/top-500/train/书/0008.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./character_classifier/data/filtered/top-500/train/书/0009.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./character_classifier/data/filtered/top-500/train/书/0010.png', np.uint8), cv2.IMREAD_UNCHANGED),
    ]
    
    
    for image in images: 
        ## add channels dimension if not present
        if (len(np.shape(image)) == 2):
            image = image[..., np.newaxis]
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        print(evaluate(image, model_name))