import cv2
import torch
from torchvision import transforms
from model.model import ChineseCharacterCNN
import json
import numpy as np
from model.scripts.crop_image import crop_image
import argparse

def evaluate(images, model_name):
    '''
    
    '''
    
    try:
        metadata_location = f'./model/exports/metadata/{model_name}-metadata.json'
        with open(metadata_location, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        n_chars = metadata['nchars']
        architecture = metadata['architecture']
        if metadata['epochs'] == 0:
            print("Model has no epochs to evaluate!")
            return
    except Exception as e:
        print("Model does not exist / Failed to fetch model data!")
        print(e)
        return
    
    batch_size = 64
    img_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(f"./model/classes/top-{n_chars}-classes.txt", 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]
        
    num_classes = len(class_names)

    model = ChineseCharacterCNN(architecture=architecture, num_classes=num_classes).to(device)
    path_to_model = f"./model/exports/checkpoints/{model_name}_best.pth"
    model.load_state_dict(torch.load(path_to_model, map_location=device))


    # Series of transformations to apply to normalize each input image
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),  # Ensure single channel (grayscale)
        transforms.Resize((img_size, img_size)),      # Resize to a consistent size
        transforms.ToTensor(),                        # Convert image to PyTorch tensor
        transforms.Normalize((0.5,), (0.5,))          # Normalize pixel values to mean=0.5, std=0.5
    ])
        
    
    images_transformed = [transform(crop_image(image)) for image in images]
    images_eval = torch.stack(images_transformed)
    
    model.eval()
    with torch.no_grad():
        output = model(images_eval) ## unsqueeze to add batch dimension (=1)
        predicted = [torch.argmax(out, 0).item() for out in output]
        
    # print(f"[{datetime.now()}] Finished Evaluation!")
    return (predicted, [class_names[p] for p in predicted])


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Parameters for Evaluating a Model on Chinese Hanzi Characters")
    
    parser.add_argument("--name", type=str, help="Name of the model to use for evaluating images", default='model-GoogLeNet-750-1.0')
    
    model_name = parser.parse_args().name    
    images_initial = [ # assuming file run from monorepo (using VS code, usually the case for name=main)
        # typed fonts, black text & white background
        cv2.imread('./model/custom_test_images/IMG_1949.jpg'),
        cv2.imread('./model/custom_test_images/IMG_1949-2.jpg'),

        # computer drawn, black text & white background
        cv2.imread('./model/custom_test_images/IMG_2000.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2001.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2001-2.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2002.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2003.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2004.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2005.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2006.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2007.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2008.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2009.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2010.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2011.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2012.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2013.jpg'), 
        cv2.imread('./model/custom_test_images/IMG_2014.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2015.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2016.jpg'),
        
        # computer drawn, non-BW background / foreground colors
        cv2.imread('./model/custom_test_images/IMG_2100.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2101.jpg'),
        
        # hand drawn
        cv2.imread('./model/custom_test_images/IMG_1975.jpg'),
        cv2.imread('./model/custom_test_images/IMG_1976.jpg'),
        cv2.imread('./model/custom_test_images/IMG_1977.jpg'),
        
        # actual training samples
        cv2.imdecode(np.fromfile('./model/data/filtered/top-500/train/十/0001.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./model/data/filtered/top-500/train/只/0002.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./model/data/filtered/top-500/train/回/0003.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./model/data/filtered/top-500/train/教/0004.png', np.uint8), cv2.IMREAD_UNCHANGED),
        # cv2.imdecode(np.fromfile('./model/data/filtered/top-500/train/书/0005.png', np.uint8), cv2.IMREAD_UNCHANGED),
        # cv2.imdecode(np.fromfile('./model/data/filtered/top-500/train/书/0006.png', np.uint8), cv2.IMREAD_UNCHANGED),
        # cv2.imdecode(np.fromfile('./model/data/filtered/top-500/train/书/0007.png', np.uint8), cv2.IMREAD_UNCHANGED),
        # cv2.imdecode(np.fromfile('./model/data/filtered/top-500/train/书/0008.png', np.uint8), cv2.IMREAD_UNCHANGED),
        # cv2.imdecode(np.fromfile('./model/data/filtered/top-500/train/书/0009.png', np.uint8), cv2.IMREAD_UNCHANGED),
        # cv2.imdecode(np.fromfile('./model/data/filtered/top-500/train/书/0010.png', np.uint8), cv2.IMREAD_UNCHANGED),
    ]
    
    images = []
    for image in images_initial: 
        
        ## add channels dimension if not present
        if (len(np.shape(image)) == 2):
            image = image[..., np.newaxis]
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        images.append(image)
            
    (ids, labels) = evaluate(images, model_name)
    for id, label in zip(ids, labels): print(str(id)+": " + label)
