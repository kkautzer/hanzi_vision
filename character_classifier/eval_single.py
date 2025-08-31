import cv2
import torch
from torchvision import transforms
from character_classifier.model import ChineseCharacterCNN
from character_classifier.scripts.crop_image import crop_image
    
def evaluate(image, model_name, n_chars, thresholded=False):
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
    
    image = crop_image(image, thresholded=thresholded)
    
    batch_size = 64
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
        output = model(transform(image).unsqueeze(0)) ## unsqueeze to add batch dimension (=1)
        predicted = torch.argmax(output, 1).item()
        
    # print(f"[{datetime.now()}] Finished Evaluation!")
    return (predicted, class_names[predicted])


if __name__ == "__main__":
    
    # replace with path to any image file
    images = [ # assuming file run from monorepo (using VS code, usually the case for name=main)
        cv2.imread('./character_classifier/custom_test_images/IMG_1949.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_1975.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_1976.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_1977.jpg'),

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
    ]
    
    
    for image in images: 
        
        print(evaluate(image, "model-GoogLeNet-500-1.0", 500))
        
    # for image in images: print(evaluate(image, "model-GoogLeNet-500-1.0", 500, 39))
    
    # for image in images[:]: show_image(image)