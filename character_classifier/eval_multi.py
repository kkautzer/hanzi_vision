import cv2
import torch
from torchvision import datasets, transforms
from character_classifier.model import ChineseCharacterCNN

def evaluate(images, model_name, epoch_num=-1):
    '''
    
    '''
    
    # this should be updated dynamically to reflect model characteristics - eventually, save
    # a mapping of the model name => nchars & other data
    # OR, check from model weight to detect the # of classes (based on final fc layer)
    data_dir = "./character_classifier/data/filtered/top-500"  # Adjust based on script location

    batch_size = 64
    img_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = datasets.ImageFolder(root=f"{data_dir}/val").classes
    num_classes = len(class_names)

    model = ChineseCharacterCNN(num_classes=num_classes).to(device)
    if (epoch_num <= 0):
        path_to_model = f"./character_classifier/checkpoints/best/{model_name}_best.pth"
    else:
        path_to_model = f"./character_classifier/checkpoints/training/{model_name}/tr_epoch{epoch_num}.pth"
    model.load_state_dict(torch.load(path_to_model, map_location=device))

    # Series of transformations to apply to normalize each input image
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),  # Ensure single channel (grayscale)
        transforms.Resize((img_size, img_size)),      # Resize to a consistent size
        transforms.ToTensor(),                        # Convert image to PyTorch tensor
        transforms.Normalize((0.5,), (0.5,))          # Normalize pixel values to mean=0.5, std=0.5
    ])
    images_transformed = [transform(image) for image in images]
    images_eval = torch.stack(images_transformed)
    model.eval()
    with torch.no_grad():
        output = model(images_eval) ## unsqueeze to add batch dimension (=1)
        predicted = [torch.argmax(out, 0).item() for out in output]
        
    # print(f"[{datetime.now()}] Finished Evaluation!")
    return (predicted, [class_names[p] for p in predicted])


if __name__ == "__main__":
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
    
    (ids, labels) = evaluate(images, "model-GoogLeNet-500-1.0")
    for id, label in zip(ids, labels): print(str(id)+": " + label)
    