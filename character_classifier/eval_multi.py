import cv2
import torch
from torchvision import transforms
from character_classifier.model import ChineseCharacterCNN
import json

from character_classifier.scripts.crop_image import crop_image

def evaluate(images, model_name):
    '''
    
    '''
    
    try:
        with open(f'./character_classifier/models/metadata/{model_name}-metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        n_chars = metadata['nchars']
        thresholded = metadata['threshold']
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

    with open(f"./character_classifier/classes/top-{n_chars}-classes.txt", 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]
        
    num_classes = len(class_names)

    model = ChineseCharacterCNN(num_classes=num_classes).to(device)
    path_to_model = f"./character_classifier/checkpoints/best/{model_name}_best.pth"
    model.load_state_dict(torch.load(path_to_model, map_location=device))


    def transform_cv2(image):
        """
        Transformations intended for thresholded-trained images
        """
        image = crop_image(image, thresholded=thresholded) # get cropped & edge detected image
        
        # otherwise, perform more extensive transformations
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to grayscale
        resized = cv2.resize(grayscale, dsize=(64, 64), interpolation=cv2.INTER_AREA) # shrink
        _, threshold = cv2.threshold(resized, thresh=40, maxval=255, type=cv2.THRESH_BINARY) # threshold
                
        # # cv2.namedWindow("Cropped", cv2.WINDOW_NORMAL)
        # # cv2.namedWindow("Grayscale", cv2.WINDOW_NORMAL)
        # # cv2.namedWindow("Resized", cv2.WINDOW_NORMAL)
        # # cv2.namedWindow("Threshold", cv2.WINDOW_NORMAL)
        # # cv2.namedWindow("Inverted", cv2.WINDOW_NORMAL)
        # # cv2.imshow("Cropped", image)
        # # cv2.imshow("Grayscale", grayscale)
        # # cv2.imshow("Resized", resized)
        # # cv2.imshow("Threshold", threshold)
        # # cv2.imshow("Inverted", inverted)
        
        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()
        
        toTensor = transforms.ToTensor()
        return toTensor(threshold) 
    
    # Series of transformations to apply to normalize each input image
    transform = transforms.Compose([
        transforms.ToTensor(),                        # Convert image to PyTorch tensor
        transforms.Normalize((0.5,), (0.5,))          # Normalize pixel values to mean=0.5, std=0.5
    ])
    
    images_transformed = [transform(transform_cv2(image, thresholded=thresholded)) for image in images]
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
    