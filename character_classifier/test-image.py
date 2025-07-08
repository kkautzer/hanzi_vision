from eval_single import evaluate
import cv2
from datetime import datetime

def test(path_to_image):
    print(f"[{datetime.now()}] Reading Image...")
    image = cv2.imread(path_to_image)
    print(f"[{datetime.now()}] Successfully read image")
    print(f"[{datetime.now()}] Evaluating image...")
    _, predicted = evaluate(image)
    print(f"[{datetime.now()}] Evaluation Finished.")
    print(f"Character prediction: {predicted}\n")
    
    
if __name__ == "__main__":
    test("character_classifier/custom_test_images/IMG_1977.jpg")

