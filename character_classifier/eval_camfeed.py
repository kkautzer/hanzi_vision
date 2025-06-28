import cv2
import torch
from eval_single import evaluate
from datetime import datetime
import time

CAMERA_INDEX = 0
cam = cv2.VideoCapture(CAMERA_INDEX)

if not cam.isOpened:
    print("Failed to open camera stream!")
    
while True:
    ret, frame = cam.read()
    if not ret:
        break

    flip_frame = torch.flip(torch.from_numpy(frame), [1]).numpy()
        
    ## evaluate frame based on custom model
    id, label = evaluate(frame)
    print(f"[{datetime.now()}] Predicted Label: {label} [id #{id}]")
    
    cv2.imshow("Basic Camera Feed", frame)
    
    ## quit program if the 'q' key is pressed or X button on window
    if cv2.waitKey(1) == ord('q') or cv2.getWindowProperty(f'Basic Camera Feed', cv2.WND_PROP_VISIBLE) < 1:
        break
    
cam.release()
cv2.destroyAllWindows()