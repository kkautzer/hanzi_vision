from flask import Flask, request
import cv2
import numpy as np
from character_classifier.eval_single import evaluate

app = Flask(__name__)

@app.route('/') # basic route to test the connection to the API
def hello_world():
    return "<p>Hello World!</p>"

@app.route('/evaluate', methods=['POST'])
def evaluate_image():
        
    if 'image' not in request.files:
        return "No image provided", 400
    
    image = request.files['image'] # must match the value of the HTML `name` attribute for the image

    if image.filename == '':
        return "No selected file", 400
    
    if image:
        try:
            image_array = np.frombuffer(image.read(), np.uint8) # read image bytes, convert to np.uint8 array
            image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED) # convert np.uint8 array to a cv2 image            
            # dictionary of form {id: [id], label: [char]}
            return dict(zip( ("id", "label"), evaluate(image, "model-GoogLeNet-500-aug-0.2") ))

        except Exception as e:
            print(e)
            return "Failed to evaluate image"
    else:
        return "No image provided [2]"
    
    
if __name__ == "__main__":
    app.run()