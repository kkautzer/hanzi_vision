from flask import Flask, request, jsonify
from flask_cors import CORS as cors
import cv2
import numpy as np
from character_classifier.eval_single import evaluate

app = Flask(__name__)
cors(app)

@app.route('/') # basic route to test the connection to the API
def hello_world():
    return "<p>Hello World!</p>"

@app.route('/evaluate', methods=['POST'])
def evaluate_image():
    print(request)
    if 'image' not in request.files:
        return jsonify({"message": "No image provided"}), 400
    
    image = request.files.get('image') # must match the value of the HTML `name` attribute for the image

    if image.filename == '':
        return jsonify({"message": "No selected file"}), 400
    
    if image:
        try:
            image_array = np.frombuffer(image.read(), np.uint8) # read image bytes, convert to np.uint8 array
            image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED) # convert np.uint8 array to a cv2 image            
            # return dictionary of form {id: [id], label: [char]}
            return jsonify(dict(zip( ("id", "label"), evaluate(image, "model-GoogLeNet-500-1.0", 500) ) )) , 200

        except Exception as e:
            print(e)
            return jsonify({"message": "Failed to evaluate image"}), 500
    else:
        return jsonify({"message": "No image provided [2]"}), 400
    
    
if __name__ == "__main__":
    app.run()