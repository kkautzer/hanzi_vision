from flask import Flask, request, jsonify
from flask_cors import CORS as cors
import cv2
import numpy as np
from character_classifier.eval_single import evaluate
import pandas as pd
import os
import json
import traceback

app = Flask(__name__)
cors(app)

training_data = pd.read_csv("./character_classifier/exports/training_data.csv")

@app.route('/') # basic route to test the connection to the API
def hello_world():
    return "<p>Hello World!</p>"

@app.route('/evaluate', methods=['POST'])
def evaluate_image():
    
    ## update to send model_name with request, based on the available models from `/models`
    if request.form.get('model') == None:
        return jsonify({"message": "No model selection provided"}), 400
    
    model_name = request.form.get('model') ## check if correct
    
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
            return jsonify(dict(zip( ("id", "label"), evaluate(image, model_name) ) )) , 200

        except Exception as e:
            print(traceback.format_exc())
            print(e)
            return jsonify({"message": "Failed to evaluate image"}), 500
    else:
        return jsonify({"message": "No image provided [2]"}), 400
    
@app.route('/models', methods=['GET'])
def get_models():
    filenames = os.listdir('./character_classifier/models/metadata_public')
    
    model_data = []
    for file in filenames:
        with open(f'./character_classifier/models/metadata_public/{file}', 'r', encoding='utf-8') as f:
            j = json.load(f)
            model_data.append(j)
            
    return model_data, 200

@app.route("/models/data/<model_name>")
def get_model_data(model_name):

    model_train_data = training_data[training_data['name'] == model_name]

    if len(model_train_data) < 1:
        return "No available data for model", 404
        
    return model_train_data.to_dict(orient='records'), 200


@app.route('/characters/<character>', methods=['GET'])
def get_char_info(character):
    df = pd.read_csv('./character_classifier/data/hanzi_db.csv')
    
    for _, entry in df.iterrows():
        if entry['character'] == character:
            print(entry)
            print(type(entry))
            return jsonify(entry.to_dict()), 200
        
    return "Character not found", 404

if __name__ == "__main__":
    app.run()