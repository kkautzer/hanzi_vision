from flask import Flask, request, jsonify
from flask_cors import CORS as cors
from werkzeug.middleware.proxy_fix import ProxyFix
import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import json
import traceback
import torch
from torchvision import transforms
from character_classifier.model import ChineseCharacterCNN
from character_classifier.scripts.crop_image import crop_image

app = Flask(__name__)

load_dotenv()

app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
#furl = os.getenv("FRONTEND_URL")
#print(f"--------------\n\n\n{furl}\n\n\n")
cors(app, 
    origins=[os.getenv("FRONTEND_URL")],
    supports_credentials=False,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"]
)


# TODO Read with csv module instead
training_data = pd.read_csv("./character_classifier/exports/training_data.csv")
training_data = training_data.replace({np.nan: None})

character_data = pd.read_csv('./character_classifier/data/hanzi_db.csv')
character_data = character_data.replace({np.nan: None})

# -----------------------------
# HanziEvaluator class (singleton)
# -----------------------------
class HanziEvaluator:
    def __init__(self, model_name):
        try:
            metadata_path = f"./character_classifier/exports/metadata_public/{model_name}-metadata.json"
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            self.n_chars = metadata['nchars']
            self.architecture = metadata['architecture']
            self.class_names = []
            with open(f"./character_classifier/classes/top-{self.n_chars}-classes.txt", "r", encoding="utf-8") as f:
                self.class_names = [line.strip() for line in f.readlines()]

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            #self.device = "cpu"
            self.model = ChineseCharacterCNN(architecture=self.architecture, num_classes=self.n_chars).to(self.device)
            self.model.load_state_dict(torch.load(
                f"./character_classifier/models/checkpoints/best/{model_name}_best.pth", 
                map_location=self.device
            ))
            self.model.eval()

            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((64,64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        except Exception as e:
            print("Failed to initialize evaluator for model:", model_name)
            print(traceback.format_exc())
            raise e

    def predict(self, image_array):
        try:
            image_tensor = self.transform(crop_image(image_array)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(image_tensor)
                predicted = torch.argmax(output, 1).item()
            return predicted, self.class_names[predicted]
        except Exception as e:
            print("Prediction failed:", e)
            print(traceback.format_exc())
            return None, None

# Cache evaluator instances for different models
evaluators = {}

def get_evaluator(model_name): # TODO Refactor this function to check the model's architecture
    if model_name not in evaluators:
        evaluators[model_name] = HanziEvaluator(model_name)
    return evaluators[model_name]


# -----------------------------
# Flask routes
# -----------------------------
@app.route('/')
def hello_world():
    return "<p>Hello World!</p>"

@app.route('/evaluate', methods=['POST'])
def evaluate_image():
    model_name = request.form.get('model')
    if not model_name:
        return jsonify({"message": "No model selection provided"}), 400

    if 'image' not in request.files:
        return jsonify({"message": "No image provided"}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    try:
        image_array = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
        evaluator = get_evaluator(model_name)
        pred_idx, pred_char = evaluator.predict(image)
        if pred_idx is None:
            return jsonify({"message": "Failed to evaluate image"}), 500
        return jsonify({"id": pred_idx, "label": pred_char}), 200
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"message": "Failed to evaluate image"}), 500

@app.route('/models', methods=['GET'])
def get_models():
    filenames = os.listdir('./character_classifier/exports/metadata_public')
    model_data = []
    for file in filenames:
        with open(f'./character_classifier/exports/metadata_public/{file}', 'r', encoding='utf-8') as f:
            j = json.load(f)
            model_data.append(j)
    return model_data, 200

@app.route("/models/data/<model_name>")
def get_model_data(model_name):
    model_train_data = training_data[training_data['name'] == model_name]
    if len(model_train_data) < 1:
        return "No available data for model", 404
    return model_train_data.to_dict(orient='records'), 200

@app.route('/characters', methods=['GET'])
def get_all_char_info():    
    return jsonify(character_data.to_dict(orient='records')), 200

@app.route('/characters/<character>', methods=['GET'])
def get_char_info(character):
    for _, entry in character_data.iterrows():
        if entry['character'] == character:
            return jsonify(entry.to_dict()), 200
    return "Character not found", 404

if __name__ == "__main__":
    app.run()
