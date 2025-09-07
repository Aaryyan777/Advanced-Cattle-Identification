import io
import time
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
import base64

# --- 1. Initialize All Models ---

# Stage 1: Zero-Shot Filter (OpenAI CLIP)
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
print(f"Initializing CLIP model ({CLIP_MODEL_NAME})...")
try:
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    print("CLIP model initialized successfully.")
except Exception as e:
    print(f"FATAL: Could not initialize CLIP model. Error: {e}")
    exit()

# Stage 2: Expert Breed Classifier (EfficientNetV2-S)
BREED_MODEL_PATH = "effnetv2s_best.keras"
BREED_CLASS_NAMES_PATH = "class_names.txt"
BREED_IMG_SIZE = 300
print(f"Initializing Breed Classifier model ({BREED_MODEL_PATH})...")
try:
    breed_model = tf.keras.models.load_model(BREED_MODEL_PATH)
    with open(BREED_CLASS_NAMES_PATH, "r") as f:
        breed_class_names = [line.strip() for line in f.readlines()]
    print("Breed Classifier initialized successfully.")
except Exception as e:
    print(f"FATAL: Could not initialize the breed classifier. Make sure '{BREED_MODEL_PATH}' and '{BREED_CLASS_NAMES_PATH}' are present. Error: {e}")
    exit()

# --- 2. Flask App Setup ---
app = Flask(__name__)

# --- 3. Preprocessing ---
def preprocess_for_breed_model(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((BREED_IMG_SIZE,BREED_IMG_SIZE), Image.NEAREST)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return tf.keras.applications.efficientnet_v2.preprocess_input(img_array)

# --- 4. Main Prediction Logic ---
@app.route('/predict', methods=['POST'])
def predict():
    started = time.time()
    img_bytes = None
    is_json_request = request.is_json

    if is_json_request:
        data = request.get_json()
        if 'imageBase64' not in data: return jsonify({'error': 'imageBase64 not found'}), 400
        img_bytes = base64.b64decode(data['imageBase64'].split(',')[-1])
    elif 'file' in request.files:
        file = request.files['file']
        if not file.filename: return jsonify({'error': 'No selected file'}), 400
        img_bytes = file.read()
    
    if not img_bytes: return jsonify({'error': 'No image data provided'}), 400

    try:
        # --- Stage 1: CLIP Filter ---
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        labels = ["a photo of a cow, bull, or cattle", "a photo of something else"]
        inputs = clip_processor(text=labels, images=img_pil, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = clip_model(**inputs)
        
        is_cattle = outputs.logits_per_image.softmax(dim=1).squeeze()[0].item() > 0.5 # Check confidence of the first label

        # --- Decision Point ---
        if not is_cattle:
            if is_json_request:
                # For web apps, return the expected format but with an empty predictions list.
                return jsonify({"model": "CLIP-Filter", "latencyMs": (time.time() - started) * 1000, "predictions": []})
            else:
                # For our test UI, return a more descriptive message.
                return jsonify({"is_cattle": False, "filter_result": {"top_prediction_label": "Not Cattle"}})

        # --- Stage 2: Breed Classifier ---
        preprocessed_img = preprocess_for_breed_model(img_bytes)
        scores = tf.nn.softmax(breed_model.predict(preprocessed_img)[0])
        
        top_k = 5
        if is_json_request and request.get_json().get('topK'):
            top_k = request.get_json()['topK']
        
        top_indices = np.argsort(scores)[-top_k:][::-1]
        predictions = [{'label': breed_class_names[i], 'confidence': float(scores[i])} for i in top_indices]

        return jsonify({
            'model': 'Two-Stage-CLIP-EffNetV2',
            'latencyMs': (time.time() - started) * 1000,
            'predictions': predictions
        })

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

# --- 5. UI Rendering ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# --- 6. Main Execution ---
if __name__ == '__main__':
    print("\n--- Universal Two-Stage Classification Server ---")
    app.run(debug=True)
