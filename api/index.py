from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import io
import json
import os

app = Flask(__name__)

# Usually Vercel root is the project root, so files in root can be accessed directly.
# Alternatively, Vercel might execute from the api folder if configured wrong, but usually it's root.
CALORIE_MAP_PATH = os.path.join(os.path.dirname(__file__), '..', 'calorie_map.json')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'best_food_model.pth')

def load_calorie_map():
    # Try parent directory first
    if os.path.exists(CALORIE_MAP_PATH):
        with open(CALORIE_MAP_PATH, 'r') as f:
            return json.load(f)
    # Fallback to current directory
    elif os.path.exists('calorie_map.json'):
         with open('calorie_map.json', 'r') as f:
            return json.load(f)
    return {}

def resolve_model_path():
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH
    if os.path.exists('best_food_model.pth'):
        return 'best_food_model.pth'
    return None

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    try:
        model_path = resolve_model_path()
        if not model_path:
            return jsonify({'error': "Model weights 'best_food_model.pth' not found. Ensure it was trained and uploaded."}), 404

        calorie_map = load_calorie_map()
        classes = sorted(list(calorie_map.keys()))

        if not classes:
            return jsonify({'error': 'No classes found in calorie map.'}), 500

        # Load image from stream
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Setup model for inference using CPU
        device = torch.device("cpu")
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, len(classes))
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Transform specific to PyTorch ResNet
        inference_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

        input_tensor = inference_transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_class = classes[predicted_idx.item()]
        conf_percent = confidence.item() * 100
        calories = calorie_map.get(predicted_class, "Unknown")

        return jsonify({
            'food': predicted_class,
            'confidence': conf_percent,
            'calories': calories
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
