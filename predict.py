import os
import sys
import json
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

def load_calorie_map(json_path='calorie_map.json'):
    with open(json_path, 'r') as f:
        return json.load(f)

def load_classes(classes_txt_path='data/food-101/meta/classes.txt'):
    if not os.path.exists(classes_txt_path):
        print(f"Classes file not found at {classes_txt_path}. Be sure dataset is downloaded.")
        return []
    with open(classes_txt_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def predict(image_path, model_path='best_food_model.pth'):
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.")
        sys.exit(1)
        
    if not os.path.exists(model_path):
        print(f"Error: Model weights '{model_path}' not found. Did you run train.py first?")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    classes = load_classes()
    if not classes:
        # Fallback to keys of calorie map if classes file is missing
        calorie_map = load_calorie_map()
        classes = sorted(list(calorie_map.keys()))

    try:
        # Initialize model architecture to match training script
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, len(classes))
        
        # Load custom weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    # Standard inference transforms
    inference_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = inference_transform(image).unsqueeze(0).to(device)

    # Classify
    print(f"Analyzing {image_path}...")
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
    predicted_class = classes[predicted_idx.item()]
    conf_percent = confidence.item() * 100

    # Get calorie info
    calorie_map = load_calorie_map()
    calories = calorie_map.get(predicted_class, "Unknown")
    
    print("-" * 30)
    print("RESULTS:")
    print(f"Food Predicted: {predicted_class.replace('_', ' ').title()}")
    print(f"Confidence:     {conf_percent:.2f}%")
    if isinstance(calories, (int, float)):
        print(f"Est. Calories:  ~{calories} kcal (per 100g)")
    else:
        print(f"Est. Calories:  {calories}")
    print("-" * 30)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image.jpg>")
        sys.exit(1)
        
    predict(sys.argv[1])
