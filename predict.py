import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import argparse
import sys

def load_model(model_path, num_classes, device):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model

def load_class_mapping(mapping_path):
    with open(mapping_path, 'r') as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return idx_to_class

def predict(image_path, model, transform, device, idx_to_class):
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        sys.exit(1)
        
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class_idx = torch.max(probabilities, 1)
        
    class_idx = top_class_idx.item()
    confidence = top_prob.item() * 100
    predicted_class = idx_to_class[class_idx]
    
    return predicted_class, confidence

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='Path to the image to classify')
    parser.add_argument('--model_path', type=str, default='model_best.pth', help='Path to the trained model')
    parser.add_argument('--class_map', type=str, default='class_mapping.json', help='Path to the class mapping JSON')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        idx_to_class = load_class_mapping(args.class_map)
    except FileNotFoundError:
        print(f"Class mapping file {args.class_map} not found. Please train the model first.")
        sys.exit(1)
        
    num_classes = len(idx_to_class)
    
    try:
        model = load_model(args.model_path, num_classes, device)
    except FileNotFoundError:
        print(f"Model file {args.model_path} not found. Please train the model first.")
        sys.exit(1)

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    prediction, confidence = predict(args.image_path, model, transform_val, device, idx_to_class)
    
    print(f"\n--- Prediction Results ---")
    print(f"Image: {args.image_path}")
    print(f"Predicted Class: {prediction}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"--------------------------\n")

if __name__ == '__main__':
    main()
