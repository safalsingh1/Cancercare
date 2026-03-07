"""
CancerCare FastAPI Backend Server

Key Functionality:
- Environment Variable Loading: Reads GEMINI_API_KEY from .env
- Model Loading: Loads a generic ResNet18 adapted for `class_mapping.json`
- Preprocessing: Torchvision transformation applied to uploaded images
- Server Setup: FastAPI app loaded with wide-open CORS for the React frontend
- Endpoints:
  * /predict: Submits image bits for ResNet inference
  * /chat: Uses Gemini SDK with environment config fallback
"""
import io
import os
import json
import torch
import torch.nn as nn
import requests
from google import genai
from dotenv import load_dotenv
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')
gemini_client = genai.Client(api_key=API_KEY) if API_KEY else None

MODEL_PATH = "model_best.pth"
CLASS_MAP_PATH = "class_mapping.json"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_class_mapping(mapping_path: str):
    try:
        with open(mapping_path, 'r') as f:
            class_to_idx = json.load(f)
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        return idx_to_class
    except FileNotFoundError:
        return None

def load_model(model_path: str, num_classes: int, device: torch.device):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        return None

IDX_TO_CLASS = load_class_mapping(CLASS_MAP_PATH)
NUM_CLASSES = len(IDX_TO_CLASS) if IDX_TO_CLASS else 30 # fallback just in case
MODEL = load_model(MODEL_PATH, NUM_CLASSES, DEVICE)

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

app = FastAPI(title="Cancer Subtype Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Welcome to the Multi-Cancer Predictor API. Model Status: " + ("Loaded" if MODEL else "Not Found")}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not MODEL or not IDX_TO_CLASS:
        raise HTTPException(status_code=500, detail="The classification model has not been trained or loaded yet.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    image_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = MODEL(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class_idx = torch.max(probabilities, 1)

    class_idx = top_class_idx.item()
    confidence = top_prob.item() * 100
    predicted_class = IDX_TO_CLASS[class_idx]
    
    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 2)
    }

class ChatRequest(BaseModel):
    user_input: str

@app.post("/chat")
def chat(req: ChatRequest):
    gemini_client = genai.Client(api_key=API_KEY) if API_KEY else genai.Client()
    
    prompt = (
        f"You are a warm, empathetic emotional support chatbot for cancer patients. "
        f"Keep your response concise (2-4 short paragraphs max). "
        f"User message: {req.user_input}"
    )
    
    models_to_try = ["gemini-3-flash-preview", "gemini-2.0-flash", "gemini-1.5-flash-8b"]
    import time
    
    for model_name in models_to_try:
        for attempt in range(2):
            try:
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return {"response": response.text if response else "I'm sorry, I couldn't get a response. Please try again."}
            except Exception as e:
                err_str = str(e)
                print(f"Chatbot Error on {model_name} (attempt {attempt}): {err_str}")
                if "429" in err_str or "404" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    time.sleep(1)
                    continue
                else:
                    break # try next model
    
    return {"response": "I'm here for you! The service is briefly busy — please send your message again in a moment."}

class GenomicRequest(BaseModel):
    query_type: str
    query: str

@app.post("/genomic")
def genomic(req: GenomicRequest):
    query = req.query
    query_type = req.query_type
    
    if not query:
        raise HTTPException(status_code=400, detail="Query input is missing")
        
    if query_type == 'gene':
        url = f'https://rest.ensembl.org/lookup/symbol/homo_sapiens/{query}?content-type=application/json'
    elif query_type == 'variant':
        url = f'https://rest.ensembl.org/variation/human/{query}?content-type=application/json'
    else:
        raise HTTPException(status_code=400, detail="Invalid query type")
        
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"{query_type.capitalize()} '{query}' not found."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)