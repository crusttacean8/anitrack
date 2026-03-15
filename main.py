import io
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import uvicorn
import traceback

app = FastAPI()

# Allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://anitrack.helioho.st"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model setup
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "my_model.pt")

model = None

@app.on_event("startup")
def load_model():
    global model
    print(f"Loading YOLO model from {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully")

# Health check endpoint
@app.get("/")
async def root():
    return {"status": "running"}

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Resize large images to reduce memory usage
        img.thumbnail((640, 640))

        results = model.predict(source=img, conf=0.5, imgsz=640)

        detections = []

        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                detections.append({
                    "label": model.names[int(box.cls)],
                    "confidence": round(float(box.conf), 2)
                })

        return {
            "success": True,
            "detections": detections
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
