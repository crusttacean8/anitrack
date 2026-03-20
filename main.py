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
    allow_origins=[
        "https://anitrack.helioho.st",
        "http://anitrack.helioho.st",
        "https://www.anitrack.helioho.st",
        "http://www.anitrack.helioho.st"
    ],
    allow_credentials=True,
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

# Health check endpoints
@app.get("/")
async def root():
    return {"status": "running"}

@app.get("/ping")
async def ping():
    return {"status": "ok"}


# yolo resize
def preprocess_image(img: Image.Image):
    # Ensure RGB
    img = img.convert("RGB")

    # Since PHP already resizes to 416x416,
    # we only ensure max size without stretching
    img.thumbnail((640, 640))

    return img


# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # Load image safely
        img = Image.open(io.BytesIO(contents))

        # Preprocess
        img = preprocess_image(img)

        # no redundant resizing
        results = model.predict(
            source=img,
            conf=0.5,
            imgsz=416,   # match PHP resize for consistency
            verbose=False
        )

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
