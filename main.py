import io
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import uvicorn
import traceback

# Initialize FastAPI
app = FastAPI()

# CORS setup: allow only your HelioHost domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://anitrack.helioho.st"],  # Update with your front-end domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model path
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "my_model.pt")

print(f"Loading YOLO model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

# Optional ping endpoint for waking Render server
@app.get("/ping")
async def ping():
    return {"status": "ok"}

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image bytes
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # YOLO prediction (conf threshold 0.5)
        results = model.predict(source=img, conf=0.5)

        # Parse detections
        detections = []
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                detections.append({
                    "label": model.names[int(box.cls)],
                    "confidence": round(float(box.conf), 2)
                })

        # Return JSON even if no detections
        return {"success": True, "detections": detections}

    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}

# Run app (Render automatically sets PORT)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
