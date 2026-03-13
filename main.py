import io
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware # Added for browser access
from ultralytics import YOLO
from PIL import Image
import uvicorn

app = FastAPI()

# Add CORS middleware to allow browser requests from your domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://anitrack.helioho.st"], # Use your specific domain instead of "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the path to your model
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "my_model.pt")

print(f"Loading YOLO model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))

        results = model.predict(source=img, conf=0.5)
        
        detections = []
        if len(results) > 0:
            for box in results[0].boxes:
                detections.append({
                    "label": model.names[int(box.cls)],
                    "confidence": round(float(box.conf), 2)
                })
        
        return {"success": True, "detections": detections}
    
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    # Get the port from the environment, default to 10000 if not provided
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)



