import io
import os
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import uvicorn

app = FastAPI()

# Get the path to your model (assuming it is in the same folder as main.py)
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "my_model.pt")

# Load the model once when the server starts
print(f"Loading YOLO model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image bytes
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))

        # Run detection (conf 0.5 to be safe, change to 0.8 later)
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
    # This runs the server on http://127.0.0.1:8000
    uvicorn.run(app, host="127.0.0.1", port=8000)