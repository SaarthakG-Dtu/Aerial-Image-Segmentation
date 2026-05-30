import io
import base64
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import torch
from torchvision import transforms as T
from PIL import Image
import numpy as np

from .model_factory import CLASS_COLORS, CLASS_NAMES, N_CLASSES, build_model

app = FastAPI()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_PATH = PROJECT_ROOT / "frontend.html"
MODEL_PATH = PROJECT_ROOT / "outputs" / "final_model.pt"
BEST_MODEL_PATH = PROJECT_ROOT / "outputs" / "best_model.pt"
INFERENCE_SIZE = (512, 768)

CLASS_COLORS_ARRAY = np.array(CLASS_COLORS, dtype=np.uint8)

model = None

@app.on_event("startup")
def load_app_model():
    global model
    print("Loading model...")
    model = build_model(n_classes=N_CLASSES, encoder_weights=None)
    checkpoint_path = MODEL_PATH if MODEL_PATH.exists() else BEST_MODEL_PATH
    if checkpoint_path.exists():
        print(f"Loading weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("Model weights not found. Running with uninitialized weights (demo mode).")
    model.to(device)
    model.eval()

@app.get("/")
def serve_frontend():
    if FRONTEND_PATH.exists():
        with FRONTEND_PATH.open("r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<html><body>Frontend not found</body></html>", status_code=404)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is still loading")
    
    # Read image
    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Unsupported or corrupt image file") from exc
    
    # Preprocess
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    t = T.Compose([T.Resize(INFERENCE_SIZE), T.ToTensor(), T.Normalize(mean, std)])
    input_tensor = t(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        masked = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
    # Generate visualization
    RGB_mask = np.zeros((*masked.shape, 3), dtype=np.uint8)
    for c in range(N_CLASSES):
        RGB_mask[masked == c] = CLASS_COLORS_ARRAY[c]
        
    mask_img = Image.fromarray(RGB_mask)
    buff = io.BytesIO()
    mask_img.save(buff, format="PNG")
    mask_base64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    
    # Calculate statistics
    total_pixels = masked.size
    class_counts = np.bincount(masked.flatten(), minlength=N_CLASSES)
    
    stats = []
    for c in range(N_CLASSES):
        if class_counts[c] > 0:
            percentage = (class_counts[c] / total_pixels) * 100
            stats.append({
                "class_id": c,
                "class_name": CLASS_NAMES[c],
                "percentage": round(percentage, 1),
                "count": int(class_counts[c]),
                "color": f"rgb({CLASS_COLORS_ARRAY[c][0]}, {CLASS_COLORS_ARRAY[c][1]}, {CLASS_COLORS_ARRAY[c][2]})"
            })
            
    # Sort stats by percentage
    stats = sorted(stats, key=lambda x: x["percentage"], reverse=True)
    
    return JSONResponse({
        "mask_base64": f"data:image/png;base64,{mask_base64}",
        "entities": stats[:6]  # Return top 6
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
