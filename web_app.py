import base64
import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from src.pipelines.inference import DocumentScanningPipeline, load_model

app = FastAPI(title="Document Scanner & Enhancer API")
pipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load all models into a registry based on their medal rankings
    try:
        models_registry = {
            # 🥇 Gold Medal Models (Ultimate Champions)
            "heatmap_v4_reg": load_model("corner_heatmap", "checkpoints/corner_heatmap_regularized_v4/best_model.pth", device, dropout_rate=0.3),
            "enh_regularized_v2": load_model("enhancement", "checkpoints/enhancement_regularized_v2/best_model.pth", device, dropout_rate=0.3),
            
            # 🥈 Silver Medal Models (Strong Baselines / Highest PSNR)
            "heatmap_v3": load_model("corner_heatmap", "checkpoints/corner_heatmap_clean_nodropout_v3/best_model.pth", device),
            "enh_clean_v2": load_model("enhancement", "checkpoints/enhancement_clean_nodropout_v2/best_model.pth", device),
            
            # 🥉 Bronze & Legacy Models (For Comparison)
            "heatmap_v2": load_model("corner_heatmap", "checkpoints/corner_heatmap_clean_nodropout_v2/best_model.pth", device),
            "enh_baseline": load_model("enhancement", "checkpoints/enhancement_clean_nodropout/best_model.pth", device),
            "regression": load_model("corner_regression", "checkpoints/corner_regression_clean_nodropout/best_model.pth", device)
        }
        pipeline = DocumentScanningPipeline(models_registry, device)
        print("Pipeline & Models initialized successfully.")
    except Exception as e:
        print(f"Warning: Could not load models during startup: {e}")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/scan")
async def scan_document(
    file: UploadFile = File(...),
    reference_file: UploadFile = File(None),
    corner_method: str = Form("ensemble"),
    enhancement_method: str = Form("enh_regularized_v2"),
    apply_binarization: str = Form("false")
):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized. Check server logs.")

    try:
        raw_bytes = await file.read()
        raw_image = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        ref_image = None
        if reference_file:
            ref_bytes = await reference_file.read()
            ref_image = cv2.imdecode(np.frombuffer(ref_bytes, np.uint8), cv2.IMREAD_COLOR)

        do_binarization = apply_binarization.lower() == 'true'

        results = pipeline.process(
            raw_image=raw_image,
            corner_method=corner_method,
            enhancement_method=enhancement_method,
            apply_binarization=do_binarization,
            reference_img=ref_image
        )

        _, buf_enh = cv2.imencode('.jpg', results['enhanced'], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        _, buf_crn = cv2.imencode('.jpg', results['corners_image'], [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        return JSONResponse(content={
            "status": "success",
            "enhanced_image": base64.b64encode(buf_enh).decode('utf-8'),
            "corners_image": base64.b64encode(buf_crn).decode('utf-8'),
            "metrics": results.get('metrics', {})
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)