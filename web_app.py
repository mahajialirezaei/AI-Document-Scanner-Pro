"""
FastAPI Web Application for Document Scanning & Enhancement.

This module provides a web-based interface for the document scanning pipeline,
allowing users to upload images and receive enhanced scans through a REST API.
"""

import io
import base64
import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path

from src.pipelines.inference import DocumentScanningPipeline

# Initialize FastAPI app
app = FastAPI(
    title="Document Scanner & Enhancer",
    description="Web API for document scanning and enhancement using deep learning",
    version="1.0.0"
)

# Global pipeline instance
pipeline = None


@app.on_event("startup")
async def startup_event():
    """Initialize the document scanning pipeline on startup."""
    global pipeline
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model paths
    enhancement_model_path = "checkpoints/enhancement/best_model.pth"
    corner_model_path = "checkpoints/corner_heat/best_model.pth"
    
    # Check if model files exist
    if not Path(enhancement_model_path).exists():
        print(f"Warning: Enhancement model not found at {enhancement_model_path}")
    if not Path(corner_model_path).exists():
        print(f"Warning: Corner model not found at {corner_model_path}")
    
    try:
        pipeline = DocumentScanningPipeline(
            corner_model_path=corner_model_path,
            enhancement_model_path=enhancement_model_path,
            corner_approach="heatmap",
            device=device
        )
        print(f"Pipeline initialized successfully on {device}")
    except Exception as e:
        print(f"Warning: Could not initialize pipeline: {e}")
        print("Pipeline will be initialized on first request if models become available")


@app.get("/")
async def root():
    """Serve the main HTML page."""
    return FileResponse("static/index.html")


# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/scan")
async def scan_document(file: UploadFile = File(...)):
    """
    Process an uploaded image through the document scanning pipeline.
    
    Args:
        file: Uploaded image file (JPEG, PNG, etc.)
        
    Returns:
        JSON response with base64-encoded enhanced image
    """
    global pipeline
    
    # Ensure pipeline is initialized
    if pipeline is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        enhancement_model_path = "checkpoints/enhancement/best_model.pth"
        corner_model_path = "checkpoints/corner_heat/best_model.pth"
        
        try:
            pipeline = DocumentScanningPipeline(
                corner_model_path=corner_model_path,
                enhancement_model_path=enhancement_model_path,
                corner_approach="heatmap",
                device=device
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize pipeline: {str(e)}")
    
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Decode image using OpenCV (returns BGR format)
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process image through pipeline (expects BGR format)
        enhanced_image = pipeline.process(image, return_intermediate=False)
        
        # Convert enhanced BGR image to JPEG format
        _, buffer = cv2.imencode('.jpg', enhanced_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        # Encode to base64
        base64_string = base64.b64encode(buffer).decode('utf-8')
        
        return JSONResponse(
            content={
                "status": "success",
                "enhanced_image": base64_string
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
