import base64
import cv2
import numpy as np
import torch
import json
import fitz
import os
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response

# FIXED: Added detect_corners_regression to the import list
from src.pipelines.inference import (
    DocumentScanningPipeline, load_model, apply_perspective_transform, 
    enhance_document, apply_adaptive_binarization, apply_ink_boost_filter,
    is_already_cropped, is_already_enhanced, detect_corners_ensemble, 
    detect_corners_heatmap, detect_corners_regression, order_corners
)

app = FastAPI(title="Document Scanner & Enhancer API")
pipeline = None
device = "cuda" if torch.cuda.is_available() else "cpu"
models_registry = {}

MODEL_PATHS = {
    "corner_heatmap_clean_nodropout": "checkpoints/corner_heatmap_clean_nodropout/best_model.pth",
    "corner_heatmap_clean_nodropout_v2": "checkpoints/corner_heatmap_clean_nodropout_v2/best_model.pth",
    "corner_heatmap_clean_nodropout_v3": "checkpoints/corner_heatmap_clean_nodropout_v3/best_model.pth",
    "corner_heatmap_clean_nodropout_v4": "checkpoints/corner_heatmap_clean_nodropout_v4/best_model.pth",
    "corner_heatmap_regularized": "checkpoints/corner_heatmap_regularized/best_model.pth",
    "corner_heatmap_regularized_v2": "checkpoints/corner_heatmap_regularized_v2/best_model.pth",
    "corner_heatmap_regularized_v3": "checkpoints/corner_heatmap_regularized_v3/best_model.pth",
    "corner_heatmap_regularized_v4": "checkpoints/corner_heatmap_regularized_v4/best_model.pth",
    "corner_regression_clean_nodropout": "checkpoints/corner_regression_clean_nodropout/best_model.pth",
    "corner_regression_regularized": "checkpoints/corner_regression_regularized/best_model.pth",
    "e2e_finetuned": "checkpoints/e2e_finetuned/best_model.pth",
    "enhancement_clean_nodropout": "checkpoints/enhancement_clean_nodropout/best_model.pth",
    "enhancement_clean_nodropout_v2": "checkpoints/enhancement_clean_nodropout_v2/best_model.pth",
    "enhancement_regularized": "checkpoints/enhancement_regularized/best_model.pth",
    "enhancement_regularized_v2": "checkpoints/enhancement_regularized_v2/best_model.pth",
}

@app.on_event("startup")
async def startup_event():
    global pipeline, models_registry
    try:
        models_registry = {
            "heatmap_v4_reg": load_model("corner_heatmap", "checkpoints/corner_heatmap_regularized_v4/best_model.pth", device, dropout_rate=0.3),
            "enh_regularized_v2": load_model("enhancement", "checkpoints/enhancement_regularized_v2/best_model.pth", device, dropout_rate=0.3),
            "heatmap_v3": load_model("corner_heatmap", "checkpoints/corner_heatmap_clean_nodropout_v3/best_model.pth", device),
            "enh_clean_v2": load_model("enhancement", "checkpoints/enhancement_clean_nodropout_v2/best_model.pth", device),
            "heatmap_v2": load_model("corner_heatmap", "checkpoints/corner_heatmap_clean_nodropout_v2/best_model.pth", device),
            "enh_baseline": load_model("enhancement", "checkpoints/enhancement_clean_nodropout/best_model.pth", device),
            "regression": load_model("corner_regression", "checkpoints/corner_regression_clean_nodropout/best_model.pth", device)
        }
        
        e2e_path = "checkpoints/e2e_finetuned/best_model.pth"
        if os.path.exists(e2e_path):
            models_registry["heatmap_e2e"] = load_model("corner_heatmap", e2e_path, device, dropout_rate=0.3)
            models_registry["enh_e2e"] = load_model("enhancement", e2e_path, device, dropout_rate=0.3)
            print("Loaded E2E Finetuned checkpoints.")
            
        pipeline = DocumentScanningPipeline(models_registry, device)
        print("Pipeline & Models initialized successfully.")
    except Exception as e:
        print(f"Warning: Could not load models during startup: {e}")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/data", StaticFiles(directory="data"), name="data")

@app.post("/api/evaluate/batch")
async def evaluate_batch(
    dataset_type: str = Form(...),
    corner_model: str = Form(""),
    enhancement_model: str = Form(...),
    use_gt_corners: str = Form("false")
):
    from evaluate import run_evaluation_pipeline
    use_gt = use_gt_corners.lower() == 'true'
    
    c_path = MODEL_PATHS.get(corner_model, "") if not use_gt else ""
    e_path = MODEL_PATHS.get(enhancement_model, "")
    
    if not e_path:
        raise HTTPException(status_code=400, detail="Invalid enhancement model selected.")
    if not use_gt and not c_path:
        raise HTTPException(status_code=400, detail="Invalid corner model selected.")
        
    task = "corner_regression" if "regression" in corner_model else "corner_heatmap"
    
    try:
        results = run_evaluation_pipeline(
            dataset_type=dataset_type,
            task=task,
            corner_ckpt=c_path,
            enhancement_ckpt=e_path,
            use_gt_corners=use_gt
        )
        if results["best_image_path"]:
            results["best_image_url"] = "/" + results["best_image_path"].replace("\\", "/")
        else:
            results["best_image_url"] = ""
            
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scan")
async def scan_document(
    file: UploadFile = File(...),
    reference_file: UploadFile = File(None),
    corner_method: str = Form("ensemble"),
    enhancement_method: str = Form("enh_e2e"),
    apply_binarization: str = Form("false"),
    apply_ink_boost: str = Form("false")
):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized.")

    try:
        file_bytes = await file.read()
        do_binarization = apply_binarization.lower() == 'true'
        do_ink_boost = apply_ink_boost.lower() == 'true'

        if file.content_type == "application/pdf":
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            out_pdf = fitz.open()
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=150)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                elif pix.n == 3: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                results = pipeline.process(
                    img, 
                    corner_method, 
                    enhancement_method, 
                    do_binarization, 
                    apply_ink_boost=do_ink_boost,
                    reference_img=None
                )
                enh_rgb = cv2.cvtColor(results['enhanced'], cv2.COLOR_BGR2RGB)
                
                pil_img = Image.fromarray(enh_rgb)
                img_byte_arr = BytesIO()
                pil_img.save(img_byte_arr, format='JPEG', quality=90)
                img_doc = fitz.open(stream=img_byte_arr.getvalue(), filetype="jpeg")
                pdf_bytes_page = img_doc.convert_to_pdf()
                out_pdf.insert_pdf(fitz.open(stream=pdf_bytes_page, filetype="pdf"))

            pdf_out_bytes = out_pdf.write()
            return Response(content=pdf_out_bytes, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=enhanced_scan.pdf"})

        raw_image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        ref_image = None
        if reference_file:
            ref_bytes = await reference_file.read()
            ref_image = cv2.imdecode(np.frombuffer(ref_bytes, np.uint8), cv2.IMREAD_COLOR)

        results = pipeline.process(
            raw_image, 
            corner_method, 
            enhancement_method, 
            do_binarization, 
            apply_ink_boost=do_ink_boost,
            reference_img=ref_image
        )

        _, buf_enh = cv2.imencode('.jpg', results['enhanced'], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        _, buf_crn = cv2.imencode('.jpg', results['corners_image'], [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        return JSONResponse(content={
            "status": "success",
            "type": "image",
            "enhanced_image": base64.b64encode(buf_enh).decode('utf-8'),
            "corners_image": base64.b64encode(buf_crn).decode('utf-8'),
            "metrics": results.get('metrics', {})
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/interactive-detect")
async def interactive_detect(
    file: UploadFile = File(...),
    corner_method: str = Form("ensemble")
):
    try:
        raw_bytes = await file.read()
        raw_image = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if is_already_cropped(raw_image):
            corners_normalized = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            return JSONResponse(content={"corners": corners_normalized})
        
        if corner_method == "ensemble":
            models_to_ensemble = [
                models_registry.get("heatmap_e2e", models_registry["heatmap_v4_reg"]), 
                models_registry["heatmap_v4_reg"], 
                models_registry["heatmap_v3"], 
                models_registry["heatmap_v2"]
            ]
            corners = detect_corners_ensemble(models_to_ensemble, raw_image, device)
        # FIXED: Ensure regression models are routed correctly to prevent OpenCV GaussianBlur crash
        elif "regression" in corner_method:
            corners, _ = detect_corners_regression(models_registry[corner_method], raw_image, device=device)
        else:
            corners, _, _ = detect_corners_heatmap(models_registry[corner_method], raw_image, device)
            
        corners = order_corners(corners)
        
        h, w = raw_image.shape[:2]
        corners_normalized = corners.copy()
        corners_normalized[:, 0] /= w
        corners_normalized[:, 1] /= h

        return JSONResponse(content={"corners": corners_normalized.tolist()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/interactive-enhance")
async def interactive_enhance(
    file: UploadFile = File(...),
    corners: str = Form(...),
    enhancement_method: str = Form("enh_e2e"),
    apply_binarization: str = Form("false"),
    apply_ink_boost: str = Form("false")
):
    try:
        raw_bytes = await file.read()
        raw_image = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        corners_norm = np.array(json.loads(corners), dtype=np.float32)
        h, w = raw_image.shape[:2]
        corners_px = corners_norm.copy()
        corners_px[:, 0] *= w
        corners_px[:, 1] *= h
        
        do_binarization = apply_binarization.lower() == 'true'
        do_ink_boost = apply_ink_boost.lower() == 'true'
        
        rectified = apply_perspective_transform(raw_image, corners_px)
        
        if is_already_enhanced(rectified):
            enhanced = rectified.copy()
        else:
            enhanced = enhance_document(models_registry[enhancement_method], rectified, device)
        
        if do_ink_boost:
            enhanced = apply_ink_boost_filter(enhanced)
            
        if do_binarization:
            enhanced = apply_adaptive_binarization(enhanced)
            
        _, buf_enh = cv2.imencode('.jpg', enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        return JSONResponse(content={
            "status": "success",
            "enhanced_image": base64.b64encode(buf_enh).decode('utf-8')
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)