from __future__ import annotations

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from api.inference import get_model_info, load_model, predict_image
from src.explainability.occlusion import generate_occlusion_for_api
from src.explainability.gradcam_render import generate_gradcam_for_api


app = FastAPI(
    title="KidneyVision AI API",
    description="Kidney CT image classification API with explainability.",
    version="1.0.0",
)


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "KidneyVision AI API",
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "service": "KidneyVision AI API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "GET /health",
            "GET /model-info",
            "POST /predict",
            "POST /explain",
            "POST /explain-gradcam",
        ],
    }


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/model-info")
def model_info():
    try:
        return get_model_info()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model information: {exc}",
        )


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file.",
        )

    try:
        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        result = predict_image(image)

        return JSONResponse(
            content=result
        )

    except Exception as exc:

        print(
            f"Prediction error: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        )


# ============================================================
# OCCLUSION EXPLANATION
# ============================================================

@app.post("/explain")
async def explain(file: UploadFile = File(...)):

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file.",
        )

    try:

        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        # load_model() returns ONLY the model
        model = load_model()

        prediction_result = predict_image(
            image
        )

        predicted_class = prediction_result[
            "prediction"
        ]

        confidence = float(
            prediction_result["confidence"]
        )

        class_names = [
            "Normal",
            "Cyst",
            "Stone",
            "Tumor",
        ]

        predicted_index = class_names.index(
            predicted_class
        )

        # Get device from model
        device = next(
            model.parameters()
        ).device

        explanation = generate_occlusion_for_api(
            model=model,
            image=image,
            device=device,
            predicted_index=predicted_index,
            predicted_class=predicted_class,
            confidence=confidence,
        )

        return StreamingResponse(
            io.BytesIO(explanation),
            media_type="image/png",
            headers={
                "X-Predicted-Class": predicted_class,
                "X-Confidence": f"{confidence:.6f}",
                "X-Explainability": "occlusion-sensitivity",
            },
        )

    except Exception as exc:

        print(
            f"Occlusion explanation error: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Explanation failed: {exc}",
        )


# ============================================================
# GRAD-CAM EXPLANATION
# ============================================================

@app.post("/explain-gradcam")
async def explain_gradcam(
    file: UploadFile = File(...)
):

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file.",
        )

    try:

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        # IMPORTANT:
        # load_model() returns ONLY the model.
        model = load_model()

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction_result = predict_image(
            image
        )

        predicted_class = prediction_result[
            "prediction"
        ]

        confidence = float(
            prediction_result["confidence"]
        )

        class_names = [
            "Normal",
            "Cyst",
            "Stone",
            "Tumor",
        ]

        predicted_index = class_names.index(
            predicted_class
        )

        print(
            f"Grad-CAM prediction: "
            f"{predicted_class} | "
            f"Confidence: {confidence:.4f}"
        )

        # ----------------------------------------------------
        # DEVICE
        # ----------------------------------------------------

        # Grad-CAM function expects device separately.
        device = next(
            model.parameters()
        ).device

        # ----------------------------------------------------
        # GENERATE GRAD-CAM
        # ----------------------------------------------------

        explanation = generate_gradcam_for_api(
            model=model,
            image=image,
            device=device,
            predicted_index=predicted_index,
            predicted_class=predicted_class,
            confidence=confidence,
        )

        print(
            "Grad-CAM generated successfully."
        )

        # ----------------------------------------------------
        # RETURN PNG
        # ----------------------------------------------------

        return StreamingResponse(
            io.BytesIO(explanation),
            media_type="image/png",
            headers={
                "X-Predicted-Class": predicted_class,
                "X-Confidence": f"{confidence:.6f}",
                "X-Explainability": "grad-cam",
            },
        )

    except Exception as exc:

        print(
            f"Grad-CAM error: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Grad-CAM generation failed: {exc}",
        )


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():

    print("=" * 50)

    print(
        "KidneyVision AI API starting..."
    )

    print(
        "Endpoints:"
    )

    print(
        "  GET  /health"
    )

    print(
        "  GET  /model-info"
    )

    print(
        "  POST /predict"
    )

    print(
        "  POST /explain"
    )

    print(
        "  POST /explain-gradcam"
    )

    print("=" * 50)