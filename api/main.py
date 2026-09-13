from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from PIL import Image
import io

from api.inference import (
    predict_image,
    get_model_info,
    get_model
)

from src.explainability.gradcam import generate_gradcam_for_api


app = FastAPI(
    title="KidneyVision API",
    description="Kidney CT image classification API",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "KidneyVision API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/model-info")
def model_info():
    try:
        return get_model_info()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load model information: {str(e)}"
        )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Only image files are allowed."
            )

        # Read uploaded image
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        # Open image
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Run prediction
        result = predict_image(image)

        return result

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/explain")
async def explain(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Only image files are allowed."
            )

        # Read uploaded image
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        # Open image
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # First get prediction
        prediction_result = predict_image(image)

        predicted_class = prediction_result["prediction"]
        confidence = prediction_result["confidence"]

        # Get class index from model probabilities
        probabilities = prediction_result["probabilities"]

        class_names = [
            "Normal",
            "Cyst",
            "Stone",
            "Tumor"
        ]

        predicted_index = class_names.index(predicted_class)

        # Generate Grad-CAM
        gradcam_bytes = generate_gradcam_for_api(
            model=get_model(),
            image=image,
            predicted_index=predicted_index,
            predicted_class=predicted_class,
            confidence=confidence
        )

        return Response(
            content=gradcam_bytes,
            media_type="image/png",
            headers={
                "X-Predicted-Class": predicted_class,
                "X-Confidence": str(confidence)
            }
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Explainability generation failed: {str(e)}"
        )