from io import BytesIO
import base64

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from api.inference import (
    predict_image,
    get_model_info,
    MODEL
)

from src.explainability.gradcam import (
    generate_gradcam_for_api
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="KidneyVision MLOps API",
    description=(
        "Deep Learning API for kidney CT image "
        "classification using Swin Transformer."
    ),
    version="1.0.0"
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_info():

    return get_model_info()


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/jpg"
    ]:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid image format. "
                "Please upload JPG or PNG."
            )
        )

    try:

        file_bytes = await file.read()

        image = Image.open(
            BytesIO(file_bytes)
        )

        result = predict_image(image)

        return {
            "filename": file.filename,
            **result
        }

    except UnidentifiedImageError:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image."
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================
# GRAD-CAM EXPLAINABILITY ENDPOINT
# ============================================================

@app.post("/explain")
async def explain(
    file: UploadFile = File(...)
):

    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/jpg"
    ]:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid image format. "
                "Please upload JPG or PNG."
            )
        )

    try:

        # -------------------------------------------------------
        # Read image
        # -------------------------------------------------------

        file_bytes = await file.read()

        image = Image.open(
            BytesIO(file_bytes)
        ).convert("RGB")

        # -------------------------------------------------------
        # First perform normal prediction
        # -------------------------------------------------------

        prediction_result = predict_image(
            image
        )

        predicted_class = (
            prediction_result["prediction"]
        )

        confidence = (
            prediction_result["confidence"]
        )

        # -------------------------------------------------------
        # Find predicted class index
        # -------------------------------------------------------

        from configs.training_config import CLASS_NAMES

        predicted_index = CLASS_NAMES.index(
            predicted_class
        )

        # -------------------------------------------------------
        # Generate Grad-CAM
        # -------------------------------------------------------

        gradcam_bytes = generate_gradcam_for_api(
            model=MODEL,
            image=image,
            predicted_index=predicted_index,
            predicted_class=predicted_class,
            confidence=confidence
        )

        # -------------------------------------------------------
        # Convert PNG to Base64
        # -------------------------------------------------------

        gradcam_base64 = base64.b64encode(
            gradcam_bytes
        ).decode("utf-8")

        # -------------------------------------------------------
        # Response
        # -------------------------------------------------------

        return {
            "filename": file.filename,
            "prediction": predicted_class,
            "confidence": confidence,
            "confidence_percent": round(
                confidence * 100,
                2
            ),
            "probabilities": (
                prediction_result["probabilities"]
            ),
            "gradcam_image_base64": gradcam_base64,
            "gradcam_media_type": "image/png"
        }

    except UnidentifiedImageError:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image."
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Explainability failed: {str(e)}"
        )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "KidneyVision MLOps",
        "message": (
            "Kidney CT Classification API is running."
        ),
        "docs": "/docs"
    }