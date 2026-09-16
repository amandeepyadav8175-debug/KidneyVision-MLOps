import io
import logging
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from api.inference import (
    get_model,
    get_model_info,
    is_model_loaded,
    predict_image,
)

from configs.training_config import CLASS_NAMES
from src.explainability.gradcam import generate_gradcam_for_api


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="KidneyVision API",
    description="Kidney CT image classification API",
    version="1.0.0",
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_IMAGE_SIZE_MB = 10
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024


# ============================================================
# HELPER: READ UPLOADED IMAGE
# ============================================================

async def read_uploaded_image(file: UploadFile) -> Image.Image:
    """
    Validate and safely convert an uploaded file into a PIL image.
    """

    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="Missing file content type.",
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image files are allowed.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(contents) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Image file is too large. "
                f"Maximum allowed size is {MAX_IMAGE_SIZE_MB} MB."
            ),
        )

    try:
        image = Image.open(io.BytesIO(contents))

        # Force image decoding so corrupt images are detected.
        image.load()

        return image.convert("RGB")

    except (UnidentifiedImageError, OSError) as exc:
        logger.warning(
            "Invalid image upload: %s",
            exc,
        )

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image.",
        ) from exc


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "KidneyVision API is running",
        "docs": "/docs",
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": is_model_loaded(),
    }


# ============================================================
# READINESS ENDPOINT
# ============================================================

@app.get("/ready")
def readiness():
    """
    Readiness endpoint.

    /health answers:
        Is the API process alive?

    /ready answers:
        Is the model loaded and ready to serve predictions?
    """

    if not is_model_loaded():

        return {
            "status": "not_ready",
            "model_loaded": False,
        }

    return {
        "status": "ready",
        "model_loaded": True,
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_info():

    try:

        return get_model_info()

    except Exception as exc:

        logger.exception(
            "Failed to retrieve model information."
        )

        raise HTTPException(
            status_code=503,
            detail="Model information is currently unavailable.",
        ) from exc


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
):

    try:

        image = await read_uploaded_image(file)

        result = predict_image(image)

        logger.info(
            "Prediction completed: %s (%.2f%%)",
            result["prediction"],
            result["confidence_percent"],
        )

        return result

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception(
            "Prediction failed."
        )

        raise HTTPException(
            status_code=503,
            detail="Prediction service is currently unavailable.",
        ) from exc


# ============================================================
# GRAD-CAM EXPLANATION ENDPOINT
# ============================================================

@app.post("/explain")
async def explain(
    file: UploadFile = File(...),
):

    try:

        image = await read_uploaded_image(file)

        # --------------------------------------------------------
        # Get prediction first
        # --------------------------------------------------------

        prediction_result = predict_image(image)

        predicted_class = prediction_result["prediction"]
        confidence = prediction_result["confidence"]

        # --------------------------------------------------------
        # Get predicted class index
        # --------------------------------------------------------

        try:

            predicted_index = CLASS_NAMES.index(
                predicted_class
            )

        except ValueError as exc:

            raise RuntimeError(
                f"Unknown predicted class: {predicted_class}"
            ) from exc

        # --------------------------------------------------------
        # Generate Grad-CAM
        # --------------------------------------------------------

        gradcam_bytes = generate_gradcam_for_api(
            model=get_model(),
            image=image,
            predicted_index=predicted_index,
            predicted_class=predicted_class,
            confidence=confidence,
        )

        logger.info(
            "Grad-CAM generated for prediction: %s",
            predicted_class,
        )

        return Response(
            content=gradcam_bytes,
            media_type="image/png",
            headers={
                "X-Predicted-Class": predicted_class,
                "X-Confidence": str(confidence),
            },
        )

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception(
            "Grad-CAM generation failed."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Explainability service is currently unavailable."
            ),
        ) from exc