from pathlib import Path
from typing import Dict
import logging
import os
import threading

import mlflow
import torch
import torch.nn.functional as F
from PIL import Image

from src.preprocessing.image_preprocessor import build_transforms
from configs.training_config import CLASS_NAMES


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MLFLOW_DB = PROJECT_ROOT / "artifacts" / "mlflow.db"

DEPLOYMENT_MODEL_DIR = (
    PROJECT_ROOT / "artifacts" / "deployment_model"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "SwinTransformer"

REGISTERED_MODEL_NAME = "KidneyVision-SwinTransformer"

MODEL_ALIAS = "champion"

NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = 224


# ============================================================
# MODEL SOURCE
# ============================================================

# registry -> Load from MLflow Model Registry
# local    -> Load bundled deployment model

MODEL_SOURCE = os.getenv(
    "MODEL_SOURCE",
    "registry"
).strip().lower()


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

TRANSFORM = build_transforms(
    image_size=IMAGE_SIZE,
    train=False
)


# ============================================================
# MODEL STATE
# ============================================================

MODEL = None

MODEL_LOAD_ERROR = None

MODEL_LOAD_LOCK = threading.Lock()


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():
    """
    Load the production model from either:

    1. MLflow Model Registry
    2. Local deployment model

    Returns:
        torch.nn.Module

    Raises:
        FileNotFoundError:
            If the local deployment model is missing.

        RuntimeError:
            If MLflow/model loading fails.

        ValueError:
            If MODEL_SOURCE is invalid.
    """

    if MODEL_SOURCE == "local":

        model_file = DEPLOYMENT_MODEL_DIR / "MLmodel"

        if not model_file.exists():

            raise FileNotFoundError(
                "Deployment model not found. "
                f"Expected MLmodel at: {model_file}"
            )

        model_uri = str(DEPLOYMENT_MODEL_DIR)

        logger.info(
            "Loading model from local deployment directory: %s",
            model_uri
        )

        try:

            model = mlflow.pytorch.load_model(
                model_uri,
                map_location=DEVICE
            )

        except Exception as exc:

            raise RuntimeError(
                "Failed to load local deployment model."
            ) from exc

    elif MODEL_SOURCE == "registry":

        mlflow.set_tracking_uri(
            f"sqlite:///{MLFLOW_DB}"
        )

        model_uri = (
            f"models:/{REGISTERED_MODEL_NAME}"
            f"@{MODEL_ALIAS}"
        )

        logger.info(
            "Loading model from MLflow Registry: %s",
            model_uri
        )

        try:

            model = mlflow.pytorch.load_model(
                model_uri,
                map_location=DEVICE
            )

        except Exception as exc:

            raise RuntimeError(
                "Failed to load model from MLflow Registry."
            ) from exc

    else:

        raise ValueError(
            f"Invalid MODEL_SOURCE: {MODEL_SOURCE}. "
            "Expected 'registry' or 'local'."
        )

    # --------------------------------------------------------
    # PREPARE MODEL
    # --------------------------------------------------------

    model.to(DEVICE)
    model.eval()

    logger.info(
        "Model loaded successfully: %s",
        MODEL_NAME
    )

    logger.info(
        "Model source: %s",
        MODEL_SOURCE
    )

    logger.info(
        "Inference device: %s",
        DEVICE
    )

    return model


# ============================================================
# LAZY MODEL LOADING
# ============================================================

def get_model():

    global MODEL
    global MODEL_LOAD_ERROR

    # Fast path:
    # Model already loaded.
    if MODEL is not None:
        return MODEL

    # Prevent multiple API requests from loading
    # the same large model simultaneously.
    with MODEL_LOAD_LOCK:

        # Another request may have loaded the model
        # while this request was waiting for the lock.
        if MODEL is not None:
            return MODEL

        try:

            MODEL = load_model()

            MODEL_LOAD_ERROR = None

        except Exception as exc:

            MODEL_LOAD_ERROR = str(exc)

            logger.exception(
                "Model loading failed."
            )

            raise RuntimeError(
                "Production model could not be loaded."
            ) from exc

    return MODEL


# ============================================================
# MODEL STATUS
# ============================================================

def is_model_loaded() -> bool:
    """
    Returns True only when the model has actually
    been successfully loaded.
    """

    return MODEL is not None


def get_model_load_error():
    """
    Returns the latest model loading error.

    This is mainly useful for health checks and
    diagnostics.
    """

    return MODEL_LOAD_ERROR


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image: Image.Image) -> Dict:
    """
    Run classification on one PIL image.

    Args:
        image:
            PIL.Image.Image

    Returns:
        Dictionary containing:
            prediction
            confidence
            confidence_percent
            probabilities
    """

    if image is None:

        raise ValueError(
            "Image cannot be None."
        )

    # --------------------------------------------------------
    # CONVERT IMAGE TO RGB
    # --------------------------------------------------------

    image = image.convert("RGB")

    # --------------------------------------------------------
    # APPLY VALIDATION/TEST PREPROCESSING
    # --------------------------------------------------------

    input_tensor = TRANSFORM(image)

    # Add batch dimension
    input_tensor = input_tensor.unsqueeze(0)

    # Move tensor to CPU/GPU
    input_tensor = input_tensor.to(DEVICE)

    # --------------------------------------------------------
    # GET MODEL
    # --------------------------------------------------------

    model = get_model()

    # --------------------------------------------------------
    # MODEL INFERENCE
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(input_tensor)

        # Some architectures return an object
        # containing logits.
        if hasattr(output, "logits"):

            output = output.logits

        # Validate output shape
        if output.ndim != 2:

            raise RuntimeError(
                "Model returned an unexpected output shape."
            )

        if output.shape[1] != NUM_CLASSES:

            raise RuntimeError(
                "Model output class count does not match "
                f"configured classes ({NUM_CLASSES})."
            )

        # Convert logits to probabilities
        probabilities = F.softmax(
            output,
            dim=1
        )

        # Get highest probability
        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

    # --------------------------------------------------------
    # CONVERT TENSORS TO PYTHON VALUES
    # --------------------------------------------------------

    predicted_index = predicted_index.item()

    confidence = confidence.item()

    probability_values = (
        probabilities[0]
        .cpu()
        .tolist()
    )

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    if not 0 <= predicted_index < len(CLASS_NAMES):

        raise RuntimeError(
            "Model returned an invalid class index."
        )

    # --------------------------------------------------------
    # CLASS PROBABILITIES
    # --------------------------------------------------------

    class_probabilities = {}

    for class_name, probability in zip(
        CLASS_NAMES,
        probability_values
    ):

        class_probabilities[class_name] = round(
            float(probability),
            6
        )

    # --------------------------------------------------------
    # FINAL RESPONSE
    # --------------------------------------------------------

    result = {

        "prediction": CLASS_NAMES[predicted_index],

        "confidence": round(
            float(confidence),
            6
        ),

        "confidence_percent": round(
            float(confidence * 100),
            2
        ),

        "probabilities": class_probabilities
    }

    return result


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_info():

    return {

        "model": MODEL_NAME,

        "registered_model": REGISTERED_MODEL_NAME,

        "alias": MODEL_ALIAS,

        "model_source": MODEL_SOURCE,

        "classes": CLASS_NAMES,

        "image_size": IMAGE_SIZE,

        "device": str(DEVICE),

        "status": (
            "loaded"
            if is_model_loaded()
            else "not_loaded"
        )
    }