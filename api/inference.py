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

MODEL_FILE = (
    DEPLOYMENT_MODEL_DIR
    / "data"
    / "model.pth"
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

# local:
#   Load the bundled PyTorch model.pth
#
# registry:
#   Load the model from MLflow Model Registry
#
# Render/Docker deployment uses:
#   MODEL_SOURCE=local

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

    1. Local PyTorch deployment model
    2. MLflow Model Registry

    Local deployment expects:

        artifacts/
        └── deployment_model/
            └── data/
                └── model.pth

    Returns:
        torch.nn.Module

    Raises:
        FileNotFoundError:
            If the local model file is missing.

        RuntimeError:
            If model loading fails.

        ValueError:
            If MODEL_SOURCE is invalid.
    """

    # ========================================================
    # LOCAL MODEL
    # ========================================================

    if MODEL_SOURCE == "local":

        if not MODEL_FILE.exists():

            raise FileNotFoundError(
                "Deployment model not found. "
                f"Expected model.pth at: {MODEL_FILE}"
            )

        logger.info(
            "Loading PyTorch model from: %s",
            MODEL_FILE
        )

        try:

            model = torch.load(
                MODEL_FILE,
                map_location=DEVICE,
                weights_only=False
            )

        except Exception as exc:

            raise RuntimeError(
                "Failed to load local PyTorch deployment model."
            ) from exc

    # ========================================================
    # MLFLOW MODEL REGISTRY
    # ========================================================

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

    # ========================================================
    # INVALID SOURCE
    # ========================================================

    else:

        raise ValueError(
            "Invalid MODEL_SOURCE. "
            f"Expected 'local' or 'registry', "
            f"got '{MODEL_SOURCE}'."
        )

    # ========================================================
    # MODEL VALIDATION
    # ========================================================

    if model is None:

        raise RuntimeError(
            "Model loader returned None."
        )

    if not isinstance(model, torch.nn.Module):

        raise TypeError(
            "Loaded object is not a PyTorch model. "
            f"Got: {type(model)}"
        )

    model = model.to(DEVICE)

    model.eval()

    logger.info(
        "Model loaded successfully: %s",
        MODEL_NAME
    )

    logger.info(
        "Device: %s",
        DEVICE
    )

    return model


# ============================================================
# THREAD-SAFE MODEL ACCESS
# ============================================================

def get_model():
    """
    Return the cached model.

    The model is loaded lazily on the first inference request.
    Thread-safe so multiple API requests cannot initialize
    the model simultaneously.
    """

    global MODEL
    global MODEL_LOAD_ERROR

    if MODEL is not None:
        return MODEL

    with MODEL_LOAD_LOCK:

        if MODEL is not None:
            return MODEL

        try:

            MODEL = load_model()

            MODEL_LOAD_ERROR = None

            return MODEL

        except Exception as exc:

            MODEL_LOAD_ERROR = str(exc)

            logger.exception(
                "Model loading failed."
            )

            raise RuntimeError(
                "Production model could not be loaded."
            ) from exc


# ============================================================
# MODEL STATUS
# ============================================================

def is_model_loaded() -> bool:
    """
    Return True when the model has been loaded successfully.
    """

    return MODEL is not None


def get_model_load_error():
    """
    Return the last model loading error, if any.
    """

    return MODEL_LOAD_ERROR


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image(image: Image.Image):
    """
    Validate uploaded image before inference.
    """

    if image is None:

        raise ValueError(
            "Image cannot be None."
        )

    if not isinstance(image, Image.Image):

        raise TypeError(
            "Input must be a PIL Image."
        )

    if image.width <= 0 or image.height <= 0:

        raise ValueError(
            "Image dimensions must be greater than zero."
        )


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image: Image.Image) -> Dict:
    """
    Run image classification.

    Args:
        image:
            PIL Image.

    Returns:
        Dictionary containing:

        prediction
        confidence
        confidence_percent
        probabilities
    """

    validate_image(image)

    model = get_model()

    # --------------------------------------------------------
    # Convert image to RGB
    # --------------------------------------------------------

    image = image.convert("RGB")

    # --------------------------------------------------------
    # Apply inference preprocessing
    # --------------------------------------------------------

    try:

        input_tensor = TRANSFORM(image)

    except Exception as exc:

        raise RuntimeError(
            "Failed to preprocess input image."
        ) from exc

    if not isinstance(input_tensor, torch.Tensor):

        raise TypeError(
            "Image transform did not return a torch.Tensor."
        )

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(DEVICE)

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    try:

        with torch.no_grad():

            output = model(input_tensor)

    except Exception as exc:

        raise RuntimeError(
            "Model inference failed."
        ) from exc

    # --------------------------------------------------------
    # Handle model output
    # --------------------------------------------------------

    if hasattr(output, "logits"):

        output = output.logits

    if not isinstance(output, torch.Tensor):

        raise TypeError(
            "Model output is not a torch.Tensor."
        )

    if output.ndim != 2:

        raise ValueError(
            "Unexpected model output shape: "
            f"{tuple(output.shape)}"
        )

    if output.shape[0] != 1:

        raise ValueError(
            "Expected a single-image prediction."
        )

    if output.shape[1] != NUM_CLASSES:

        raise ValueError(
            "Unexpected number of model outputs. "
            f"Expected {NUM_CLASSES}, "
            f"got {output.shape[1]}."
        )

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    probabilities_tensor = F.softmax(
        output,
        dim=1
    )[0]

    predicted_index = int(
        torch.argmax(
            probabilities_tensor
        ).item()
    )

    confidence = float(
        probabilities_tensor[predicted_index].item()
    )

    # --------------------------------------------------------
    # Class probabilities
    # --------------------------------------------------------

    probabilities = {}

    for index, class_name in enumerate(CLASS_NAMES):

        probabilities[class_name] = float(
            probabilities_tensor[index].item()
        )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {
        "prediction": CLASS_NAMES[predicted_index],
        "confidence": confidence,
        "confidence_percent": round(
            confidence * 100,
            2
        ),
        "probabilities": probabilities
    }

    logger.info(
        "Prediction: %s | Confidence: %.4f",
        result["prediction"],
        result["confidence"]
    )

    return result


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_info() -> Dict:
    """
    Return model/deployment information for the API.
    """

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