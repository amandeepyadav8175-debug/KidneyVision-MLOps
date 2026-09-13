from pathlib import Path
from typing import Dict
import os

import mlflow
import torch
import torch.nn.functional as F
from PIL import Image

from src.preprocessing.image_preprocessor import build_transforms
from configs.training_config import CLASS_NAMES


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

# Possible values:
#
# registry -> Load model from MLflow Model Registry
# local    -> Load bundled model from artifacts/deployment_model
#
# Local development uses MLflow Registry by default.
# Docker will set MODEL_SOURCE=local.

MODEL_SOURCE = os.getenv("MODEL_SOURCE", "registry").strip().lower()


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
# LOAD MODEL
# ============================================================

def load_model():

    # --------------------------------------------------------
    # OPTION 1: LOAD FROM LOCAL DEPLOYMENT MODEL
    # --------------------------------------------------------

    if MODEL_SOURCE == "local":

        model_file = DEPLOYMENT_MODEL_DIR / "MLmodel"

        if not model_file.exists():
            raise FileNotFoundError(
                f"Deployment model not found.\n"
                f"Expected MLmodel at:\n"
                f"{model_file}"
            )

        model_uri = str(DEPLOYMENT_MODEL_DIR)

        print(
            f"Loading model from local deployment directory:\n"
            f"{model_uri}"
        )

        model = mlflow.pytorch.load_model(
            model_uri,
            map_location=DEVICE
        )

    # --------------------------------------------------------
    # OPTION 2: LOAD FROM MLflow MODEL REGISTRY
    # --------------------------------------------------------

    elif MODEL_SOURCE == "registry":

        mlflow.set_tracking_uri(
            f"sqlite:///{MLFLOW_DB}"
        )

        model_uri = (
            f"models:/{REGISTERED_MODEL_NAME}"
            f"@{MODEL_ALIAS}"
        )

        print(
            f"Loading model from MLflow Registry:\n"
            f"{model_uri}"
        )

        model = mlflow.pytorch.load_model(
            model_uri,
            map_location=DEVICE
        )

    # --------------------------------------------------------
    # INVALID MODEL SOURCE
    # --------------------------------------------------------

    else:

        raise ValueError(
            f"Invalid MODEL_SOURCE: {MODEL_SOURCE}\n"
            f"Use either 'registry' or 'local'."
        )

    # --------------------------------------------------------
    # PREPARE MODEL
    # --------------------------------------------------------

    model.to(DEVICE)
    model.eval()

    print(f"Model loaded: {MODEL_NAME}")
    print(f"Model source: {MODEL_SOURCE}")
    print(f"Device: {DEVICE}")

    return model


# ============================================================
# LOAD MODEL ON API STARTUP
# ============================================================

MODEL = load_model()


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image: Image.Image) -> Dict:

    # Convert image to RGB
    image = image.convert("RGB")

    # Apply validation/test preprocessing
    input_tensor = TRANSFORM(image)

    # Add batch dimension
    input_tensor = input_tensor.unsqueeze(0)

    # Move tensor to CPU/GPU
    input_tensor = input_tensor.to(DEVICE)

    # --------------------------------------------------------
    # MODEL INFERENCE
    # --------------------------------------------------------

    with torch.no_grad():

        output = MODEL(input_tensor)

        # Some architectures return an object
        # containing logits.
        if hasattr(output, "logits"):
            output = output.logits

        # Convert logits to probabilities
        probabilities = F.softmax(output, dim=1)

        # Get highest probability
        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

    # Convert tensors to Python values
    predicted_index = predicted_index.item()

    confidence = confidence.item()

    probability_values = (
        probabilities[0]
        .cpu()
        .tolist()
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
            probability,
            6
        )

    # --------------------------------------------------------
    # FINAL RESPONSE
    # --------------------------------------------------------

    result = {

        "prediction": CLASS_NAMES[predicted_index],

        "confidence": round(
            confidence,
            6
        ),

        "confidence_percent": round(
            confidence * 100,
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

        "status": "loaded"
    }