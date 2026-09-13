import sys
from pathlib import Path

import pytest
import torch


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from configs.training_config import CLASS_NAMES
from src.models.model_factory import (
    create_model,
    count_parameters,
)


# ============================================================
# TEST CONFIGURATION
# ============================================================

MODEL_NAMES = [
    "VGG16",
    "ResNet50",
    "InceptionV3",
    "EfficientNet-B0",
    "EANet",
    "CCT",
    "SwinTransformer",
]

NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# TEST 1
# ============================================================

@pytest.mark.parametrize(
    "model_name",
    MODEL_NAMES
)
def test_model_creation(model_name):

    model = create_model(
        model_name=model_name,
        num_classes=NUM_CLASSES,
        pretrained=False
    )

    assert model is not None

    total_parameters, trainable_parameters = count_parameters(model)

    assert total_parameters > 0
    assert trainable_parameters > 0


# ============================================================
# TEST 2
# ============================================================

@pytest.mark.parametrize(
    "model_name",
    MODEL_NAMES
)
def test_model_output_shape(model_name):

    model = create_model(
        model_name=model_name,
        num_classes=NUM_CLASSES,
        pretrained=False
    )

    model.eval()

    # InceptionV3 expects 299x299.
    # Other models use 224x224.

    if model_name == "InceptionV3":

        image_size = 299

    else:

        image_size = 224

    input_tensor = torch.randn(
        1,
        3,
        image_size,
        image_size
    )

    with torch.no_grad():

        output = model(input_tensor)

    # Some torchvision models may return
    # an object containing logits.

    if hasattr(output, "logits"):

        output = output.logits

    assert output.shape == (
        1,
        NUM_CLASSES
    )


# ============================================================
# TEST 3
# ============================================================

def test_class_configuration():

    assert CLASS_NAMES == [
        "Normal",
        "Cyst",
        "Stone",
        "Tumor",
    ]

    assert NUM_CLASSES == 4