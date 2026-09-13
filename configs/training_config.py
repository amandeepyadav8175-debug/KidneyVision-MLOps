"""
KidneyVision-MLOps
Central training configuration.

All important training settings are kept here so that
experiments remain reproducible and easy to modify.
"""

from pathlib import Path
import random

import numpy as np
import torch


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_DIR = PROCESSED_DIR / "manifests"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CONFIG_DIR = PROJECT_ROOT / "configs"


# ============================================================
# DATASET CONFIGURATION
# ============================================================

CLASS_NAMES = [
    "Normal",
    "Cyst",
    "Stone",
    "Tumor",
]

NUM_CLASSES = len(CLASS_NAMES)

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}

INDEX_TO_CLASS = {
    index: class_name
    for class_name, index in CLASS_TO_INDEX.items()
}


# ============================================================
# DATA SPLIT CONFIGURATION
# ============================================================

TRAIN_SPLIT = 0.70
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

RANDOM_SEED = 42


# ============================================================
# IMAGE CONFIGURATION
# ============================================================

DEFAULT_IMAGE_SIZE = 224

# InceptionV3 normally uses 299x299.
INCEPTION_IMAGE_SIZE = 299


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

DEFAULT_BATCH_SIZE = 16

DEFAULT_EPOCHS = 10

DEFAULT_LEARNING_RATE = 1e-4

DEFAULT_WEIGHT_DECAY = 1e-4


# ============================================================
# OPTIMIZER
# ============================================================

OPTIMIZER_NAME = "AdamW"


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

SCHEDULER_NAME = "ReduceLROnPlateau"

SCHEDULER_FACTOR = 0.1

SCHEDULER_PATIENCE = 2


# ============================================================
# EARLY STOPPING
# ============================================================

EARLY_STOPPING_PATIENCE = 4

MONITOR_METRIC = "val_loss"


# ============================================================
# MODEL CONFIGURATION
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


# ============================================================
# PRETRAINED MODEL CONFIGURATION
# ============================================================

USE_PRETRAINED_WEIGHTS = True


# ============================================================
# DEVICE CONFIGURATION
# ============================================================

def get_device() -> torch.device:
    """
    Select GPU when CUDA is available.
    Otherwise use CPU.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


DEVICE = get_device()


# ============================================================
# DATA LOADING
# ============================================================

NUM_WORKERS = 0

PIN_MEMORY = torch.cuda.is_available()


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set random seeds for reproducible experiments.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Makes CUDA operations deterministic where possible.
    # This can reduce performance slightly but improves
    # experiment reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_required_directories() -> None:
    """
    Create directories required by the project.
    """

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONFIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

def validate_config() -> None:
    """
    Validate important configuration values.
    """

    split_sum = (
        TRAIN_SPLIT
        + VALIDATION_SPLIT
        + TEST_SPLIT
    )

    if abs(split_sum - 1.0) > 1e-9:
        raise ValueError(
            "Train/validation/test splits must sum to 1.0"
        )

    if NUM_CLASSES != 4:
        raise ValueError(
            f"Expected 4 classes, found {NUM_CLASSES}"
        )

    if DEFAULT_IMAGE_SIZE <= 0:
        raise ValueError(
            "Image size must be greater than zero."
        )

    if DEFAULT_BATCH_SIZE <= 0:
        raise ValueError(
            "Batch size must be greater than zero."
        )

    if DEFAULT_EPOCHS <= 0:
        raise ValueError(
            "Epochs must be greater than zero."
        )

    if DEFAULT_LEARNING_RATE <= 0:
        raise ValueError(
            "Learning rate must be greater than zero."
        )

    if DEFAULT_WEIGHT_DECAY < 0:
        raise ValueError(
            "Weight decay cannot be negative."
        )

    if not MODEL_NAMES:
        raise ValueError(
            "At least one model must be configured."
        )


# ============================================================
# CONFIGURATION DISPLAY
# ============================================================

def print_config() -> None:
    """
    Print the active configuration.
    """

    print("=" * 70)
    print("KIDNEYVISION-MLOPS - TRAINING CONFIGURATION")
    print("=" * 70)

    print("\nPROJECT")
    print(f"Project root       : {PROJECT_ROOT}")
    print(f"Artifacts directory: {ARTIFACTS_DIR}")

    print("\nDATASET")
    print(f"Classes             : {CLASS_NAMES}")
    print(f"Number of classes   : {NUM_CLASSES}")

    print("\nDATA SPLIT")
    print(f"Train               : {TRAIN_SPLIT:.0%}")
    print(f"Validation          : {VALIDATION_SPLIT:.0%}")
    print(f"Test                : {TEST_SPLIT:.0%}")

    print("\nIMAGE")
    print(f"Default image size  : {DEFAULT_IMAGE_SIZE}")
    print(f"InceptionV3 size    : {INCEPTION_IMAGE_SIZE}")

    print("\nTRAINING")
    print(f"Batch size          : {DEFAULT_BATCH_SIZE}")
    print(f"Epochs              : {DEFAULT_EPOCHS}")
    print(f"Learning rate       : {DEFAULT_LEARNING_RATE}")
    print(f"Weight decay        : {DEFAULT_WEIGHT_DECAY}")

    print("\nOPTIMIZER")
    print(f"Optimizer           : {OPTIMIZER_NAME}")

    print("\nSCHEDULER")
    print(f"Scheduler            : {SCHEDULER_NAME}")
    print(f"Factor               : {SCHEDULER_FACTOR}")
    print(f"Patience             : {SCHEDULER_PATIENCE}")

    print("\nEARLY STOPPING")
    print(f"Patience             : {EARLY_STOPPING_PATIENCE}")
    print(f"Monitor              : {MONITOR_METRIC}")

    print("\nMODELS")
    for model_name in MODEL_NAMES:
        print(f"  - {model_name}")

    print("\nPRETRAINED WEIGHTS")
    print(f"Use pretrained       : {USE_PRETRAINED_WEIGHTS}")

    print("\nHARDWARE")
    print(f"Device               : {DEVICE}")
    print(f"CUDA available       : {torch.cuda.is_available()}")
    print(f"DataLoader workers   : {NUM_WORKERS}")
    print(f"Pin memory           : {PIN_MEMORY}")

    print("\nREPRODUCIBILITY")
    print(f"Random seed          : {RANDOM_SEED}")

    print("\nCONFIGURATION VALIDATION: PASSED")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    set_seed()

    create_required_directories()

    validate_config()

    print_config()