"""
KidneyVision-MLOps
Reusable Training Engine + MLflow Tracking

Responsibilities:
1. Load prepared dataset manifests.
2. Create model using the model factory.
3. Train one model.
4. Validate after every epoch.
5. Track metrics with MLflow.
6. Save the best checkpoint.
7. Calculate classification metrics.
8. Support CPU/GPU automatically.

IMPORTANT:
This file provides the reusable training engine.
Actual multi-model benchmarking will be done later.
"""

from pathlib import Path
import argparse
import json
import sys
import time
from typing import Dict, Tuple

import mlflow
import mlflow.pytorch
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ============================================================
# PROJECT IMPORT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from configs.training_config import (
    CLASS_NAMES,
    NUM_CLASSES,
    RANDOM_SEED,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WEIGHT_DECAY,
    OPTIMIZER_NAME,
    SCHEDULER_NAME,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    EARLY_STOPPING_PATIENCE,
    MONITOR_METRIC,
    DEVICE,
    NUM_WORKERS,
    PIN_MEMORY,
    USE_PRETRAINED_WEIGHTS,
    ARTIFACTS_DIR,
    set_seed,
    validate_config,
)

from src.models.model_factory import (
    create_model,
    get_model_image_size,
    count_parameters,
)

from src.preprocessing.image_preprocessor import (
    KidneyCTDataset,
)


# ============================================================
# PATHS
# ============================================================

MLFLOW_DIR = ARTIFACTS_DIR / "mlruns"

CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"

METRICS_DIR = ARTIFACTS_DIR / "metrics"


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_training_directories() -> None:
    """
    Create directories required by training.
    """

    MLFLOW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

def configure_mlflow() -> None:
    """
    Configure local MLflow tracking using SQLite.
    The MLflow database and artifacts remain inside the project.
    """

    mlflow_db = ARTIFACTS_DIR / "mlflow.db"
    tracking_uri = f"sqlite:///{mlflow_db.resolve().as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(
        "KidneyVision-Model-Benchmark"
    )

    print(
        f"MLflow tracking URI: {tracking_uri}"
    )


# ============================================================
# DATA LOADERS
# ============================================================

def create_loaders(
    image_size: int,
    batch_size: int,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    """
    Create train/validation/test DataLoaders.

    Uses the manifests generated during Step 3.
    """

    manifest_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "manifests"
    )

    train_manifest = (
        manifest_dir / "train.csv"
    )

    validation_manifest = (
        manifest_dir / "validation.csv"
    )

    test_manifest = (
        manifest_dir / "test.csv"
    )

    train_dataset = KidneyCTDataset(
        train_manifest,
        image_size=image_size,
        train=True,
    )

    validation_dataset = KidneyCTDataset(
        validation_manifest,
        image_size=image_size,
        train=False,
    )

    test_dataset = KidneyCTDataset(
        test_manifest,
        image_size=image_size,
        train=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
    )


# ============================================================
# LOSS
# ============================================================

def create_loss_function() -> nn.Module:
    """
    Cross-entropy loss for four-class classification.
    """

    return nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

def create_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
):
    """
    Create the configured optimizer.
    """

    if OPTIMIZER_NAME.lower() == "adamw":

        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    raise ValueError(
        f"Unsupported optimizer: {OPTIMIZER_NAME}"
    )


# ============================================================
# SCHEDULER
# ============================================================

def create_scheduler(
    optimizer,
):
    """
    Create the configured learning-rate scheduler.
    """

    if SCHEDULER_NAME.lower() == "reducelronplateau":

        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
        )

    raise ValueError(
        f"Unsupported scheduler: {SCHEDULER_NAME}"
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
) -> Tuple[float, float]:

    model.train()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )
        outputs = model(images)

        if hasattr(outputs, "logits"):
            loss = criterion(outputs.logits, labels)

            if outputs.aux_logits is not None:
               aux_loss = criterion(outputs.aux_logits, labels)
               loss = loss + 0.4 * aux_loss
        else:
            loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        if hasattr(outputs, "logits"):
            prediction_outputs = outputs.logits
        else:
            prediction_outputs = outputs

        predictions = torch.argmax(
            prediction_outputs,
            dim=1,
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )

    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    return (
        epoch_loss,
        epoch_accuracy,
    )


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict:

    model.eval()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            outputs = model(images)

            if hasattr(outputs, "logits"):
                outputs = outputs.logits

            loss = criterion(outputs, labels)

            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
    )

    return {
        "loss": float(epoch_loss),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": matrix.tolist(),
        "predictions": all_predictions,
        "labels": all_labels,
    }


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model: nn.Module,
    model_name: str,
    epoch: int,
    validation_loss: float,
    optimizer,
) -> Path:

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"{safe_name}_best.pt"
    )

    checkpoint = {
        "model_name": model_name,
        "epoch": epoch,
        "validation_loss": validation_loss,
        "class_names": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    return checkpoint_path


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(
    model_name: str,
    metrics: Dict,
) -> Path:

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    metrics_path = (
        METRICS_DIR
        / f"{safe_name}_metrics.json"
    )

    serializable_metrics = {
        key: value
        for key, value in metrics.items()
        if key not in {
            "predictions",
            "labels",
        }
    }

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            serializable_metrics,
            file,
            indent=4,
        )

    return metrics_path


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    model_name: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
) -> Dict:

    set_seed(RANDOM_SEED)

    validate_config()

    create_training_directories()

    configure_mlflow()

    image_size = get_model_image_size(
        model_name
    )

    print("\n" + "=" * 70)
    print(
        f"TRAINING MODEL: {model_name}"
    )
    print("=" * 70)

    print(f"Device       : {DEVICE}")
    print(f"Image size   : {image_size}x{image_size}")
    print(f"Batch size   : {batch_size}")
    print(f"Epochs       : {epochs}")
    print(f"Learning rate: {learning_rate}")

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    print("\nCreating DataLoaders...")

    (
        train_loader,
        validation_loader,
        test_loader,
    ) = create_loaders(
        image_size=image_size,
        batch_size=batch_size,
    )

    print(
        f"Train samples      : {len(train_loader.dataset)}"
    )

    print(
        f"Validation samples : {len(validation_loader.dataset)}"
    )

    print(
        f"Test samples       : {len(test_loader.dataset)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nCreating model...")

    model = create_model(
        model_name=model_name,
        num_classes=NUM_CLASSES,
        pretrained=USE_PRETRAINED_WEIGHTS,
    )

    model = model.to(DEVICE)

    total_parameters, trainable_parameters = (
        count_parameters(model)
    )

    print(
        f"Total parameters    : {total_parameters:,}"
    )

    print(
        f"Trainable parameters: {trainable_parameters:,}"
    )

    # --------------------------------------------------------
    # Training components
    # --------------------------------------------------------

    criterion = create_loss_function()

    optimizer = create_optimizer(
        model,
        learning_rate,
        weight_decay,
    )

    scheduler = create_scheduler(
        optimizer
    )

    best_validation_loss = float("inf")

    best_epoch = 0

    epochs_without_improvement = 0

    history = []

    # --------------------------------------------------------
    # MLflow Run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name=model_name
    ):

        mlflow.log_params({
            "model_name": model_name,
            "num_classes": NUM_CLASSES,
            "image_size": image_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "optimizer": OPTIMIZER_NAME,
            "scheduler": SCHEDULER_NAME,
            "random_seed": RANDOM_SEED,
            "device": str(DEVICE),
            "pretrained": USE_PRETRAINED_WEIGHTS,
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
        })

        # ----------------------------------------------------
        # Epoch loop
        # ----------------------------------------------------

        for epoch in range(
            1,
            epochs + 1,
        ):

            start_time = time.time()

            train_loss, train_accuracy = (
                train_one_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    DEVICE,
                )
            )

            validation_metrics = evaluate(
                model,
                validation_loader,
                criterion,
                DEVICE,
            )

            validation_loss = (
                validation_metrics["loss"]
            )

            scheduler.step(
                validation_loss
            )

            current_lr = optimizer.param_groups[0]["lr"]

            epoch_time = (
                time.time()
                - start_time
            )

            # ------------------------------------------------
            # Log epoch metrics
            # ------------------------------------------------

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": validation_metrics["loss"],
                    "val_accuracy": validation_metrics["accuracy"],
                    "val_precision": validation_metrics["precision"],
                    "val_recall": validation_metrics["recall"],
                    "val_f1": validation_metrics["f1"],
                    "learning_rate": current_lr,
                    "epoch_time_seconds": epoch_time,
                },
                step=epoch,
            )

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": validation_metrics["loss"],
                "val_accuracy": validation_metrics["accuracy"],
                "val_precision": validation_metrics["precision"],
                "val_recall": validation_metrics["recall"],
                "val_f1": validation_metrics["f1"],
                "learning_rate": current_lr,
                "epoch_time_seconds": epoch_time,
            }

            history.append(
                epoch_record
            )

            print(
                f"\nEpoch {epoch}/{epochs}"
            )

            print(
                f"  Train Loss     : {train_loss:.4f}"
            )

            print(
                f"  Train Accuracy : {train_accuracy:.4f}"
            )

            print(
                f"  Val Loss       : {validation_metrics['loss']:.4f}"
            )

            print(
                f"  Val Accuracy   : {validation_metrics['accuracy']:.4f}"
            )

            print(
                f"  Val Precision  : {validation_metrics['precision']:.4f}"
            )

            print(
                f"  Val Recall     : {validation_metrics['recall']:.4f}"
            )

            print(
                f"  Val F1         : {validation_metrics['f1']:.4f}"
            )

            print(
                f"  Learning Rate  : {current_lr:.6f}"
            )

            # ------------------------------------------------
            # Best checkpoint
            # ------------------------------------------------

            if validation_loss < best_validation_loss:

                best_validation_loss = validation_loss

                best_epoch = epoch

                epochs_without_improvement = 0

                checkpoint_path = save_checkpoint(
                    model=model,
                    model_name=model_name,
                    epoch=epoch,
                    validation_loss=validation_loss,
                    optimizer=optimizer,
                )

                mlflow.log_artifact(
                    str(checkpoint_path)
                )

                print(
                    "  Best checkpoint: SAVED"
                )

            else:

                epochs_without_improvement += 1

                print(
                    "  Best checkpoint: unchanged"
                )

            # ------------------------------------------------
            # Early stopping
            # ------------------------------------------------

            if (
                epochs_without_improvement
                >= EARLY_STOPPING_PATIENCE
            ):

                print(
                    "\nEarly stopping triggered."
                )

                break

        # ----------------------------------------------------
        # Final validation metrics
        # ----------------------------------------------------

        final_validation = evaluate(
            model,
            validation_loader,
            criterion,
            DEVICE,
        )

        final_test = evaluate(
            model,
            test_loader,
            criterion,
            DEVICE,
        )

        # ----------------------------------------------------
        # Final MLflow metrics
        # ----------------------------------------------------

        mlflow.log_metrics({
            "best_epoch": float(best_epoch),

            "best_val_loss": float(
                best_validation_loss
            ),

            "final_val_loss": float(
                final_validation["loss"]
            ),

            "final_val_accuracy": float(
                final_validation["accuracy"]
            ),

            "final_val_precision": float(
                final_validation["precision"]
            ),

            "final_val_recall": float(
                final_validation["recall"]
            ),

            "final_val_f1": float(
                final_validation["f1"]
            ),

            "test_loss": float(
                final_test["loss"]
            ),

            "test_accuracy": float(
                final_test["accuracy"]
            ),

            "test_precision": float(
                final_test["precision"]
            ),

            "test_recall": float(
                final_test["recall"]
            ),

            "test_f1": float(
                final_test["f1"]
            ),
        })

        # ----------------------------------------------------
        # Save metrics
        # ----------------------------------------------------

        test_metrics = {
            "model_name": model_name,
            "best_epoch": best_epoch,
            "best_val_loss": best_validation_loss,
            "validation": {
                "loss": final_validation["loss"],
                "accuracy": final_validation["accuracy"],
                "precision": final_validation["precision"],
                "recall": final_validation["recall"],
                "f1": final_validation["f1"],
                "confusion_matrix": final_validation[
                    "confusion_matrix"
                ],
            },
            "test": {
                "loss": final_test["loss"],
                "accuracy": final_test["accuracy"],
                "precision": final_test["precision"],
                "recall": final_test["recall"],
                "f1": final_test["f1"],
                "confusion_matrix": final_test[
                    "confusion_matrix"
                ],
            },
            "history": history,
        }

        metrics_path = save_metrics(
            model_name,
            test_metrics,
        )

        mlflow.log_artifact(
            str(metrics_path)
        )

        # ----------------------------------------------------
        # Log model
        # ----------------------------------------------------

        input_example = torch.zeros(
            1,
            3,
            image_size,
            image_size,
        )

        signature = ModelSignature(
            inputs=Schema([
                TensorSpec(
                    np.dtype(np.float32),
                    [-1, 3, image_size, image_size],
                )
            ]),
            outputs=Schema([
                TensorSpec(
                    np.dtype(np.float32),
                    [-1, NUM_CLASSES],
                )
            ]),
        )

        mlflow.pytorch.log_model(
            model,
            name="model",
            input_example=input_example,
            signature=signature,
            serialization_format="pickle",
        )

        print("\n" + "=" * 70)
        print(
            f"TRAINING COMPLETE: {model_name}"
        )
        print("=" * 70)

        print(
            f"Best epoch       : {best_epoch}"
        )

        print(
            f"Best val loss    : {best_validation_loss:.4f}"
        )

        print(
            f"Test accuracy    : {final_test['accuracy']:.4f}"
        )

        print(
            f"Test precision   : {final_test['precision']:.4f}"
        )

        print(
            f"Test recall      : {final_test['recall']:.4f}"
        )

        print(
            f"Test F1          : {final_test['f1']:.4f}"
        )

        print(
            f"Metrics saved    : {metrics_path}"
        )

    return test_metrics


# ============================================================
# SMOKE TEST
# ============================================================

def run_smoke_test() -> None:
    """
    Validate the training infrastructure without
    performing actual model training.

    This test:
        - validates configuration
        - creates directories
        - configures MLflow
        - loads manifests
        - creates DataLoaders
        - creates one model
        - runs one forward pass
        - verifies output shape

    No weights are downloaded.
    No training occurs.
    """

    print("=" * 70)
    print("STEP 7 - TRAINING ENGINE SMOKE TEST")
    print("=" * 70)

    set_seed(RANDOM_SEED)

    validate_config()

    create_training_directories()

    configure_mlflow()

    model_name = "ResNet50"

    image_size = get_model_image_size(
        model_name
    )

    print(
        f"\nTesting model     : {model_name}"
    )

    print(
        f"Image size        : {image_size}x{image_size}"
    )

    print(
        f"Device            : {DEVICE}"
    )

    print("\nCreating DataLoaders...")

    (
        train_loader,
        validation_loader,
        test_loader,
    ) = create_loaders(
        image_size=image_size,
        batch_size=DEFAULT_BATCH_SIZE,
    )

    print(
        f"Train samples      : {len(train_loader.dataset)}"
    )

    print(
        f"Validation samples : {len(validation_loader.dataset)}"
    )

    print(
        f"Test samples       : {len(test_loader.dataset)}"
    )

    print("\nCreating model...")

    # IMPORTANT:
    # No pretrained download during smoke test.
    model = create_model(
        model_name=model_name,
        num_classes=NUM_CLASSES,
        pretrained=False,
    )

    model = model.to(DEVICE)

    total_parameters, trainable_parameters = (
        count_parameters(model)
    )

    print(
        f"Total parameters    : {total_parameters:,}"
    )

    print(
        f"Trainable parameters: {trainable_parameters:,}"
    )

    print("\nLoading one batch...")

    images, labels = next(
        iter(train_loader)
    )

    images = images.to(DEVICE)

    print(
        f"Input shape        : {tuple(images.shape)}"
    )

    print(
        f"Labels shape       : {tuple(labels.shape)}"
    )

    print("\nRunning forward pass...")

    model.eval()

    with torch.no_grad():

        outputs = model(images)

    print(
        f"Output shape       : {tuple(outputs.shape)}"
    )

    expected_shape = (
        images.shape[0],
        NUM_CLASSES,
    )

    if tuple(outputs.shape) != expected_shape:

        raise RuntimeError(
            f"Unexpected output shape.\n"
            f"Expected: {expected_shape}\n"
            f"Received: {tuple(outputs.shape)}"
        )

    print("\nTesting MLflow run...")

    with mlflow.start_run(
        run_name="training_engine_smoke_test"
    ):

        mlflow.log_params({
            "test": True,
            "model_name": model_name,
            "num_classes": NUM_CLASSES,
            "image_size": image_size,
        })

        mlflow.log_metric(
            "smoke_test",
            1.0,
        )

    print(
        "\nMLflow smoke test: PASS"
    )

    print("\n" + "=" * 70)
    print("TRAINING ENGINE SMOKE TEST PASSED")
    print("=" * 70)


# ============================================================
# COMMAND LINE
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "KidneyVision-MLOps training engine"
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default="ResNet50",
        help="Model name to train.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training batch size.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate.",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run infrastructure test without training.",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    args = parse_arguments()

    if args.smoke_test:

        run_smoke_test()

    else:

        train_model(
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )