"""
KidneyVision-MLOps
Model Evaluation Pipeline

Evaluates a trained model on the untouched test set.

Metrics:
- Accuracy
- Precision
- Recall / Sensitivity
- Specificity
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion Matrix
- Per-class metrics
- Classification Report

Artifacts:
- evaluation JSON
- classification report CSV
- confusion matrix CSV
- confusion matrix PNG
- ROC curve PNG
- PR curve PNG

The evaluator does NOT retrain the model.
"""

from pathlib import Path
import json
import sys

import mlflow
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize


# ---------------------------------------------------------------------
# PROJECT IMPORTS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

from configs.training_config import (
    CLASS_NAMES,
    get_device,
)

from src.models.model_factory import create_model

from src.preprocessing.image_preprocessor import (
    KidneyCTDataset,
    build_transforms,
)


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

MANIFEST_DIR = PROJECT_ROOT / "data" / "processed" / "manifests"

CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"

EVALUATION_DIR = PROJECT_ROOT / "artifacts" / "evaluation"

MLFLOW_DB = PROJECT_ROOT / "artifacts" / "mlflow.db"


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

MODEL_NAME = "SwinTransformer"

CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "swintransformer_best.pt"
)

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0

NUM_CLASSES = len(CLASS_NAMES)


# ---------------------------------------------------------------------
# DIRECTORY SETUP
# ---------------------------------------------------------------------

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------------------
# DATA LOADER
# ---------------------------------------------------------------------

def create_test_loader():
    """
    Create deterministic test DataLoader.
    """

    from torch.utils.data import DataLoader

    test_manifest = MANIFEST_DIR / "test.csv"

    if not test_manifest.exists():
        raise FileNotFoundError(
            f"Test manifest not found:\n{test_manifest}"
        )

    dataset = KidneyCTDataset(
        manifest_path=test_manifest,
        image_size=IMAGE_SIZE,
        train=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return loader


# ---------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------

def load_checkpoint_model(
    model_name: str,
    checkpoint_path: Path,
    device: torch.device,
):
    """
    Recreate model architecture and load trained weights.
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    print("\nLoading model...")
    print(f"Model      : {model_name}")
    print(f"Checkpoint : {checkpoint_path}")

    model = create_model(
        model_name=model_name,
        num_classes=NUM_CLASSES,
        pretrained=False,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # -------------------------------------------------------------
    # Support both:
    # 1. state_dict directly
    # 2. dictionary containing "model_state_dict"
    # -------------------------------------------------------------

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            # Some checkpoints may directly contain state_dict
            state_dict = checkpoint
    else:
        raise ValueError(
            "Unsupported checkpoint format."
        )

    # Remove possible DataParallel prefix
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned_state_dict[key] = value

    missing_keys, unexpected_keys = model.load_state_dict(
        cleaned_state_dict,
        strict=False,
    )

    if missing_keys:
        print(
            f"Warning: missing keys: {missing_keys}"
        )

    if unexpected_keys:
        print(
            f"Warning: unexpected keys: {unexpected_keys}"
        )

    model.to(device)

    model.eval()

    print("Model loaded successfully.")

    return model


# ---------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------

def collect_predictions(
    model,
    test_loader,
    device,
):
    """
    Run inference on the complete test set.
    """

    all_labels = []
    all_predictions = []
    all_probabilities = []

    print("\nRunning inference on test set...")

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            # Inception-like models may return special output objects
            if hasattr(outputs, "logits"):
                outputs = outputs.logits

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            predictions = torch.argmax(
                probabilities,
                dim=1,
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    y_true = np.array(all_labels)

    y_pred = np.array(all_predictions)

    y_prob = np.array(all_probabilities)

    print(f"Test samples: {len(y_true)}")

    return y_true, y_pred, y_prob


# ---------------------------------------------------------------------
# SPECIFICITY
# ---------------------------------------------------------------------

def calculate_specificity(
    y_true,
    y_pred,
):
    """
    Calculate one-vs-rest specificity for every class.

    Specificity = TN / (TN + FP)
    """

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=range(NUM_CLASSES),
    )

    specificities = {}

    for class_index, class_name in enumerate(CLASS_NAMES):

        true_positive = cm[
            class_index,
            class_index
        ]

        false_positive = (
            cm[:, class_index].sum()
            - true_positive
        )

        false_negative = (
            cm[class_index, :].sum()
            - true_positive
        )

        true_negative = (
            cm.sum()
            - true_positive
            - false_positive
            - false_negative
        )

        denominator = (
            true_negative
            + false_positive
        )

        if denominator == 0:
            specificity = 0.0
        else:
            specificity = (
                true_negative / denominator
            )

        specificities[class_name] = float(
            specificity
        )

    macro_specificity = float(
        np.mean(
            list(specificities.values())
        )
    )

    return specificities, macro_specificity


# ---------------------------------------------------------------------
# ROC-AUC
# ---------------------------------------------------------------------

def calculate_roc_auc(
    y_true,
    y_prob,
):
    """
    Calculate multiclass ROC-AUC using One-vs-Rest.
    """

    try:

        y_true_binary = label_binarize(
            y_true,
            classes=range(NUM_CLASSES),
        )

        score = roc_auc_score(
            y_true_binary,
            y_prob,
            multi_class="ovr",
            average="macro",
        )

        return float(score)

    except ValueError as error:

        print(
            f"ROC-AUC could not be calculated: {error}"
        )

        return None


# ---------------------------------------------------------------------
# PR-AUC
# ---------------------------------------------------------------------

def calculate_pr_auc(
    y_true,
    y_prob,
):
    """
    Calculate macro-average Precision-Recall AUC.
    """

    try:

        y_true_binary = label_binarize(
            y_true,
            classes=range(NUM_CLASSES),
        )

        average_precisions = []

        for class_index in range(NUM_CLASSES):

            precision, recall, _ = (
                precision_recall_curve(
                    y_true_binary[:, class_index],
                    y_prob[:, class_index],
                )
            )

            ap = average_precision_score(
                y_true_binary[:, class_index],
                y_prob[:, class_index],
            )

            average_precisions.append(
                float(ap)
            )

        macro_pr_auc = float(
            np.mean(average_precisions)
        )

        return (
            macro_pr_auc,
            average_precisions,
        )

    except ValueError as error:

        print(
            f"PR-AUC could not be calculated: {error}"
        )

        return None, []


# ---------------------------------------------------------------------
# CONFUSION MATRIX PLOT
# ---------------------------------------------------------------------

def save_confusion_matrix(
    cm,
    output_path,
):
    """
    Save confusion matrix visualization.
    """

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    image = ax.imshow(cm)

    ax.set_title(
        f"{MODEL_NAME} - Confusion Matrix"
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks(
        range(NUM_CLASSES)
    )

    ax.set_yticks(
        range(NUM_CLASSES)
    )

    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)

    for i in range(NUM_CLASSES):

        for j in range(NUM_CLASSES):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

    fig.colorbar(image)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# ROC CURVE
# ---------------------------------------------------------------------

def save_roc_curve(
    y_true,
    y_prob,
    output_path,
):
    """
    Save one-vs-rest ROC curves.
    """

    y_true_binary = label_binarize(
        y_true,
        classes=range(NUM_CLASSES),
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for class_index, class_name in enumerate(CLASS_NAMES):

        # ROC requires both positive and negative examples
        if len(
            np.unique(
                y_true_binary[:, class_index]
            )
        ) < 2:
            continue

        fpr, tpr, _ = roc_curve(
            y_true_binary[:, class_index],
            y_prob[:, class_index],
        )

        auc_value = roc_auc_score(
            y_true_binary[:, class_index],
            y_prob[:, class_index],
        )

        ax.plot(
            fpr,
            tpr,
            label=f"{class_name} (AUC={auc_value:.4f})",
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random",
    )

    ax.set_title(
        f"{MODEL_NAME} - ROC Curves"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.legend()

    ax.grid(True)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# PRECISION-RECALL CURVE
# ---------------------------------------------------------------------

def save_pr_curve(
    y_true,
    y_prob,
    output_path,
):
    """
    Save one-vs-rest Precision-Recall curves.
    """

    y_true_binary = label_binarize(
        y_true,
        classes=range(NUM_CLASSES),
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for class_index, class_name in enumerate(CLASS_NAMES):

        if len(
            np.unique(
                y_true_binary[:, class_index]
            )
        ) < 2:
            continue

        precision, recall, _ = (
            precision_recall_curve(
                y_true_binary[:, class_index],
                y_prob[:, class_index],
            )
        )

        ap = average_precision_score(
            y_true_binary[:, class_index],
            y_prob[:, class_index],
        )

        ax.plot(
            recall,
            precision,
            label=f"{class_name} (AP={ap:.4f})",
        )

    ax.set_title(
        f"{MODEL_NAME} - Precision-Recall Curves"
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    ax.legend()

    ax.grid(True)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------------------------

def evaluate_model():

    print("=" * 70)
    print("STEP 8 - MODEL EVALUATION")
    print("=" * 70)

    print(f"\nProject root : {PROJECT_ROOT}")
    print(f"Model        : {MODEL_NAME}")
    print(f"Checkpoint   : {CHECKPOINT_PATH}")

    # -------------------------------------------------------------
    # Device
    # -------------------------------------------------------------

    device = get_device()

    print(f"Device       : {device}")

    # -------------------------------------------------------------
    # Test loader
    # -------------------------------------------------------------

    test_loader = create_test_loader()

    print(
        f"Test samples : {len(test_loader.dataset)}"
    )

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

    model = load_checkpoint_model(
        model_name=MODEL_NAME,
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
    )

    # -------------------------------------------------------------
    # Predictions
    # -------------------------------------------------------------

    y_true, y_pred, y_prob = (
        collect_predictions(
            model,
            test_loader,
            device,
        )
    )

    # -------------------------------------------------------------
    # Basic metrics
    # -------------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    # -------------------------------------------------------------
    # Specificity
    # -------------------------------------------------------------

    class_specificity, macro_specificity = (
        calculate_specificity(
            y_true,
            y_pred,
        )
    )

    # -------------------------------------------------------------
    # Confusion matrix
    # -------------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=range(NUM_CLASSES),
    )

    # -------------------------------------------------------------
    # ROC-AUC
    # -------------------------------------------------------------

    roc_auc = calculate_roc_auc(
        y_true,
        y_prob,
    )

    # -------------------------------------------------------------
    # PR-AUC
    # -------------------------------------------------------------

    pr_auc, class_pr_auc = calculate_pr_auc(
        y_true,
        y_prob,
    )

    # -------------------------------------------------------------
    # Classification report
    # -------------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        labels=range(NUM_CLASSES),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    # -------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"Accuracy            : {accuracy:.4f}"
    )

    print(
        f"Macro Precision     : {precision:.4f}"
    )

    print(
        f"Macro Recall        : {recall:.4f}"
    )

    print(
        f"Macro Specificity   : {macro_specificity:.4f}"
    )

    print(
        f"Macro F1            : {f1:.4f}"
    )

    if roc_auc is not None:
        print(
            f"Macro ROC-AUC       : {roc_auc:.4f}"
        )
    else:
        print(
            "Macro ROC-AUC       : N/A"
        )

    if pr_auc is not None:
        print(
            f"Macro PR-AUC        : {pr_auc:.4f}"
        )
    else:
        print(
            "Macro PR-AUC        : N/A"
        )

    print("\nPer-class specificity:")

    for class_name, value in (
        class_specificity.items()
    ):
        print(
            f"  {class_name:<10}: {value:.4f}"
        )

    print("\nConfusion Matrix:")

    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=range(NUM_CLASSES),
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    # -------------------------------------------------------------
    # Output paths
    # -------------------------------------------------------------

    model_eval_dir = (
        EVALUATION_DIR
        / MODEL_NAME.lower().replace("-", "_")
    )

    model_eval_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        model_eval_dir
        / "evaluation.json"
    )

    report_csv_path = (
        model_eval_dir
        / "classification_report.csv"
    )

    cm_csv_path = (
        model_eval_dir
        / "confusion_matrix.csv"
    )

    cm_png_path = (
        model_eval_dir
        / "confusion_matrix.png"
    )

    roc_png_path = (
        model_eval_dir
        / "roc_curve.png"
    )

    pr_png_path = (
        model_eval_dir
        / "pr_curve.png"
    )

    # -------------------------------------------------------------
    # Save classification report
    # -------------------------------------------------------------

    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(
        report_csv_path
    )

    # -------------------------------------------------------------
    # Save confusion matrix
    # -------------------------------------------------------------

    cm_df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    cm_df.to_csv(
        cm_csv_path
    )

    # -------------------------------------------------------------
    # Save plots
    # -------------------------------------------------------------

    save_confusion_matrix(
        cm,
        cm_png_path,
    )

    save_roc_curve(
        y_true,
        y_prob,
        roc_png_path,
    )

    save_pr_curve(
        y_true,
        y_prob,
        pr_png_path,
    )

    # -------------------------------------------------------------
    # Build JSON
    # -------------------------------------------------------------

    evaluation_results = {
        "model": MODEL_NAME,
        "checkpoint": str(
            CHECKPOINT_PATH
        ),
        "test_samples": int(
            len(y_true)
        ),
        "device": str(device),

        "metrics": {
            "accuracy": float(
                accuracy
            ),
            "macro_precision": float(
                precision
            ),
            "macro_recall_sensitivity": float(
                recall
            ),
            "macro_specificity": float(
                macro_specificity
            ),
            "macro_f1": float(
                f1
            ),
            "macro_roc_auc": (
                float(roc_auc)
                if roc_auc is not None
                else None
            ),
            "macro_pr_auc": (
                float(pr_auc)
                if pr_auc is not None
                else None
            ),
        },

        "per_class_specificity": (
            class_specificity
        ),

        "per_class_pr_auc": {
            CLASS_NAMES[index]: value
            for index, value
            in enumerate(class_pr_auc)
        },

        "confusion_matrix": cm.tolist(),

        "classification_report": report,
    }

    # -------------------------------------------------------------
    # Save JSON
    # -------------------------------------------------------------

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            evaluation_results,
            file,
            indent=4,
        )

    # -------------------------------------------------------------
    # MLflow
    # -------------------------------------------------------------

    mlflow.set_tracking_uri(
        f"sqlite:///{MLFLOW_DB.as_posix()}"
    )

    mlflow.set_experiment(
        "KidneyVision-Model-Benchmark"
    )

    with mlflow.start_run(
        run_name=f"Evaluation-{MODEL_NAME}"
    ):

        mlflow.log_param(
            "model",
            MODEL_NAME,
        )

        mlflow.log_param(
            "checkpoint",
            str(CHECKPOINT_PATH),
        )

        mlflow.log_param(
            "test_samples",
            len(y_true),
        )

        mlflow.log_metric(
            "test_accuracy",
            float(accuracy),
        )

        mlflow.log_metric(
            "test_precision_macro",
            float(precision),
        )

        mlflow.log_metric(
            "test_recall_macro",
            float(recall),
        )

        mlflow.log_metric(
            "test_specificity_macro",
            float(macro_specificity),
        )

        mlflow.log_metric(
            "test_f1_macro",
            float(f1),
        )

        if roc_auc is not None:

            mlflow.log_metric(
                "test_roc_auc_macro",
                float(roc_auc),
            )

        if pr_auc is not None:

            mlflow.log_metric(
                "test_pr_auc_macro",
                float(pr_auc),
            )

        mlflow.log_artifacts(
            str(model_eval_dir)
        )

    # -------------------------------------------------------------
    # Final
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print("EVALUATION ARTIFACTS SAVED")
    print("=" * 70)

    print(
        f"Evaluation JSON      : {json_path}"
    )

    print(
        f"Classification CSV   : {report_csv_path}"
    )

    print(
        f"Confusion Matrix CSV : {cm_csv_path}"
    )

    print(
        f"Confusion Matrix PNG : {cm_png_path}"
    )

    print(
        f"ROC Curve PNG        : {roc_png_path}"
    )

    print(
        f"PR Curve PNG         : {pr_png_path}"
    )

    print("\nEVALUATION COMPLETE")


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------

if __name__ == "__main__":
    evaluate_model()