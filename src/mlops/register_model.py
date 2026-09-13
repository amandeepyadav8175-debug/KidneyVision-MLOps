from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MLFLOW_DB = PROJECT_ROOT / "artifacts" / "mlflow.db"

EXPERIMENT_NAME = "KidneyVision-Model-Benchmark"
REGISTERED_MODEL_NAME = "KidneyVision-SwinTransformer"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 11 - MODEL REGISTRATION")
    print("=" * 70)

    tracking_uri = f"sqlite:///{MLFLOW_DB.resolve().as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)

    print(f"\nMLflow DB : {MLFLOW_DB}")
    print(f"Experiment: {EXPERIMENT_NAME}")

    # --------------------------------------------------------
    # Find experiment
    # --------------------------------------------------------

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise RuntimeError(
            f"Experiment '{EXPERIMENT_NAME}' not found."
        )

    print(f"Experiment ID: {experiment.experiment_id}")

    # --------------------------------------------------------
    # Find successful SwinTransformer runs
    # --------------------------------------------------------

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "attributes.status = 'FINISHED' "
            "AND params.model_name = 'SwinTransformer'"
        ),
        order_by=["attributes.start_time DESC"],
    )

    if not runs:
        raise RuntimeError(
            "No finished SwinTransformer training run found."
        )

    run = runs[0]

    run_id = run.info.run_id

    print("\nSelected training run:")
    print(f"Run ID   : {run_id}")
    print(f"Run Name : {run.data.tags.get('mlflow.runName')}")
    print(f"Status   : {run.info.status}")

    # --------------------------------------------------------
    # Display important metrics
    # --------------------------------------------------------

    print("\nTraining metrics:")

    for metric_name in [
        "best_epoch",
        "best_val_loss",
        "final_val_accuracy",
        "final_val_f1",
        "test_accuracy",
        "test_f1",
    ]:
        value = run.data.metrics.get(metric_name)

        if value is not None:
            print(f"  {metric_name}: {value:.4f}")

    # --------------------------------------------------------
    # Find MLflow Logged Models
    # --------------------------------------------------------

    print("\nSearching MLflow logged models...")

    logged_models = client.search_logged_models(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"source_run_id = '{run_id}'",
    )

    if not logged_models:
        raise RuntimeError(
            f"No MLflow logged model found for run {run_id}."
        )

    # Find model named "model"
    logged_model = None

    for candidate in logged_models:

        if getattr(candidate, "name", None) == "model":
            logged_model = candidate
            break

    # Fallback: use first logged model
    if logged_model is None:
        logged_model = logged_models[0]

    print("\nLogged model found:")
    print(f"Logged Model ID : {logged_model.model_id}")
    print(f"Model Name      : {logged_model.name}")

    # --------------------------------------------------------
    # Model URI
    # --------------------------------------------------------

    model_uri = f"models:/{logged_model.model_id}"

    print(f"\nModel URI:")
    print(model_uri)

    # --------------------------------------------------------
    # Create Registered Model if needed
    # --------------------------------------------------------

    existing_models = client.search_registered_models()

    existing_names = {
        model.name for model in existing_models
    }

    if REGISTERED_MODEL_NAME not in existing_names:

        print("\nCreating registered model...")

        client.create_registered_model(
            REGISTERED_MODEL_NAME,
            description=(
                "Swin Transformer model for four-class "
                "kidney CT image classification: "
                "Normal, Cyst, Stone and Tumor."
            ),
        )

        print(
            f"Registered model created: "
            f"{REGISTERED_MODEL_NAME}"
        )

    else:

        print(
            f"\nRegistered model already exists: "
            f"{REGISTERED_MODEL_NAME}"
        )

    # --------------------------------------------------------
    # Register model version
    # --------------------------------------------------------

    print("\nRegistering model version...")

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    version = model_version.version

    print(f"Model version created: {version}")

    # --------------------------------------------------------
    # Add metadata
    # --------------------------------------------------------

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key="model_type",
        value="SwinTransformer",
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key="task",
        value="4-class kidney CT classification",
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key="source_run_id",
        value=run_id,
    )

    # --------------------------------------------------------
    # Champion alias
    # --------------------------------------------------------

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=version,
    )

    print(
        f"Champion alias -> version {version}"
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL REGISTRATION COMPLETE")
    print("=" * 70)

    print(f"\nRegistered Model : {REGISTERED_MODEL_NAME}")
    print(f"Version          : {version}")
    print("Alias            : champion")
    print(f"Source Run       : {run_id}")
    print(f"Model URI        : {model_uri}")


if __name__ == "__main__":
    main()