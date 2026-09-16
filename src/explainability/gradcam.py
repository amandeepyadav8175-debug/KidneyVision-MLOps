from pathlib import Path
import sys
import io

import numpy as np
import torch
import torch.nn.functional as F

# -------------------------------------------------------------------
# IMPORTANT:
# Use a non-GUI backend because this code also runs inside FastAPI
# and Docker/server environments.
# -------------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image

# -------------------------------------------------------------------
# Project root
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.training_config import CLASS_NAMES
from src.models.model_factory import create_model
from src.preprocessing.image_preprocessor import build_transforms


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

MODEL_NAME = "SwinTransformer"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "swintransformer_best.pt"
)

IMAGE_SIZE = 224

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "explainability"
    / "gradcam"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# Device
# -------------------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -------------------------------------------------------------------
# Load model
# -------------------------------------------------------------------

def load_model():
    print("Loading model...")

    model = create_model(
        model_name=MODEL_NAME,
        num_classes=len(CLASS_NAMES),
        pretrained=False,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    # ---------------------------------------------------------------
    # Handle different checkpoint formats
    # ---------------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # ---------------------------------------------------------------
    # Remove possible DataParallel prefix
    # ---------------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    # ---------------------------------------------------------------
    # Load weights
    # ---------------------------------------------------------------

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    print(f"Model loaded: {MODEL_NAME}")
    print(f"Device: {DEVICE}")

    return model


# -------------------------------------------------------------------
# Find target layer
# -------------------------------------------------------------------

def get_target_layer(model):
    """
    For Swin Transformer we use the final feature block.

    TorchVision Swin produces spatial features in the form:
        [batch, height, width, channels]

    We retain this spatial representation for Grad-CAM.
    """

    target_layer = model.features[-1]

    print(
        f"Grad-CAM target layer: "
        f"{target_layer.__class__.__name__}"
    )

    return target_layer


# -------------------------------------------------------------------
# Grad-CAM
# -------------------------------------------------------------------

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(
            self.save_activation
        )

        self.backward_handle = target_layer.register_full_backward_hook(
            self.save_gradient
        )

    # ---------------------------------------------------------------
    # Save activation
    # ---------------------------------------------------------------

    def save_activation(self, module, input, output):

        self.activations = output

    # ---------------------------------------------------------------
    # Save gradient
    # ---------------------------------------------------------------

    def save_gradient(self, module, grad_input, grad_output):

        self.gradients = grad_output[0]

    # ---------------------------------------------------------------
    # Remove hooks
    # ---------------------------------------------------------------

    def remove_hooks(self):

        if self.forward_handle is not None:
            self.forward_handle.remove()
            self.forward_handle = None

        if self.backward_handle is not None:
            self.backward_handle.remove()
            self.backward_handle = None

    # ---------------------------------------------------------------
    # Context manager support
    # ---------------------------------------------------------------

    def __enter__(self):

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):

        self.remove_hooks()

    # ---------------------------------------------------------------
    # Generate CAM
    # ---------------------------------------------------------------

    def generate(
        self,
        input_tensor,
        class_index,
    ):

        # Clear old activation/gradient data.
        self.activations = None
        self.gradients = None

        self.model.zero_grad(set_to_none=True)

        output = self.model(input_tensor)

        # -----------------------------------------------------------
        # Handle models that return objects such as Inception output
        # -----------------------------------------------------------

        if hasattr(output, "logits"):
            output = output.logits

        if not isinstance(output, torch.Tensor):
            raise RuntimeError(
                "Model output is not a torch.Tensor."
            )

        if output.ndim != 2:
            raise RuntimeError(
                f"Unexpected model output shape: "
                f"{tuple(output.shape)}"
            )

        if (
            class_index < 0
            or class_index >= output.shape[1]
        ):
            raise ValueError(
                f"Invalid class index: {class_index}. "
                f"Model has {output.shape[1]} classes."
            )

        # -----------------------------------------------------------
        # Backpropagate target class score
        # -----------------------------------------------------------

        score = output[:, class_index].sum()

        score.backward()

        activations = self.activations
        gradients = self.gradients

        if activations is None:
            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        if gradients is None:
            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        # -----------------------------------------------------------
        # Swin output:
        # [B, H, W, C]
        #
        # Convert to:
        # [B, C, H, W]
        # -----------------------------------------------------------

        if activations.ndim == 4:

            if (
                activations.shape[-1] == gradients.shape[-1]
                and activations.shape[1] < activations.shape[-1]
            ):

                activations = activations.permute(
                    0,
                    3,
                    1,
                    2,
                )

                gradients = gradients.permute(
                    0,
                    3,
                    1,
                    2,
                )

        else:

            raise RuntimeError(
                f"Unexpected activation shape: "
                f"{tuple(activations.shape)}"
            )

        # -----------------------------------------------------------
        # Global average pooling of gradients
        # -----------------------------------------------------------

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        # -----------------------------------------------------------
        # Weighted activation maps
        # -----------------------------------------------------------

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        # -----------------------------------------------------------
        # ReLU
        # -----------------------------------------------------------

        cam = F.relu(cam)

        # -----------------------------------------------------------
        # Resize CAM to input image size
        # -----------------------------------------------------------

        cam = F.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze()

        # -----------------------------------------------------------
        # Normalize between 0 and 1
        # -----------------------------------------------------------

        cam_min = cam.min()
        cam_max = cam.max()

        if (cam_max - cam_min) > 1e-8:

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        else:

            cam = torch.zeros_like(cam)

        return output.detach(), cam.detach()


# -------------------------------------------------------------------
# Prepare image
# -------------------------------------------------------------------

def prepare_image(image_path):

    transform = build_transforms(
        image_size=IMAGE_SIZE,
        train=False,
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    input_tensor = transform(image)

    input_tensor = input_tensor.unsqueeze(0)

    return image, input_tensor.to(DEVICE)


# -------------------------------------------------------------------
# Create visualization
# -------------------------------------------------------------------

def save_gradcam_visualization(
    original_image,
    cam,
    predicted_class,
    confidence,
    output_path,
):

    # ---------------------------------------------------------------
    # Resize original image
    # ---------------------------------------------------------------

    original_image = original_image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    original_array = (
        np.asarray(original_image)
        .astype(np.float32)
        / 255.0
    )

    cam_array = cam.cpu().numpy()

    figure = plt.figure(
        figsize=(12, 4)
    )

    # ---------------------------------------------------------------
    # Original image
    # ---------------------------------------------------------------

    plt.subplot(1, 3, 1)

    plt.imshow(original_array)

    plt.title(
        "Original CT Image"
    )

    plt.axis("off")

    # ---------------------------------------------------------------
    # Grad-CAM
    # ---------------------------------------------------------------

    plt.subplot(1, 3, 2)

    plt.imshow(
        cam_array,
        cmap="jet",
    )

    plt.title(
        "Grad-CAM"
    )

    plt.axis("off")

    # ---------------------------------------------------------------
    # Overlay
    # ---------------------------------------------------------------

    plt.subplot(1, 3, 3)

    plt.imshow(
        original_array
    )

    plt.imshow(
        cam_array,
        cmap="jet",
        alpha=0.45,
    )

    plt.title(
        f"Prediction: {predicted_class}\n"
        f"Confidence: {confidence:.2%}"
    )

    plt.axis("off")

    plt.tight_layout()

    # ---------------------------------------------------------------
    # Save figure
    # ---------------------------------------------------------------

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


# -------------------------------------------------------------------
# Generate Grad-CAM for API
# -------------------------------------------------------------------

def generate_gradcam_for_api(
    model,
    image,
    predicted_index,
    predicted_class,
    confidence,
):
    """
    Generate a Grad-CAM visualization for a PIL image.

    This function is designed to be called by the FastAPI backend.

    The already-loaded model is used instead of loading the checkpoint
    again.

    Hooks are automatically removed after Grad-CAM generation.
    """

    # ---------------------------------------------------------------
    # Prepare image
    # ---------------------------------------------------------------

    transform = build_transforms(
        image_size=IMAGE_SIZE,
        train=False,
    )

    input_tensor = transform(
        image.convert("RGB")
    )

    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(DEVICE)

    # ---------------------------------------------------------------
    # Target layer
    # ---------------------------------------------------------------

    target_layer = get_target_layer(model)

    # ---------------------------------------------------------------
    # Grad-CAM
    #
    # Context manager guarantees hook cleanup even if an exception
    # occurs during generation.
    # ---------------------------------------------------------------

    try:

        with GradCAM(
            model,
            target_layer,
        ) as gradcam:

            with torch.enable_grad():

                _, cam = gradcam.generate(
                    input_tensor,
                    class_index=predicted_index,
                )

        # -----------------------------------------------------------
        # Create image in memory
        # -----------------------------------------------------------

        original_image = image.convert(
            "RGB"
        ).resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        original_array = (
            np.asarray(original_image)
            .astype(np.float32)
            / 255.0
        )

        cam_array = cam.cpu().numpy()

        figure = plt.figure(
            figsize=(12, 4)
        )

        # -----------------------------------------------------------
        # Original
        # -----------------------------------------------------------

        plt.subplot(1, 3, 1)

        plt.imshow(
            original_array
        )

        plt.title(
            "Original CT Image"
        )

        plt.axis("off")

        # -----------------------------------------------------------
        # Grad-CAM
        # -----------------------------------------------------------

        plt.subplot(1, 3, 2)

        plt.imshow(
            cam_array,
            cmap="jet",
        )

        plt.title(
            "Grad-CAM"
        )

        plt.axis("off")

        # -----------------------------------------------------------
        # Overlay
        # -----------------------------------------------------------

        plt.subplot(1, 3, 3)

        plt.imshow(
            original_array
        )

        plt.imshow(
            cam_array,
            cmap="jet",
            alpha=0.45,
        )

        plt.title(
            f"Prediction: {predicted_class}\n"
            f"Confidence: {confidence:.2%}"
        )

        plt.axis("off")

        plt.tight_layout()

        # -----------------------------------------------------------
        # Save figure into memory
        # -----------------------------------------------------------

        image_buffer = io.BytesIO()

        figure.savefig(
            image_buffer,
            format="png",
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(figure)

        image_buffer.seek(0)

        return image_buffer.getvalue()

    except Exception:

        # Make absolutely sure matplotlib figures are not left open.
        plt.close("all")

        raise


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():

    print("=" * 70)
    print("STEP 12 - GRAD-CAM EXPLAINABILITY")
    print("=" * 70)

    print()
    print(
        f"Checkpoint : {CHECKPOINT_PATH}"
    )

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    # ---------------------------------------------------------------
    # Find one test image
    # ---------------------------------------------------------------

    test_manifest = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "manifests"
        / "test.csv"
    )

    if not test_manifest.exists():

        raise FileNotFoundError(
            f"Test manifest not found:\n"
            f"{test_manifest}"
        )

    import pandas as pd

    test_df = pd.read_csv(
        test_manifest
    )

    if len(test_df) == 0:

        raise RuntimeError(
            "Test manifest is empty."
        )

    image_path = (
        PROJECT_ROOT
        / test_df.iloc[0]["image_path"]
    )

    print(
        f"Image      : {image_path}"
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Test image not found:\n"
            f"{image_path}"
        )

    # ---------------------------------------------------------------
    # Load model
    # ---------------------------------------------------------------

    model = load_model()

    target_layer = get_target_layer(
        model
    )

    # ---------------------------------------------------------------
    # Prepare image
    # ---------------------------------------------------------------

    original_image, input_tensor = prepare_image(
        image_path
    )

    print(
        f"Input tensor shape: "
        f"{tuple(input_tensor.shape)}"
    )

    # ---------------------------------------------------------------
    # Generate prediction + Grad-CAM
    # ---------------------------------------------------------------

    try:

        with GradCAM(
            model,
            target_layer,
        ) as gradcam:

            with torch.enable_grad():

                # First pass: get prediction.
                output, _ = gradcam.generate(
                    input_tensor,
                    class_index=0,
                )

                probabilities = torch.softmax(
                    output,
                    dim=1,
                )

                predicted_index = int(
                    torch.argmax(
                        probabilities,
                        dim=1,
                    ).item()
                )

                predicted_class = CLASS_NAMES[
                    predicted_index
                ]

                confidence = float(
                    probabilities[
                        0,
                        predicted_index,
                    ].item()
                )

                # Second pass: generate CAM for
                # the actual predicted class.
                output, cam = gradcam.generate(
                    input_tensor,
                    class_index=predicted_index,
                )

    finally:

        plt.close("all")

    print()
    print(
        f"Predicted class : {predicted_class}"
    )

    print(
        f"Confidence      : {confidence:.4f}"
    )

    print(
        f"CAM shape       : {tuple(cam.shape)}"
    )

    # ---------------------------------------------------------------
    # Save result
    # ---------------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / "gradcam_example.png"
    )

    save_gradcam_visualization(
        original_image=original_image,
        cam=cam,
        predicted_class=predicted_class,
        confidence=confidence,
        output_path=output_path,
    )

    print()
    print(
        f"Grad-CAM saved  : {output_path}"
    )

    print()
    print("=" * 70)
    print("GRAD-CAM COMPLETE")
    print("=" * 70)


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()