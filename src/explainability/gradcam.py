"""
KidneyVision-MLOps
Grad-CAM Explainability

Production-safe Grad-CAM implementation for:
- FastAPI inference
- EfficientNet-B0
- Swin Transformer
- CPU deployment
- Render free-tier deployment

Important:
The FastAPI endpoint passes both:
    predicted_index
    predicted_class

This module accepts both forms safely.
"""

from pathlib import Path
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from configs.training_config import CLASS_NAMES
from src.preprocessing.image_preprocessor import build_transforms


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_SIZE = 224

GRADCAM_IMAGE_SIZE = 160

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "explainability"
    / "gradcam"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CLASS HELPERS
# ============================================================

def class_name_to_index(class_name):
    """
    Convert class name to class index.

    Example:
        Cyst -> 1
        Normal -> 0
        Stone -> 2
        Tumor -> 3
    """

    if not isinstance(class_name, str):
        raise TypeError(
            "class_name must be a string."
        )

    if class_name not in CLASS_NAMES:
        raise ValueError(
            f"Unknown class '{class_name}'. "
            f"Expected one of: {CLASS_NAMES}"
        )

    return CLASS_NAMES.index(
        class_name
    )


def normalize_class_index(
    predicted_index=None,
    predicted_class=None,
):
    """
    Resolve the final integer class index.

    The API normally supplies:

        predicted_index = 1
        predicted_class = "Cyst"

    This function also safely handles cases where
    predicted_index accidentally arrives as a class name.
    """

    # --------------------------------------------------------
    # If predicted_index is already an integer
    # --------------------------------------------------------

    if isinstance(
        predicted_index,
        (int, np.integer),
    ):

        index = int(
            predicted_index
        )

    # --------------------------------------------------------
    # If predicted_index is a string
    #
    # Example:
    # predicted_index = "Cyst"
    #
    # Never do int("Cyst").
    # --------------------------------------------------------

    elif isinstance(
        predicted_index,
        str,
    ):

        if predicted_index in CLASS_NAMES:

            index = class_name_to_index(
                predicted_index
            )

        else:

            try:

                index = int(
                    predicted_index
                )

            except ValueError:

                if predicted_class in CLASS_NAMES:

                    index = class_name_to_index(
                        predicted_class
                    )

                else:

                    raise ValueError(
                        "Could not resolve predicted class/index. "
                        f"predicted_index={predicted_index!r}, "
                        f"predicted_class={predicted_class!r}, "
                        f"expected classes={CLASS_NAMES}"
                    )

    # --------------------------------------------------------
    # If no index was supplied, use predicted_class
    # --------------------------------------------------------

    elif predicted_index is None:

        if predicted_class in CLASS_NAMES:

            index = class_name_to_index(
                predicted_class
            )

        else:

            raise ValueError(
                "predicted_index is None and "
                "predicted_class is invalid."
            )

    else:

        raise TypeError(
            "predicted_index must be an integer, "
            "string, or None."
        )

    # --------------------------------------------------------
    # Validate range
    # --------------------------------------------------------

    if index < 0 or index >= len(CLASS_NAMES):

        raise ValueError(
            f"Invalid class index {index}. "
            f"Expected range 0-{len(CLASS_NAMES) - 1}."
        )

    return index


# ============================================================
# MODEL OUTPUT HELPER
# ============================================================

def extract_logits(output):
    """
    Extract logits from different TorchVision model outputs.

    Supports:
    - Tensor
    - Inception-style objects with .logits
    - tuples/lists
    """

    # --------------------------------------------------------
    # Normal Tensor
    # --------------------------------------------------------

    if isinstance(
        output,
        torch.Tensor,
    ):

        return output

    # --------------------------------------------------------
    # Inception / named output object
    # --------------------------------------------------------

    if hasattr(
        output,
        "logits",
    ):

        return output.logits

    # --------------------------------------------------------
    # Tuple/list
    # --------------------------------------------------------

    if isinstance(
        output,
        (tuple, list),
    ):

        if len(output) == 0:

            raise RuntimeError(
                "Model returned an empty tuple/list."
            )

        first_output = output[0]

        if isinstance(
            first_output,
            torch.Tensor,
        ):

            return first_output

        if hasattr(
            first_output,
            "logits",
        ):

            return first_output.logits

    raise RuntimeError(
        "Could not extract logits from model output."
    )


# ============================================================
# TARGET LAYER
# ============================================================

def get_target_layer(model):
    """
    Select a spatial feature layer suitable for Grad-CAM.

    EfficientNet:
        model.features[-1]

    Swin:
        model.features[-1]

    Generic CNN:
        Searches backwards for a suitable module.

    The selected layer must produce a spatial feature map.
    """

    # --------------------------------------------------------
    # EfficientNet / Swin / similar TorchVision models
    # --------------------------------------------------------

    if hasattr(
        model,
        "features",
    ):

        features = model.features

        if len(features) > 0:

            target_layer = features[-1]

            print(
                "Grad-CAM target layer:",
                target_layer.__class__.__name__,
            )

            return target_layer

    # --------------------------------------------------------
    # ResNet-style models
    # --------------------------------------------------------

    if hasattr(
        model,
        "layer4",
    ):

        target_layer = model.layer4[-1]

        print(
            "Grad-CAM target layer:",
            target_layer.__class__.__name__,
        )

        return target_layer

    # --------------------------------------------------------
    # VGG-style models
    # --------------------------------------------------------

    if hasattr(
        model,
        "features",
    ):

        for module in reversed(
            list(model.features)
        ):

            if isinstance(
                module,
                torch.nn.Conv2d,
            ):

                print(
                    "Grad-CAM target layer:",
                    module.__class__.__name__,
                )

                return module

    # --------------------------------------------------------
    # Generic fallback
    # --------------------------------------------------------

    candidate_layers = []

    for module in model.modules():

        if isinstance(
            module,
            torch.nn.Conv2d,
        ):

            candidate_layers.append(
                module
            )

    if candidate_layers:

        target_layer = candidate_layers[-1]

        print(
            "Grad-CAM target layer:",
            target_layer.__class__.__name__,
        )

        return target_layer

    raise RuntimeError(
        "Could not find a suitable Grad-CAM target layer."
    )


# ============================================================
# GRAD-CAM CLASS
# ============================================================

class GradCAM:
    """
    Memory-conscious Grad-CAM implementation.

    Only one forward hook is kept.
    Gradients are obtained directly with autograd.grad().
    """

    def __init__(
        self,
        model,
        target_layer,
    ):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.activation_handle = None

        self.activation_handle = (
            self.target_layer.register_forward_hook(
                self._save_activation
            )
        )

    # --------------------------------------------------------
    # Forward hook
    # --------------------------------------------------------

    def _save_activation(
        self,
        module,
        inputs,
        output,
    ):

        self.activations = output

    # --------------------------------------------------------
    # Remove hook
    # --------------------------------------------------------

    def close(self):

        if (
            self.activation_handle
            is not None
        ):

            self.activation_handle.remove()

            self.activation_handle = None

        self.activations = None

    # --------------------------------------------------------
    # Context manager
    # --------------------------------------------------------

    def __enter__(self):

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):

        self.close()

        return False

    # --------------------------------------------------------
    # Generate CAM
    # --------------------------------------------------------

    def generate(
        self,
        input_tensor,
        class_index,
    ):

        if not isinstance(
            class_index,
            int,
        ):

            class_index = normalize_class_index(
                predicted_index=class_index
            )

        self.model.zero_grad(
            set_to_none=True
        )

        self.activations = None

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        output = self.model(
            input_tensor
        )

        logits = extract_logits(
            output
        )

        if logits.ndim != 2:

            raise RuntimeError(
                "Model output must have shape "
                "[batch, classes]. "
                f"Received: {tuple(logits.shape)}"
            )

        if class_index < 0:

            raise ValueError(
                f"Invalid class index: {class_index}"
            )

        if class_index >= logits.shape[1]:

            raise ValueError(
                f"Class index {class_index} is outside "
                f"model output range 0-{logits.shape[1] - 1}."
            )

        # ----------------------------------------------------
        # Make sure target activation exists
        # ----------------------------------------------------

        activations = self.activations

        if activations is None:

            raise RuntimeError(
                "Grad-CAM target layer did not produce "
                "activations."
            )

        if not isinstance(
            activations,
            torch.Tensor,
        ):

            raise RuntimeError(
                "Grad-CAM target layer output is not a Tensor."
            )

        # ----------------------------------------------------
        # Select target score
        # ----------------------------------------------------

        score = logits[
            0,
            class_index
        ]

        # ----------------------------------------------------
        # Direct gradient calculation
        #
        # This avoids register_full_backward_hook(),
        # which is more fragile for deployment.
        # ----------------------------------------------------

        gradients = torch.autograd.grad(
            outputs=score,
            inputs=activations,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]

        # ----------------------------------------------------
        # Convert Swin layout:
        #
        # [B, H, W, C]
        #
        # into:
        #
        # [B, C, H, W]
        # ----------------------------------------------------

        if activations.ndim == 4:

            if (
                activations.shape[1]
                == activations.shape[2]
                and
                activations.shape[3]
                != activations.shape[1]
            ):

                # Already B,C,H,W
                activation_tensor = activations

                gradient_tensor = gradients

            elif (
                activations.shape[1]
                != activations.shape[3]
                and
                activations.shape[2]
                != activations.shape[3]
            ):

                # Usually B,H,W,C
                activation_tensor = (
                    activations
                    .permute(
                        0,
                        3,
                        1,
                        2,
                    )
                    .contiguous()
                )

                gradient_tensor = (
                    gradients
                    .permute(
                        0,
                        3,
                        1,
                        2,
                    )
                    .contiguous()
                )

            else:

                # Default to B,C,H,W
                activation_tensor = activations
                gradient_tensor = gradients

        else:

            raise RuntimeError(
                "Grad-CAM target layer must produce "
                "a 4D feature map. "
                f"Received shape: {tuple(activations.shape)}"
            )

        # ----------------------------------------------------
        # Global average pooling of gradients
        # ----------------------------------------------------

        weights = gradient_tensor.mean(
            dim=(2, 3),
            keepdim=True,
        )

        # ----------------------------------------------------
        # Weighted feature maps
        # ----------------------------------------------------

        cam = (
            weights
            * activation_tensor
        ).sum(
            dim=1,
            keepdim=True,
        )

        # ----------------------------------------------------
        # ReLU
        # ----------------------------------------------------

        cam = F.relu(
            cam
        )

        # ----------------------------------------------------
        # Resize CAM
        # ----------------------------------------------------

        cam = F.interpolate(
            cam,
            size=(
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[
            0,
            0
        ]

        # ----------------------------------------------------
        # Normalize 0-1
        # ----------------------------------------------------

        cam_min = cam.min()
        cam_max = cam.max()

        difference = (
            cam_max
            - cam_min
        )

        if float(
            difference.detach().cpu()
        ) > 1e-8:

            cam = (
                cam
                - cam_min
            ) / difference

        else:

            cam = torch.zeros_like(
                cam
            )

        # ----------------------------------------------------
        # Detach everything that is no longer needed
        # ----------------------------------------------------

        output_copy = logits.detach()

        cam_copy = cam.detach()

        self.activations = None

        del output
        del logits
        del score
        del gradients
        del activation_tensor
        del gradient_tensor
        del weights
        del cam

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        return (
            output_copy,
            cam_copy,
        )


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_image(
    image,
    image_size=IMAGE_SIZE,
    device=None,
):
    """
    Convert PIL image into model input tensor.
    """

    if image is None:

        raise ValueError(
            "Image cannot be None."
        )

    if not isinstance(
        image,
        Image.Image,
    ):

        raise TypeError(
            "image must be a PIL.Image.Image."
        )

    if device is None:

        device = DEVICE

    transform = build_transforms(
        image_size=image_size,
        train=False,
    )

    rgb_image = image.convert(
        "RGB"
    )

    input_tensor = transform(
        rgb_image
    )

    input_tensor = input_tensor.unsqueeze(
        0
    )

    input_tensor = input_tensor.to(
        device
    )

    return (
        rgb_image,
        input_tensor,
    )


# ============================================================
# HEATMAP COLORIZATION
# ============================================================

def create_heatmap(
    cam,
):
    """
    Convert normalized CAM [0,1] into RGB heatmap.

    This intentionally uses NumPy/PIL instead of Matplotlib,
    keeping the API lightweight for Render.
    """

    cam_array = (
        cam.detach()
        .cpu()
        .numpy()
    )

    cam_array = np.clip(
        cam_array,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Simple blue -> cyan -> yellow -> red heatmap
    # --------------------------------------------------------

    red = np.clip(
        2.0 * cam_array,
        0.0,
        1.0,
    )

    green = np.clip(
        2.0
        * (
            1.0
            - np.abs(
                cam_array
                - 0.5
            )
            * 2.0
        ),
        0.0,
        1.0,
    )

    blue = np.clip(
        2.0
        * (
            1.0
            - cam_array
        ),
        0.0,
        1.0,
    )

    heatmap_array = np.stack(
        [
            red,
            green,
            blue,
        ],
        axis=-1,
    )

    heatmap_array = (
        heatmap_array
        * 255.0
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        heatmap_array,
        mode="RGB",
    )


# ============================================================
# CREATE OVERLAY
# ============================================================

def create_gradcam_overlay(
    original_image,
    cam,
    predicted_class,
    confidence,
):
    """
    Create a lightweight 3-panel PNG:

    1. Original image
    2. Grad-CAM heatmap
    3. Grad-CAM overlay

    Returns:
        PNG bytes
    """

    # --------------------------------------------------------
    # Resize original
    # --------------------------------------------------------

    original = original_image.convert(
        "RGB"
    ).resize(
        (
            GRADCAM_IMAGE_SIZE,
            GRADCAM_IMAGE_SIZE,
        ),
        Image.Resampling.BILINEAR,
    )

    # --------------------------------------------------------
    # Resize CAM
    # --------------------------------------------------------

    cam_image = create_heatmap(
        cam
    ).resize(
        (
            GRADCAM_IMAGE_SIZE,
            GRADCAM_IMAGE_SIZE,
        ),
        Image.Resampling.BILINEAR,
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = Image.blend(
        original,
        cam_image,
        alpha=0.45,
    )

    # --------------------------------------------------------
    # Create canvas
    #
    # No Matplotlib.
    # --------------------------------------------------------

    panel_width = GRADCAM_IMAGE_SIZE
    panel_height = GRADCAM_IMAGE_SIZE

    canvas = Image.new(
        "RGB",
        (
            panel_width * 3,
            panel_height,
        ),
        "white",
    )

    canvas.paste(
        original,
        (
            0,
            0,
        ),
    )

    canvas.paste(
        cam_image,
        (
            panel_width,
            0,
        ),
    )

    canvas.paste(
        overlay,
        (
            panel_width * 2,
            0,
        ),
    )

    # --------------------------------------------------------
    # Save PNG in memory
    # --------------------------------------------------------

    image_buffer = io.BytesIO()

    canvas.save(
        image_buffer,
        format="PNG",
        optimize=True,
    )

    image_buffer.seek(
        0
    )

    return image_buffer.getvalue()


# ============================================================
# API GRAD-CAM
# ============================================================

def generate_gradcam_for_api(
    model,
    image,
    predicted_index=None,
    predicted_class=None,
    confidence=0.0,
):
    """
    Generate Grad-CAM for the FastAPI /explain endpoint.

    IMPORTANT:
    FastAPI passes both:

        predicted_index=<integer>
        predicted_class=<string>

    Example:

        predicted_index=1
        predicted_class="Cyst"

    The function safely resolves the integer index.

    Returns:
        PNG image bytes.
    """

    if model is None:

        raise ValueError(
            "Model cannot be None."
        )

    if image is None:

        raise ValueError(
            "Image cannot be None."
        )

    # --------------------------------------------------------
    # Resolve class index
    # --------------------------------------------------------

    class_index = normalize_class_index(
        predicted_index=predicted_index,
        predicted_class=predicted_class,
    )

    # --------------------------------------------------------
    # Resolve class name
    # --------------------------------------------------------

    resolved_class = CLASS_NAMES[
        class_index
    ]

    # --------------------------------------------------------
    # Use model's actual device
    # --------------------------------------------------------

    try:

        model_device = next(
            model.parameters()
        ).device

    except StopIteration:

        model_device = DEVICE

    # --------------------------------------------------------
    # Prepare image
    # --------------------------------------------------------

    original_image, input_tensor = (
        prepare_image(
            image=image,
            image_size=IMAGE_SIZE,
            device=model_device,
        )
    )

    # --------------------------------------------------------
    # Target layer
    # --------------------------------------------------------

    target_layer = get_target_layer(
        model
    )

    # --------------------------------------------------------
    # Generate CAM
    # --------------------------------------------------------

    try:

        with GradCAM(
            model=model,
            target_layer=target_layer,
        ) as gradcam:

            with torch.enable_grad():

                _, cam = gradcam.generate(
                    input_tensor=input_tensor,
                    class_index=class_index,
                )

    finally:

        # ----------------------------------------------------
        # Release input tensor
        # ----------------------------------------------------

        del input_tensor

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Create PNG
    # --------------------------------------------------------

    png_bytes = create_gradcam_overlay(
        original_image=original_image,
        cam=cam,
        predicted_class=resolved_class,
        confidence=float(
            confidence
        ),
    )

    # --------------------------------------------------------
    # Release CAM
    # --------------------------------------------------------

    del cam

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return png_bytes


# ============================================================
# LOCAL MODEL LOADER
# ============================================================

def load_local_deployment_model():
    """
    Load the bundled deployment model.

    This is only used when running this file directly.

    The FastAPI application has its own model loader.
    """

    deployment_model = (
        PROJECT_ROOT
        / "artifacts"
        / "deployment_model"
        / "data"
        / "model.pth"
    )

    if deployment_model.exists():

        print(
            f"Loading deployment model:\n"
            f"{deployment_model}"
        )

        model = torch.load(
            deployment_model,
            map_location=DEVICE,
            weights_only=False,
        )

        if not isinstance(
            model,
            torch.nn.Module,
        ):

            raise RuntimeError(
                "Deployment file does not contain "
                "a torch.nn.Module."
            )

        model.to(
            DEVICE
        )

        model.eval()

        return model

    raise FileNotFoundError(
        "Deployment model not found:\n"
        f"{deployment_model}"
    )


# ============================================================
# LOCAL TEST
# ============================================================

def main():
    """
    Local Grad-CAM smoke test.

    Uses the deployment model if available.
    """

    print(
        "=" * 70
    )

    print(
        "STEP 12 - GRAD-CAM EXPLAINABILITY"
    )

    print(
        "=" * 70
    )

    print()

    # --------------------------------------------------------
    # Find sample image
    # --------------------------------------------------------

    image_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "images"
        / "New folder9999"
        / "cystfolder"
        / "Cyst- (70).jpg"
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Sample image not found:\n"
            f"{image_path}"
        )

    print(
        f"Image      : {image_path}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_local_deployment_model()

    print(
        f"Model type : {model.__class__.__name__}"
    )

    print(
        f"Device     : {DEVICE}"
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    # --------------------------------------------------------
    # Prepare image
    # --------------------------------------------------------

    original_image, input_tensor = (
        prepare_image(
            image=image,
            image_size=IMAGE_SIZE,
            device=DEVICE,
        )
    )

    print(
        f"Input tensor shape: "
        f"{tuple(input_tensor.shape)}"
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        output = model(
            input_tensor
        )

        logits = extract_logits(
            output
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        confidence_tensor, predicted_index_tensor = (
            torch.max(
                probabilities,
                dim=1,
            )
        )

        predicted_index = int(
            predicted_index_tensor.item()
        )

        confidence = float(
            confidence_tensor.item()
        )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    print(
        f"Predicted class : "
        f"{predicted_class}"
    )

    print(
        f"Confidence      : "
        f"{confidence:.4f}"
    )

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    png_bytes = generate_gradcam_for_api(
        model=model,
        image=original_image,
        predicted_index=predicted_index,
        predicted_class=predicted_class,
        confidence=confidence,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / "gradcam_example.png"
    )

    output_path.write_bytes(
        png_bytes
    )

    print()

    print(
        f"Grad-CAM saved  : "
        f"{output_path}"
    )

    print(
        f"File size       : "
        f"{len(png_bytes):,} bytes"
    )

    print()

    print(
        "=" * 70
    )

    print(
        "GRAD-CAM COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()