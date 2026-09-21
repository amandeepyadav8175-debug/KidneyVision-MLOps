from pathlib import Path
import io
import sys

import numpy as np
import torch
import torch.nn.functional as F
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
        weights_only=False,
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

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
    Select a spatial feature layer suitable for Grad-CAM.

    EfficientNet:
        final convolutional feature block

    Swin:
        final feature block

    The API deployment model can therefore use the same Grad-CAM
    implementation regardless of which supported model object is loaded.
    """

    if hasattr(model, "features"):

        features = model.features

        if len(features) == 0:
            raise RuntimeError(
                "Model features container is empty."
            )

        # EfficientNet:
        # model.features[-1] is normally the final Conv2dNormActivation.
        #
        # If the final block is a Sequential container, use its final
        # convolutional module where possible.
        last_block = features[-1]

        if isinstance(last_block, torch.nn.Sequential):

            for module in reversed(list(last_block.children())):

                if isinstance(
                    module,
                    (
                        torch.nn.Conv2d,
                        torch.nn.modules.conv.Conv2d,
                    ),
                ):
                    target_layer = module
                    break

            else:
                target_layer = last_block

        else:
            target_layer = last_block

    else:
        raise RuntimeError(
            "Unable to automatically locate a Grad-CAM target layer "
            f"for model type: {type(model).__name__}"
        )

    print(
        "Grad-CAM target layer:",
        target_layer.__class__.__name__,
    )

    return target_layer


# -------------------------------------------------------------------
# Grad-CAM
# -------------------------------------------------------------------

class GradCAM:

    def __init__(
        self,
        model,
        target_layer,
    ):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(
            self.save_activation
        )

        self.backward_handle = None

    # ---------------------------------------------------------------
    # Save activation
    # ---------------------------------------------------------------

    def save_activation(
        self,
        module,
        input_data,
        output,
    ):

        self.activations = output

        # Instead of using a module backward hook, attach a tensor
        # gradient hook directly to the activation tensor.
        #
        # This is lighter and avoids some backward-hook overhead.

        if isinstance(output, torch.Tensor) and output.requires_grad:

            output.register_hook(
                self.save_tensor_gradient
            )

    # ---------------------------------------------------------------
    # Save gradient
    # ---------------------------------------------------------------

    def save_tensor_gradient(self, gradient):

        self.gradients = gradient

        return gradient

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
    # Context manager
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

        self.activations = None
        self.gradients = None

    # ---------------------------------------------------------------
    # Generate CAM
    # ---------------------------------------------------------------

    def generate(
        self,
        input_tensor,
        class_index,
    ):

        self.activations = None
        self.gradients = None

        self.model.zero_grad(set_to_none=True)

        output = self.model(input_tensor)

        if hasattr(output, "logits"):
            output = output.logits

        if not isinstance(output, torch.Tensor):

            raise RuntimeError(
                "Model output is not a torch.Tensor."
            )

        if output.ndim != 2:

            raise RuntimeError(
                "Unexpected model output shape: "
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

        if self.activations is None:

            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        score = output[:, class_index].sum()

        score.backward()

        gradients = self.gradients
        activations = self.activations

        if gradients is None:

            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        # -----------------------------------------------------------
        # Convert feature layout
        # -----------------------------------------------------------

        if activations.ndim != 4:

            raise RuntimeError(
                "Expected 4D Grad-CAM activations, got: "
                f"{tuple(activations.shape)}"
            )

        if gradients.ndim != 4:

            raise RuntimeError(
                "Expected 4D Grad-CAM gradients, got: "
                f"{tuple(gradients.shape)}"
            )

        # Swin:
        # [B, H, W, C]
        #
        # CNN:
        # [B, C, H, W]

        if (
            activations.shape[1] < activations.shape[-1]
            and activations.shape[2] < activations.shape[-1]
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
        # Resize CAM
        # -----------------------------------------------------------

        cam = F.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze(0).squeeze(0)

        # -----------------------------------------------------------
        # Normalize
        # -----------------------------------------------------------

        cam_min = cam.min()
        cam_max = cam.max()

        denominator = cam_max - cam_min

        if denominator > 1e-8:

            cam = (
                cam - cam_min
            ) / denominator

        else:

            cam = torch.zeros_like(cam)

        # Detach immediately to release autograd graph.
        output = output.detach()
        cam = cam.detach()

        self.activations = None
        self.gradients = None

        return output, cam


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
# Simple heatmap generation without Matplotlib
# -------------------------------------------------------------------

def create_heatmap(cam_array):
    """
    Convert normalized CAM [0,1] into an RGB heatmap.

    This intentionally avoids Matplotlib because the API runs on
    low-memory deployment instances.
    """

    cam_array = np.clip(
        cam_array,
        0.0,
        1.0,
    )

    # Simple blue -> cyan -> yellow -> red style gradient.
    r = np.clip(
        2.0 * cam_array - 0.5,
        0.0,
        1.0,
    )

    g = np.clip(
        2.0 - np.abs(4.0 * cam_array - 2.0),
        0.0,
        1.0,
    )

    b = np.clip(
        1.5 - 2.0 * cam_array,
        0.0,
        1.0,
    )

    heatmap = np.stack(
        [r, g, b],
        axis=-1,
    )

    return (
        heatmap * 255.0
    ).astype(np.uint8)


# -------------------------------------------------------------------
# Create visualization without Matplotlib
# -------------------------------------------------------------------

def create_gradcam_image(
    original_image,
    cam,
    predicted_class,
    confidence,
):
    """
    Create a compact Grad-CAM visualization entirely with PIL/NumPy.

    Returns:
        PNG bytes
    """

    original_image = original_image.convert(
        "RGB"
    ).resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )

    original_array = np.asarray(
        original_image,
        dtype=np.float32,
    )

    cam_array = cam.cpu().numpy()

    cam_array = np.clip(
        cam_array,
        0.0,
        1.0,
    )

    # ---------------------------------------------------------------
    # Heatmap
    # ---------------------------------------------------------------

    heatmap_array = create_heatmap(
        cam_array
    )

    heatmap_image = Image.fromarray(
        heatmap_array,
        mode="RGB",
    )

    # ---------------------------------------------------------------
    # Overlay
    # ---------------------------------------------------------------

    overlay_array = (
        0.55 * original_array
        + 0.45 * heatmap_array.astype(np.float32)
    )

    overlay_array = np.clip(
        overlay_array,
        0,
        255,
    ).astype(np.uint8)

    overlay_image = Image.fromarray(
        overlay_array,
        mode="RGB",
    )

    # ---------------------------------------------------------------
    # Build 3-panel image using PIL
    # ---------------------------------------------------------------

    panel_width = IMAGE_SIZE
    panel_height = IMAGE_SIZE + 35

    canvas = Image.new(
        "RGB",
        (
            panel_width * 3,
            panel_height,
        ),
        "white",
    )

    canvas.paste(
        original_image,
        (0, 0),
    )

    canvas.paste(
        heatmap_image,
        (panel_width, 0),
    )

    canvas.paste(
        overlay_image,
        (panel_width * 2, 0),
    )

    # Add simple labels.
    try:

        from PIL import ImageDraw

        draw = ImageDraw.Draw(canvas)

        draw.text(
            (10, IMAGE_SIZE + 8),
            "Original CT Image",
            fill="black",
        )

        draw.text(
            (panel_width + 10, IMAGE_SIZE + 8),
            "Grad-CAM",
            fill="black",
        )

        draw.text(
            (panel_width * 2 + 10, IMAGE_SIZE + 8),
            f"{predicted_class} {confidence:.2%}",
            fill="black",
        )

    except Exception:
        pass

    # ---------------------------------------------------------------
    # Encode PNG in memory
    # ---------------------------------------------------------------

    image_buffer = io.BytesIO()

    canvas.save(
        image_buffer,
        format="PNG",
        optimize=True,
    )

    image_bytes = image_buffer.getvalue()

    image_buffer.close()

    return image_bytes


# -------------------------------------------------------------------
# Save visualization for local CLI
# -------------------------------------------------------------------

def save_gradcam_visualization(
    original_image,
    cam,
    predicted_class,
    confidence,
    output_path,
):

    image_bytes = create_gradcam_image(
        original_image=original_image,
        cam=cam,
        predicted_class=predicted_class,
        confidence=confidence,
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_bytes(
        image_bytes
    )


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
    Generate a Grad-CAM PNG for the FastAPI backend.

    Important:
    - Does not load another model.
    - Does not import/use Matplotlib.
    - Performs exactly one forward/backward Grad-CAM pass.
    - Removes hooks automatically.
    """

    transform = build_transforms(
        image_size=IMAGE_SIZE,
        train=False,
    )

    input_tensor = transform(
        image.convert("RGB")
    )

    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(
        DEVICE
    )

    target_layer = get_target_layer(
        model
    )

    try:

        with GradCAM(
            model=model,
            target_layer=target_layer,
        ) as gradcam:

            with torch.enable_grad():

                _, cam = gradcam.generate(
                    input_tensor=input_tensor,
                    class_index=int(
                        predicted_index
                    ),
                )

        # -----------------------------------------------------------
        # Create compact visualization
        # -----------------------------------------------------------

        image_bytes = create_gradcam_image(
            original_image=image,
            cam=cam,
            predicted_class=predicted_class,
            confidence=confidence,
        )

        return image_bytes

    finally:

        # Release references immediately.
        input_tensor = None
        cam = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
    # Find test image
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
    # Prediction first
    # ---------------------------------------------------------------

    with torch.no_grad():

        output = model(
            input_tensor
        )

        if hasattr(output, "logits"):
            output = output.logits

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

    print(
        f"Predicted class : {predicted_class}"
    )

    print(
        f"Confidence      : {confidence:.4f}"
    )

    # ---------------------------------------------------------------
    # Grad-CAM
    # ---------------------------------------------------------------

    target_layer = get_target_layer(
        model
    )

    with GradCAM(
        model=model,
        target_layer=target_layer,
    ) as gradcam:

        with torch.enable_grad():

            _, cam = gradcam.generate(
                input_tensor=input_tensor,
                class_index=predicted_index,
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