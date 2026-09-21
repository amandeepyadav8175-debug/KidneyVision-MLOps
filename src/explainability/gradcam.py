from pathlib import Path
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.models.model_factory import create_model
from src.preprocessing.image_preprocessor import build_transforms


# ============================================================
# PROJECT CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLASS_NAMES = [
    "Normal",
    "Cyst",
    "Stone",
    "Tumor",
]

IMAGE_SIZE = 224

# Smaller visualization size keeps memory usage low,
# especially on Render Free tier.
GRADCAM_IMAGE_SIZE = 160

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Local CLI checkpoint only.
# FastAPI does NOT use this path.
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "swintransformer_best.pt"
)


# ============================================================
# TARGET LAYER SELECTION
# ============================================================

def get_target_layer(model):
    """
    Select a suitable convolutional/feature layer for Grad-CAM.

    EfficientNet:
        Uses the final convolutional feature block.

    SwinTransformer:
        Uses the final feature block.

    Generic fallback:
        Searches backward through the model for a suitable
        Conv2d or Sequential layer.
    """

    # --------------------------------------------------------
    # EfficientNet
    # --------------------------------------------------------
    if hasattr(model, "features"):
        features = model.features

        if isinstance(features, torch.nn.Sequential):
            for layer in reversed(list(features)):
                if layer is not None:
                    return layer

        return features

    # --------------------------------------------------------
    # Swin Transformer / models with feature blocks
    # --------------------------------------------------------
    if hasattr(model, "features"):
        return model.features

    # --------------------------------------------------------
    # Generic search
    # --------------------------------------------------------
    modules = list(model.named_modules())

    # First preference: Conv2d
    for name, module in reversed(modules):
        if isinstance(module, torch.nn.Conv2d):
            return module

    # Second preference: Sequential
    for name, module in reversed(modules):
        if isinstance(module, torch.nn.Sequential):
            return module

    raise RuntimeError(
        "Could not find a suitable Grad-CAM target layer."
    )


# ============================================================
# GRAD-CAM ENGINE
# ============================================================

class GradCAM:
    """
    Lightweight Grad-CAM implementation.

    Uses a forward hook to capture activations.

    Gradients are obtained with torch.autograd.grad()
    instead of registering a backward hook. This avoids
    unnecessary graph retention and reduces memory usage.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.hook = None

        self._register_hook()

    # --------------------------------------------------------
    # Forward hook
    # --------------------------------------------------------

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _register_hook(self):
        self.hook = self.target_layer.register_forward_hook(
            self._forward_hook
        )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    def remove(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    # --------------------------------------------------------
    # Generate CAM
    # --------------------------------------------------------

    def generate(self, input_tensor, target_class):
        """
        Generate Grad-CAM for one image.

        Returns:
            CAM tensor with shape [1, H, W]
        """

        self.activations = None

        # Make sure gradients are enabled.
        with torch.enable_grad():

            # Forward pass.
            output = self.model(input_tensor)

            # Handle models returning objects with .logits.
            if hasattr(output, "logits"):
                logits = output.logits

            # Handle models returning tuples.
            elif isinstance(output, tuple):
                logits = output[0]

            else:
                logits = output

            if logits.ndim != 2:
                raise RuntimeError(
                    f"Expected logits shape [B, C], "
                    f"got {tuple(logits.shape)}"
                )

            # Target score.
            score = logits[:, target_class].sum()

            if self.activations is None:
                raise RuntimeError(
                    "Grad-CAM target layer did not produce "
                    "activations."
                )

            activations = self.activations

            # ------------------------------------------------
            # Convert activation layout if necessary.
            #
            # CNN:
            #   [B, C, H, W]
            #
            # Transformer:
            #   [B, H, W, C]
            # ------------------------------------------------

            if activations.ndim == 4:

                # CNN-style tensor.
                if (
                    activations.shape[1] <= 2048
                    and activations.shape[2] > 1
                    and activations.shape[3] > 1
                ):
                    activation_tensor = activations

                else:
                    # Transformer-style [B,H,W,C]
                    activation_tensor = (
                        activations.permute(0, 3, 1, 2)
                        .contiguous()
                    )

            elif activations.ndim == 3:
                # Possible transformer sequence:
                # [B, N, C]
                batch, tokens, channels = activations.shape

                side = int(np.sqrt(tokens))

                if side * side != tokens:
                    raise RuntimeError(
                        "Cannot reshape transformer activation "
                        f"with {tokens} tokens into a square map."
                    )

                activation_tensor = (
                    activations
                    .transpose(1, 2)
                    .contiguous()
                    .view(batch, channels, side, side)
                )

            else:
                raise RuntimeError(
                    "Unsupported activation shape: "
                    f"{tuple(activations.shape)}"
                )

            # ------------------------------------------------
            # Get gradients with respect to activations.
            # ------------------------------------------------

            gradients = torch.autograd.grad(
                outputs=score,
                inputs=activations,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]

            # Match gradient layout with activation layout.
            if gradients.ndim == 4:

                if (
                    gradients.shape[1] <= 2048
                    and gradients.shape[2] > 1
                    and gradients.shape[3] > 1
                ):
                    gradient_tensor = gradients

                else:
                    gradient_tensor = (
                        gradients.permute(0, 3, 1, 2)
                        .contiguous()
                    )

            elif gradients.ndim == 3:

                batch, tokens, channels = gradients.shape

                side = int(np.sqrt(tokens))

                gradient_tensor = (
                    gradients
                    .transpose(1, 2)
                    .contiguous()
                    .view(batch, channels, side, side)
                )

            else:
                raise RuntimeError(
                    "Unsupported gradient shape: "
                    f"{tuple(gradients.shape)}"
                )

            # ------------------------------------------------
            # Global average pooling over spatial dimensions.
            # ------------------------------------------------

            weights = gradient_tensor.mean(
                dim=(2, 3),
                keepdim=True
            )

            # ------------------------------------------------
            # Weighted activation maps.
            # ------------------------------------------------

            cam = (
                weights * activation_tensor
            ).sum(dim=1, keepdim=True)

            # ReLU.
            cam = F.relu(cam)

            # Resize to input image size.
            cam = F.interpolate(
                cam,
                size=(
                    input_tensor.shape[-2],
                    input_tensor.shape[-1],
                ),
                mode="bilinear",
                align_corners=False,
            )

            # Remove channel dimension.
            cam = cam[:, 0]

            # Normalize each image independently.
            cam_min = cam.amin(
                dim=(1, 2),
                keepdim=True
            )

            cam_max = cam.amax(
                dim=(1, 2),
                keepdim=True
            )

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min + 1e-8
            )

            return cam.detach()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):
    """
    Convert PIL image to model input tensor.
    """

    if image.mode != "RGB":
        image = image.convert("RGB")

    transform = build_transforms(
        image_size=IMAGE_SIZE,
        train=False,
    )

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    return tensor.to(DEVICE)


# ============================================================
# HEATMAP CREATION
# ============================================================

def create_heatmap(cam):
    """
    Convert normalized CAM [H,W] into an RGB heatmap.

    This intentionally avoids Matplotlib so that the API
    does not carry the Matplotlib rendering stack.
    """

    cam = np.asarray(cam, dtype=np.float32)

    cam = np.clip(cam, 0.0, 1.0)

    # Simple blue -> cyan -> yellow -> red gradient.
    heatmap = np.zeros(
        (cam.shape[0], cam.shape[1], 3),
        dtype=np.uint8,
    )

    # Red channel.
    heatmap[:, :, 0] = (
        np.clip(
            255.0 * (cam * 1.5 - 0.5),
            0,
            255,
        )
    ).astype(np.uint8)

    # Green channel.
    heatmap[:, :, 1] = (
        np.clip(
            255.0 * (cam * 1.5),
            0,
            255,
        )
    ).astype(np.uint8)

    # Blue channel.
    heatmap[:, :, 2] = (
        np.clip(
            255.0 * (1.0 - cam * 1.5),
            0,
            255,
        )
    ).astype(np.uint8)

    return Image.fromarray(
        heatmap,
        mode="RGB",
    )


# ============================================================
# GRAD-CAM VISUALIZATION
# ============================================================

def create_visualization(
    original_image,
    cam,
    predicted_class,
    confidence,
):
    """
    Create a lightweight 3-panel visualization:

        Original | Heatmap | Overlay

    Returns:
        PIL.Image
    """

    # Resize original image.
    original = original_image.convert("RGB")

    original = original.resize(
        (
            GRADCAM_IMAGE_SIZE,
            GRADCAM_IMAGE_SIZE,
        ),
        Image.Resampling.BILINEAR,
    )

    # CAM -> heatmap.
    cam_image = Image.fromarray(
        (
            np.clip(cam, 0.0, 1.0) * 255
        ).astype(np.uint8),
        mode="L",
    )

    cam_image = cam_image.resize(
        (
            GRADCAM_IMAGE_SIZE,
            GRADCAM_IMAGE_SIZE,
        ),
        Image.Resampling.BILINEAR,
    )

    cam_array = (
        np.asarray(cam_image)
        .astype(np.float32)
        / 255.0
    )

    heatmap = create_heatmap(
        cam_array
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = Image.blend(
        original,
        heatmap,
        alpha=0.45,
    )

    # --------------------------------------------------------
    # Create horizontal canvas.
    # --------------------------------------------------------

    width = GRADCAM_IMAGE_SIZE * 3
    height = GRADCAM_IMAGE_SIZE

    canvas = Image.new(
        "RGB",
        (width, height),
        "black",
    )

    canvas.paste(
        original,
        (0, 0),
    )

    canvas.paste(
        heatmap,
        (GRADCAM_IMAGE_SIZE, 0),
    )

    canvas.paste(
        overlay,
        (GRADCAM_IMAGE_SIZE * 2, 0),
    )

    return canvas


# ============================================================
# API GRAD-CAM FUNCTION
# ============================================================

def generate_gradcam_for_api(
    model,
    image,
    predicted_class,
    confidence,
):
    """
    Generate Grad-CAM PNG bytes for FastAPI.

    IMPORTANT:
        The model is supplied by api.inference.

        This function does NOT load another model.

        This prevents duplicate model memory usage on Render.
    """

    if not isinstance(image, Image.Image):
        raise TypeError(
            "image must be a PIL.Image.Image"
        )

    if not isinstance(predicted_class, int):
        predicted_class = int(predicted_class)

    if predicted_class < 0:
        predicted_class = 0

    if predicted_class >= len(CLASS_NAMES):
        predicted_class = len(CLASS_NAMES) - 1

    # --------------------------------------------------------
    # Original image.
    # --------------------------------------------------------

    original_image = image.convert("RGB")

    # --------------------------------------------------------
    # Preprocess.
    # --------------------------------------------------------

    input_tensor = preprocess_image(
        original_image
    )

    # --------------------------------------------------------
    # Model mode.
    # --------------------------------------------------------

    model.eval()

    # --------------------------------------------------------
    # Find target layer.
    # --------------------------------------------------------

    target_layer = get_target_layer(
        model
    )

    print(
        "Grad-CAM target layer:",
        target_layer.__class__.__name__,
    )

    # --------------------------------------------------------
    # Grad-CAM.
    # --------------------------------------------------------

    cam_engine = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    try:

        cam = cam_engine.generate(
            input_tensor=input_tensor,
            target_class=predicted_class,
        )

        cam = cam[0].cpu().numpy()

        # ----------------------------------------------------
        # Create visualization.
        # ----------------------------------------------------

        visualization = create_visualization(
            original_image=original_image,
            cam=cam,
            predicted_class=predicted_class,
            confidence=confidence,
        )

        # ----------------------------------------------------
        # Encode PNG.
        # ----------------------------------------------------

        output = io.BytesIO()

        visualization.save(
            output,
            format="PNG",
            optimize=True,
        )

        output.seek(0)

        return output.getvalue()

    finally:

        # Remove hook immediately.
        cam_engine.remove()

        # Release references.
        cam_engine.activations = None

        del input_tensor

        # CPU deployment:
        # this is harmless and helps when CUDA is used locally.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# LOCAL MODEL LOADING
# ============================================================

def load_local_swin_model():
    """
    Load the locally trained SwinTransformer checkpoint.

    This function is ONLY for the local Grad-CAM CLI.

    FastAPI does not use this function.
    """

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    model = create_model(
        model_name="SwinTransformer",
        num_classes=len(CLASS_NAMES),
        pretrained=False,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get(
            "model_state_dict"
        )

        if state_dict is None:
            raise RuntimeError(
                "Checkpoint does not contain "
                "'model_state_dict'."
            )

        model.load_state_dict(
            state_dict
        )

    elif isinstance(checkpoint, torch.nn.Module):
        model = checkpoint

    else:
        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# LOCAL CLI
# ============================================================

def run_local_gradcam(
    image_path,
    output_path,
):
    """
    Run Grad-CAM locally using the SwinTransformer checkpoint.
    """

    image_path = Path(image_path)
    output_path = Path(output_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    print("=" * 70)
    print("GRAD-CAM EXPLAINABILITY")
    print("=" * 70)

    print(
        "Checkpoint:",
        CHECKPOINT_PATH,
    )

    print(
        "Image:",
        image_path,
    )

    print(
        "Device:",
        DEVICE,
    )

    # --------------------------------------------------------
    # Load image.
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    # --------------------------------------------------------
    # Load model.
    # --------------------------------------------------------

    print("Loading local SwinTransformer model...")

    model = load_local_swin_model()

    print(
        "Model loaded:",
        model.__class__.__name__,
    )

    # --------------------------------------------------------
    # Predict first.
    # --------------------------------------------------------

    input_tensor = preprocess_image(
        image
    )

    with torch.no_grad():

        output = model(
            input_tensor
        )

        if hasattr(output, "logits"):
            logits = output.logits

        elif isinstance(output, tuple):
            logits = output[0]

        else:
            logits = output

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predicted_class = int(
            probabilities.argmax(
                dim=1
            ).item()
        )

        confidence = float(
            probabilities[
                0,
                predicted_class,
            ].item()
        )

    del input_tensor

    print(
        "Predicted class:",
        CLASS_NAMES[predicted_class],
    )

    print(
        "Confidence:",
        f"{confidence:.4f}",
    )

    # --------------------------------------------------------
    # Generate Grad-CAM.
    # --------------------------------------------------------

    result = generate_gradcam_for_api(
        model=model,
        image=image,
        predicted_class=predicted_class,
        confidence=confidence,
    )

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_bytes(
        result
    )

    print(
        "Grad-CAM saved:",
        output_path,
    )

    print(
        "GRAD-CAM COMPLETE"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    default_image = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "images"
        / "New folder9999"
        / "cystfolder"
        / "Cyst- (70).jpg"
    )

    default_output = (
        PROJECT_ROOT
        / "artifacts"
        / "explainability"
        / "gradcam"
        / "gradcam_example.png"
    )

    run_local_gradcam(
        image_path=default_image,
        output_path=default_output,
    )