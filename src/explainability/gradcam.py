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

# Keep visualization small to reduce memory usage.
GRADCAM_IMAGE_SIZE = 160

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Used only by the local Grad-CAM CLI.
# FastAPI receives an already-loaded model from api.inference.
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "swintransformer_best.pt"
)


# ============================================================
# TARGET LAYER
# ============================================================

def get_target_layer(model):
    """
    Find a suitable Grad-CAM target layer.

    Works with:
    - EfficientNet
    - Swin Transformer
    - other CNN-style models with Conv2d layers
    """

    # --------------------------------------------------------
    # Models with a `features` Sequential block.
    # EfficientNet uses this.
    # --------------------------------------------------------

    if hasattr(model, "features"):

        features = model.features

        if isinstance(features, torch.nn.Sequential):

            for layer in reversed(
                list(features)
            ):

                if layer is not None:
                    return layer

        if features is not None:
            return features

    # --------------------------------------------------------
    # Generic fallback: search for the last Conv2d.
    # --------------------------------------------------------

    modules = list(
        model.named_modules()
    )

    for _, module in reversed(modules):

        if isinstance(
            module,
            torch.nn.Conv2d,
        ):
            return module

    # --------------------------------------------------------
    # Generic Sequential fallback.
    # --------------------------------------------------------

    for _, module in reversed(modules):

        if isinstance(
            module,
            torch.nn.Sequential,
        ):
            return module

    raise RuntimeError(
        "Could not find a suitable Grad-CAM target layer."
    )


# ============================================================
# GRAD-CAM CLASS
# ============================================================

class GradCAM:
    """
    Lightweight Grad-CAM implementation.

    Uses a forward hook to capture activations.

    Gradients are obtained with torch.autograd.grad()
    instead of a backward hook to reduce memory overhead.
    """

    def __init__(
        self,
        model,
        target_layer,
    ):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.hook = None

        self._register_hook()

    # --------------------------------------------------------
    # Register forward hook
    # --------------------------------------------------------

    def _register_hook(self):

        self.hook = (
            self.target_layer.register_forward_hook(
                self._forward_hook
            )
        )

    # --------------------------------------------------------
    # Forward hook
    # --------------------------------------------------------

    def _forward_hook(
        self,
        module,
        inputs,
        output,
    ):
        self.activations = output

    # --------------------------------------------------------
    # Remove hook
    # --------------------------------------------------------

    def remove(self):

        if self.hook is not None:

            self.hook.remove()
            self.hook = None

    # --------------------------------------------------------
    # Generate Grad-CAM
    # --------------------------------------------------------

    def generate(
        self,
        input_tensor,
        target_class,
    ):
        """
        Generate Grad-CAM.

        Returns:
            Tensor with shape [B, H, W]
        """

        self.activations = None

        with torch.enable_grad():

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            output = self.model(
                input_tensor
            )

            # ------------------------------------------------
            # Extract logits
            # ------------------------------------------------

            if hasattr(
                output,
                "logits",
            ):

                logits = output.logits

            elif isinstance(
                output,
                tuple,
            ):

                logits = output[0]

            else:

                logits = output

            if logits.ndim != 2:

                raise RuntimeError(
                    "Expected model output with shape "
                    f"[B, C], got {tuple(logits.shape)}"
                )

            # ------------------------------------------------
            # Target score
            # ------------------------------------------------

            score = logits[
                :,
                target_class,
            ].sum()

            # ------------------------------------------------
            # Activations must exist.
            # ------------------------------------------------

            if self.activations is None:

                raise RuntimeError(
                    "Grad-CAM target layer did not "
                    "produce activations."
                )

            activations = self.activations

            # ------------------------------------------------
            # Convert activation layout.
            #
            # CNN:
            # [B, C, H, W]
            #
            # Transformer:
            # [B, H, W, C]
            # or [B, N, C]
            # ------------------------------------------------

            if activations.ndim == 4:

                # Most CNN models.
                if (
                    activations.shape[1] <= 2048
                    and activations.shape[2] > 1
                    and activations.shape[3] > 1
                ):

                    activation_tensor = activations

                else:

                    # Transformer layout:
                    # [B, H, W, C]
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

            elif activations.ndim == 3:

                # Transformer sequence:
                # [B, N, C]

                batch_size, tokens, channels = (
                    activations.shape
                )

                side = int(
                    np.sqrt(tokens)
                )

                if side * side != tokens:

                    raise RuntimeError(
                        "Cannot reshape transformer "
                        f"activation with {tokens} tokens "
                        "into a square feature map."
                    )

                activation_tensor = (
                    activations
                    .transpose(
                        1,
                        2,
                    )
                    .contiguous()
                    .view(
                        batch_size,
                        channels,
                        side,
                        side,
                    )
                )

            else:

                raise RuntimeError(
                    "Unsupported activation shape: "
                    f"{tuple(activations.shape)}"
                )

            # ------------------------------------------------
            # Gradient of target score with respect to
            # captured activations.
            # ------------------------------------------------

            gradients = torch.autograd.grad(
                outputs=score,
                inputs=activations,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]

            # ------------------------------------------------
            # Match gradient layout.
            # ------------------------------------------------

            if gradients.ndim == 4:

                if (
                    gradients.shape[1] <= 2048
                    and gradients.shape[2] > 1
                    and gradients.shape[3] > 1
                ):

                    gradient_tensor = gradients

                else:

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

            elif gradients.ndim == 3:

                batch_size, tokens, channels = (
                    gradients.shape
                )

                side = int(
                    np.sqrt(tokens)
                )

                gradient_tensor = (
                    gradients
                    .transpose(
                        1,
                        2,
                    )
                    .contiguous()
                    .view(
                        batch_size,
                        channels,
                        side,
                        side,
                    )
                )

            else:

                raise RuntimeError(
                    "Unsupported gradient shape: "
                    f"{tuple(gradients.shape)}"
                )

            # ------------------------------------------------
            # Global average pooling.
            # ------------------------------------------------

            weights = gradient_tensor.mean(
                dim=(2, 3),
                keepdim=True,
            )

            # ------------------------------------------------
            # Weighted activation map.
            # ------------------------------------------------

            cam = (
                weights * activation_tensor
            ).sum(
                dim=1,
                keepdim=True,
            )

            # ReLU.
            cam = F.relu(cam)

            # ------------------------------------------------
            # Resize CAM to input image dimensions.
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Normalize CAM to [0, 1].
            # ------------------------------------------------

            cam_min = cam.amin(
                dim=(1, 2),
                keepdim=True,
            )

            cam_max = cam.amax(
                dim=(1, 2),
                keepdim=True,
            )

            cam = (
                cam - cam_min
            ) / (
                cam_max
                - cam_min
                + 1e-8
            )

            return cam.detach()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):
    """
    Convert a PIL image into a model input tensor.
    """

    if image.mode != "RGB":

        image = image.convert(
            "RGB"
        )

    transform = build_transforms(
        image_size=IMAGE_SIZE,
        train=False,
    )

    tensor = transform(
        image
    )

    tensor = tensor.unsqueeze(0)

    return tensor.to(
        DEVICE
    )


# ============================================================
# HEATMAP
# ============================================================

def create_heatmap(cam):
    """
    Convert normalized CAM into an RGB heatmap.

    No Matplotlib is used.
    """

    cam = np.asarray(
        cam,
        dtype=np.float32,
    )

    cam = np.clip(
        cam,
        0.0,
        1.0,
    )

    heatmap = np.zeros(
        (
            cam.shape[0],
            cam.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    # Red.
    heatmap[:, :, 0] = (
        np.clip(
            255.0
            * (
                cam * 1.5
                - 0.5
            ),
            0,
            255,
        )
    ).astype(
        np.uint8
    )

    # Green.
    heatmap[:, :, 1] = (
        np.clip(
            255.0
            * (
                cam * 1.5
            ),
            0,
            255,
        )
    ).astype(
        np.uint8
    )

    # Blue.
    heatmap[:, :, 2] = (
        np.clip(
            255.0
            * (
                1.0
                - cam * 1.5
            ),
            0,
            255,
        )
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        heatmap,
        mode="RGB",
    )


# ============================================================
# VISUALIZATION
# ============================================================

def create_visualization(
    original_image,
    cam,
    predicted_class,
    confidence,
):
    """
    Create:

        Original | Heatmap | Overlay

    Returns:
        PIL.Image
    """

    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    original = (
        original_image
        .convert("RGB")
        .resize(
            (
                GRADCAM_IMAGE_SIZE,
                GRADCAM_IMAGE_SIZE,
            ),
            Image.Resampling.BILINEAR,
        )
    )

    # --------------------------------------------------------
    # CAM grayscale
    # --------------------------------------------------------

    cam_image = Image.fromarray(
        (
            np.clip(
                cam,
                0.0,
                1.0,
            )
            * 255
        ).astype(
            np.uint8
        ),
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
        np.asarray(
            cam_image
        ).astype(
            np.float32
        )
        / 255.0
    )

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

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
    # Canvas
    # --------------------------------------------------------

    width = (
        GRADCAM_IMAGE_SIZE
        * 3
    )

    height = GRADCAM_IMAGE_SIZE

    canvas = Image.new(
        "RGB",
        (
            width,
            height,
        ),
        "black",
    )

    canvas.paste(
        original,
        (
            0,
            0,
        ),
    )

    canvas.paste(
        heatmap,
        (
            GRADCAM_IMAGE_SIZE,
            0,
        ),
    )

    canvas.paste(
        overlay,
        (
            GRADCAM_IMAGE_SIZE * 2,
            0,
        ),
    )

    return canvas


# ============================================================
# API GRAD-CAM
# ============================================================

def generate_gradcam_for_api(
    model,
    image,
    predicted_class=None,
    confidence=0.0,
    predicted_index=None,
):
    """
    Generate Grad-CAM PNG bytes for FastAPI.

    Supports both:

        predicted_class=
        predicted_index=

    This keeps compatibility with the existing
    api.main.py implementation.
    """

    # --------------------------------------------------------
    # Backward compatibility.
    # --------------------------------------------------------

    if predicted_class is None:

        if predicted_index is None:

            raise ValueError(
                "Either predicted_class or "
                "predicted_index must be provided."
            )

        predicted_class = predicted_index

    predicted_class = int(
        predicted_class
    )

    # --------------------------------------------------------
    # Validate class index.
    # --------------------------------------------------------

    if predicted_class < 0:

        predicted_class = 0

    if predicted_class >= len(
        CLASS_NAMES
    ):

        predicted_class = (
            len(CLASS_NAMES)
            - 1
        )

    # --------------------------------------------------------
    # Validate image.
    # --------------------------------------------------------

    if not isinstance(
        image,
        Image.Image,
    ):

        raise TypeError(
            "image must be a PIL.Image.Image"
        )

    original_image = image.convert(
        "RGB"
    )

    # --------------------------------------------------------
    # Prepare tensor.
    # --------------------------------------------------------

    input_tensor = preprocess_image(
        original_image
    )

    # --------------------------------------------------------
    # Model.
    # --------------------------------------------------------

    model.eval()

    # --------------------------------------------------------
    # Target layer.
    # --------------------------------------------------------

    target_layer = get_target_layer(
        model
    )

    print(
        "Grad-CAM target layer:",
        target_layer.__class__.__name__,
    )

    # --------------------------------------------------------
    # Grad-CAM engine.
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

        cam = (
            cam[0]
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Visualization.
        # ----------------------------------------------------

        visualization = create_visualization(
            original_image=original_image,
            cam=cam,
            predicted_class=predicted_class,
            confidence=confidence,
        )

        # ----------------------------------------------------
        # PNG bytes.
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

        # Remove forward hook immediately.
        cam_engine.remove()

        cam_engine.activations = None

        # Release input tensor.
        del input_tensor

        # CUDA cleanup if applicable.
        if torch.cuda.is_available():

            torch.cuda.empty_cache()


# ============================================================
# LOCAL SWIN CHECKPOINT
# ============================================================

def load_local_swin_model():
    """
    Load the locally trained SwinTransformer checkpoint.

    Used only by the local CLI.

    FastAPI does NOT use this function.
    """

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{CHECKPOINT_PATH}"
        )

    model = create_model(
        model_name="SwinTransformer",
        num_classes=len(
            CLASS_NAMES
        ),
        pretrained=False,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    # --------------------------------------------------------
    # Training checkpoint.
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict,
    ):

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

    # --------------------------------------------------------
    # Complete model object.
    # --------------------------------------------------------

    elif isinstance(
        checkpoint,
        torch.nn.Module,
    ):

        model = checkpoint

    else:

        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    model.to(
        DEVICE
    )

    model.eval()

    return model


# ============================================================
# LOCAL GRAD-CAM
# ============================================================

def run_local_gradcam(
    image_path,
    output_path,
):
    """
    Run Grad-CAM locally using the SwinTransformer
    checkpoint.
    """

    image_path = Path(
        image_path
    )

    output_path = Path(
        output_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: "
            f"{image_path}"
        )

    print("=" * 70)
    print(
        "GRAD-CAM EXPLAINABILITY"
    )
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
    ).convert(
        "RGB"
    )

    # --------------------------------------------------------
    # Load model.
    # --------------------------------------------------------

    print(
        "Loading local SwinTransformer model..."
    )

    model = load_local_swin_model()

    print(
        "Model loaded:",
        model.__class__.__name__,
    )

    # --------------------------------------------------------
    # Prediction.
    # --------------------------------------------------------

    input_tensor = preprocess_image(
        image
    )

    with torch.no_grad():

        output = model(
            input_tensor
        )

        if hasattr(
            output,
            "logits",
        ):

            logits = output.logits

        elif isinstance(
            output,
            tuple,
        ):

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
        CLASS_NAMES[
            predicted_class
        ],
    )

    print(
        "Confidence:",
        f"{confidence:.4f}",
    )

    # --------------------------------------------------------
    # Grad-CAM.
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