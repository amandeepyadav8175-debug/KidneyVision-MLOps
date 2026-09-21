from __future__ import annotations

import io
import gc
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "deployment_model"
    / "data"
    / "model.pth"
)

CLASS_NAMES = [
    "Normal",
    "Cyst",
    "Stone",
    "Tumor",
]

NUM_CLASSES = len(CLASS_NAMES)

# IMPORTANT:
# Render Free has limited RAM.
# Grad-CAM will use a smaller image than prediction.
GRADCAM_IMAGE_SIZE = 128

DEVICE = torch.device("cpu")


# ============================================================
# MEMORY SETTINGS
# ============================================================

def configure_cpu_memory() -> None:
    """
    Keep CPU thread usage low.

    This is important on small Render instances because
    PyTorch can otherwise create many worker threads and
    increase memory usage.
    """
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


# ============================================================
# LOGITS EXTRACTION
# ============================================================

def extract_logits(output):
    """
    Supports:
    - Tensor
    - Inception-style objects with .logits
    - tuple/list outputs
    """

    if torch.is_tensor(output):
        return output

    if hasattr(output, "logits"):
        return output.logits

    if isinstance(output, (tuple, list)):
        if len(output) == 0:
            raise RuntimeError("Model returned an empty tuple/list.")

        first = output[0]

        if torch.is_tensor(first):
            return first

        if hasattr(first, "logits"):
            return first.logits

    raise RuntimeError(
        f"Unsupported model output type: {type(output)}"
    )


# ============================================================
# CLASS NAME / INDEX HANDLING
# ============================================================

def normalize_class_index(
    predicted_index: Optional[int] = None,
    predicted_class: Optional[str | int] = None,
) -> int:
    """
    Convert prediction information into a valid class index.

    Examples:
        "Cyst" -> 1
        "Stone" -> 2
        1 -> 1
        "1" -> 1
    """

    if predicted_index is not None:

        if isinstance(predicted_index, str):
            if predicted_index in CLASS_NAMES:
                return CLASS_NAMES.index(predicted_index)

            try:
                predicted_index = int(predicted_index)
            except ValueError:
                raise ValueError(
                    f"Invalid predicted_index: {predicted_index}"
                )

        index = int(predicted_index)

    elif predicted_class is not None:

        if isinstance(predicted_class, str):

            if predicted_class in CLASS_NAMES:
                index = CLASS_NAMES.index(predicted_class)

            else:
                try:
                    index = int(predicted_class)
                except ValueError:
                    raise ValueError(
                        f"Unknown predicted class: {predicted_class}"
                    )

        else:
            index = int(predicted_class)

    else:
        raise ValueError(
            "Either predicted_index or predicted_class must be provided."
        )

    if index < 0 or index >= NUM_CLASSES:
        raise ValueError(
            f"Class index {index} is outside valid range "
            f"0-{NUM_CLASSES - 1}."
        )

    return index


# ============================================================
# TARGET LAYER
# ============================================================

def get_target_layer(model: nn.Module) -> nn.Module:
    """
    Select a relatively low-memory convolutional layer.

    EfficientNet-B0:
        features[-2]

    ResNet:
        layer4[-1]

    VGG:
        features[-1]

    Generic fallback:
        last Conv2d layer
    """

    # --------------------------------------------------------
    # EfficientNet
    # --------------------------------------------------------

    if hasattr(model, "features"):

        features = model.features

        if len(features) >= 2:

            # Use the previous feature block rather than the
            # final 1280-channel layer.
            #
            # This substantially reduces Grad-CAM memory.
            layer = features[-2]

            print(
                f"Grad-CAM target layer: "
                f"{layer.__class__.__name__}"
            )

            return layer

        layer = features[-1]

        print(
            f"Grad-CAM target layer: "
            f"{layer.__class__.__name__}"
        )

        return layer

    # --------------------------------------------------------
    # ResNet
    # --------------------------------------------------------

    if hasattr(model, "layer4"):

        layer = model.layer4[-1]

        print(
            f"Grad-CAM target layer: "
            f"{layer.__class__.__name__}"
        )

        return layer

    # --------------------------------------------------------
    # VGG
    # --------------------------------------------------------

    if hasattr(model, "features"):

        layer = model.features[-1]

        print(
            f"Grad-CAM target layer: "
            f"{layer.__class__.__name__}"
        )

        return layer

    # --------------------------------------------------------
    # Generic Conv2d fallback
    # --------------------------------------------------------

    for layer in reversed(list(model.modules())):

        if isinstance(layer, nn.Conv2d):

            print(
                f"Grad-CAM target layer: "
                f"{layer.__class__.__name__}"
            )

            return layer

    raise RuntimeError(
        "Could not find a suitable Grad-CAM target layer."
    )


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:
    """
    Low-memory Grad-CAM implementation.

    Uses torch.autograd.grad() instead of a full backward hook.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
    ):

        self.model = model
        self.target_layer = target_layer

        self.activations = None

        self.forward_handle = target_layer.register_forward_hook(
            self._forward_hook
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output,
    ):
        self.activations = output

    def remove_hooks(self):

        if self.forward_handle is not None:

            self.forward_handle.remove()

            self.forward_handle = None

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
    ) -> np.ndarray:

        self.activations = None

        self.model.eval()

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        output = self.model(input_tensor)

        logits = extract_logits(output)

        if self.activations is None:

            raise RuntimeError(
                "Grad-CAM target layer did not produce activations."
            )

        # ----------------------------------------------------
        # Select target score
        # ----------------------------------------------------

        target_score = logits[:, target_class].sum()

        # ----------------------------------------------------
        # Gradient only for target score
        # ----------------------------------------------------

        gradients = torch.autograd.grad(
            outputs=target_score,
            inputs=self.activations,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]

        activations = self.activations

        # ----------------------------------------------------
        # Convert activation format
        # ----------------------------------------------------

        if activations.ndim != 4:

            raise RuntimeError(
                f"Expected 4D activations, got "
                f"{activations.shape}"
            )

        # CNN:
        # B, C, H, W
        #
        # Some transformer-style layers can produce:
        # B, H, W, C
        #
        # Detect the likely channel dimension.

        if activations.shape[1] <= 2048:

            # Standard CNN layout.
            activations_cf = activations

            gradients_cf = gradients

        else:

            # NHWC layout.
            activations_cf = activations.permute(
                0,
                3,
                1,
                2,
            )

            gradients_cf = gradients.permute(
                0,
                3,
                1,
                2,
            )

        # ----------------------------------------------------
        # Global average pooling of gradients
        # ----------------------------------------------------

        weights = gradients_cf.mean(
            dim=(2, 3),
            keepdim=True,
        )

        # ----------------------------------------------------
        # Weighted activation maps
        # ----------------------------------------------------

        cam = (
            weights * activations_cf
        ).sum(
            dim=1,
            keepdim=True,
        )

        cam = F.relu(cam)

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        cam = F.interpolate(
            cam,
            size=(
                GRADCAM_IMAGE_SIZE,
                GRADCAM_IMAGE_SIZE,
            ),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0]

        cam_min = cam.min()
        cam_max = cam.max()

        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min + 1e-8
        )

        result = (
            cam.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # ----------------------------------------------------
        # Release tensors immediately
        # ----------------------------------------------------

        del output
        del logits
        del target_score
        del gradients
        del activations
        del activations_cf
        del gradients_cf
        del weights
        del cam

        gc.collect()

        return result


# ============================================================
# IMAGE CONVERSION
# ============================================================

def prepare_input_image(
    image,
) -> torch.Tensor:
    """
    Accept either:

    1. PIL.Image
    2. Tensor [C,H,W]
    3. Tensor [B,C,H,W]

    and return:

        [1,3,128,128]
    """

    if isinstance(image, Image.Image):

        pil_image = image.convert("RGB")

        pil_image = pil_image.resize(
            (
                GRADCAM_IMAGE_SIZE,
                GRADCAM_IMAGE_SIZE,
            ),
            Image.Resampling.BILINEAR,
        )

        array = np.asarray(
            pil_image,
            dtype=np.float32,
        ) / 255.0

        tensor = torch.from_numpy(
            array
        ).permute(
            2,
            0,
            1,
        )

        tensor = tensor.unsqueeze(0)

    elif torch.is_tensor(image):

        tensor = image.detach()

        if tensor.ndim == 3:

            tensor = tensor.unsqueeze(0)

        if tensor.ndim != 4:

            raise ValueError(
                f"Expected image tensor with 3 or 4 dimensions, "
                f"got {tensor.shape}"
            )

        tensor = tensor.to(
            dtype=torch.float32
        )

        tensor = F.interpolate(
            tensor,
            size=(
                GRADCAM_IMAGE_SIZE,
                GRADCAM_IMAGE_SIZE,
            ),
            mode="bilinear",
            align_corners=False,
        )

    else:

        raise TypeError(
            "image must be a PIL image or torch.Tensor."
        )

    # --------------------------------------------------------
    # If image is raw [0,1], normalize using ImageNet stats.
    #
    # If API already supplied a normalized tensor, values will
    # normally be outside [0,1], so do not normalize twice.
    # --------------------------------------------------------

    minimum = float(tensor.min())
    maximum = float(tensor.max())

    if minimum >= 0.0 and maximum <= 1.0:

        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=tensor.dtype,
        ).view(
            1,
            3,
            1,
            1,
        )

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=tensor.dtype,
        ).view(
            1,
            3,
            1,
            1,
        )

        tensor = (
            tensor - mean
        ) / std

    return tensor.to(DEVICE)


# ============================================================
# HEATMAP CREATION
# ============================================================

def create_heatmap_overlay(
    original_image: Image.Image,
    cam: np.ndarray,
) -> bytes:
    """
    Create a lightweight Grad-CAM PNG using PIL/NumPy.

    No Matplotlib is used here to reduce memory usage.
    """

    original_image = original_image.convert("RGB")

    width, height = original_image.size

    # --------------------------------------------------------
    # CAM -> 0..255
    # --------------------------------------------------------

    cam = np.clip(
        cam,
        0.0,
        1.0,
    )

    cam_uint8 = (
        cam * 255.0
    ).astype(
        np.uint8
    )

    cam_image = Image.fromarray(
        cam_uint8,
        mode="L",
    )

    cam_image = cam_image.resize(
        (
            width,
            height,
        ),
        Image.Resampling.BILINEAR,
    )

    # Slight smoothing makes the tiny 128x128 CAM less noisy.
    cam_image = cam_image.filter(
        ImageFilter.GaussianBlur(
            radius=1.0
        )
    )

    cam_array = np.asarray(
        cam_image,
        dtype=np.float32,
    ) / 255.0

    # --------------------------------------------------------
    # Simple JET-like color mapping
    # --------------------------------------------------------

    red = np.clip(
        1.5 * cam_array - 0.5,
        0.0,
        1.0,
    )

    green = np.clip(
        1.5 - np.abs(
            4.0 * cam_array - 2.0
        ),
        0.0,
        1.0,
    )

    blue = np.clip(
        1.5 - 1.5 * cam_array,
        0.0,
        1.0,
    )

    heatmap = np.stack(
        [
            red,
            green,
            blue,
        ],
        axis=2,
    )

    heatmap = (
        heatmap * 255.0
    ).astype(
        np.uint8
    )

    heatmap_image = Image.fromarray(
        heatmap,
        mode="RGB",
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    original_array = np.asarray(
        original_image,
        dtype=np.float32,
    )

    heatmap_array = np.asarray(
        heatmap_image,
        dtype=np.float32,
    )

    overlay = (
        0.55 * original_array
        + 0.45 * heatmap_array
    )

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(
        np.uint8
    )

    result_image = Image.fromarray(
        overlay,
        mode="RGB",
    )

    # --------------------------------------------------------
    # Add lightweight border
    # --------------------------------------------------------

    from PIL import ImageOps

    result_image = ImageOps.expand(
        result_image,
        border=3,
        fill=(255, 255, 255),
    )

    # --------------------------------------------------------
    # Encode PNG
    # --------------------------------------------------------

    buffer = io.BytesIO()

    result_image.save(
        buffer,
        format="PNG",
        optimize=True,
    )

    result = buffer.getvalue()

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del cam
    del cam_array
    del heatmap
    del heatmap_array
    del original_array
    del overlay

    gc.collect()

    return result


# ============================================================
# PUBLIC API FUNCTION
# ============================================================

def generate_gradcam_for_api(
    model: nn.Module,
    image,
    predicted_index: Optional[int] = None,
    predicted_class: Optional[str | int] = None,
    confidence: float = 0.0,
) -> bytes:

    configure_cpu_memory()

    # --------------------------------------------------------
    # Resolve class
    # --------------------------------------------------------

    target_class = normalize_class_index(
        predicted_index=predicted_index,
        predicted_class=predicted_class,
    )

    # --------------------------------------------------------
    # Prepare original image for final overlay
    # --------------------------------------------------------

    if isinstance(image, Image.Image):

        original_image = image.convert("RGB")

    elif torch.is_tensor(image):

        tensor_for_image = image.detach().cpu()

        if tensor_for_image.ndim == 4:

            tensor_for_image = tensor_for_image[0]

        if tensor_for_image.ndim != 3:

            raise ValueError(
                "Image tensor must have shape [C,H,W] "
                "or [B,C,H,W]."
            )

        # Undo ImageNet normalization if necessary.
        tensor_for_image = tensor_for_image.clone()

        mean = torch.tensor(
            [0.485, 0.456, 0.406],
        ).view(
            3,
            1,
            1,
        )

        std = torch.tensor(
            [0.229, 0.224, 0.225],
        ).view(
            3,
            1,
            1,
        )

        tensor_for_image = (
            tensor_for_image * std
        ) + mean

        tensor_for_image = torch.clamp(
            tensor_for_image,
            0.0,
            1.0,
        )

        array = (
            tensor_for_image
            .permute(1, 2, 0)
            .numpy()
            * 255.0
        ).astype(
            np.uint8
        )

        original_image = Image.fromarray(
            array,
            mode="RGB",
        )

        del tensor_for_image
        del array

    else:

        raise TypeError(
            "image must be PIL.Image or torch.Tensor."
        )

    # --------------------------------------------------------
    # Prepare SMALL Grad-CAM input
    # --------------------------------------------------------

    input_tensor = prepare_input_image(
        image
    )

    input_tensor.requires_grad_(True)

    # --------------------------------------------------------
    # Target layer
    # --------------------------------------------------------

    target_layer = get_target_layer(
        model
    )

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    try:

        # ----------------------------------------------------
        # Generate CAM
        # ----------------------------------------------------

        cam = gradcam.generate(
            input_tensor=input_tensor,
            target_class=target_class,
        )

    finally:

        gradcam.remove_hooks()

    # --------------------------------------------------------
    # Create final PNG
    # --------------------------------------------------------

    result = create_heatmap_overlay(
        original_image=original_image,
        cam=cam,
    )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del input_tensor
    del original_image
    del cam
    del target_layer
    del gradcam

    gc.collect()

    return result


# ============================================================
# LOCAL SMOKE TEST
# ============================================================

def load_local_model() -> nn.Module:

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    print(
        f"Loading model:\n{MODEL_PATH}"
    )

    model = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if not isinstance(
        model,
        nn.Module,
    ):

        raise TypeError(
            "Deployment file does not contain "
            "a torch.nn.Module."
        )

    model.to(DEVICE)

    model.eval()

    return model


def main():

    configure_cpu_memory()

    print("=" * 60)
    print("KidneyVision Grad-CAM Local Smoke Test")
    print("=" * 60)

    model = load_local_model()

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
            f"Test image not found:\n{image_path}"
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    print(
        f"Image: {image_path}"
    )

    print(
        f"Image size: {image.size}"
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize(
                (
                    224,
                    224,
                )
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    prediction_tensor = transform(
        image
    ).unsqueeze(0)

    with torch.no_grad():

        output = model(
            prediction_tensor
        )

        logits = extract_logits(
            output
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predicted_index = int(
            probabilities.argmax(
                dim=1
            ).item()
        )

        confidence = float(
            probabilities[
                0,
                predicted_index
            ].item()
        )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    print(
        f"Prediction: {predicted_class}"
    )

    print(
        f"Confidence: {confidence:.4f}"
    )

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    output_bytes = generate_gradcam_for_api(
        model=model,
        image=image,
        predicted_index=predicted_index,
        predicted_class=predicted_class,
        confidence=confidence,
    )

    output_path = (
        PROJECT_ROOT
        / "artifacts"
        / "explainability"
        / "gradcam"
        / "gradcam_efficientnet_low_memory.png"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_bytes(
        output_bytes
    )

    print(
        f"Grad-CAM saved to:\n{output_path}"
    )

    print(
        f"Output size: {len(output_bytes):,} bytes"
    )

    print("=" * 60)
    print("GRAD-CAM SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()