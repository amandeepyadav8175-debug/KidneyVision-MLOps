from __future__ import annotations

import gc
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================

CLASS_NAMES = [
    "Normal",
    "Cyst",
    "Stone",
    "Tumor",
]

# Lightweight visualization for CPU / Render Free.
IMAGE_SIZE = 128

# EfficientNet-B0:
# Use the final convolutional feature block before
# AdaptiveAvgPool + Linear classifier.
TARGET_LAYER_INDEX = -1


# ============================================================
# CPU CONFIGURATION
# ============================================================

def configure_cpu() -> None:
    """
    Keep Grad-CAM lightweight on CPU.
    """

    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


# ============================================================
# PREPROCESSING
# ============================================================

def build_preprocess(
    image_size: int = IMAGE_SIZE,
):
    """
    Inference preprocessing used by EfficientNet-B0.
    """

    return transforms.Compose(
        [
            transforms.Resize(
                (
                    image_size,
                    image_size,
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


# ============================================================
# TARGET LAYER
# ============================================================

def find_target_layer(model):
    """
    Find the final convolutional feature block of
    EfficientNet-B0.

    features[-1] is used instead of the classifier.
    """

    try:
        target_layer = model.features[
            TARGET_LAYER_INDEX
        ]

    except (
        AttributeError,
        IndexError,
        TypeError,
    ) as exc:

        raise RuntimeError(
            "Could not locate EfficientNet-B0 "
            "Grad-CAM target layer."
        ) from exc

    print(
        "Grad-CAM target layer: "
        f"{target_layer.__class__.__name__}"
    )

    return target_layer


# ============================================================
# GRAD-CAM
# ============================================================

class EfficientNetGradCAM:

    def __init__(
        self,
        model,
        target_layer,
    ):

        self.model = model
        self.target_layer = target_layer

        self.activation = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self._save_activation
            )
        )

    def _save_activation(
        self,
        module,
        inputs,
        output,
    ):

        self.activation = output

    def remove(self):

        if self.forward_handle is not None:

            self.forward_handle.remove()

            self.forward_handle = None

    def generate(
        self,
        input_tensor,
        class_index: int,
    ):
        """
        Generate a Grad-CAM map for one class.
        """

        self.activation = None

        self.model.zero_grad(
            set_to_none=True
        )

        # Grad-CAM requires gradients.
        with torch.enable_grad():

            output = self.model(
                input_tensor
            )

            # Some torchvision models may return
            # an object containing .logits.
            if hasattr(
                output,
                "logits",
            ):

                output = output.logits

            if output.ndim != 2:

                raise RuntimeError(
                    "Unexpected model output shape: "
                    f"{tuple(output.shape)}"
                )

            # Make sure the forward hook captured
            # the target feature map.
            if self.activation is None:

                raise RuntimeError(
                    "Grad-CAM activation was not captured."
                )

            target_score = output[
                0,
                class_index,
            ]

            # ------------------------------------------------
            # Gradient of target class score with respect
            # to the target feature map.
            # ------------------------------------------------

            gradients = torch.autograd.grad(
                outputs=target_score,
                inputs=self.activation,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]

            activation = self.activation

            # Expected EfficientNet feature shape:
            # [batch, channels, height, width]
            if activation.ndim != 4:

                raise RuntimeError(
                    "Unexpected EfficientNet activation shape: "
                    f"{tuple(activation.shape)}"
                )

            # ------------------------------------------------
            # Standard Grad-CAM
            # ------------------------------------------------

            weights = gradients.mean(
                dim=(2, 3),
                keepdim=True,
            )

            raw_cam = (
                activation
                * weights
            ).sum(
                dim=1,
                keepdim=True,
            )

            # Standard Grad-CAM keeps positive influence.
            cam = F.relu(
                raw_cam
            )

            # ------------------------------------------------
            # Robust fallback
            # ------------------------------------------------
            #
            # On some images/models the positive Grad-CAM
            # response can become almost zero.
            #
            # Instead of showing a completely blank blue
            # image, use the magnitude of the weighted
            # activation as a visualization fallback.
            # ------------------------------------------------

            cam_max_value = float(
                cam.max()
                .detach()
                .cpu()
            )

            if cam_max_value <= 1e-8:

                print(
                    "Grad-CAM positive response was "
                    "near zero; using magnitude fallback."
                )

                cam = raw_cam.abs()

            # ------------------------------------------------
            # Resize CAM
            # ------------------------------------------------

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
                0,
            ]

            # ------------------------------------------------
            # Normalize 0 -> 1
            # ------------------------------------------------

            cam_min = cam.min()
            cam_max = cam.max()

            range_value = float(
                (
                    cam_max
                    - cam_min
                )
                .detach()
                .cpu()
            )

            if range_value > 1e-8:

                cam = (
                    cam
                    - cam_min
                ) / (
                    cam_max
                    - cam_min
                )

            else:

                # Completely constant CAM.
                # This should be extremely unusual.
                cam = torch.zeros_like(
                    cam
                )

            return (
                output.detach(),
                cam.detach(),
            )


# ============================================================
# JET COLORMAP
# ============================================================

def jet_colormap(
    values: np.ndarray,
) -> np.ndarray:
    """
    Lightweight NumPy-only Jet colormap.

    Avoids Matplotlib dependency inside FastAPI.
    """

    values = np.clip(
        values,
        0.0,
        1.0,
    ).astype(
        np.float32
    )

    red = np.clip(
        1.5
        - np.abs(
            4.0 * values
            - 3.0
        ),
        0.0,
        1.0,
    )

    green = np.clip(
        1.5
        - np.abs(
            4.0 * values
            - 2.0
        ),
        0.0,
        1.0,
    )

    blue = np.clip(
        1.5
        - np.abs(
            4.0 * values
            - 1.0
        ),
        0.0,
        1.0,
    )

    rgb = np.stack(
        [
            red,
            green,
            blue,
        ],
        axis=-1,
    )

    return np.uint8(
        np.clip(
            rgb * 255.0,
            0,
            255,
        )
    )


# ============================================================
# FONT
# ============================================================

def get_font(
    size: int,
):
    """
    Use a system font when available.
    """

    font_candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]

    for font_path in font_candidates:

        try:

            return ImageFont.truetype(
                font_path,
                size=size,
            )

        except OSError:

            continue

    return ImageFont.load_default()


# ============================================================
# 3-PANEL VISUALIZATION
# ============================================================

def create_three_panel_visualization(
    image: Image.Image,
    cam: torch.Tensor,
    predicted_class: str,
    confidence: float,
) -> bytes:
    """
    Create:

    Original CT Image | Grad-CAM | Prediction Overlay
    """

    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    original = (
        image
        .convert("RGB")
        .resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            Image.Resampling.BILINEAR,
        )
    )

    original_array = (
        np.asarray(
            original
        )
        .astype(
            np.float32
        )
    )

    # --------------------------------------------------------
    # CAM
    # --------------------------------------------------------

    cam_array = (
        cam
        .detach()
        .cpu()
        .numpy()
    )

    cam_array = np.clip(
        cam_array,
        0.0,
        1.0,
    )

    heatmap_array = jet_colormap(
        cam_array
    )

    heatmap = Image.fromarray(
        heatmap_array,
        mode="RGB",
    )

    # --------------------------------------------------------
    # Prediction overlay
    # --------------------------------------------------------

    heat_float = (
        heatmap_array
        .astype(
            np.float32
        )
    )

    alpha = (
        0.45
        * cam_array[
            ...,
            None,
        ]
    )

    overlay_array = (
        original_array
        * (1.0 - alpha)
        + heat_float
        * alpha
    )

    overlay_array = np.uint8(
        np.clip(
            overlay_array,
            0,
            255,
        )
    )

    overlay = Image.fromarray(
        overlay_array,
        mode="RGB",
    )

    # --------------------------------------------------------
    # Figure dimensions
    # --------------------------------------------------------

    panel_size = IMAGE_SIZE

    horizontal_gap = 28

    top_area = 55

    bottom_area = 34

    total_width = (
        panel_size * 3
        + horizontal_gap * 2
    )

    total_height = (
        top_area
        + panel_size
        + bottom_area
    )

    canvas = Image.new(
        "RGB",
        (
            total_width,
            total_height,
        ),
        "white",
    )

    # --------------------------------------------------------
    # Fonts
    # --------------------------------------------------------

    title_font = get_font(
        15
    )

    confidence_font = get_font(
        13
    )

    footer_font = get_font(
        11
    )

    # --------------------------------------------------------
    # Panel positions
    # --------------------------------------------------------

    x1 = 0

    x2 = (
        panel_size
        + horizontal_gap
    )

    x3 = (
        panel_size * 2
        + horizontal_gap * 2
    )

    y = top_area

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    canvas.paste(
        original,
        (
            x1,
            y,
        ),
    )

    canvas.paste(
        heatmap,
        (
            x2,
            y,
        ),
    )

    canvas.paste(
        overlay,
        (
            x3,
            y,
        ),
    )

    draw = ImageDraw.Draw(
        canvas
    )

    # --------------------------------------------------------
    # Centered text helper
    # --------------------------------------------------------

    def centered_text(
        text,
        center_x,
        y_position,
        font,
    ):

        bbox = draw.textbbox(
            (
                0,
                0,
            ),
            text,
            font=font,
        )

        text_width = (
            bbox[2]
            - bbox[0]
        )

        draw.text(
            (
                center_x
                - text_width / 2,
                y_position,
            ),
            text,
            fill="black",
            font=font,
        )

    # --------------------------------------------------------
    # Titles
    # --------------------------------------------------------

    centered_text(
        "Original CT Image",
        x1
        + panel_size / 2,
        10,
        title_font,
    )

    centered_text(
        "Grad-CAM",
        x2
        + panel_size / 2,
        10,
        title_font,
    )

    centered_text(
        f"Prediction: {predicted_class}",
        x3
        + panel_size / 2,
        5,
        title_font,
    )

    centered_text(
        f"Confidence: {confidence:.2%}",
        x3
        + panel_size / 2,
        24,
        confidence_font,
    )

    # --------------------------------------------------------
    # Footer
    # --------------------------------------------------------

    centered_text(
        "Grad-CAM Explanation",
        total_width / 2,
        top_area
        + panel_size
        + 10,
        footer_font,
    )

    # --------------------------------------------------------
    # PNG
    # --------------------------------------------------------

    buffer = io.BytesIO()

    canvas.save(
        buffer,
        format="PNG",
        optimize=True,
    )

    return buffer.getvalue()


# ============================================================
# API FUNCTION
# ============================================================

def generate_gradcam_for_api(
    model,
    image,
    predicted_index,
    predicted_class,
    confidence,
    device=None,
):
    """
    Generate the complete three-panel Grad-CAM PNG.

    The model is already loaded by FastAPI.
    """

    configure_cpu()

    image = image.convert(
        "RGB"
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if device is None:

        try:

            device = next(
                model.parameters()
            ).device

        except StopIteration:

            device = torch.device(
                "cpu"
            )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    preprocess = build_preprocess(
        IMAGE_SIZE
    )

    input_tensor = (
        preprocess(
            image
        )
        .unsqueeze(
            0
        )
        .to(
            device
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model.eval()

    target_layer = find_target_layer(
        model
    )

    gradcam = EfficientNetGradCAM(
        model,
        target_layer,
    )

    try:

        print(
            "Grad-CAM prediction: "
            f"{predicted_class} | "
            f"Confidence: "
            f"{confidence:.4f}"
        )

        _, cam = gradcam.generate(
            input_tensor,
            class_index=int(
                predicted_index
            ),
        )

        # ----------------------------------------------------
        # Important debugging information
        # ----------------------------------------------------

        cam_min = float(
            cam.min()
            .detach()
            .cpu()
        )

        cam_max = float(
            cam.max()
            .detach()
            .cpu()
        )

        cam_mean = float(
            cam.mean()
            .detach()
            .cpu()
        )

        print(
            "Grad-CAM map range: "
            f"min={cam_min:.6f}, "
            f"max={cam_max:.6f}, "
            f"mean={cam_mean:.6f}"
        )

        result = (
            create_three_panel_visualization(
                image=image,
                cam=cam,
                predicted_class=predicted_class,
                confidence=float(
                    confidence
                ),
            )
        )

        print(
            "Grad-CAM generated successfully."
        )

        return result

    finally:

        gradcam.remove()

        gradcam.activation = None

        gc.collect()


# ============================================================
# LOCAL SMOKE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Grad-CAM render module loaded successfully."
    )

    print(
        "Target layer index:",
        TARGET_LAYER_INDEX,
    )

    print(
        "Visualization size:",
        IMAGE_SIZE,
    )