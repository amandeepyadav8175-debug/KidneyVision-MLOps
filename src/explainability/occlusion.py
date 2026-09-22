from __future__ import annotations

import io
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms


CLASS_NAMES = ["Normal", "Cyst", "Stone", "Tumor"]
IMAGE_SIZE = 224


def build_preprocess(image_size: int = IMAGE_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _predict(
    model: torch.nn.Module,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Run inference without constructing a gradient graph.
    """
    with torch.inference_mode():
        output = model(tensor)

        if hasattr(output, "logits"):
            output = output.logits

        probabilities = torch.softmax(output, dim=1)

    return probabilities


def _get_prediction(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    image_size: int = IMAGE_SIZE,
) -> Tuple[int, float, torch.Tensor]:
    preprocess = build_preprocess(image_size)

    tensor = preprocess(image).unsqueeze(0).to(device)

    probabilities = _predict(model, tensor)

    confidence, predicted_index = torch.max(probabilities[0], dim=0)

    return (
        int(predicted_index.item()),
        float(confidence.item()),
        probabilities[0].detach().cpu(),
    )


def _occlusion_heatmap(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    predicted_index: int,
    image_size: int = IMAGE_SIZE,
    grid_size: int = 4,
    patch_size: int = 56,
) -> np.ndarray:
    """
    Gradient-free occlusion sensitivity.

    The image is divided into a small number of regions.
    Each region is covered with a neutral patch and the change
    in the predicted-class probability is measured.

    Larger probability drops indicate regions that were more
    important for the prediction.
    """

    preprocess = build_preprocess(image_size)

    resized = image.convert("RGB").resize(
        (image_size, image_size),
        Image.Resampling.BILINEAR,
    )

    base_tensor = preprocess(resized).unsqueeze(0).to(device)

    with torch.inference_mode():
        base_output = model(base_tensor)

        if hasattr(base_output, "logits"):
            base_output = base_output.logits

        base_probability = torch.softmax(
            base_output,
            dim=1,
        )[0, predicted_index].item()

    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)

    stride = max(1, image_size // grid_size)

    for row in range(grid_size):
        for col in range(grid_size):

            x1 = col * stride
            y1 = row * stride

            x2 = min(image_size, x1 + patch_size)
            y2 = min(image_size, y1 + patch_size)

            occluded = resized.copy()

            draw = ImageDraw.Draw(occluded)

            # Neutral gray patch.
            draw.rectangle(
                [x1, y1, x2, y2],
                fill=(128, 128, 128),
            )

            occluded_tensor = preprocess(occluded).unsqueeze(0).to(device)

            with torch.inference_mode():
                output = model(occluded_tensor)

                if hasattr(output, "logits"):
                    output = output.logits

                probability = torch.softmax(
                    output,
                    dim=1,
                )[0, predicted_index].item()

            # Positive value = prediction confidence decreased
            # after hiding this region.
            drop = max(0.0, base_probability - probability)

            heatmap[row, col] = drop

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Upscale the small grid into a smooth image.
    heatmap_image = Image.fromarray(
        np.uint8(heatmap * 255),
        mode="L",
    ).resize(
        (image_size, image_size),
        Image.Resampling.BICUBIC,
    )

    return np.asarray(heatmap_image).astype(np.float32) / 255.0


def _create_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    predicted_class: str,
    confidence: float,
) -> bytes:
    """
    Create a lightweight PIL-based explanation image.

    No matplotlib is used so the API does not need to allocate
    a large plotting backend on low-memory deployments.
    """

    original = image.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )

    original_array = np.asarray(original).astype(np.float32)

    # Simple red/yellow heat visualization.
    red = np.full_like(original_array, 255.0)

    heat = heatmap[..., None]

    overlay = original_array * (1.0 - 0.45 * heat) + red * (
        0.45 * heat
    )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    result = Image.fromarray(overlay, mode="RGB")

    # Small label strip.
    canvas = Image.new(
        "RGB",
        (IMAGE_SIZE, IMAGE_SIZE + 34),
        "black",
    )

    canvas.paste(result, (0, 34))

    draw = ImageDraw.Draw(canvas)

    label = (
        f"{predicted_class} | "
        f"{confidence * 100:.2f}% | "
        f"Occlusion explanation"
    )

    draw.text(
        (6, 9),
        label,
        fill="white",
    )

    buffer = io.BytesIO()

    canvas.save(
        buffer,
        format="PNG",
        optimize=True,
    )

    return buffer.getvalue()


def generate_occlusion_for_api(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    predicted_index: int,
    predicted_class: str,
    confidence: float,
) -> bytes:
    """
    Production-safe explainability method.

    Unlike Grad-CAM, this method does not calculate gradients.
    That substantially reduces memory usage on low-memory
    CPU deployments such as Render Free.
    """

    heatmap = _occlusion_heatmap(
        model=model,
        image=image,
        device=device,
        predicted_index=predicted_index,
        image_size=IMAGE_SIZE,
        grid_size=4,
        patch_size=56,
    )

    return _create_overlay(
        image=image,
        heatmap=heatmap,
        predicted_class=predicted_class,
        confidence=confidence,
    )