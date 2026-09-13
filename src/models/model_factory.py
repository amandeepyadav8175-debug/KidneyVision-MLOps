"""
KidneyVision-MLOps
Model Factory

Creates all benchmark models used in the project.

Paper models:
    - VGG16
    - ResNet50
    - InceptionV3
    - EANet
    - CCT
    - Swin Transformer

Additional benchmark:
    - EfficientNet-B0

Important:
EANet and CCT are implemented as PyTorch classification
architectures based on the architectures described in the
kidney CT research paper.

EfficientNet-B0 is our additional benchmark model and is
NOT part of the original six-model paper benchmark.
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn
from torchvision import models


# ============================================================
# PROJECT IMPORT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from configs.training_config import (
    CLASS_NAMES,
    NUM_CLASSES,
    DEFAULT_IMAGE_SIZE,
    INCEPTION_IMAGE_SIZE,
    USE_PRETRAINED_WEIGHTS,
)


# ============================================================
# EANET - EXTERNAL ATTENTION
# ============================================================

class ExternalAttention(nn.Module):
    """
    External Attention block.

    Instead of computing attention between every pair of
    input tokens, the input interacts with a small learned
    external memory.

    This follows the external-attention concept described
    for EANet in the kidney CT paper.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        memory_size: int = 64,
    ):
        super().__init__()

        self.memory_size = memory_size

        self.to_memory = nn.Linear(
            embed_dim,
            memory_size,
            bias=False,
        )

        self.from_memory = nn.Linear(
            memory_size,
            embed_dim,
            bias=False,
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x -> [batch, tokens, embed_dim]

        Output:
            same shape
        """

        residual = x

        # Input -> external memory attention scores
        attention = self.to_memory(x)

        # Normalize attention across memory dimension
        attention = torch.softmax(
            attention,
            dim=-1,
        )

        # Normalize across token dimension
        attention = attention / (
            attention.sum(
                dim=1,
                keepdim=True,
            )
            + 1e-6
        )

        # External memory -> feature space
        output = self.from_memory(attention)

        output = self.norm(
            output + residual
        )

        return output


class EANetClassifier(nn.Module):
    """
    External-Attention Transformer classifier.

    Designed for the four-class kidney CT classification task.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        image_size: int = DEFAULT_IMAGE_SIZE,
        embed_dim: int = 64,
        memory_size: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()

        self.image_size = image_size

        # Patch extraction
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=16,
            stride=16,
        )

        num_patches = (
            image_size // 16
        ) ** 2

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                num_patches,
                embed_dim,
            )
        )

        # External attention
        self.external_attention = ExternalAttention(
            embed_dim=embed_dim,
            memory_size=memory_size,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(
                embed_dim,
                num_classes,
            ),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.trunc_normal_(
            self.position_embedding,
            std=0.02,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # [B, 3, H, W]
        x = self.patch_embedding(x)

        # [B, C, H', W']
        x = x.flatten(2)

        # [B, C, tokens]
        x = x.transpose(1, 2)

        # Add positional information
        x = x + self.position_embedding

        # External attention
        x = self.external_attention(x)

        # Transformer encoder
        x = self.transformer(x)

        # Sequence pooling
        x = x.mean(dim=1)

        # Classification
        return self.classifier(x)


# ============================================================
# CCT - COMPACT CONVOLUTIONAL TRANSFORMER
# ============================================================

class SequencePooling(nn.Module):
    """
    Attention-based sequence pooling used by CCT-style models.
    """

    def __init__(self, embed_dim: int):
        super().__init__()

        self.attention = nn.Linear(
            embed_dim,
            1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # x = [B, tokens, embed_dim]

        attention = self.attention(x)

        attention = torch.softmax(
            attention,
            dim=1,
        )

        pooled = torch.sum(
            x * attention,
            dim=1,
        )

        return pooled


class CCTClassifier(nn.Module):
    """
    Compact Convolutional Transformer classifier.

    Architecture:
        image
          ↓
        convolutional tokenization
          ↓
        transformer encoder
          ↓
        sequence pooling
          ↓
        classifier
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()

        # Convolutional tokenization
        self.tokenizer = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(
                32,
                embed_dim,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
            nn.BatchNorm2d(embed_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.sequence_pooling = SequencePooling(
            embed_dim
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(
                embed_dim,
                num_classes,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # Convolutional tokenization
        x = self.tokenizer(x)

        # [B, C, H, W] -> [B, tokens, C]
        x = x.flatten(2)

        x = x.transpose(1, 2)

        # Transformer
        x = self.transformer(x)

        # Sequence pooling
        x = self.sequence_pooling(x)

        # Classification
        return self.classifier(x)


# ============================================================
# CLASSIFIER REPLACEMENT HELPERS
# ============================================================

def build_vgg16(
    num_classes: int,
    pretrained: bool,
) -> nn.Module:

    weights = (
        models.VGG16_Weights.DEFAULT
        if pretrained
        else None
    )

    model = models.vgg16(
        weights=weights
    )

    input_features = model.classifier[6].in_features

    model.classifier[6] = nn.Linear(
        input_features,
        num_classes,
    )

    return model


def build_resnet50(
    num_classes: int,
    pretrained: bool,
) -> nn.Module:

    weights = (
        models.ResNet50_Weights.DEFAULT
        if pretrained
        else None
    )

    model = models.resnet50(
        weights=weights
    )

    input_features = model.fc.in_features

    model.fc = nn.Linear(
        input_features,
        num_classes,
    )

    return model


def build_inception_v3(
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build InceptionV3 for 4-class kidney disease classification.

    InceptionV3 requires aux_logits=True when pretrained
    torchvision weights are requested.
    """

    weights = (
        models.Inception_V3_Weights.DEFAULT
        if pretrained
        else None
    )

    model = models.inception_v3(
        weights=weights,
        aux_logits=True,
    )

    # Replace the main classifier
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    # Replace the auxiliary classifier
    if model.AuxLogits is not None:
        model.AuxLogits.fc = nn.Linear(
            model.AuxLogits.fc.in_features,
            num_classes,
        )

    return model


def build_efficientnet_b0(
    num_classes: int,
    pretrained: bool,
) -> nn.Module:

    weights = (
        models.EfficientNet_B0_Weights.DEFAULT
        if pretrained
        else None
    )

    model = models.efficientnet_b0(
        weights=weights
    )

    input_features = (
        model.classifier[1].in_features
    )

    model.classifier[1] = nn.Linear(
        input_features,
        num_classes,
    )

    return model


def build_swin_transformer(
    num_classes: int,
    pretrained: bool,
) -> nn.Module:

    weights = (
        models.Swin_T_Weights.DEFAULT
        if pretrained
        else None
    )

    model = models.swin_t(
        weights=weights
    )

    input_features = model.head.in_features

    model.head = nn.Linear(
        input_features,
        num_classes,
    )

    return model


# ============================================================
# MAIN MODEL FACTORY
# ============================================================

def create_model(
    model_name: str,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = USE_PRETRAINED_WEIGHTS,
) -> nn.Module:
    """
    Create a model by name.

    Parameters
    ----------
    model_name:
        Name of the model.

    num_classes:
        Number of output classes.

    pretrained:
        Whether to use ImageNet pretrained weights.

    Returns
    -------
    torch.nn.Module
    """

    normalized_name = (
        model_name
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )

    if normalized_name == "vgg16":

        return build_vgg16(
            num_classes,
            pretrained,
        )

    if normalized_name == "resnet50":

        return build_resnet50(
            num_classes,
            pretrained,
        )

    if normalized_name == "inceptionv3":

        return build_inception_v3(
            num_classes,
            pretrained,
        )

    if normalized_name == "efficientnetb0":

        return build_efficientnet_b0(
            num_classes,
            pretrained,
        )

    if normalized_name == "eanet":

        return EANetClassifier(
            num_classes=num_classes,
            image_size=DEFAULT_IMAGE_SIZE,
        )

    if normalized_name == "cct":

        return CCTClassifier(
            num_classes=num_classes,
        )

    if normalized_name in {
        "swintransformer",
        "swint",
        "swintiny",
    }:

        return build_swin_transformer(
            num_classes,
            pretrained,
        )

    raise ValueError(
        f"Unknown model: {model_name}\n"
        f"Supported models:\n"
        f"  - VGG16\n"
        f"  - ResNet50\n"
        f"  - InceptionV3\n"
        f"  - EfficientNet-B0\n"
        f"  - EANet\n"
        f"  - CCT\n"
        f"  - SwinTransformer"
    )


# ============================================================
# MODEL INPUT SIZE
# ============================================================

def get_model_image_size(
    model_name: str,
) -> int:
    """
    Return the appropriate image size for a model.
    """

    normalized_name = (
        model_name
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )

    if normalized_name == "inceptionv3":
        return INCEPTION_IMAGE_SIZE

    return DEFAULT_IMAGE_SIZE


# ============================================================
# PARAMETER COUNT
# ============================================================

def count_parameters(
    model: nn.Module,
) -> tuple[int, int]:
    """
    Return:
        total parameters
        trainable parameters
    """

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


# ============================================================
# SMOKE TEST
# ============================================================

def run_smoke_test() -> None:

    print("=" * 70)
    print("STEP 6 - MODEL FACTORY TEST")
    print("=" * 70)

    device = torch.device("cpu")

    model_names = [
        "VGG16",
        "ResNet50",
        "InceptionV3",
        "EfficientNet-B0",
        "EANet",
        "CCT",
        "SwinTransformer",
    ]

    for model_name in model_names:

        print("\n" + "-" * 70)
        print(f"Testing: {model_name}")

        image_size = get_model_image_size(
            model_name
        )

        print(f"Input size: {image_size}x{image_size}")

        # IMPORTANT:
        # Smoke test does NOT download pretrained weights.
        model = create_model(
            model_name=model_name,
            num_classes=NUM_CLASSES,
            pretrained=False,
        )

        model = model.to(device)
        model.eval()

        total_params, trainable_params = count_parameters(
            model
        )

        dummy_input = torch.randn(
            1,
            3,
            image_size,
            image_size,
            device=device,
        )

        with torch.no_grad():

            output = model(dummy_input)

        print(
            f"Total parameters    : {total_params:,}"
        )

        print(
            f"Trainable parameters: {trainable_params:,}"
        )

        print(
            f"Output shape        : {tuple(output.shape)}"
        )

        if output.shape != (1, NUM_CLASSES):

            raise RuntimeError(
                f"Unexpected output shape for "
                f"{model_name}: {output.shape}"
            )

        print("STATUS              : PASS")

        del model

    print("\n" + "=" * 70)
    print("ALL MODEL FACTORY TESTS PASSED")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_smoke_test()