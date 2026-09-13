"""
KidneyVision-MLOps
Image preprocessing and PyTorch dataset pipeline.

Responsibilities:
1. Define training augmentation.
2. Define deterministic validation/test preprocessing.
3. Load images from prepared manifests.
4. Convert images to RGB tensors.
5. Apply ImageNet normalization for pretrained models.
6. Create PyTorch DataLoaders.
7. Provide a preprocessing smoke test.

Important:
- Raw images are NEVER modified.
- Images are transformed on-the-fly.
- Training augmentation is applied ONLY to the training set.
- Validation and test sets use deterministic preprocessing.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_DIR = PROJECT_ROOT / "data" / "processed" / "manifests"


# ============================================================
# CONSTANTS
# ============================================================

CLASS_NAMES = [
    "Normal",
    "Cyst",
    "Stone",
    "Tumor",
]

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}

INDEX_TO_CLASS = {
    index: class_name
    for class_name, index in CLASS_TO_INDEX.items()
}


# ImageNet statistics.
# These are used because our benchmark contains
# pretrained ImageNet models.
IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


# ============================================================
# TRANSFORMS
# ============================================================

def build_transforms(
    image_size: int = 224,
    train: bool = False,
) -> transforms.Compose:
    """
    Build preprocessing pipeline.

    Training:
        - Resize
        - Small rotation
        - Small translation
        - Small scale variation
        - Convert to tensor
        - ImageNet normalization

    Validation/Test:
        - Resize
        - Convert to tensor
        - ImageNet normalization

    We intentionally avoid aggressive color augmentation because
    CT image intensity information can be important.
    """

    if train:
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size)
                ),

                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=IMAGENET_MEAN,
                    std=IMAGENET_STD,
                ),
            ]
        )

    else:
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size)
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=IMAGENET_MEAN,
                    std=IMAGENET_STD,
                ),
            ]
        )

    return transform


# ============================================================
# DATASET
# ============================================================

class KidneyCTDataset(Dataset):
    """
    PyTorch Dataset for kidney CT images.

    Expected manifest information:
        - image_id
        - image_path/path
        - label/Class

    The implementation supports both the names used by our
    prepared manifests and the original CSV naming.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        image_size: int = 224,
        train: bool = False,
    ):
        self.manifest_path = Path(manifest_path)

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found:\n{self.manifest_path}"
            )

        self.data = pd.read_csv(self.manifest_path)

        if self.data.empty:
            raise ValueError(
                f"Manifest is empty:\n{self.manifest_path}"
            )

        self.image_path_column = self._find_column(
            [
                "image_path",
                "path",
                "filepath",
                "file_path",
            ]
        )

        self.label_column = self._find_column(
            [
                "label",
                "Class",
                "class",
                "diagnosis",
                "diag",
            ]
        )

        self.image_id_column = self._find_optional_column(
            [
                "image_id",
                "id",
            ]
        )

        self.transform = build_transforms(
            image_size=image_size,
            train=train,
        )

        self._validate_manifest()

    # --------------------------------------------------------
    # COLUMN HELPERS
    # --------------------------------------------------------

    def _find_column(self, candidates: list[str]) -> str:
        """
        Find the first available column from candidates.
        """

        for column in candidates:
            if column in self.data.columns:
                return column

        raise ValueError(
            f"Could not find required column.\n"
            f"Available columns: {list(self.data.columns)}\n"
            f"Expected one of: {candidates}"
        )

    def _find_optional_column(
        self,
        candidates: list[str],
    ) -> str | None:
        """
        Find an optional column.
        """

        for column in candidates:
            if column in self.data.columns:
                return column

        return None

    # --------------------------------------------------------
    # MANIFEST VALIDATION
    # --------------------------------------------------------

    def _validate_manifest(self) -> None:
        """
        Validate image paths and labels before training.
        """

        missing_images = []
        invalid_labels = []

        for _, row in self.data.iterrows():

            image_path = self._resolve_image_path(
                row[self.image_path_column]
            )

            if not image_path.exists():
                missing_images.append(str(image_path))

            label = str(row[self.label_column]).strip()

            if label not in CLASS_TO_INDEX:
                invalid_labels.append(label)

        if missing_images:
            preview = "\n".join(
                missing_images[:10]
            )

            raise FileNotFoundError(
                f"{len(missing_images)} image files "
                f"were not found.\n"
                f"First files:\n{preview}"
            )

        if invalid_labels:
            unique_invalid = sorted(
                set(invalid_labels)
            )

            raise ValueError(
                f"Invalid labels found: {unique_invalid}\n"
                f"Expected labels: {CLASS_NAMES}"
            )

    # --------------------------------------------------------
    # PATH RESOLUTION
    # --------------------------------------------------------

    def _resolve_image_path(
        self,
        image_path_value,
    ) -> Path:
        """
        Convert manifest path into an actual local path.

        Prepared manifests use project-relative paths.
        Absolute paths are also supported.
        """

        raw_path = Path(
            str(image_path_value)
        )

        if raw_path.is_absolute():
            return raw_path

        return PROJECT_ROOT / raw_path

    # --------------------------------------------------------
    # DATASET METHODS
    # --------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, int]:
        """
        Return:

            image_tensor
            label_index
        """

        row = self.data.iloc[index]

        image_path = self._resolve_image_path(
            row[self.image_path_column]
        )

        label_name = str(
            row[self.label_column]
        ).strip()

        label_index = CLASS_TO_INDEX[
            label_name
        ]

        try:
            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as error:
            raise RuntimeError(
                f"Failed to load image:\n"
                f"{image_path}\n"
                f"Error: {error}"
            ) from error

        image_tensor = self.transform(
            image
        )

        return image_tensor, label_index


# ============================================================
# DATALOADERS
# ============================================================

def create_dataloaders(
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation and test DataLoaders.

    Default num_workers=0 is intentionally used because it is
    reliable on Windows. It can be increased later when needed.
    """

    train_manifest = (
        MANIFEST_DIR / "train.csv"
    )

    validation_manifest = (
        MANIFEST_DIR / "validation.csv"
    )

    test_manifest = (
        MANIFEST_DIR / "test.csv"
    )

    train_dataset = KidneyCTDataset(
        manifest_path=train_manifest,
        image_size=image_size,
        train=True,
    )

    validation_dataset = KidneyCTDataset(
        manifest_path=validation_manifest,
        image_size=image_size,
        train=False,
    )

    test_dataset = KidneyCTDataset(
        manifest_path=test_manifest,
        image_size=image_size,
        train=False,
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
    )


# ============================================================
# SMOKE TEST
# ============================================================

def run_smoke_test() -> None:
    """
    Verify that:

    1. Manifests can be loaded.
    2. Images can be opened.
    3. Transformations work.
    4. DataLoaders produce correct tensors.
    5. Labels are valid.
    """

    print("=" * 70)
    print("STEP 4 - IMAGE PREPROCESSING PIPELINE TEST")
    print("=" * 70)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Manifest dir : {MANIFEST_DIR}")

    print("\nCreating DataLoaders...")

    train_loader, validation_loader, test_loader = (
        create_dataloaders(
            image_size=224,
            batch_size=16,
            num_workers=0,
        )
    )

    print("\nDataset sizes:")
    print(
        f"Train      : "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Validation : "
        f"{len(validation_loader.dataset)}"
    )

    print(
        f"Test       : "
        f"{len(test_loader.dataset)}"
    )

    print("\nLoading one training batch...")

    images, labels = next(
        iter(train_loader)
    )

    print(
        f"Image tensor shape : "
        f"{tuple(images.shape)}"
    )

    print(
        f"Label tensor shape : "
        f"{tuple(labels.shape)}"
    )

    print(
        f"Image tensor dtype : "
        f"{images.dtype}"
    )

    print(
        f"Label tensor dtype : "
        f"{labels.dtype}"
    )

    print(
        f"Image tensor min   : "
        f"{images.min().item():.4f}"
    )

    print(
        f"Image tensor max   : "
        f"{images.max().item():.4f}"
    )

    print(
        f"Labels in batch    : "
        f"{labels.tolist()}"
    )

    print("\nClass mapping:")

    for index, class_name in INDEX_TO_CLASS.items():
        print(
            f"  {index} -> {class_name}"
        )

    print("\nExpected image shape:")
    print(
        "(batch_size, 3, 224, 224)"
    )

    print("\nPREPROCESSING PIPELINE TEST PASSED")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_smoke_test()