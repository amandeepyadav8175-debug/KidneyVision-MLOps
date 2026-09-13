import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.training_config import CLASS_NAMES
from src.preprocessing.image_preprocessor import (
    KidneyCTDataset,
    create_dataloaders,
)


MANIFEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "manifests"
)

TRAIN_MANIFEST = MANIFEST_DIR / "train.csv"
VAL_MANIFEST = MANIFEST_DIR / "validation.csv"
TEST_MANIFEST = MANIFEST_DIR / "test.csv"


def test_manifest_files_exist():
    assert TRAIN_MANIFEST.exists()
    assert VAL_MANIFEST.exists()
    assert TEST_MANIFEST.exists()


def test_manifest_sizes():
    train_df = pd.read_csv(TRAIN_MANIFEST)
    val_df = pd.read_csv(VAL_MANIFEST)
    test_df = pd.read_csv(TEST_MANIFEST)

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15

    assert len(train_df) + len(val_df) + len(test_df) == 100


def test_manifest_columns():

    required_columns = {
        "image_id",
        "image_path",
        "label",
    }

    for manifest_path in [
        TRAIN_MANIFEST,
        VAL_MANIFEST,
        TEST_MANIFEST,
    ]:

        df = pd.read_csv(manifest_path)

        assert required_columns.issubset(
            set(df.columns)
        )


def test_class_labels_are_valid():

    valid_classes = set(CLASS_NAMES)

    for manifest_path in [
        TRAIN_MANIFEST,
        VAL_MANIFEST,
        TEST_MANIFEST,
    ]:

        df = pd.read_csv(manifest_path)

        labels = set(
            df["label"].astype(str).tolist()
        )

        assert labels.issubset(
            valid_classes
        )


def test_dataset_loading():

    dataset = KidneyCTDataset(
        manifest_path=TRAIN_MANIFEST,
        image_size=224,
        train=False
    )

    assert len(dataset) == 70

    image, label = dataset[0]

    assert isinstance(
        image,
        torch.Tensor
    )

    assert isinstance(
        label,
        int
    )

    assert image.shape == (
        3,
        224,
        224
    )

    assert 0 <= label < len(CLASS_NAMES)


def test_all_dataset_images_are_readable():

    for manifest_path, expected_size in [
        (TRAIN_MANIFEST, 70),
        (VAL_MANIFEST, 15),
        (TEST_MANIFEST, 15),
    ]:

        dataset = KidneyCTDataset(
            manifest_path=manifest_path,
            image_size=224,
            train=False
        )

        assert len(dataset) == expected_size

        for index in range(len(dataset)):

            image, label = dataset[index]

            assert image.shape == (
                3,
                224,
                224
            )

            assert torch.isfinite(
                image
            ).all()

            assert isinstance(
                label,
                int
            )

            assert 0 <= label < len(CLASS_NAMES)


def test_dataloaders():

    train_loader, val_loader, test_loader = (
        create_dataloaders(
            batch_size=16,
            image_size=224
        )
    )

    train_images, train_labels = next(
        iter(train_loader)
    )

    val_images, val_labels = next(
        iter(val_loader)
    )

    test_images, test_labels = next(
        iter(test_loader)
    )

    assert train_images.ndim == 4
    assert val_images.ndim == 4
    assert test_images.ndim == 4

    assert train_images.shape[1:] == (
        3,
        224,
        224
    )

    assert val_images.shape[1:] == (
        3,
        224,
        224
    )

    assert test_images.shape[1:] == (
        3,
        224,
        224
    )

    assert train_labels.ndim == 1
    assert val_labels.ndim == 1
    assert test_labels.ndim == 1

    assert train_labels.dtype == torch.int64
    assert val_labels.dtype == torch.int64
    assert test_labels.dtype == torch.int64


def test_no_split_overlap():

    train_df = pd.read_csv(TRAIN_MANIFEST)
    val_df = pd.read_csv(VAL_MANIFEST)
    test_df = pd.read_csv(TEST_MANIFEST)

    train_paths = set(
        train_df["image_path"]
    )

    val_paths = set(
        val_df["image_path"]
    )

    test_paths = set(
        test_df["image_path"]
    )

    assert train_paths.isdisjoint(
        val_paths
    )

    assert train_paths.isdisjoint(
        test_paths
    )

    assert val_paths.isdisjoint(
        test_paths
    )