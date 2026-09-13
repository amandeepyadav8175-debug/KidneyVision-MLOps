"""
KidneyVision-MLOps
Dataset Preparation

Purpose:
- Load verified local images
- Match images with CSV metadata
- Verify labels
- Create reproducible stratified train/validation/test split
- Generate CSV manifests
- Verify no image appears in multiple splits

Current dataset:
- 100 locally available images
- 4 classes:
    Normal, Cyst, Stone, Tumor

Split:
- Train      : 70%
- Validation : 15%
- Test       : 15%

Important:
- Raw data is never modified.
- Only verified local images are included.
- CSV records without local images are ignored for image manifests.
"""

from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "metadata"
    / "kidneyData.csv"
)

IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "images"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

MANIFEST_DIR = (
    PROCESSED_DIR
    / "manifests"
)

MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

EXPECTED_CLASSES = [
    "Normal",
    "Cyst",
    "Stone",
    "Tumor",
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_label(value):
    """Normalize class labels."""

    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    label_map = {
        "normal": "Normal",
        "cyst": "Cyst",
        "stone": "Stone",
        "tumor": "Tumor",
    }

    return label_map.get(
        value,
        str(value).strip()
    )


def normalize_filename(value):
    """
    Convert image_id / filename into a common key.

    Examples:
        Tumor- (1044)       -> tumor- (1044)
        Tumor- (1044).jpg   -> tumor- (1044)
    """

    if pd.isna(value):
        return None

    value = (
        str(value)
        .strip()
        .replace("\\", "/")
    )

    name = Path(value).name

    return Path(name).stem.lower()


def infer_folder_class(image_path):
    """Infer class from the image folder name."""

    parts = [
        part.lower()
        for part in image_path.parts
    ]

    for part in parts:

        if "normal" in part:
            return "Normal"

        if "cyst" in part:
            return "Cyst"

        if "stone" in part:
            return "Stone"

        if "tumor" in part:
            return "Tumor"

    return "Unknown"


def discover_images():
    """Find all supported local image files."""

    if not IMAGE_ROOT.exists():

        raise FileNotFoundError(
            f"Image directory not found:\n{IMAGE_ROOT}"
        )

    image_paths = [
        path
        for path in IMAGE_ROOT.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]

    return sorted(image_paths)


# ============================================================
# LOAD AND MATCH DATA
# ============================================================

def load_verified_dataset():

    print("\n" + "=" * 70)
    print("1. LOADING VERIFIED DATASET")
    print("=" * 70)

    if not CSV_PATH.exists():

        raise FileNotFoundError(
            f"CSV not found:\n{CSV_PATH}"
        )

    df = pd.read_csv(CSV_PATH)

    print(
        f"CSV metadata records : {len(df):,}"
    )

    required_columns = {
        "image_id",
        "Class",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Missing required CSV columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    image_paths = discover_images()

    print(
        f"Local image files    : {len(image_paths):,}"
    )

    # --------------------------------------------------------
    # Create CSV lookup
    # --------------------------------------------------------

    csv_lookup = {}

    for _, row in df.iterrows():

        image_key = normalize_filename(
            row["image_id"]
        )

        if image_key is None:
            continue

        if image_key in csv_lookup:

            raise ValueError(
                f"Duplicate image_id detected: "
                f"{image_key}"
            )

        csv_lookup[image_key] = row

    # --------------------------------------------------------
    # Match local images to CSV
    # --------------------------------------------------------

    records = []
    missing_from_csv = []

    for image_path in image_paths:

        image_key = normalize_filename(
            image_path.name
        )

        row = csv_lookup.get(
            image_key
        )

        if row is None:

            missing_from_csv.append(
                image_path.name
            )

            continue

        csv_class = normalize_label(
            row["Class"]
        )

        folder_class = normalize_label(
            infer_folder_class(image_path)
        )

        if csv_class != folder_class:

            raise ValueError(
                "Label mismatch detected:\n"
                f"Image : {image_path.name}\n"
                f"Folder: {folder_class}\n"
                f"CSV   : {csv_class}"
            )

        if csv_class not in EXPECTED_CLASSES:

            raise ValueError(
                f"Unexpected class '{csv_class}' "
                f"for image {image_path.name}"
            )

        # Store project-relative path.
        # This makes manifests portable.
        relative_path = image_path.relative_to(
            PROJECT_ROOT
        )

        records.append({
            "image_id": str(
                row["image_id"]
            ),
            "image_path": str(
                relative_path
            ).replace("\\", "/"),
            "label": csv_class,
        })

    if missing_from_csv:

        raise ValueError(
            "Some local images were not found "
            "in the CSV:\n"
            + "\n".join(
                missing_from_csv
            )
        )

    if not records:

        raise ValueError(
            "No verified images were available "
            "for dataset preparation."
        )

    dataset = pd.DataFrame(
        records
    )

    print(
        f"Verified image/CSV matches: "
        f"{len(dataset):,}"
    )

    return dataset


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset(dataset):

    print("\n" + "=" * 70)
    print("2. PRE-SPLIT VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Duplicate image IDs
    # --------------------------------------------------------

    duplicate_ids = dataset[
        dataset["image_id"].duplicated(
            keep=False
        )
    ]

    if not duplicate_ids.empty:

        raise ValueError(
            "Duplicate image IDs found before splitting."
        )

    # --------------------------------------------------------
    # Duplicate image paths
    # --------------------------------------------------------

    duplicate_paths = dataset[
        dataset["image_path"].duplicated(
            keep=False
        )
    ]

    if not duplicate_paths.empty:

        raise ValueError(
            "Duplicate image paths found before splitting."
        )

    # --------------------------------------------------------
    # Missing labels
    # --------------------------------------------------------

    if dataset["label"].isna().any():

        raise ValueError(
            "Missing labels found."
        )

    # --------------------------------------------------------
    # Class validation
    # --------------------------------------------------------

    unexpected_classes = set(
        dataset["label"]
    ) - set(
        EXPECTED_CLASSES
    )

    if unexpected_classes:

        raise ValueError(
            "Unexpected classes found: "
            + ", ".join(
                sorted(unexpected_classes)
            )
        )

    print("Duplicate image IDs : 0")
    print("Duplicate image paths: 0")
    print("Missing labels       : 0")

    print("\nVerified class distribution:")

    counts = (
        dataset["label"]
        .value_counts()
        .reindex(
            EXPECTED_CLASSES,
            fill_value=0
        )
    )

    for class_name in EXPECTED_CLASSES:

        print(
            f"  {class_name:7s}: "
            f"{counts[class_name]:,}"
        )


# ============================================================
# STRATIFIED SPLIT
# ============================================================

def create_splits(dataset):

    print("\n" + "=" * 70)
    print("3. CREATING STRATIFIED SPLIT")
    print("=" * 70)

    print(
        f"Random seed       : {RANDOM_SEED}"
    )

    print(
        "Train / Validation / Test: "
        f"{TRAIN_SIZE:.0%} / "
        f"{VALIDATION_SIZE:.0%} / "
        f"{TEST_SIZE:.0%}"
    )

    # --------------------------------------------------------
    # First split:
    #
    # 70% train
    # 30% temporary
    # --------------------------------------------------------

    train_df, temp_df = train_test_split(
        dataset,
        test_size=(
            VALIDATION_SIZE
            + TEST_SIZE
        ),
        random_state=RANDOM_SEED,
        stratify=dataset["label"],
    )

    # --------------------------------------------------------
    # Second split:
    #
    # Temporary 30%
    # -> 15% validation
    # -> 15% test
    #
    # Therefore:
    # validation fraction within temp = 0.5
    # --------------------------------------------------------

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=temp_df["label"],
    )

    # --------------------------------------------------------
    # Reset indexes
    # --------------------------------------------------------

    train_df = train_df.reset_index(
        drop=True
    )

    validation_df = validation_df.reset_index(
        drop=True
    )

    test_df = test_df.reset_index(
        drop=True
    )

    return (
        train_df,
        validation_df,
        test_df,
    )


# ============================================================
# SPLIT VALIDATION
# ============================================================

def validate_splits(
    train_df,
    validation_df,
    test_df,
):

    print("\n" + "=" * 70)
    print("4. SPLIT VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check sizes
    # --------------------------------------------------------

    total = (
        len(train_df)
        + len(validation_df)
        + len(test_df)
    )

    print(
        f"Train      : {len(train_df):,}"
    )

    print(
        f"Validation : {len(validation_df):,}"
    )

    print(
        f"Test       : {len(test_df):,}"
    )

    print(
        f"Total      : {total:,}"
    )

    # --------------------------------------------------------
    # Check total
    # --------------------------------------------------------

    if total != (
        len(train_df)
        + len(validation_df)
        + len(test_df)
    ):

        raise ValueError(
            "Split size calculation failed."
        )

    # --------------------------------------------------------
    # Check cross-split leakage
    # --------------------------------------------------------

    train_ids = set(
        train_df["image_id"]
    )

    validation_ids = set(
        validation_df["image_id"]
    )

    test_ids = set(
        test_df["image_id"]
    )

    train_validation_overlap = (
        train_ids
        & validation_ids
    )

    train_test_overlap = (
        train_ids
        & test_ids
    )

    validation_test_overlap = (
        validation_ids
        & test_ids
    )

    if train_validation_overlap:

        raise ValueError(
            "Data leakage detected between "
            "train and validation."
        )

    if train_test_overlap:

        raise ValueError(
            "Data leakage detected between "
            "train and test."
        )

    if validation_test_overlap:

        raise ValueError(
            "Data leakage detected between "
            "validation and test."
        )

    print(
        "Train ↔ Validation overlap : 0"
    )

    print(
        "Train ↔ Test overlap        : 0"
    )

    print(
        "Validation ↔ Test overlap   : 0"
    )

    # --------------------------------------------------------
    # Class distributions
    # --------------------------------------------------------

    print("\nClass distribution by split:")

    for split_name, split_df in [
        ("Train", train_df),
        ("Validation", validation_df),
        ("Test", test_df),
    ]:

        print(
            f"\n{split_name}:"
        )

        counts = (
            split_df["label"]
            .value_counts()
            .reindex(
                EXPECTED_CLASSES,
                fill_value=0
            )
        )

        for class_name in EXPECTED_CLASSES:

            print(
                f"  {class_name:7s}: "
                f"{counts[class_name]:,}"
            )


# ============================================================
# SAVE MANIFESTS
# ============================================================

def save_manifests(
    train_df,
    validation_df,
    test_df,
):

    print("\n" + "=" * 70)
    print("5. SAVING DATASET MANIFESTS")
    print("=" * 70)

    manifests = {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }

    for split_name, split_df in manifests.items():

        output_path = (
            MANIFEST_DIR
            / f"{split_name}.csv"
        )

        split_df.to_csv(
            output_path,
            index=False
        )

        print(
            f"{split_name:10s}: "
            f"{output_path}"
        )


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary(
    dataset,
    train_df,
    validation_df,
    test_df,
):

    summary = {
        "random_seed": RANDOM_SEED,
        "split_ratio": {
            "train": TRAIN_SIZE,
            "validation": VALIDATION_SIZE,
            "test": TEST_SIZE,
        },
        "total_verified_images": len(dataset),
        "splits": {
            "train": len(train_df),
            "validation": len(validation_df),
            "test": len(test_df),
        },
        "class_distribution": {},
    }

    for split_name, split_df in [
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ]:

        counts = (
            split_df["label"]
            .value_counts()
            .reindex(
                EXPECTED_CLASSES,
                fill_value=0
            )
        )

        summary["class_distribution"][
            split_name
        ] = {
            class_name: int(
                counts[class_name]
            )
            for class_name in EXPECTED_CLASSES
        }

    output_path = (
        PROCESSED_DIR
        / "split_summary.json"
    )

    import json

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print(
        f"\nSplit summary saved to:\n"
        f"{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print(
        " KIDNEYVISION-MLOPS DATASET PREPARATION"
    )
    print("=" * 70)

    try:

        dataset = load_verified_dataset()

        validate_dataset(
            dataset
        )

        (
            train_df,
            validation_df,
            test_df,
        ) = create_splits(
            dataset
        )

        validate_splits(
            train_df,
            validation_df,
            test_df,
        )

        save_manifests(
            train_df,
            validation_df,
            test_df,
        )

        save_summary(
            dataset,
            train_df,
            validation_df,
            test_df,
        )

        print("\n" + "=" * 70)
        print("DATASET PREPARATION COMPLETE")
        print("=" * 70)

    except Exception as error:

        print("\n" + "=" * 70)
        print("DATASET PREPARATION FAILED")
        print("=" * 70)

        print(
            f"\nError: {error}"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()