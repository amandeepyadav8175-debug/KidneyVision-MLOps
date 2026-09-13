import os
from pathlib import Path
import csv

from PIL import Image


# ============================================================
# CI SAFETY GUARD
# ============================================================

if os.getenv("GITHUB_ACTIONS") != "true":
    raise RuntimeError(
        "\n"
        "ERROR: create_ci_dataset.py is CI-only.\n"
        "This script must NOT be executed on your local machine.\n"
        "It is intended to run only inside GitHub Actions.\n"
    )


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "images"
    / "New folder9999"
)

METADATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "metadata"
)

CSV_PATH = METADATA_DIR / "kidneyData.csv"


# ============================================================
# CI DATASET CONFIGURATION
# ============================================================

CLASS_CONFIG = {
    "Normal": {
        "folder": "Normalfolder",
        "target": 1,
        "diag": "Normal",
    },
    "Cyst": {
        "folder": "cystfolder",
        "target": 0,
        "diag": "Cyst",
    },
    "Stone": {
        "folder": "stonefolder",
        "target": 2,
        "diag": "Stone",
    },
    "Tumor": {
        "folder": "Tumorfolder",
        "target": 3,
        "diag": "Tumor",
    },
}

IMAGES_PER_CLASS = 25


# ============================================================
# CREATE SYNTHETIC IMAGE
# ============================================================

def create_image(path: Path, class_index: int):
    """
    Create a small valid RGB JPEG image.

    Used only inside GitHub Actions CI.
    """

    width = 224
    height = 224

    base_value = 40 + (class_index * 40)

    image = Image.new(
        "RGB",
        (width, height),
        color=(
            base_value,
            base_value,
            base_value,
        ),
    )

    image.save(
        path,
        format="JPEG",
        quality=90,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("Creating synthetic CI dataset...")
    print("Environment: GitHub Actions")
    print("Real medical dataset will NOT be used in CI.")

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    IMAGE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Remove only files in the disposable CI runner.
    # GitHub Actions runs on a fresh temporary machine.
    # --------------------------------------------------------

    for config in CLASS_CONFIG.values():

        class_dir = IMAGE_ROOT / config["folder"]

        class_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for image_file in class_dir.glob("*.jpg"):
            image_file.unlink()

    # --------------------------------------------------------
    # Create images + metadata
    # --------------------------------------------------------

    rows = []

    image_counter = 1

    class_names = list(CLASS_CONFIG.keys())

    for class_index, class_name in enumerate(class_names):

        config = CLASS_CONFIG[class_name]

        class_dir = IMAGE_ROOT / config["folder"]

        for number in range(
            1,
            IMAGES_PER_CLASS + 1,
        ):

            image_id = (
                f"CI-{class_name}-{number:03d}"
            )

            filename = f"{image_id}.jpg"

            image_path = class_dir / filename

            create_image(
                image_path,
                class_index,
            )

            relative_path = image_path.relative_to(
                PROJECT_ROOT
            ).as_posix()

            rows.append(
                {
                    "Unnamed: 0": image_counter,
                    "image_id": image_id,
                    "path": relative_path,
                    "diag": config["diag"],
                    "target": config["target"],
                    "Class": class_name,
                }
            )

            image_counter += 1

    # --------------------------------------------------------
    # Write CI metadata CSV
    # --------------------------------------------------------

    fieldnames = [
        "Unnamed: 0",
        "image_id",
        "path",
        "diag",
        "target",
        "Class",
    ]

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)

    print("")
    print("Synthetic CI dataset created successfully.")
    print(f"Images per class : {IMAGES_PER_CLASS}")
    print(f"Total images     : {len(rows)}")
    print(f"Metadata CSV     : {CSV_PATH}")

    print("")
    print("Class distribution:")

    for class_name in class_names:
        print(
            f"  {class_name}: "
            f"{IMAGES_PER_CLASS}"
        )


if __name__ == "__main__":
    main()