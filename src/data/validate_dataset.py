from pathlib import Path
import hashlib
import json
import sys

import pandas as pd
from PIL import Image, UnidentifiedImageError


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = PROJECT_ROOT / "data" / "raw" / "metadata" / "kidneyData.csv"
IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "images"

REPORT_DIR = PROJECT_ROOT / "data" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXPECTED CLASSES
# ============================================================

EXPECTED_CLASSES = {
    "normal": "Normal",
    "cyst": "Cyst",
    "stone": "Stone",
    "tumor": "Tumor",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# HELPERS
# ============================================================

def normalize_label(value):
    if pd.isna(value):
        return None

    return str(value).strip().lower()


def normalize_filename(value):
  
   
    if pd.isna(value):
        return None

    value = str(value).strip().replace("\\", "/")
    name = Path(value).name

    # Remove image extension so CSV image_id and local image filename match
    return Path(name).stem.lower()


def file_hash(path, chunk_size=1024 * 1024):
    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# LOAD CSV
# ============================================================

def load_csv():

    print("\n" + "=" * 70)
    print("1. CSV VALIDATION")
    print("=" * 70)

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV not found:\n{CSV_PATH}"
        )

    df = pd.read_csv(CSV_PATH)

    print(f"CSV path : {CSV_PATH}")
    print(f"Rows     : {len(df):,}")
    print(f"Columns  : {len(df.columns)}")

    print("\nColumns:")

    for column in df.columns:
        print(f"  - {column}")

    return df


# ============================================================
# CSV QUALITY CHECKS
# ============================================================

def validate_csv(df):

    print("\n" + "=" * 70)
    print("2. CSV QUALITY CHECKS")
    print("=" * 70)

    required_columns = {
        "image_id",
        "path",
        "diag",
        "target",
        "Class",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:

        print("WARNING: Required columns missing:")

        for column in sorted(missing_columns):
            print(f"  - {column}")

    else:
        print("Required columns: PASS")

    print("\nMissing values:")

    missing_values = df.isnull().sum()

    for column, count in missing_values.items():
        print(f"  {column}: {count}")

    duplicate_rows = df.duplicated().sum()

    print(
        f"\nDuplicate complete records: "
        f"{duplicate_rows:,}"
    )

    if "image_id" in df.columns:

        duplicate_image_ids = df["image_id"].duplicated().sum()

        print(
            f"Duplicate image_id values: "
            f"{duplicate_image_ids:,}"
        )

    if "path" in df.columns:

        duplicate_paths = df["path"].duplicated().sum()

        print(
            f"Duplicate path values: "
            f"{duplicate_paths:,}"
        )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def class_distribution(df):

    print("\n" + "=" * 70)
    print("3. CSV CLASS DISTRIBUTION")
    print("=" * 70)

    if "Class" not in df.columns:

        print("Class column not available.")

        return {}

    distribution = df["Class"].value_counts(dropna=False)

    for label, count in distribution.items():

        print(f"  {label}: {count:,}")

    return {
        str(label): int(count)
        for label, count in distribution.items()
    }


# ============================================================
# DISCOVER IMAGES
# ============================================================

def discover_images():

    print("\n" + "=" * 70)
    print("4. IMAGE DISCOVERY")
    print("=" * 70)

    if not IMAGE_ROOT.exists():

        raise FileNotFoundError(
            f"Image directory not found:\n{IMAGE_ROOT}"
        )

    image_paths = []

    for path in IMAGE_ROOT.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ):

            image_paths.append(path)

    image_paths.sort()

    print(f"Image root  : {IMAGE_ROOT}")
    print(f"Images found: {len(image_paths):,}")

    return image_paths


# ============================================================
# FOLDER CLASS DETECTION
# ============================================================

def infer_folder_class(image_path):

    parts = [
        part.lower()
        for part in image_path.parts
    ]

    for key, canonical_name in EXPECTED_CLASSES.items():

        for part in parts:

            if key in part:
                return canonical_name

    return None


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_images(image_paths):

    print("\n" + "=" * 70)
    print("5. IMAGE VALIDATION")
    print("=" * 70)

    valid_images = []
    corrupt_images = []
    image_info = []

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        try:

            with Image.open(image_path) as image:
                image.verify()

            with Image.open(image_path) as image:

                width, height = image.size
                mode = image.mode
                image_format = image.format

            folder_class = infer_folder_class(
                image_path
            )

            valid_images.append(image_path)

            image_info.append({
                "filename": image_path.name,
                "relative_path": str(
                    image_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "width": width,
                "height": height,
                "mode": mode,
                "format": image_format,
                "folder_class": folder_class,
            })

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as error:

            corrupt_images.append({
                "path": str(image_path),
                "error": str(error),
            })

        if index % 25 == 0 or index == len(image_paths):

            print(
                f"Checked "
                f"{index:,}/"
                f"{len(image_paths):,}"
            )

    print(f"\nValid images   : {len(valid_images):,}")
    print(f"Corrupt images : {len(corrupt_images):,}")

    return (
        valid_images,
        corrupt_images,
        image_info,
    )


# ============================================================
# IMAGE CLASS DISTRIBUTION
# ============================================================

def image_class_distribution(image_info):

    print("\n" + "=" * 70)
    print("6. IMAGE FOLDER CLASS DISTRIBUTION")
    print("=" * 70)

    distribution = {}

    for item in image_info:

        label = item["folder_class"]

        if label is None:
            label = "UNKNOWN"

        distribution[label] = (
            distribution.get(label, 0) + 1
        )

    for label, count in sorted(
        distribution.items()
    ):

        print(f"  {label}: {count:,}")

    return distribution


# ============================================================
# CSV ↔ IMAGE MAPPING
# ============================================================

def validate_mapping(df, image_paths):

    print("\n" + "=" * 70)
    print("7. CSV ↔ IMAGE MAPPING")
    print("=" * 70)

    if "image_id" not in df.columns:
        print("image_id column unavailable.")
        return {}

    if "Class" not in df.columns:
        print("Class column unavailable.")
        return {}

    # ---------------------------------------------------------
    # Create lookup from CSV image_id
    # Example:
    # Tumor- (1044) -> tumor- (1044)
    # ---------------------------------------------------------
    csv_by_filename = {}

    for _, row in df.iterrows():

        image_id = normalize_filename(row["image_id"])

        if image_id:
            csv_by_filename.setdefault(
                image_id,
                []
            ).append(row)

    # ---------------------------------------------------------
    # Normalize local image filenames too
    # Example:
    # Tumor- (1044).jpg -> tumor- (1044)
    # ---------------------------------------------------------
    image_filename_set = {
        normalize_filename(path.name)
        for path in image_paths
    }

    matched = []
    missing_csv_records = []
    label_mismatches = []
    duplicate_csv_matches = []

    # ---------------------------------------------------------
    # Compare every local image with CSV
    # ---------------------------------------------------------
    for image_path in image_paths:

        filename = normalize_filename(
            image_path.name
        )

        records = csv_by_filename.get(
            filename,
            []
        )

        if not records:

            missing_csv_records.append(
                image_path.name
            )

            continue

        # More than one CSV record has same image_id
        if len(records) > 1:

            duplicate_csv_matches.append({
                "filename": image_path.name,
                "records": len(records),
            })

        # -----------------------------------------------------
        # Compare folder label with CSV Class
        # -----------------------------------------------------
        folder_class = infer_folder_class(
            image_path
        )

        for row in records:

            csv_class = str(
                row["Class"]
            ).strip()

            if (
                normalize_label(folder_class)
                != normalize_label(csv_class)
            ):

                label_mismatches.append({
                    "filename": image_path.name,
                    "folder_class": folder_class,
                    "csv_class": csv_class,
                })

        matched.append(
            image_path.name
        )

    # ---------------------------------------------------------
    # Find CSV records for which local image is unavailable
    # ---------------------------------------------------------
    csv_filenames = set(
        csv_by_filename.keys()
    )

    csv_without_local_image = sorted(
        csv_filenames - image_filename_set
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------
    print(
        f"Local images                     : "
        f"{len(image_paths):,}"
    )

    print(
        f"Images matched to CSV            : "
        f"{len(matched):,}"
    )

    print(
        f"Local images missing from CSV    : "
        f"{len(missing_csv_records):,}"
    )

    print(
        f"CSV filenames without local image: "
        f"{len(csv_without_local_image):,}"
    )

    print(
        f"Label mismatches                 : "
        f"{len(label_mismatches):,}"
    )

    print(
        f"CSV duplicate filename matches   : "
        f"{len(duplicate_csv_matches):,}"
    )

    return {
        "matched": matched,
        "missing_csv_records": missing_csv_records,
        "csv_without_local_image": csv_without_local_image,
        "label_mismatches": label_mismatches,
        "duplicate_csv_matches": duplicate_csv_matches,
    }

# ============================================================
# EXACT DUPLICATE IMAGE DETECTION
# ============================================================

def detect_duplicate_images(valid_images):

    print("\n" + "=" * 70)
    print("8. EXACT DUPLICATE IMAGE DETECTION")
    print("=" * 70)

    hashes = {}
    errors = []

    for image_path in valid_images:

        try:

            digest = file_hash(image_path)

            hashes.setdefault(
                digest,
                []
            ).append(
                str(
                    image_path.relative_to(
                        PROJECT_ROOT
                    )
                )
            )

        except OSError as error:

            errors.append({
                "path": str(image_path),
                "error": str(error),
            })

    duplicate_groups = {
        digest: paths
        for digest, paths in hashes.items()
        if len(paths) > 1
    }

    duplicate_image_count = sum(
        len(paths) - 1
        for paths in duplicate_groups.values()
    )

    print(
        f"Unique image hashes : "
        f"{len(hashes):,}"
    )

    print(
        f"Duplicate image groups: "
        f"{len(duplicate_groups):,}"
    )

    print(
        f"Duplicate image files: "
        f"{duplicate_image_count:,}"
    )

    return duplicate_groups, errors


# ============================================================
# IMAGE DIMENSION ANALYSIS
# ============================================================

def dimension_analysis(image_info):

    print("\n" + "=" * 70)
    print("9. IMAGE DIMENSION ANALYSIS")
    print("=" * 70)

    if not image_info:
        return {}

    dimensions = {}

    for item in image_info:

        key = (
            item["width"],
            item["height"]
        )

        dimensions[key] = (
            dimensions.get(key, 0) + 1
        )

    sorted_dimensions = sorted(
        dimensions.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    print("Top image dimensions:")

    for (
        (width, height),
        count
    ) in sorted_dimensions[:20]:

        print(
            f"  {width} x {height}: "
            f"{count:,}"
        )

    return {
        f"{width}x{height}": count
        for (
            width,
            height
        ), count in dimensions.items()
    }


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    df,
    image_info,
    corrupt_images,
    csv_distribution,
    image_distribution,
    mapping,
    duplicate_groups,
    dimension_data,
):

    report = {

        "csv": {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "duplicate_rows": int(
                df.duplicated().sum()
            ),
        },

        "images": {
            "total_discovered": len(image_info),
            "corrupt_images": len(
                corrupt_images
            ),
        },

        "csv_class_distribution":
            csv_distribution,

        "image_folder_class_distribution":
            image_distribution,

        "mapping": {

            "matched": len(
                mapping.get(
                    "matched",
                    []
                )
            ),

            "missing_csv_records": len(
                mapping.get(
                    "missing_csv_records",
                    []
                )
            ),

            "csv_without_local_image": len(
                mapping.get(
                    "csv_without_local_image",
                    []
                )
            ),

            "label_mismatches": len(
                mapping.get(
                    "label_mismatches",
                    []
                )
            ),

            "duplicate_csv_matches": len(
                mapping.get(
                    "duplicate_csv_matches",
                    []
                )
            ),
        },

        "duplicate_image_groups":
            len(duplicate_groups),

        "dimensions":
            dimension_data,
    }

    report_path = (
        REPORT_DIR /
        "dataset_validation_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    print(
        f"\nValidation report saved to:\n"
        f"{report_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print(" KIDNEYVISION-MLOPS DATASET VALIDATOR")
    print("=" * 70)

    try:

        df = load_csv()

        validate_csv(df)

        csv_distribution = class_distribution(
            df
        )

        image_paths = discover_images()

        (
            valid_images,
            corrupt_images,
            image_info,
        ) = validate_images(
            image_paths
        )

        image_distribution = (
            image_class_distribution(
                image_info
            )
        )

        mapping = validate_mapping(
            df,
            valid_images
        )

        (
            duplicate_groups,
            hash_errors,
        ) = detect_duplicate_images(
            valid_images
        )

        dimension_data = dimension_analysis(
            image_info
        )

        save_report(
            df=df,
            image_info=image_info,
            corrupt_images=corrupt_images,
            csv_distribution=csv_distribution,
            image_distribution=image_distribution,
            mapping=mapping,
            duplicate_groups=duplicate_groups,
            dimension_data=dimension_data,
        )

        print("\n" + "=" * 70)
        print("VALIDATION COMPLETE")
        print("=" * 70)

    except Exception as error:

        print("\nVALIDATION FAILED")
        print("-" * 70)
        print(str(error))

        sys.exit(1)


if __name__ == "__main__":
    main()