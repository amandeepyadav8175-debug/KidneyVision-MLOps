"""
KidneyVision-MLOps
Exploratory Data Analysis (EDA)

Purpose:
- Analyze CSV metadata
- Analyze available local images
- Class distribution
- Image dimensions
- Image mode / format
- Pixel statistics
- Sample image visualization
- Generate EDA reports and plots

Important:
The CSV contains 12,446 metadata records, but only 100 image files
are currently available locally. Image-level analysis is therefore
performed only on the 100 locally available images.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


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

REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

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
    """Normalize class labels for comparison."""

    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    label_map = {
        "normal": "Normal",
        "cyst": "Cyst",
        "stone": "Stone",
        "tumor": "Tumor",
    }

    return label_map.get(value, str(value).strip())


def infer_folder_class(image_path):
    """
    Infer class from folder name.

    Expected folders:
        cystfolder
        Normalfolder
        stonefolder
        Tumorfolder
    """

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
    """Recursively discover all supported image files."""

    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(
            f"Image directory not found:\n{IMAGE_ROOT}"
        )

    image_paths = [
        path
        for path in IMAGE_ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(image_paths)


def save_json(data, filename):
    """Save dictionary as formatted JSON."""

    output_path = REPORT_DIR / filename

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return output_path


# ============================================================
# 1. CSV ANALYSIS
# ============================================================

def analyze_csv():

    print("\n" + "=" * 70)
    print("1. CSV METADATA ANALYSIS")
    print("=" * 70)

    if not CSV_PATH.exists():

        raise FileNotFoundError(
            f"CSV not found:\n{CSV_PATH}"
        )

    df = pd.read_csv(CSV_PATH)

    print(
        f"CSV rows    : {len(df):,}"
    )

    print(
        f"CSV columns : {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    if "Class" in df.columns:

        class_counts = (
            df["Class"]
            .value_counts()
            .reindex(
                EXPECTED_CLASSES,
                fill_value=0
            )
        )

        class_percentages = (
            class_counts
            / len(df)
            * 100
        )

        print("\nCSV class distribution:")

        for class_name in EXPECTED_CLASSES:

            print(
                f"  {class_name:7s}: "
                f"{class_counts[class_name]:,} "
                f"({class_percentages[class_name]:.2f}%)"
            )

    else:

        class_counts = pd.Series(
            dtype="int64"
        )

        class_percentages = pd.Series(
            dtype="float64"
        )

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "class_counts": {
            str(k): int(v)
            for k, v in class_counts.items()
        },
        "class_percentages": {
            str(k): round(float(v), 4)
            for k, v in class_percentages.items()
        },
    }


# ============================================================
# 2. IMAGE ANALYSIS
# ============================================================

def analyze_images(image_paths):

    print("\n" + "=" * 70)
    print("2. IMAGE ANALYSIS")
    print("=" * 70)

    print(
        f"Local images available: {len(image_paths):,}"
    )

    dimensions = []
    modes = {}
    formats = {}
    class_counts = {}

    pixel_values = []

    valid_images = 0
    corrupt_images = []

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        try:

            with Image.open(image_path) as image:

                # Force image decoding
                image.load()

                width, height = image.size

                dimensions.append(
                    (width, height)
                )

                mode = image.mode

                modes[mode] = (
                    modes.get(mode, 0) + 1
                )

                image_format = (
                    image.format
                    if image.format
                    else "Unknown"
                )

                formats[image_format] = (
                    formats.get(image_format, 0) + 1
                )

                class_name = infer_folder_class(
                    image_path
                )

                class_counts[class_name] = (
                    class_counts.get(class_name, 0)
                    + 1
                )

                # ------------------------------------------------
                # Pixel statistics
                # ------------------------------------------------

                # Convert to grayscale for a consistent
                # brightness analysis.
                grayscale = image.convert("L")

                pixels = np.asarray(
                    grayscale,
                    dtype=np.float32
                )

                pixel_values.append(
                    pixels.flatten()
                )

                valid_images += 1

        except Exception as error:

            corrupt_images.append({
                "file": str(image_path),
                "error": str(error),
            })

        if index % 25 == 0:
            print(
                f"Analyzed {index}/{len(image_paths)}"
            )

    # --------------------------------------------------------
    # Dimension statistics
    # --------------------------------------------------------

    dimension_counts = {}

    for width, height in dimensions:

        key = f"{width}x{height}"

        dimension_counts[key] = (
            dimension_counts.get(key, 0)
            + 1
        )

    dimension_counts = dict(
        sorted(
            dimension_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    # --------------------------------------------------------
    # Pixel statistics
    # --------------------------------------------------------

    if pixel_values:

        all_pixels = np.concatenate(
            pixel_values
        )

        pixel_statistics = {
            "mean": round(
                float(np.mean(all_pixels)),
                4
            ),
            "std": round(
                float(np.std(all_pixels)),
                4
            ),
            "min": round(
                float(np.min(all_pixels)),
                4
            ),
            "max": round(
                float(np.max(all_pixels)),
                4
            ),
            "median": round(
                float(np.median(all_pixels)),
                4
            ),
        }

    else:

        pixel_statistics = {}

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nImage class distribution:")

    for class_name in EXPECTED_CLASSES:

        print(
            f"  {class_name:7s}: "
            f"{class_counts.get(class_name, 0):,}"
        )

    print("\nImage modes:")

    for mode, count in sorted(
        modes.items()
    ):

        print(
            f"  {mode}: {count:,}"
        )

    print("\nImage formats:")

    for image_format, count in sorted(
        formats.items()
    ):

        print(
            f"  {image_format}: {count:,}"
        )

    print("\nTop image dimensions:")

    for dimension, count in list(
        dimension_counts.items()
    )[:15]:

        print(
            f"  {dimension}: {count:,}"
        )

    print("\nPixel statistics (grayscale):")

    for key, value in pixel_statistics.items():

        print(
            f"  {key:7s}: {value}"
        )

    print(
        f"\nValid images   : {valid_images:,}"
    )

    print(
        f"Corrupt images : {len(corrupt_images):,}"
    )

    return {
        "total_images": len(image_paths),
        "valid_images": valid_images,
        "corrupt_images": corrupt_images,
        "class_counts": class_counts,
        "modes": modes,
        "formats": formats,
        "dimensions": dimension_counts,
        "pixel_statistics": pixel_statistics,
    }


# ============================================================
# 3. CLASS DISTRIBUTION PLOT
# ============================================================

def create_class_distribution_plot(
    image_analysis
):

    print("\nCreating class distribution plot...")

    class_counts = [
        image_analysis["class_counts"].get(
            class_name,
            0
        )
        for class_name in EXPECTED_CLASSES
    ]

    plt.figure(
        figsize=(8, 6)
    )

    plt.bar(
        EXPECTED_CLASSES,
        class_counts
    )

    plt.title(
        "Local Image Class Distribution"
    )

    plt.xlabel(
        "Class"
    )

    plt.ylabel(
        "Number of Images"
    )

    plt.tight_layout()

    output_path = (
        REPORT_DIR
        / "class_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# 4. IMAGE DIMENSION PLOT
# ============================================================

def create_dimension_plot(
    image_analysis
):

    print(
        "Creating image dimension plot..."
    )

    dimensions = image_analysis[
        "dimensions"
    ]

    labels = list(
        dimensions.keys()
    )

    counts = list(
        dimensions.values()
    )

    # Keep plot readable if many dimensions exist.
    labels = labels[:15]
    counts = counts[:15]

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        labels,
        counts
    )

    plt.title(
        "Image Dimension Distribution"
    )

    plt.xlabel(
        "Image Dimensions"
    )

    plt.ylabel(
        "Number of Images"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    output_path = (
        REPORT_DIR
        / "image_dimensions.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# 5. PIXEL DISTRIBUTION PLOT
# ============================================================

def create_pixel_statistics_plot(
    image_paths
):

    print(
        "Creating pixel intensity distribution..."
    )

    # Sample pixels from each image so that
    # memory usage remains reasonable.
    sampled_pixels = []

    rng = np.random.default_rng(
        seed=42
    )

    for image_path in image_paths:

        try:

            with Image.open(image_path) as image:

                grayscale = image.convert(
                    "L"
                )

                pixels = np.asarray(
                    grayscale,
                    dtype=np.uint8
                ).flatten()

                # Sample at most 5,000 pixels
                # from each image.
                sample_size = min(
                    5000,
                    len(pixels)
                )

                if sample_size > 0:

                    indices = rng.choice(
                        len(pixels),
                        size=sample_size,
                        replace=False,
                    )

                    sampled_pixels.extend(
                        pixels[indices]
                    )

        except Exception:
            continue

    if not sampled_pixels:

        print(
            "No pixels available for histogram."
        )

        return

    plt.figure(
        figsize=(8, 6)
    )

    plt.hist(
        sampled_pixels,
        bins=50
    )

    plt.title(
        "Grayscale Pixel Intensity Distribution"
    )

    plt.xlabel(
        "Pixel Intensity"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.tight_layout()

    output_path = (
        REPORT_DIR
        / "pixel_intensity_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# 6. SAMPLE IMAGE CONTACT SHEET
# ============================================================

def create_sample_contact_sheet(
    image_paths
):

    print(
        "Creating sample image contact sheet..."
    )

    images_by_class = {
        class_name: []
        for class_name in EXPECTED_CLASSES
    }

    for image_path in image_paths:

        class_name = infer_folder_class(
            image_path
        )

        if class_name in images_by_class:

            images_by_class[
                class_name
            ].append(image_path)

    # --------------------------------------------------------
    # Select up to 4 images per class
    # --------------------------------------------------------

    selected = []

    for class_name in EXPECTED_CLASSES:

        class_images = images_by_class[
            class_name
        ]

        selected.extend(
            class_images[:4]
        )

    if not selected:
        print(
            "No sample images available."
        )
        return

    columns = 4
    rows = int(
        np.ceil(
            len(selected) / columns
        )
    )

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(12, 3 * rows)
    )

    axes = np.array(
        axes
    ).reshape(
        -1
    )

    for axis in axes:

        axis.axis("off")

    for index, image_path in enumerate(
        selected
    ):

        try:

            with Image.open(
                image_path
            ) as image:

                axes[index].imshow(
                    image.convert("L"),
                    cmap="gray"
                )

                class_name = infer_folder_class(
                    image_path
                )

                axes[index].set_title(
                    class_name
                )

                axes[index].axis("off")

        except Exception:

            axes[index].axis("off")

    fig.suptitle(
        "Kidney CT Image Samples",
        fontsize=16
    )

    plt.tight_layout()

    output_path = (
        REPORT_DIR
        / "sample_images.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# 7. FINAL EDA REPORT
# ============================================================

def create_final_report(
    csv_analysis,
    image_analysis
):

    report = {
        "project": "KidneyVision-MLOps",
        "analysis_type": "Exploratory Data Analysis",
        "data_scope": {
            "csv_metadata_records": (
                csv_analysis["rows"]
            ),
            "local_images_analyzed": (
                image_analysis["total_images"]
            ),
        },
        "csv_analysis": csv_analysis,
        "image_analysis": {
            key: value
            for key, value
            in image_analysis.items()
        },
        "important_note": (
            "The CSV contains metadata for 12,446 "
            "records, while only 100 corresponding "
            "image files are currently available "
            "locally. Image-level EDA is therefore "
            "based only on the 100 available images."
        ),
    }

    output_path = save_json(
        report,
        "eda_report.json"
    )

    print(
        f"\nEDA report saved to:\n{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print(
        " KIDNEYVISION-MLOPS EXPLORATORY DATA ANALYSIS"
    )
    print("=" * 70)

    try:

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        csv_analysis = analyze_csv()

        # ----------------------------------------------------
        # Images
        # ----------------------------------------------------

        image_paths = discover_images()

        image_analysis = analyze_images(
            image_paths
        )

        # ----------------------------------------------------
        # Visualizations
        # ----------------------------------------------------

        create_class_distribution_plot(
            image_analysis
        )

        create_dimension_plot(
            image_analysis
        )

        create_pixel_statistics_plot(
            image_paths
        )

        create_sample_contact_sheet(
            image_paths
        )

        # ----------------------------------------------------
        # Final report
        # ----------------------------------------------------

        create_final_report(
            csv_analysis,
            image_analysis
        )

        print("\n" + "=" * 70)
        print("EDA COMPLETE")
        print("=" * 70)

    except Exception as error:

        print("\n" + "=" * 70)
        print("EDA FAILED")
        print("=" * 70)

        print(
            f"\nError: {error}"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()