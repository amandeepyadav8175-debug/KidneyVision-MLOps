import zipfile
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ZIP_PATH = Path(
    r"C:\Users\yadav\Downloads\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone4567.zip"
)

MANIFEST_DIR = PROJECT_ROOT / "data" / "processed" / "manifests"
IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "images" / "New folder9999"


def main():
    print("Restoring available original CT images...")
    print(f"ZIP: {ZIP_PATH}")

    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"ZIP not found: {ZIP_PATH}")

    manifests = [
        MANIFEST_DIR / "train.csv",
        MANIFEST_DIR / "validation.csv",
        MANIFEST_DIR / "test.csv",
    ]

    df = pd.concat(
        [pd.read_csv(path) for path in manifests],
        ignore_index=True,
    )

    if len(df) != 100 or df["image_id"].nunique() != 100:
        raise RuntimeError(
            f"Expected 100 unique manifest images, "
            f"found {len(df)} rows and {df['image_id'].nunique()} unique IDs."
        )

    folder_map = {
        "Cyst": "cystfolder",
        "Normal": "Normalfolder",
        "Stone": "stonefolder",
        "Tumor": "Tumorfolder",
    }

    required_files = {}

    for _, row in df.iterrows():
        image_id = str(row["image_id"]).strip()
        label = str(row["label"]).strip()

        if label not in folder_map:
            raise ValueError(f"Unknown label: {label}")

        filename = f"{image_id}.jpg"
        required_files[filename] = folder_map[label]

    print(f"Required project images: {len(required_files)}")

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        zip_members = {}

        for member in z.namelist():
            if member.lower().endswith(".jpg"):
                basename = Path(member).name

                if basename in required_files:
                    zip_members[basename] = member

        missing = sorted(set(required_files) - set(zip_members))

        print(f"Found in ZIP: {len(zip_members)}")
        print(f"Missing from ZIP: {len(missing)}")

        if missing:
            print("\nMissing images:")
            for filename in missing:
                print(f"  {filename}")

        # Remove current JPG files because these are the accidental
        # synthetic replacements.
        for folder in folder_map.values():
            target_folder = IMAGE_ROOT / folder

            if target_folder.exists():
                for image in target_folder.glob("*.jpg"):
                    image.unlink()

            target_folder.mkdir(parents=True, exist_ok=True)

        # Restore only images that actually exist in the original ZIP.
        restored = 0

        for filename, folder in required_files.items():
            if filename not in zip_members:
                continue

            member = zip_members[filename]
            target = IMAGE_ROOT / folder / filename

            with z.open(member) as source, open(target, "wb") as destination:
                destination.write(source.read())

            restored += 1

    print("\nRESTORATION SUMMARY")
    print("-------------------")
    print(f"Required : 100")
    print(f"Restored : {restored}")
    print(f"Missing  : {len(missing)}")

    print("\nMissing files:")
    for filename in missing:
        print(f"  {filename}")

    if restored != 92 or len(missing) != 8:
        raise RuntimeError(
            f"Unexpected result: restored={restored}, missing={len(missing)}"
        )

    print("\n92 ORIGINAL IMAGES RESTORED SUCCESSFULLY.")
    print("The remaining 8 are intentionally left unresolved.")


if __name__ == "__main__":
    main()