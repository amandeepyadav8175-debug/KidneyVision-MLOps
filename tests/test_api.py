import io
import sys
from pathlib import Path

from PIL import Image
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from api.main import app


client = TestClient(app)


IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "images"
    / "New folder9999"
    / "cystfolder"
    / "Cyst- (70).jpg"
)


def test_root_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert "message" in data


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


def test_model_info_endpoint():

    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "SwinTransformer"

    assert data["registered_model"] == (
        "KidneyVision-SwinTransformer"
    )

    assert data["alias"] == "champion"

    assert data["image_size"] == 224

    assert data["status"] == "loaded"

    assert set(data["classes"]) == {
        "Normal",
        "Cyst",
        "Stone",
        "Tumor"
    }


def test_predict_endpoint():

    assert IMAGE_PATH.exists()

    with open(IMAGE_PATH, "rb") as image_file:

        response = client.post(
            "/predict",
            files={
                "file": (
                    "Cyst- (70).jpg",
                    image_file,
                    "image/jpeg"
                )
            }
        )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data

    assert "confidence" in data

    assert "confidence_percent" in data

    assert "probabilities" in data

    assert data["prediction"] in {
        "Normal",
        "Cyst",
        "Stone",
        "Tumor"
    }

    assert 0.0 <= data["confidence"] <= 1.0

    assert 0.0 <= data["confidence_percent"] <= 100.0

    probabilities = data["probabilities"]

    assert set(probabilities.keys()) == {
        "Normal",
        "Cyst",
        "Stone",
        "Tumor"
    }

    for probability in probabilities.values():

        assert 0.0 <= probability <= 1.0


def test_explain_endpoint():

    assert IMAGE_PATH.exists()

    with open(IMAGE_PATH, "rb") as image_file:

        response = client.post(
            "/explain",
            files={
                "file": (
                    "Cyst- (70).jpg",
                    image_file,
                    "image/jpeg"
                )
            }
        )

    assert response.status_code == 200

    data = response.json()

    assert data["filename"] == "Cyst- (70).jpg"

    assert "prediction" in data

    assert "confidence" in data

    assert "confidence_percent" in data

    assert "probabilities" in data

    assert "gradcam_image_base64" in data

    assert "gradcam_media_type" in data

    assert data["prediction"] in {
        "Normal",
        "Cyst",
        "Stone",
        "Tumor"
    }

    assert 0.0 <= data["confidence"] <= 1.0

    assert data["gradcam_media_type"] == "image/png"

    assert len(data["gradcam_image_base64"]) > 100


def test_predict_rejects_non_image():

    fake_file = io.BytesIO(
        b"This is not an image"
    )

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.txt",
                fake_file,
                "text/plain"
            )
        }
    )

    assert response.status_code >= 400


def test_explain_rejects_non_image():

    fake_file = io.BytesIO(
        b"This is not an image"
    )

    response = client.post(
        "/explain",
        files={
            "file": (
                "test.txt",
                fake_file,
                "text/plain"
            )
        }
    )

    assert response.status_code >= 400