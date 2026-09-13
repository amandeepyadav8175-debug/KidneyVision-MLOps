import io
import base64

from PIL import Image
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


# ============================================================
# TEST IMAGE
# ============================================================

def create_test_image():
    """
    Creates a small valid RGB JPEG image in memory.
    No real dataset image is required for API tests.
    """

    image = Image.new(
        "RGB",
        (64, 64),
        color=(120, 120, 120)
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG"
    )

    buffer.seek(0)

    return buffer


# ============================================================
# MOCK PREDICTION
# ============================================================

def mock_predict_image(image):
    """
    Deterministic fake prediction for testing.
    """

    return {
        "prediction": "Cyst",
        "confidence": 0.998,
        "confidence_percent": 99.8,
        "probabilities": {
            "Normal": 0.001,
            "Cyst": 0.998,
            "Stone": 0.0005,
            "Tumor": 0.0005
        }
    }


# ============================================================
# MOCK GRAD-CAM
# ============================================================

def mock_gradcam(
    model,
    image,
    predicted_index,
    predicted_class,
    confidence
):
    """
    Fake Grad-CAM generator used only during tests.

    The function signature MUST match the real
    generate_gradcam_for_api() function.
    """

    buffer = io.BytesIO()

    fake_image = Image.new(
        "RGB",
        (16, 16),
        color=(255, 0, 0)
    )

    fake_image.save(
        buffer,
        format="PNG"
    )

    return buffer.getvalue()


# ============================================================
# ROOT ENDPOINT
# ============================================================

def test_root_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert "message" in data
    assert "docs" in data


# ============================================================
# HEALTH ENDPOINT
# ============================================================

def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


# ============================================================
# MODEL INFO ENDPOINT
# ============================================================

def test_model_info_endpoint(monkeypatch):

    fake_model_info = {
        "model_name": "SwinTransformer",
        "version": "1",
        "classes": [
            "Normal",
            "Cyst",
            "Stone",
            "Tumor"
        ]
    }

    monkeypatch.setattr(
        "api.main.get_model_info",
        lambda: fake_model_info
    )

    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "SwinTransformer"
    assert data["version"] == "1"
    assert len(data["classes"]) == 4


# ============================================================
# PREDICT ENDPOINT
# ============================================================

def test_predict_endpoint(monkeypatch):

    # Replace real model inference
    monkeypatch.setattr(
        "api.main.predict_image",
        mock_predict_image
    )

    image_buffer = create_test_image()

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.jpg",
                image_buffer,
                "image/jpeg"
            )
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == "Cyst"
    assert data["confidence"] == 0.998
    assert data["confidence_percent"] == 99.8

    assert "probabilities" in data

    assert "Normal" in data["probabilities"]
    assert "Cyst" in data["probabilities"]
    assert "Stone" in data["probabilities"]
    assert "Tumor" in data["probabilities"]


# ============================================================
# EXPLAIN ENDPOINT
# ============================================================

def test_explain_endpoint(monkeypatch):

    # Replace real model inference
    monkeypatch.setattr(
        "api.main.predict_image",
        mock_predict_image
    )

    # Replace real Grad-CAM generation
    monkeypatch.setattr(
        "api.main.generate_gradcam_for_api",
        mock_gradcam
    )

    # Prevent loading the real MLflow model
    monkeypatch.setattr(
        "api.main.get_model",
        lambda: object()
    )

    image_buffer = create_test_image()

    response = client.post(
        "/explain",
        files={
            "file": (
                "test.jpg",
                image_buffer,
                "image/jpeg"
            )
        }
    )

    assert response.status_code == 200

    # Grad-CAM endpoint should return PNG
    assert response.headers["content-type"] == "image/png"

    # Verify prediction information is returned in headers
    assert response.headers["x-predicted-class"] == "Cyst"

    assert float(
        response.headers["x-confidence"]
    ) == 0.998

    # Verify returned content is not empty
    assert len(response.content) > 0

    # Verify it is actually a valid PNG
    result_image = Image.open(
        io.BytesIO(response.content)
    )

    assert result_image.format == "PNG"


# ============================================================
# INVALID FILE - PREDICT
# ============================================================

def test_predict_rejects_non_image():

    text_file = io.BytesIO(
        b"This is not an image."
    )

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.txt",
                text_file,
                "text/plain"
            )
        }
    )

    assert response.status_code == 400


# ============================================================
# INVALID FILE - EXPLAIN
# ============================================================

def test_explain_rejects_non_image():

    text_file = io.BytesIO(
        b"This is not an image."
    )

    response = client.post(
        "/explain",
        files={
            "file": (
                "test.txt",
                text_file,
                "text/plain"
            )
        }
    )

    assert response.status_code == 400