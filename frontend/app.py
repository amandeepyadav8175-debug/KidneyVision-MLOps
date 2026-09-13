import base64
import io
import os

import requests
import streamlit as st
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)

PREDICT_ENDPOINT = f"{API_URL}/predict"
EXPLAIN_ENDPOINT = f"{API_URL}/explain"
HEALTH_ENDPOINT = f"{API_URL}/health"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="KidneyVision MLOps",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .prediction-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-top: 15px;
    }

    .prediction-label {
        font-size: 16px;
        color: #666;
    }

    .prediction-value {
        font-size: 32px;
        font-weight: 700;
    }

    .confidence-value {
        font-size: 26px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🩺 KidneyVision MLOps</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Kidney CT Image Classification using Swin Transformer
        with MLflow and Grad-CAM Explainability
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About the Model")

    st.write(
        """
        This application uses a trained Swin Transformer
        model to classify kidney CT images into four classes.
        """
    )

    st.markdown("### Classes")

    st.write("🟢 Normal")
    st.write("🔵 Cyst")
    st.write("🟡 Stone")
    st.write("🔴 Tumor")

    st.markdown("---")

    st.markdown("### Backend")

    try:

        health_response = requests.get(
            HEALTH_ENDPOINT,
            timeout=5
        )

        if health_response.status_code == 200:
            st.success("API Online")
        else:
            st.error("API Error")

    except requests.exceptions.RequestException:
        st.error("API Offline")

    st.markdown("---")

    st.caption(
        "KidneyVision MLOps Project"
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.header("Upload Kidney CT Image")

uploaded_file = st.file_uploader(
    "Choose a CT image",
    type=["jpg", "jpeg", "png"],
    help="Upload a JPG or PNG kidney CT image."
)


# ============================================================
# MAIN PROCESSING
# ============================================================

if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception:

        st.error(
            "The uploaded file is not a valid image."
        )

        st.stop()


    # ========================================================
    # DISPLAY UPLOADED IMAGE
    # ========================================================

    st.markdown("### Uploaded Image")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption=uploaded_file.name,
            width="stretch"
        )

    with col2:

        st.info(
            f"""
            **Filename:** {uploaded_file.name}

            **Image size:** {image.size[0]} × {image.size[1]}

            **Format:** {uploaded_file.type}

            **Mode:** {image.mode}
            """
        )

    st.markdown("---")


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    if st.button(
        "🔍 Analyze Image",
        type="primary",
        width="stretch"
    ):

        with st.spinner(
            "Running Swin Transformer inference and generating Grad-CAM..."
        ):

            try:

                # ------------------------------------------------
                # PREPARE IMAGE FILE
                # ------------------------------------------------

                files = {
                    "file": (
                        uploaded_file.name,
                        image_bytes,
                        uploaded_file.type
                    )
                }


                # =================================================
                # STEP 1: PREDICTION
                # =================================================

                predict_response = requests.post(
                    PREDICT_ENDPOINT,
                    files=files,
                    timeout=120
                )


                # ------------------------------------------------
                # HANDLE PREDICTION API ERROR
                # ------------------------------------------------

                if predict_response.status_code != 200:

                    try:

                        error_detail = predict_response.json()

                    except Exception:

                        error_detail = predict_response.text

                    st.error(
                        f"Prediction API request failed "
                        f"(HTTP {predict_response.status_code})"
                    )

                    st.code(
                        str(error_detail)
                    )

                    st.stop()


                # ------------------------------------------------
                # READ PREDICTION JSON
                # ------------------------------------------------

                result = predict_response.json()

                prediction = result.get(
                    "prediction",
                    "Unknown"
                )

                confidence = float(
                    result.get(
                        "confidence",
                        0
                    )
                )

                confidence_percent = float(
                    result.get(
                        "confidence_percent",
                        confidence * 100
                    )
                )

                probabilities = result.get(
                    "probabilities",
                    {}
                )


                # =================================================
                # STEP 2: GRAD-CAM
                # =================================================

                explain_response = requests.post(
                    EXPLAIN_ENDPOINT,
                    files=files,
                    timeout=120
                )


                # ------------------------------------------------
                # HANDLE GRAD-CAM RESPONSE
                # ------------------------------------------------

                if explain_response.status_code == 200:

                    # /explain returns PNG image bytes.
                    # It does NOT return JSON.

                    gradcam_base64 = base64.b64encode(
                        explain_response.content
                    ).decode("utf-8")

                else:

                    gradcam_base64 = None

                    st.warning(
                        "Prediction succeeded, but "
                        "Grad-CAM could not be generated."
                    )


                # =================================================
                # SAVE RESULT
                # =================================================

                result["gradcam_image_base64"] = (
                    gradcam_base64
                )

                st.session_state["result"] = result


                # =================================================
                # PREDICTION RESULT
                # =================================================

                st.markdown("---")

                st.header("Prediction Result")

                result_col1, result_col2 = st.columns(2)


                with result_col1:

                    st.markdown(
                        f"""
                        <div class="prediction-box">

                            <div class="prediction-label">
                                Predicted Class
                            </div>

                            <div class="prediction-value">
                                {prediction}
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                with result_col2:

                    st.markdown(
                        f"""
                        <div class="prediction-box">

                            <div class="prediction-label">
                                Confidence
                            </div>

                            <div class="confidence-value">
                                {confidence_percent:.2f}%
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                # =================================================
                # CLASS PROBABILITIES
                # =================================================

                st.markdown("### Class Probabilities")

                if probabilities:

                    probability_columns = st.columns(
                        len(probabilities)
                    )

                    for column, (
                        class_name,
                        probability
                    ) in zip(
                        probability_columns,
                        probabilities.items()
                    ):

                        probability = float(
                            probability
                        )

                        probability_percent = (
                            probability * 100
                        )

                        with column:

                            st.metric(
                                label=class_name,
                                value=(
                                    f"{probability_percent:.2f}%"
                                )
                            )

                            st.progress(
                                min(
                                    max(
                                        probability,
                                        0.0
                                    ),
                                    1.0
                                )
                            )


                # =================================================
                # GRAD-CAM
                # =================================================

                st.markdown("---")

                st.header(
                    "Grad-CAM Explainability"
                )

                st.write(
                    """
                    Grad-CAM highlights image regions that
                    contributed to the model's prediction.
                    """
                )


                if gradcam_base64:

                    try:

                        gradcam_bytes = base64.b64decode(
                            gradcam_base64
                        )

                        gradcam_image = Image.open(
                            io.BytesIO(
                                gradcam_bytes
                            )
                        )

                        st.image(
                            gradcam_image,
                            caption="Grad-CAM Explanation",
                            width="stretch"
                        )

                    except Exception as error:

                        st.error(
                            "Could not decode Grad-CAM image."
                        )

                        st.exception(error)

                else:

                    st.warning(
                        "Grad-CAM image was not returned by the API."
                    )


                # =================================================
                # RAW API RESPONSE
                # =================================================

                with st.expander(
                    "View API Response"
                ):

                    st.json(result)


            # ====================================================
            # REQUEST ERROR HANDLING
            # ====================================================

            except requests.exceptions.Timeout:

                st.error(
                    """
                    The API request timed out.

                    Make sure the FastAPI server is running
                    and the model is loaded.
                    """
                )


            except requests.exceptions.ConnectionError:

                st.error(
                    f"""
                    Could not connect to FastAPI.

                    Current API URL:

                    {API_URL}
                    """
                )


            except requests.exceptions.RequestException as error:

                st.error(
                    f"Request failed: {error}"
                )


            except Exception as error:

                st.error(
                    "An unexpected error occurred."
                )

                st.exception(error)


# ============================================================
# INITIAL SCREEN
# ============================================================

else:

    st.info(
        """
        Upload a kidney CT image above to start classification.

        The application will return:

        • Predicted kidney condition
        • Prediction confidence
        • Probability for each class
        • Grad-CAM visual explanation
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "KidneyVision MLOps • Swin Transformer • MLflow • FastAPI • Grad-CAM"
)