import os
import requests
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

DEFAULT_API_URL = "https://kidneyvision-mlops.onrender.com"
API_URL = os.getenv("API_URL", DEFAULT_API_URL).rstrip("/")

CLASS_NAMES = ["Normal", "Cyst", "Stone", "Tumor"]

st.set_page_config(
    page_title="KidneyVision AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
<style>

html, body, [class*="css"] {
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 8% 0%,
            rgba(30, 136, 229, 0.16),
            transparent 26%
        ),
        radial-gradient(
            circle at 92% 8%,
            rgba(0, 188, 212, 0.12),
            transparent 25%
        ),
        linear-gradient(
            135deg,
            #f5f9ff 0%,
            #eef5fc 50%,
            #f8fbff 100%
        );
}

.main .block-container {
    max-width: 1380px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}

/* Hide Streamlit chrome */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #071b35 0%,
            #0b2850 48%,
            #0d3564 100%
        );
}

section[data-testid="stSidebar"] > div {
    background: transparent;
}

section[data-testid="stSidebar"] * {
    color: white;
}

/* File uploader */

[data-testid="stFileUploader"] {
    background: transparent;
}

[data-testid="stFileUploaderDropzone"] {
    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.98),
            rgba(247,251,255,0.98)
        ) !important;

    border: 2px dashed #8eb6dc !important;
    border-radius: 20px !important;
    min-height: 190px !important;

    transition: all 0.2s ease;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #1677c8 !important;
    box-shadow:
        0 12px 30px rgba(20,80,140,0.10);
}

/* Buttons */

.stButton > button {
    border: none !important;
    border-radius: 13px !important;

    background:
        linear-gradient(
            135deg,
            #0d4f91,
            #1688d5
        ) !important;

    color: white !important;

    font-weight: 800 !important;
    font-size: 0.95rem !important;

    min-height: 48px;

    box-shadow:
        0 10px 25px rgba(13,79,145,0.20);

    transition:
        transform 0.18s ease,
        box-shadow 0.18s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 14px 32px rgba(13,79,145,0.28);
}

/* Progress */

.stProgress > div > div > div {
    background:
        linear-gradient(
            90deg,
            #1769aa,
            #12a6e8
        );
}

/* Image */

[data-testid="stImage"] img {
    border-radius: 18px;
}

/* Alerts */

div[data-testid="stAlert"] {
    border-radius: 14px;
}

/* Mobile */

@media (max-width: 900px) {

    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HTML HELPER
# ============================================================

def render_html(content: str):
    """
    Uses Streamlit's native HTML renderer.

    This prevents custom HTML from appearing as raw text/code.
    """
    st.html(content)


# ============================================================
# API FUNCTIONS
# ============================================================

def check_api():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=10,
        )

        return response.ok

    except Exception:
        return False


def predict_image(uploaded_file):

    try:

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "image/jpeg",
            )
        }

        response = requests.post(
            f"{API_URL}/predict",
            files=files,
            timeout=90,
        )

        if response.ok:
            return response.json()

        return None

    except Exception:
        return None


def explain_image(uploaded_file):

    try:

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "image/jpeg",
            )
        }

        response = requests.post(
            f"{API_URL}/explain",
            files=files,
            timeout=150,
        )

        if response.ok:
            return response.content

        return None

    except Exception:
        return None

def explain_gradcam_image(uploaded_file):

    try:
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "image/jpeg",
            )
        }

        response = requests.post(
            f"{API_URL}/explain-gradcam",
            files=files,
            timeout=180,
        )

        if response.ok:
            return response.content

        return None

    except Exception:
        return None




# ============================================================
# SIDEBAR
# ============================================================

api_online = check_api()

with st.sidebar:

    render_html(
        """
        <div style="
            padding:10px 4px 24px 4px;
            border-bottom:1px solid rgba(255,255,255,0.14);
            margin-bottom:22px;
        ">

            <div style="
                font-size:2.2rem;
                margin-bottom:7px;
            ">
                🩺
            </div>

            <div style="
                font-size:1.35rem;
                font-weight:850;
                letter-spacing:-0.4px;
            ">
                KidneyVision AI
            </div>

            <div style="
                color:#b9cbe1;
                font-size:0.78rem;
                margin-top:5px;
            ">
                Deep Learning + MLOps Platform
            </div>

        </div>
        """
    )

    # System status

    status_color = "#62e6a4" if api_online else "#ff8585"
    status_text = "ONLINE" if api_online else "OFFLINE"

    render_html(
        f"""
        <div style="
            background:rgba(255,255,255,0.075);
            border:1px solid rgba(255,255,255,0.11);
            border-radius:16px;
            padding:16px;
            margin-bottom:14px;
        ">

            <div style="
                color:#8faac7;
                font-size:0.68rem;
                font-weight:800;
                letter-spacing:1px;
                text-transform:uppercase;
            ">
                System Status
            </div>

            <div style="
                color:{status_color};
                font-size:0.92rem;
                font-weight:800;
                margin-top:7px;
            ">
                ● API {status_text}
            </div>

        </div>
        """
    )

    # Model

    render_html(
        """
        <div style="
            background:rgba(255,255,255,0.075);
            border:1px solid rgba(255,255,255,0.11);
            border-radius:16px;
            padding:16px;
            margin-bottom:14px;
        ">

            <div style="
                color:#8faac7;
                font-size:0.68rem;
                font-weight:800;
                letter-spacing:1px;
                text-transform:uppercase;
            ">
                Model
            </div>

            <div style="
                color:white;
                font-size:0.95rem;
                font-weight:800;
                margin-top:7px;
            ">
                EfficientNet-B0
            </div>

        </div>
        """
    )

    # Task

    render_html(
        """
        <div style="
            background:rgba(255,255,255,0.075);
            border:1px solid rgba(255,255,255,0.11);
            border-radius:16px;
            padding:16px;
            margin-bottom:14px;
        ">

            <div style="
                color:#8faac7;
                font-size:0.68rem;
                font-weight:800;
                letter-spacing:1px;
                text-transform:uppercase;
            ">
                Task
            </div>

            <div style="
                color:white;
                font-size:0.91rem;
                font-weight:750;
                margin-top:7px;
                line-height:1.4;
            ">
                4-Class Kidney CT Classification
            </div>

        </div>
        """
    )

    # Classes

    render_html(
        """
        <div style="
            background:rgba(255,255,255,0.075);
            border:1px solid rgba(255,255,255,0.11);
            border-radius:16px;
            padding:16px;
            margin-bottom:14px;
        ">

            <div style="
                color:#8faac7;
                font-size:0.68rem;
                font-weight:800;
                letter-spacing:1px;
                text-transform:uppercase;
            ">
                Classes
            </div>

            <div style="
                color:white;
                font-size:0.88rem;
                font-weight:700;
                margin-top:7px;
                line-height:1.6;
            ">
                Normal · Cyst · Stone · Tumor
            </div>

        </div>
        """
    )

    # Explainability

    render_html(
        """
        <div style="
            background:rgba(255,255,255,0.075);
            border:1px solid rgba(255,255,255,0.11);
            border-radius:16px;
            padding:16px;
        ">

            <div style="
                color:#8faac7;
                font-size:0.68rem;
                font-weight:800;
                letter-spacing:1px;
                text-transform:uppercase;
            ">
                Explainability
            </div>

            <div style="
                color:white;
                font-size:0.9rem;
                font-weight:750;
                margin-top:7px;
            ">
                Occlusion + Grad-CAM
            </div>

        </div>
        """
    )


# ============================================================
# TOP HEADER
# ============================================================

render_html(
    f"""
    <div style="
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:20px;

        background:rgba(255,255,255,0.82);
        border:1px solid #dbe7f3;
        border-radius:18px;

        padding:14px 20px;
        margin-bottom:20px;

        box-shadow:
            0 8px 28px rgba(20,50,90,0.07);
    ">

        <div style="
            display:flex;
            align-items:center;
            gap:12px;
        ">

            <div style="
                width:44px;
                height:44px;

                display:flex;
                align-items:center;
                justify-content:center;

                border-radius:13px;

                background:
                    linear-gradient(
                        135deg,
                        #0d3f77,
                        #1687d2
                    );

                color:white;
                font-size:1.45rem;

                box-shadow:
                    0 8px 18px rgba(13,63,119,0.18);
            ">
                🩺
            </div>

            <div>

                <div style="
                    color:#0b2443;
                    font-size:1.05rem;
                    font-weight:850;
                ">
                    KidneyVision AI
                </div>

                <div style="
                    color:#71849a;
                    font-size:0.72rem;
                    margin-top:2px;
                ">
                    Intelligent Kidney CT Analysis Platform
                </div>

            </div>

        </div>

        <div style="
            display:flex;
            align-items:center;
            gap:8px;

            background:
                {'#e9fff4' if api_online else '#fff0f0'};

            color:
                {'#138653' if api_online else '#c03939'};

            border:1px solid
                {'#b9efd5' if api_online else '#ffd0d0'};

            border-radius:999px;

            padding:8px 13px;

            font-size:0.72rem;
            font-weight:850;
            letter-spacing:0.5px;
        ">

            <span style="
                width:8px;
                height:8px;
                border-radius:50%;
                background:
                    {'#20bd72' if api_online else '#e55353'};
                display:inline-block;
            "></span>

            {'LIVE API' if api_online else 'API OFFLINE'}

        </div>

    </div>
    """
)


# ============================================================
# HERO
# ============================================================

render_html(
    """
    <div style="
        position:relative;
        overflow:hidden;

        background:
            linear-gradient(
                135deg,
                #071c37 0%,
                #0d3c70 52%,
                #087dbd 100%
            );

        border-radius:26px;

        padding:46px 48px;

        margin-bottom:28px;

        box-shadow:
            0 22px 55px rgba(10,48,88,0.20);
    ">

        <div style="
            position:absolute;
            width:320px;
            height:320px;
            right:-120px;
            top:-150px;
            border-radius:50%;
            background:rgba(255,255,255,0.08);
        "></div>

        <div style="
            position:absolute;
            width:180px;
            height:180px;
            right:120px;
            bottom:-120px;
            border-radius:50%;
            background:rgba(0,210,255,0.08);
        "></div>

        <div style="
            position:relative;
            z-index:2;
        ">

            <div style="
                display:inline-block;

                background:rgba(255,255,255,0.10);

                border:1px solid rgba(255,255,255,0.18);

                border-radius:999px;

                padding:8px 14px;

                color:#dff3ff;

                font-size:0.73rem;

                font-weight:850;

                letter-spacing:0.7px;
            ">
                🧠 AI-POWERED MEDICAL IMAGE ANALYSIS
            </div>

            <div style="
                color:white;

                font-size:3rem;

                line-height:1.05;

                font-weight:900;

                letter-spacing:-1.8px;

                margin-top:17px;
            ">
                KidneyVision AI
            </div>

            <div style="
                max-width:760px;

                color:#d6e9fa;

                font-size:1rem;

                line-height:1.7;

                margin-top:15px;
            ">
                An end-to-end deep learning and MLOps platform
                for classifying kidney CT images into
                <b style="color:white;">Normal</b>,
                <b style="color:white;">Cyst</b>,
                <b style="color:white;">Stone</b>,
                and <b style="color:white;">Tumor</b>
                categories.
            </div>

            <div style="
                display:flex;
                flex-wrap:wrap;
                gap:9px;

                margin-top:22px;
            ">

                <span style="
                    background:rgba(255,255,255,0.10);
                    border:1px solid rgba(255,255,255,0.13);
                    border-radius:999px;
                    padding:7px 11px;
                    color:#e4f4ff;
                    font-size:0.72rem;
                    font-weight:700;
                ">
                    PyTorch
                </span>

                <span style="
                    background:rgba(255,255,255,0.10);
                    border:1px solid rgba(255,255,255,0.13);
                    border-radius:999px;
                    padding:7px 11px;
                    color:#e4f4ff;
                    font-size:0.72rem;
                    font-weight:700;
                ">
                    EfficientNet-B0
                </span>

                <span style="
                    background:rgba(255,255,255,0.10);
                    border:1px solid rgba(255,255,255,0.13);
                    border-radius:999px;
                    padding:7px 11px;
                    color:#e4f4ff;
                    font-size:0.72rem;
                    font-weight:700;
                ">
                    MLflow
                </span>

                <span style="
                    background:rgba(255,255,255,0.10);
                    border:1px solid rgba(255,255,255,0.13);
                    border-radius:999px;
                    padding:7px 11px;
                    color:#e4f4ff;
                    font-size:0.72rem;
                    font-weight:700;
                ">
                    FastAPI
                </span>

                <span style="
                    background:rgba(255,255,255,0.10);
                    border:1px solid rgba(255,255,255,0.13);
                    border-radius:999px;
                    padding:7px 11px;
                    color:#e4f4ff;
                    font-size:0.72rem;
                    font-weight:700;
                ">
                    Docker
                </span>

                <span style="
                    background:rgba(255,255,255,0.10);
                    border:1px solid rgba(255,255,255,0.13);
                    border-radius:999px;
                    padding:7px 11px;
                    color:#e4f4ff;
                    font-size:0.72rem;
                    font-weight:700;
                ">
                    Explainable AI
                </span>

            </div>

        </div>

    </div>
    """
)


# ============================================================
# UPLOAD SECTION
# ============================================================

render_html(
    """
    <div style="
        margin-top:5px;
        margin-bottom:7px;

        color:#0b2748;

        font-size:1.45rem;

        font-weight:900;
    ">
        🔬 Analyze Kidney CT Scan
    </div>

    <div style="
        color:#6b7f95;
        font-size:0.88rem;
        margin-bottom:16px;
    ">
        Upload a CT image and run the deployed AI model.
    </div>
    """
)


# ============================================================
# UPLOAD CARD
# ============================================================

render_html(
    """
    <div style="
        background:rgba(255,255,255,0.88);

        border:1px solid #d9e6f3;

        border-radius:22px;

        padding:20px;

        box-shadow:
            0 10px 30px rgba(20,50,90,0.07);

        margin-bottom:18px;
    ">
    """
)

uploaded_file = st.file_uploader(
    "Upload kidney CT image",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible",
)

render_html(
    """
    <div style="
        text-align:center;

        color:#71849a;

        font-size:0.78rem;

        margin-top:8px;
    ">
        Supported formats: JPG · JPEG · PNG
        &nbsp;&nbsp;•&nbsp;&nbsp;
        Medical/research prototype
    </div>

    </div>
    """
)


# ============================================================
# PREVIEW
# ============================================================

analyze = False

if uploaded_file is not None:

    st.write("")

    preview_col, action_col = st.columns(
        [1.45, 1],
        gap="large",
    )

    with preview_col:

        render_html(
            """
            <div style="
                color:#0b2748;
                font-size:0.82rem;
                font-weight:850;
                text-transform:uppercase;
                letter-spacing:0.8px;
                margin-bottom:8px;
            ">
                Uploaded Scan
            </div>
            """
        )

        st.image(
            uploaded_file,
            caption=uploaded_file.name,
            width="stretch",
        )

    with action_col:

        render_html(
            """
            <div style="
                background:
                    linear-gradient(
                        135deg,
                        #f2f8ff,
                        #ffffff
                    );

                border:1px solid #dbe8f4;

                border-radius:18px;

                padding:22px;

                margin-top:25px;
            ">

                <div style="
                    font-size:2rem;
                    margin-bottom:10px;
                ">
                    🧠
                </div>

                <div style="
                    color:#0b2748;
                    font-size:1.1rem;
                    font-weight:850;
                ">
                    Ready for AI Analysis
                </div>

                <div style="
                    color:#71849a;
                    font-size:0.82rem;
                    line-height:1.6;
                    margin-top:8px;
                ">
                    The deployed EfficientNet-B0 model
                    will classify the scan into four categories.
                </div>

            </div>
            """
        )

        st.write("")

        analyze = st.button(
            "🔍  Analyze CT Scan",
            use_container_width=True,
        )


# ============================================================
# ANALYSIS
# ============================================================

if uploaded_file is not None and analyze:

    if not api_online:

        st.error(
            "Prediction API is currently unavailable. "
            "Please check the Render backend."
        )

        st.stop()

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with st.spinner(
        "🧠 Running EfficientNet-B0 inference..."
    ):

        prediction_result = predict_image(uploaded_file)

    if prediction_result is None:

        st.error(
            "Prediction failed. Please try another image."
        )

        st.stop()

    prediction = prediction_result.get(
        "prediction",
        "Unknown",
    )

    confidence = float(
        prediction_result.get(
            "confidence",
            0,
        )
    )

    confidence_percent = float(
        prediction_result.get(
            "confidence_percent",
            confidence * 100,
        )
    )

    probabilities = prediction_result.get(
        "probabilities",
        {},
    )

    # --------------------------------------------------------
    # RESULT HEADER
    # --------------------------------------------------------

    render_html(
        """
        <div style="
            margin-top:34px;
            margin-bottom:7px;

            color:#0b2748;

            font-size:1.45rem;

            font-weight:900;
        ">
            📊 Analysis Result
        </div>

        <div style="
            color:#6b7f95;
            font-size:0.88rem;
            margin-bottom:18px;
        ">
            Output returned by the deployed AI inference service.
        </div>
        """
    )

    # --------------------------------------------------------
    # RESULT CARDS
    # --------------------------------------------------------

    result_col, confidence_col = st.columns(
        2,
        gap="large",
    )

    with result_col:

        render_html(
            f"""
            <div style="
                background:white;

                border:1px solid #dbe7f3;

                border-radius:20px;

                padding:25px;

                min-height:150px;

                box-shadow:
                    0 10px 28px rgba(20,50,90,0.07);
            ">

                <div style="
                    color:#74869b;

                    font-size:0.68rem;

                    text-transform:uppercase;

                    letter-spacing:1px;

                    font-weight:850;
                ">
                    Predicted Class
                </div>

                <div style="
                    color:#0b2748;

                    font-size:2.35rem;

                    font-weight:900;

                    margin-top:9px;

                    letter-spacing:-1px;
                ">
                    {prediction}
                </div>

                <div style="
                    color:#7b8da2;

                    font-size:0.78rem;

                    margin-top:7px;
                ">
                    Highest probability class from the model.
                </div>

            </div>
            """
        )

    with confidence_col:

        render_html(
            f"""
            <div style="
                background:white;

                border:1px solid #dbe7f3;

                border-radius:20px;

                padding:25px;

                min-height:150px;

                box-shadow:
                    0 10px 28px rgba(20,50,90,0.07);
            ">

                <div style="
                    color:#74869b;

                    font-size:0.68rem;

                    text-transform:uppercase;

                    letter-spacing:1px;

                    font-weight:850;
                ">
                    Model Confidence
                </div>

                <div style="
                    color:#0877bd;

                    font-size:2.35rem;

                    font-weight:900;

                    margin-top:9px;

                    letter-spacing:-1px;
                ">
                    {confidence_percent:.2f}%
                </div>

                <div style="
                    color:#7b8da2;

                    font-size:0.78rem;

                    margin-top:7px;
                ">
                    Softmax probability for the predicted class.
                </div>

            </div>
            """
        )

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.write("")

    render_html(
        """
        <div style="
            color:#0b2748;
            font-size:1.05rem;
            font-weight:850;
            margin-top:15px;
            margin-bottom:12px;
        ">
            ⚙️ Model Information
        </div>
        """
    )

    info1, info2, info3, info4 = st.columns(4)

    info_data = [
        (
            info1,
            "🧠",
            "Architecture",
            "EfficientNet-B0",
        ),
        (
            info2,
            "🖼️",
            "Input Size",
            "224 × 224 RGB",
        ),
        (
            info3,
            "🎯",
            "Classes",
            "4 Categories",
        ),
        (
            info4,
            "🔎",
            "Explainability",
            "Occlusion + Grad-CAM",
        ),
    ]

    for column, icon, label, value in info_data:

        with column:

            render_html(
                f"""
                <div style="
                    background:white;

                    border:1px solid #dbe7f3;

                    border-radius:16px;

                    padding:18px;

                    min-height:105px;

                    box-shadow:
                        0 7px 20px rgba(20,50,90,0.05);
                ">

                    <div style="
                        font-size:1.25rem;
                    ">
                        {icon}
                    </div>

                    <div style="
                        color:#71849a;

                        font-size:0.67rem;

                        text-transform:uppercase;

                        letter-spacing:0.7px;

                        font-weight:800;

                        margin-top:7px;
                    ">
                        {label}
                    </div>

                    <div style="
                        color:#0b2748;

                        font-size:0.91rem;

                        font-weight:850;

                        margin-top:4px;
                    ">
                        {value}
                    </div>

                </div>
                """
            )

    # --------------------------------------------------------
    # PROBABILITY DISTRIBUTION
    # --------------------------------------------------------

    st.write("")

    render_html(
        """
        <div style="
            color:#0b2748;
            font-size:1.25rem;
            font-weight:900;
            margin-top:25px;
        ">
            📈 Class Probability Distribution
        </div>

        <div style="
            color:#6b7f95;
            font-size:0.84rem;
            margin-top:4px;
            margin-bottom:15px;
        ">
            Relative probabilities generated by the neural network.
        </div>
        """
    )

    probability_icons = {
        "Normal": "🟢",
        "Cyst": "🟡",
        "Stone": "🟠",
        "Tumor": "🔴",
    }

    for class_name in CLASS_NAMES:

        value = float(
            probabilities.get(
                class_name,
                0,
            )
        )

        percentage = value * 100

        left, right = st.columns(
            [5, 1],
            gap="small",
        )

        with left:

            render_html(
                f"""
                <div style="
                    color:#263c55;

                    font-size:0.88rem;

                    font-weight:800;

                    margin-top:8px;
                ">
                    {probability_icons[class_name]}
                    &nbsp;&nbsp;
                    {class_name}
                </div>
                """
            )

        with right:

            render_html(
                f"""
                <div style="
                    text-align:right;

                    color:#0b2748;

                    font-size:0.88rem;

                    font-weight:900;

                    margin-top:8px;
                ">
                    {percentage:.2f}%
                </div>
                """
            )

        st.progress(
            max(
                0.0,
                min(
                    1.0,
                    value,
                ),
            )
        )

    # --------------------------------------------------------
    # EXPLAINABILITY
    # --------------------------------------------------------

    st.write("")

    render_html(
        """
        <div style="
            color:#0b2748;
            font-size:1.25rem;
            font-weight:900;
            margin-top:30px;
        ">
            🔎 Explainable AI
        </div>

        <div style="
            color:#6b7f95;
            font-size:0.84rem;
            margin-top:4px;
            margin-bottom:15px;
        ">
            Visual explanations generated using occlusion sensitivity and Grad-CAM.
        </div>
        """
    )

    with st.spinner(
        "🧠 Generating explainability maps..."
    ):

        explanation = explain_image(
            uploaded_file
        )

        gradcam_explanation = explain_gradcam_image(
            uploaded_file
        )

    # --------------------------------------------------------
    # OCCLUSION SENSITIVITY
    # --------------------------------------------------------

    if explanation:

        render_html(
            """
            <div style="
                background:white;
                border:1px solid #dbe7f3;
                border-radius:20px;
                padding:22px;
                box-shadow:
                    0 10px 28px rgba(20,50,90,0.07);
                margin-bottom:12px;
            ">

                <div style="
                    color:#0b2748;
                    font-size:1rem;
                    font-weight:850;
                ">
                    🧠 Occlusion Sensitivity Map
                </div>

                <div style="
                    color:#6b7f95;
                    font-size:0.8rem;
                    line-height:1.6;
                    margin-top:6px;
                ">
                    Highlighted regions represent areas where
                    masking parts of the image produced a stronger
                    change in the model's predicted probability.
                </div>

            </div>
            """
        )

        st.image(
            explanation,
            width="stretch",
        )

    else:

        st.warning(
            "Prediction succeeded, but the occlusion visualization "
            "could not be generated."
        )

    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    if gradcam_explanation:

        render_html(
            """
            <div style="
                background:white;
                border:1px solid #dbe7f3;
                border-radius:20px;
                padding:22px;
                box-shadow:
                    0 10px 28px rgba(20,50,90,0.07);
                margin-top:18px;
                margin-bottom:12px;
            ">

                <div style="
                    color:#0b2748;
                    font-size:1rem;
                    font-weight:850;
                ">
                    🔥 Grad-CAM Visualization
                </div>

                <div style="
                    color:#6b7f95;
                    font-size:0.8rem;
                    line-height:1.6;
                    margin-top:6px;
                ">
                    Gradient-based visualization showing the
                    image regions that contributed most strongly
                    to the model's predicted class.
                </div>

            </div>
            """
        )

        # API returns the complete 3-panel PNG:
        # Original CT Image | Grad-CAM | Prediction Overlay.
        # Streamlit renders the image natively; no image HTML is used.
        st.image(
            gradcam_explanation,
            width="stretch",
        )

    else:

        st.warning(
            "Grad-CAM visualization could not be generated."
        )



# ============================================================
# FOOTER
# ============================================================

render_html(
    """
    <div style="
        margin-top:55px;

        padding-top:22px;

        border-top:1px solid #d9e5f0;

        text-align:center;

        color:#7a8da2;

        font-size:0.76rem;

        line-height:1.8;
    ">

        <div style="
            color:#304b68;
            font-weight:850;
            font-size:0.86rem;
        ">
            KidneyVision AI
        </div>

        <div>
            Deep Learning · MLOps · Explainable AI
        </div>

        <div style="
            margin-top:8px;
        ">
            PyTorch · FastAPI · MLflow · Docker · Streamlit
        </div>

        <div style="
            margin-top:12px;
            color:#8a9aad;
        ">
            ⚠️ Academic/research prototype.
            Not intended for clinical diagnosis.
        </div>

    </div>
    """
)