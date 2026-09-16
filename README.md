# KidneyVision-MLOps

Production-style deep learning and MLOps pipeline for **4-class kidney disease classification from CT images**.

> **Important:** This project is an engineering/research prototype, not a clinical diagnostic system. The current local test set contains only 15 images, so the reported test result should not be interpreted as evidence of clinical or real-world generalization.

---

## 📌 Project Overview

KidneyVision-MLOps classifies kidney CT images into four categories:

- **Normal**
- **Cyst**
- **Stone**
- **Tumor**

The project combines deep learning, experiment tracking, model registry, explainable AI, API serving, frontend visualization, Docker, DVC, automated testing, and GitHub Actions CI/CD.

### End-to-end pipeline

```text
Kaggle Dataset
      │
      ▼
Dataset Validation
      │
      ▼
EDA + Data Quality Checks
      │
      ▼
Reproducible Train / Validation / Test Split
      │
      ▼
Image Preprocessing + Augmentation
      │
      ▼
7 Deep Learning Models
      │
      ├── VGG16
      ├── ResNet50
      ├── InceptionV3
      ├── EfficientNet-B0
      ├── EANet
      ├── CCT
      └── Swin Transformer
      │
      ▼
MLflow Experiment Tracking
      │
      ▼
Model Benchmarking
      │
      ▼
Swin Transformer
      │
      ├── Detailed Evaluation
      ├── MLflow Model Registry
      └── Grad-CAM Explainability
      │
      ▼
FastAPI Inference API
      │
      ▼
Streamlit Frontend
      │
      ▼
Docker / Docker Compose
      │
      ▼
GitHub Actions CI/CD
```

---

## 🏗️ Architecture

```text
                         ┌──────────────────────┐
                         │   CT Kidney Images   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Dataset Validation  │
                         │ + Data Quality      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ EDA + Data Analysis  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │ Train / Validation / Test    │
                    │        70 / 15 / 15          │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ Image Preprocessing           │
                    │ Resize + Augmentation         │
                    │ ImageNet Normalization       │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ 7 Model Benchmark             │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ MLflow Tracking + Registry    │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ Swin Transformer              │
                    │ Production Candidate          │
                    └──────────────┬───────────────┘
                                   │
                       ┌───────────┴───────────┐
                       ▼                       ▼
              ┌─────────────────┐     ┌─────────────────┐
              │ FastAPI         │     │ Grad-CAM        │
              │ Inference API   │     │ Explainability  │
              └────────┬────────┘     └─────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Streamlit       │
              │ Frontend        │
              └─────────────────┘
```

---

## 📂 Project Structure

```text
KidneyVision-MLOps/
│
├── api/
│   ├── __init__.py
│   ├── inference.py
│   └── main.py
│
├── artifacts/
│   ├── benchmark/
│   ├── checkpoints/
│   ├── deployment_model/
│   ├── evaluation/
│   ├── explainability/
│   └── mlflow.db
│
├── configs/
│   └── training_config.py
│
├── data/
│   ├── raw/
│   │   ├── images/
│   │   └── metadata/
│   ├── processed/
│   │   └── manifests/
│   └── reports/
│
├── frontend/
│   └── app.py
│
├── notebooks/
│
├── scripts/
│
├── src/
│   ├── data/
│   │   ├── validate_dataset.py
│   │   ├── eda.py
│   │   └── prepare_dataset.py
│   │
│   ├── evaluation/
│   │   └── evaluator.py
│   │
│   ├── explainability/
│   │   └── gradcam.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── model_factory.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── image_preprocessor.py
│   │
│   └── training/
│       ├── __init__.py
│       ├── train.py
│       └── benchmark.py
│
├── tests/
│   ├── fixtures/
│   └── test_api.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .dvc/
├── .gitignore
├── data/raw.dvc
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

> Large datasets, checkpoints, generated evaluation artifacts, MLflow databases, and deployment model files are intentionally excluded from Git tracking where appropriate.

---

## 🧬 Dataset

The project uses a kidney CT image dataset with metadata containing:

```text
Unnamed: 0
image_id
path
diag
target
Class
```

The metadata contains **12,446 records**.

The currently uploaded/verified local image subset contains **100 images**, distributed equally across the four classes:

| Class | Local images |
|---|---:|
| Normal | 25 |
| Cyst | 25 |
| Stone | 25 |
| Tumor | 25 |
| **Total** | **100** |

The metadata distribution is different from the local 100-image subset:

| Class | Metadata records |
|---|---:|
| Normal | 5,077 |
| Cyst | 3,709 |
| Tumor | 2,283 |
| Stone | 1,377 |
| **Total** | **12,446** |

### Dataset quality checks

The validation pipeline checks:

- Required CSV columns
- Missing values
- Duplicate records
- Duplicate image IDs
- Duplicate paths
- Corrupt images
- Image/CSV mapping
- Folder-label vs CSV-label mismatches
- Exact duplicate images
- Image dimensions
- Train/validation/test overlap

For the verified 100-image subset:

- 100/100 images mapped to metadata
- 0 corrupt images
- 0 duplicate image hashes
- 0 label mismatches
- 0 cross-split overlap

---

## 🔬 Exploratory Data Analysis

The EDA pipeline generates:

```text
data/reports/class_distribution.png
data/reports/image_dimensions.png
data/reports/pixel_intensity_distribution.png
data/reports/sample_images.png
data/reports/eda_report.json
```

The verified image subset contains RGB JPEG images.

Pixel statistics and image dimensions are reported by the EDA script rather than assumed in advance.

---

## ✂️ Dataset Preparation

A reproducible stratified split is generated using:

```text
Random seed: 42

Training:   70%
Validation: 15%
Testing:    15%
```

For the current 100-image subset:

```text
Train:       70
Validation: 15
Test:        15
Total:      100
```

Generated manifests:

```text
data/processed/manifests/train.csv
data/processed/manifests/validation.csv
data/processed/manifests/test.csv
data/processed/split_summary.json
```

---

## 🖼️ Image Preprocessing

The PyTorch preprocessing pipeline uses:

### Training

- Resize to `224 × 224`
- Random affine augmentation
- Small rotation
- Small translation
- Small scale variation
- Tensor conversion
- ImageNet normalization

### Validation / Test

- Resize to `224 × 224`
- Tensor conversion
- ImageNet normalization

InceptionV3 uses its required larger input resolution.

---

## 🤖 Deep Learning Models

Seven architectures are included in the benchmark:

1. VGG16
2. ResNet50
3. InceptionV3
4. EfficientNet-B0
5. EANet
6. CCT
7. Swin Transformer

### Research basis

Six architectures were taken from the primary research-paper direction:

- VGG16
- ResNet50
- InceptionV3
- EANet
- CCT
- Swin Transformer

**EfficientNet-B0 was added separately as an additional benchmark model and was not claimed as one of the models from the primary paper.**

The EANet and CCT implementations in this project are project-level implementations inspired by the corresponding concepts; they should not be described as exact reproductions of the original research implementations.

---

## 📊 Model Benchmarking

All seven models can be trained sequentially through:

```text
src/training/benchmark.py
```

Benchmark outputs are stored under:

```text
artifacts/benchmark/
```

The benchmark is intended for engineering comparison. Results can vary with:

- dataset size
- random seed
- hardware
- training duration
- preprocessing
- initialization
- hyperparameters

The current benchmark used a small local image subset and a short training schedule, so it should not be treated as a definitive scientific comparison.

---

## 🧠 Final Model

The current production candidate is:

**Swin Transformer**

The final local training run used:

```text
Epochs:          10
Batch size:      16
Learning rate:   0.0001
Weight decay:    0.0001
Optimizer:       AdamW
Scheduler:       ReduceLROnPlateau
Early stopping: enabled
Random seed:     42
```

The best validation checkpoint was obtained at **epoch 9**.

Checkpoint:

```text
artifacts/checkpoints/swintransformer_best.pt
```

---

## 📈 Evaluation

The detailed evaluator reports:

- Accuracy
- Precision
- Recall / Sensitivity
- Specificity
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix
- Per-class metrics
- ROC curve
- Precision-Recall curve
- Classification report

### Current local evaluation

On the current 15-image test split:

```text
Accuracy          : 93.33%
Macro Precision   : 95.00%
Macro Recall      : 93.75%
Macro Specificity : 97.73%
Macro F1          : 93.65%
Macro ROC-AUC     : 1.00
Macro PR-AUC      : 1.00
```

Confusion matrix:

```text
              Predicted
             N   C   S   T

Actual N     4   0   0   0
       C     0   3   1   0
       S     0   0   4   0
       T     0   0   0   3
```

### ⚠️ Evaluation limitation

The test set contains only **15 images**.

Therefore:

```text
93.33% = 14 / 15 correct predictions
```

This result is useful for validating the project pipeline, but it is **not sufficient to establish clinical performance, safety, or real-world generalization**.

A larger independent holdout dataset should be used for meaningful external evaluation.

---

## 🧪 MLflow

MLflow is used for:

- Experiment tracking
- Parameters
- Training metrics
- Validation metrics
- Test metrics
- Checkpoints
- Model logging
- Model Registry

Local MLflow backend:

```text
artifacts/mlflow.db
```

Start the MLflow UI:

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///artifacts/mlflow.db --host 127.0.0.1 --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

### Registered model

```text
KidneyVision-SwinTransformer
```

Current alias:

```text
champion
```

The deployment container uses a bundled model artifact so that Docker does not depend on the developer's local MLflow database.

---

## 🔎 Explainable AI — Grad-CAM

Grad-CAM is integrated to provide a visual explanation of the model's prediction.

Endpoint:

```text
POST /explain
```

The API returns a Grad-CAM visualization showing:

- Original CT image
- Activation heatmap
- Overlay

Example output:

```text
artifacts/explainability/gradcam/gradcam_example.png
```

Grad-CAM should be interpreted as a model-attribution visualization, not as a medically validated segmentation or diagnostic explanation.

---

## 🚀 FastAPI

The inference API is implemented using FastAPI.

### Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Liveness status |
| GET | `/ready` | Model readiness |
| GET | `/model-info` | Model metadata |
| POST | `/predict` | CT classification |
| POST | `/explain` | Grad-CAM explanation |

### Example health response

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Example model information

```json
{
  "model": "SwinTransformer",
  "registered_model": "KidneyVision-SwinTransformer",
  "alias": "champion",
  "model_source": "local",
  "classes": [
    "Normal",
    "Cyst",
    "Stone",
    "Tumor"
  ],
  "image_size": 224,
  "device": "cpu",
  "status": "loaded"
}
```

---

## 🎨 Streamlit Frontend

The Streamlit application provides:

- CT image upload
- API health display
- Predicted class
- Prediction confidence
- Class probabilities
- Grad-CAM visualization
- Raw API response

Run locally:

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

Open:

```text
http://localhost:8501
```

---

## 🐳 Docker

The API is containerized using Docker.

Build:

```powershell
docker build -t kidneyvision-api:local .
```

Run:

```powershell
docker run --rm -p 8000:8000 --name kidneyvision-api kidneyvision-api:local
```

API:

```text
http://localhost:8000
```

Health check:

```text
http://localhost:8000/health
```

The Docker image uses the bundled deployment model rather than requiring the local MLflow Registry database.

---

## 🐳 Docker Compose

Run the complete local application:

```powershell
docker compose up --build
```

Services:

```text
API       → http://localhost:8000
Frontend  → http://localhost:8501
```

Stop:

```powershell
docker compose down
```

Architecture:

```text
Browser
   │
   ▼
Streamlit :8501
   │
   │ HTTP
   ▼
FastAPI :8000
   │
   ▼
Swin Transformer
   │
   ├── Prediction
   └── Grad-CAM
```

---

## 🧪 Testing

The project contains automated tests for the API and supporting project functionality.

Run all tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Current local test status:

```text
31 passed
7 warnings
```

The warnings are dependency/framework warnings and did not cause test failures.

---

## 🔄 DVC

DVC is used for dataset versioning.

The raw dataset is tracked through:

```text
data/raw.dvc
```

The configured remote is Google Drive.

Typical commands:

```powershell
dvc status
dvc pull
dvc push
```

The actual dataset is intentionally not committed directly to GitHub.

---

## 🔐 Git and Security

Sensitive/local files are excluded using `.gitignore`, including:

```text
.env
.dvc/config.local
.venv/
data/raw/
*.zip
*.pt
*.pth
artifacts/mlflow.db
```

DVC's local credential configuration is not committed.

Large datasets and trained model binaries are kept outside normal Git source tracking.

---

## ⚙️ GitHub Actions CI/CD

The repository includes:

```text
.github/workflows/ci.yml
```

The CI pipeline:

1. Checks out the repository
2. Installs Python dependencies
3. Creates a small CI-only synthetic dataset
4. Prepares CI manifests
5. Verifies the project structure
6. Runs the automated test suite
7. Builds the Docker image

The synthetic dataset is used only inside GitHub Actions so the real dataset does not need to be uploaded to GitHub.

---

## 🛠️ Local Setup

### 1. Clone the repository

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd KidneyVision-MLOps
```

### 2. Create/activate virtual environment

```powershell
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare dataset

Place the dataset according to the project structure and run:

```powershell
python src/data/validate_dataset.py
python src/data/eda.py
python src/data/prepare_dataset.py
```

### 5. Run tests

```powershell
python -m pytest tests/ -v
```

### 6. Start API

```powershell
uvicorn api.main:app --reload
```

### 7. Start frontend

```powershell
streamlit run frontend/app.py
```

---

## 🧑‍💻 Training

Train an individual model:

```powershell
python src/training/train.py --model SwinTransformer --epochs 10 --batch-size 16 --learning-rate 0.0001
```

Available models:

```text
VGG16
ResNet50
InceptionV3
EfficientNet-B0
EANet
CCT
SwinTransformer
```

Run benchmark:

```powershell
python src/training/benchmark.py
```

---

## 📁 Important Artifacts

| Artifact | Location |
|---|---|
| Dataset validation report | `data/reports/dataset_validation_report.json` |
| EDA report | `data/reports/eda_report.json` |
| Train manifest | `data/processed/manifests/train.csv` |
| Validation manifest | `data/processed/manifests/validation.csv` |
| Test manifest | `data/processed/manifests/test.csv` |
| MLflow database | `artifacts/mlflow.db` |
| Swin checkpoint | `artifacts/checkpoints/swintransformer_best.pt` |
| Evaluation results | `artifacts/evaluation/swintransformer/` |
| Grad-CAM example | `artifacts/explainability/gradcam/gradcam_example.png` |
| Deployment model | `artifacts/deployment_model/` |

---

## 📚 Research Basis

The project is based primarily on the research direction of:

**"Vision transformer and explainable transfer learning models for auto detection of kidney cyst, stone and tumor from CT-radiography"**

Scientific Reports, 2022.

The project uses the paper's model direction and explainability concept as a research basis while adding an independent MLOps/production pipeline.

The reported paper results should **not** be confused with this project's own results. This project does not claim to reproduce the paper's reported performance.

---

## 🔮 Future Scope

Potential next-stage improvements:

- Larger independent test dataset
- External validation
- Patient-level split to reduce leakage risk
- Cross-validation
- Class imbalance strategies on the full dataset
- Hyperparameter optimization
- More robust calibration analysis
- Model monitoring
- Data drift detection
- Automated model retraining
- Cloud deployment
- HTTPS and authentication
- Container orchestration
- Production observability
- Medical expert validation
- Prospective clinical evaluation

---

## ⚠️ Limitations

1. The currently verified local image subset is small.
2. The final test split contains only 15 images.
3. The current evaluation is not sufficient for clinical conclusions.
4. Image-level performance may not represent patient-level performance.
5. The model has not undergone prospective clinical validation.
6. Grad-CAM is an attribution method and should not be interpreted as a definitive medical explanation.
7. The custom EANet/CCT implementations are project-level implementations rather than guaranteed exact reproductions of original research code.
8. Results depend on the dataset, preprocessing, random seed, training configuration, and hardware.

---

## 👨‍💻 Project Goals

This project demonstrates an end-to-end workflow covering:

```text
Deep Learning
      +
Computer Vision
      +
Medical Image Classification
      +
Experiment Tracking
      +
Model Registry
      +
Explainable AI
      +
REST API
      +
Frontend
      +
Docker
      +
DVC
      +
Automated Testing
      +
CI/CD
```

---

## 📜 License

Add the appropriate license for your project and verify that the underlying dataset's license permits the intended use and redistribution.
