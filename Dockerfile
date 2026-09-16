FROM python:3.13-slim

# Application directory
WORKDIR /app

# Prevent Python from creating .pyc files
# and make logs appear immediately in Docker
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy dependency file first for Docker layer caching
COPY requirements.txt .

# Install production dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and required production model
COPY . .

# Create a dedicated non-root user
RUN useradd \
    --create-home \
    --shell /bin/bash \
    appuser

# Give the application user read/write ownership
# of the application directory
RUN chown -R appuser:appuser /app

# Container uses the bundled production model
# instead of requiring the local MLflow database
ENV MODEL_SOURCE=local

# Docker automatically checks API liveness
HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=60s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

# FastAPI port
EXPOSE 8000

# Security: never run the application as root
USER appuser

# Start FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]