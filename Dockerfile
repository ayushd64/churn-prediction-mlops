# ---- Base image: a lean, official Python 3.12 ----
FROM python:3.12-slim

# ---- Working directory inside the container ----
WORKDIR /app

# ---- Install dependencies FIRST (for Docker layer caching) ----
COPY requirements-serving.txt .
RUN pip install --upgrade pip && pip install --default-timeout=1000 --no-cache-dir -r requirements-serving.txt

# ---- Copy app code, config, and the exported champion model ----
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/champion_model/ ./models/champion_model/

# ---- Document the port the API listens on ----
EXPOSE 8000

# ---- Launch the API (0.0.0.0 = reachable from outside the container) ----
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
