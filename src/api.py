"""
Model serving API.

A FastAPI service that loads the champion churn model from the MLflow Model
Registry and exposes /health and /predict endpoints. Raw customer features
come in as JSON; a churn prediction and probability go back out.
"""

from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.config import load_config
from src.logger import get_logger
from src.preprocessing import clean_data
from pathlib import Path

logger = get_logger(__name__)

# Simple in-memory store for the loaded model (populated at startup).
ml_models = {}


class CustomerData(BaseModel):
    """Schema for one customer's raw features (the /predict request body)."""
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class PredictionResponse(BaseModel):
    """Schema for the /predict response body."""
    churn: bool
    churn_label: str
    churn_probability: float


# @asynccontextmanager
# def lifespan(app: FastAPI):
    # """Load the champion model once when the API starts up."""
    # config = load_config()
    # mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    # model_uri = f"models:/{config['registry']['model_name']}@{config['registry']['champion_alias']}"
    # logger.info(f"Loading model from {model_uri}")
    # ml_models["model"] = mlflow.sklearn.load_model(model_uri)
    # ml_models["id_column"] = config["columns"]["id_column"]
    # logger.info("Model loaded and ready ✅")

    # yield  # <-- API serves requests while suspended here

    # ml_models.clear()  # cleanup on shutdown
    # config = load_config()

    # Prefer the locally exported model (used inside the Docker container, which
    # has no MLflow backend). Fall back to the registry for local development.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the champion model once when the API starts up."""
    config = load_config()

    # Prefer the locally exported model (used inside the Docker container, which
    # has no MLflow backend). Fall back to the registry for local development.
    local_path = Path(config["serving"]["model_path"])
    if local_path.exists():
        logger.info(f"Loading model from local export: {local_path}")
        ml_models["model"] = mlflow.sklearn.load_model(str(local_path))
    else:
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        model_uri = f"models:/{config['registry']['model_name']}@{config['registry']['champion_alias']}"
        logger.info(f"Local export not found — loading from registry: {model_uri}")
        ml_models["model"] = mlflow.sklearn.load_model(model_uri)

    ml_models["id_column"] = config["columns"]["id_column"]
    logger.info("Model loaded and ready ✅")

    yield  # <-- API serves requests while suspended here

    ml_models.clear()  # cleanup on shutdown



app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    """Liveness/readiness check."""
    return {"status": "ok", "model_loaded": "model" in ml_models}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    """Predict churn for a single customer."""
    model = ml_models["model"]

    # Pydantic object -> single-row DataFrame, then apply the same cleaning as training.
    df = pd.DataFrame([customer.model_dump()])
    df = clean_data(df, ml_models["id_column"])

    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0, 1])

    return PredictionResponse(
        churn=bool(prediction),
        churn_label="Churn" if prediction else "No Churn",
        churn_probability=round(probability, 4),
    )
