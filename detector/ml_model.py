import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = Path(
    os.environ.get("ML_MODEL_PATH", BASE_DIR / "models" / "random_forest_model.pkl")
)

FEATURE_NAMES = [
    "amount",
    "transaction_hour",
    "merchant_risk",
    "device_risk",
    "location_risk",
    "customer_tenure_days",
    "avg_transaction_amount",
    "ip_risk",
]


@lru_cache(maxsize=1)
def _load_model_from_path(model_path: str):
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained Random Forest model not found at {path}."
        )
    return joblib.load(path)


def load_model(model_path: str | Path | None = None):
    """Load and cache the trained Random Forest artifact; never train per request."""
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    return _load_model_from_path(str(path.resolve()))


def predict_transaction(data: Mapping[str, Any], model_path: str | Path | None = None):
    """Return a structured fraud evaluation for a transaction payload."""
    model = load_model(model_path)

    if hasattr(model, "feature_names_in_"):
        feature_names = [str(name) for name in model.feature_names_in_]
    else:
        feature_names = FEATURE_NAMES

    features = [float(data.get(name, 0.0)) for name in feature_names]
    prediction = model.predict([features])[0]
    probabilities = model.predict_proba([features])[0]
    classes = list(getattr(model, "classes_", []))
    fraud_class_index = next(
        (
            index
            for index, label in enumerate(classes)
            if str(label).strip().lower() in {"1", "fraud", "fraudulent", "true", "yes"}
        ),
        1 if len(probabilities) > 1 else 0,
    )
    risk_score = float(probabilities[fraud_class_index])
    prediction_is_fraud = str(prediction).strip().lower() in {
        "1", "fraud", "fraudulent", "true", "yes"
    }

    return {
        "prediction": prediction.item() if hasattr(prediction, "item") else prediction,
        "label": "Fraudulent" if prediction_is_fraud else "Legitimate",
        "risk_score": round(risk_score, 4),
    }
