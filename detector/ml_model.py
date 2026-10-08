import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
_MODEL_PATH_SETTING = os.environ.get("ML_MODEL_PATH")
_APP_MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"
_WORKSPACE_MODEL_PATH = BASE_DIR.parent / "models" / "random_forest_model.pkl"
DEFAULT_MODEL_PATH = (
    Path(_MODEL_PATH_SETTING)
    if _MODEL_PATH_SETTING
    else _APP_MODEL_PATH if _APP_MODEL_PATH.is_file() else _WORKSPACE_MODEL_PATH
)

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


def train_random_forest(
    dataframe: pd.DataFrame,
    target_column: str,
    model_path: str | Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, float]:
    """Train, evaluate, and save a Random Forest pipeline from a labeled DataFrame."""
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' is not in the dataset.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    dataset = dataframe.dropna(subset=[target_column]).copy()
    if dataset.empty:
        raise ValueError("The dataset has no rows with a target value.")

    features = dataset.drop(columns=[target_column])
    target = dataset[target_column]
    if features.empty or target.nunique() < 2:
        raise ValueError("Training requires at least one feature and two target classes.")

    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.columns.difference(numeric_columns).tolist()
    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    stratify = target if target.value_counts().min() >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError as exc:
        raise ValueError(f"Could not create a train/test split: {exc}") from exc

    def build_pipeline() -> Pipeline:
        preprocessing = ColumnTransformer(transformers=transformers)
        return Pipeline(
            [
                ("preprocessing", preprocessing),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    evaluation_model = build_pipeline()
    evaluation_model.fit(X_train, y_train)
    predictions = evaluation_model.predict(X_test)
    positive_label = next(
        (
            label
            for label in evaluation_model.classes_
            if str(label).strip().lower()
            in {"1", "fraud", "fraudulent", "true", "yes"}
        ),
        evaluation_model.classes_[-1],
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, pos_label=positive_label, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, pos_label=positive_label, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, pos_label=positive_label, zero_division=0)
        ),
    }

    final_model = build_pipeline()
    final_model.fit(features, target)
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, path)
    _load_model_from_path.cache_clear()
    return metrics


def predict_transaction(data: Mapping[str, Any], model_path: str | Path | None = None):
    """Return a structured fraud evaluation for a transaction payload."""
    model = load_model(model_path)

    if not hasattr(model, "feature_names_in_"):
        raise ValueError("The saved model does not include its training feature names.")

    feature_names = [str(name) for name in model.feature_names_in_]
    missing_features = [name for name in feature_names if name not in data]
    if missing_features:
        raise ValueError(
            "The transaction form is missing model features: " + ", ".join(missing_features)
        )

    transaction = pd.DataFrame([{name: data[name] for name in feature_names}])
    prediction = model.predict(transaction)[0]
    probabilities = model.predict_proba(transaction)[0]
    classes = list(model.classes_)
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
