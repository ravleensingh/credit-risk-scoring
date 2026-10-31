from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.runtime import PROJECT_ROOT
from utils.preprocessing import (
    CATEGORICAL_COLS,
    FEATURE_ORDER,
    NUMERICAL_COLS,
    TARGET_COL,
    load_data,
    prepare_features,
    preprocess_data,
)


MODEL_ARTIFACT_PATH = PROJECT_ROOT / "models" / "credit_risk_pipeline.joblib"
ARTIFACT_VERSION = 2
MODEL_RANDOM_STATE = 42
MODEL_N_ESTIMATORS = 100
MODEL_MAX_DEPTH = 10
MODEL_TEST_SIZE = 0.2


@dataclass
class TrainingArtifacts:
    pipeline: Pipeline
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_pred: np.ndarray
    y_pred_proba: np.ndarray
    accuracy: float
    roc_auc: float


def _dataset_signature() -> dict[str, Any]:
    dataset_path = PROJECT_ROOT / "dataset" / "original_dataset.csv"
    stats = dataset_path.stat()
    return {
        "path": str(dataset_path),
        "size": stats.st_size,
        "mtime_ns": stats.st_mtime_ns,
    }


def _artifact_metadata() -> dict[str, Any]:
    return {
        "artifact_version": ARTIFACT_VERSION,
        "dataset_signature": _dataset_signature(),
        "feature_order": FEATURE_ORDER,
        "categorical_columns": CATEGORICAL_COLS,
        "numerical_columns": NUMERICAL_COLS,
        "model_type": "RandomForest",
        "n_estimators": MODEL_N_ESTIMATORS,
        "max_depth": MODEL_MAX_DEPTH,
        "random_state": MODEL_RANDOM_STATE,
    }


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
            ("numeric", StandardScaler(), NUMERICAL_COLS),
        ]
    )


def build_random_forest_pipeline(
    *,
    n_estimators: int = MODEL_N_ESTIMATORS,
    max_depth: int = MODEL_MAX_DEPTH,
    random_state: int = MODEL_RANDOM_STATE,
) -> Pipeline:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def get_train_test_data():
    df = load_data()
    df_processed, _ = preprocess_data(df)
    X, y = prepare_features(df_processed)
    return train_test_split(
        X,
        y,
        test_size=MODEL_TEST_SIZE,
        random_state=MODEL_RANDOM_STATE,
        stratify=y,
    )


def train_random_forest_pipeline(*, save_artifact: bool = False) -> TrainingArtifacts:
    X_train, X_test, y_train, y_test = get_train_test_data()
    pipeline = build_random_forest_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    if save_artifact:
        MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "metadata": _artifact_metadata(),
                "pipeline": pipeline,
            },
            MODEL_ARTIFACT_PATH,
        )

    return TrainingArtifacts(
        pipeline=pipeline,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        accuracy=accuracy,
        roc_auc=roc_auc,
    )


def load_or_train_pipeline() -> Pipeline:
    if MODEL_ARTIFACT_PATH.exists():
        try:
            artifact = joblib.load(MODEL_ARTIFACT_PATH)
        except Exception:
            artifact = None

        if isinstance(artifact, dict):
            metadata = artifact.get("metadata", {})
            pipeline = artifact.get("pipeline")
            if (
                pipeline is not None
                and metadata.get("artifact_version") == ARTIFACT_VERSION
                and metadata.get("dataset_signature") == _dataset_signature()
                and metadata.get("feature_order") == FEATURE_ORDER
            ):
                return pipeline

    artifacts = train_random_forest_pipeline(save_artifact=True)
    return artifacts.pipeline


def validate_prediction_input(input_data: dict[str, Any]) -> dict[str, Any]:
    missing_fields = [column for column in FEATURE_ORDER if column not in input_data]
    if missing_fields:
        raise ValueError(f"Missing prediction fields: {missing_fields}")

    normalized = {column: input_data[column] for column in FEATURE_ORDER}
    for column in CATEGORICAL_COLS:
        normalized[column] = str(normalized[column]).strip()

    return normalized


def predict_credit_risk(input_data: dict[str, Any], pipeline: Pipeline | None = None):
    model_pipeline = pipeline or load_or_train_pipeline()
    normalized_input = validate_prediction_input(input_data)
    input_frame = pd.DataFrame([[normalized_input[column] for column in FEATURE_ORDER]], columns=FEATURE_ORDER)

    prediction = int(model_pipeline.predict(input_frame)[0])
    probabilities = model_pipeline.predict_proba(input_frame)[0]
    return prediction, float(probabilities[1]), float(probabilities[0])


def _transformed_feature_mapping(preprocessor: ColumnTransformer) -> list[str]:
    categorical_encoder = preprocessor.named_transformers_["categorical"]
    categorical_names = categorical_encoder.get_feature_names_out(CATEGORICAL_COLS)

    mapping: list[str] = []
    for transformed_name in categorical_names:
        transformed_name = str(transformed_name)
        original_name = next(
            column for column in CATEGORICAL_COLS if transformed_name.startswith(f"{column}_")
        )
        mapping.append(original_name)

    mapping.extend(NUMERICAL_COLS)
    return mapping


def aggregate_feature_contributions(pipeline: Pipeline) -> pd.Series:
    preprocessor = pipeline.named_steps["preprocessor"]
    model: ClassifierMixin = pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        raw_contributions = np.asarray(model.feature_importances_, dtype=float)
    else:
        raw_contributions = np.abs(np.asarray(model.coef_[0], dtype=float))

    feature_mapping = _transformed_feature_mapping(preprocessor)
    aggregated = (
        pd.Series(raw_contributions, index=feature_mapping)
        .groupby(level=0)
        .sum()
        .reindex(FEATURE_ORDER)
        .sort_values(ascending=False)
    )
    return aggregated


def compute_permutation_feature_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    scoring: str = "roc_auc",
    n_repeats: int = 5,
) -> pd.Series:
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=MODEL_RANDOM_STATE,
        n_jobs=1,
    )
    importance = pd.Series(result.importances_mean, index=X_test.columns)
    importance = importance.clip(lower=0).sort_values(ascending=False)

    total_importance = float(importance.sum())
    if total_importance > 0:
        importance = importance / total_importance

    return importance
