from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "latest_model.joblib"
DEFAULT_METADATA_PATH = MODEL_DIR / "latest_model.json"

SUPPORTED_DATASET_EXTENSIONS = {".csv", ".txt", ".json"}
DEFAULT_TARGET_CANDIDATES = ("target", "label", "class", "output", "outcome", "y")
DEFAULT_TASK_TYPES = {"auto", "classification", "regression"}


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _normalize_name(value):
    return str(value).strip().lower().replace(" ", "").replace("_", "")


def _resolve_column_name(columns: Iterable[str], requested: str | None):
    if requested is None:
        return None

    lookup = {_normalize_name(column): column for column in columns}
    key = _normalize_name(requested)
    if key in lookup:
        return lookup[key]

    if requested in columns:
        return requested

    raise ValueError(f"Column '{requested}' was not found in the dataset.")


def parse_name_list(raw):
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        return [str(item).strip() for item in raw if str(item).strip()]
    if not isinstance(raw, str):
        return [str(raw).strip()] if str(raw).strip() else []

    value = raw.strip()
    if not value:
        return []

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]

    return [part.strip() for part in value.split(",") if part.strip()]


def ensure_model_dir():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_tabular_dataframe(source, filename=None):
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_DATASET_EXTENSIONS:
            raise ValueError(
                f"Unsupported dataset format '{suffix}'. Use CSV or JSON tabular data."
            )
        if suffix in {".csv", ".txt"}:
            return pd.read_csv(path)
        return pd.read_json(path)

    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_DATASET_EXTENSIONS:
        suffix = ".csv"

    if hasattr(source, "seek"):
        try:
            source.seek(0)
        except Exception:
            pass

    if suffix in {".csv", ".txt"}:
        return pd.read_csv(source)

    if hasattr(source, "read"):
        raw = source.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return pd.read_json(io.StringIO(raw))

    raise ValueError("Unable to read the supplied dataset.")


def _expand_datetime_features(frame):
    expanded = frame.copy()
    datetime_columns = []

    for column in list(expanded.columns):
        series = expanded[column]
        parsed = None

        if pd.api.types.is_datetime64_any_dtype(series):
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            candidate = pd.to_datetime(series, errors="coerce", format="mixed")
            valid_ratio = float(candidate.notna().mean()) if len(candidate) else 0.0
            if candidate.notna().sum() >= 5 and valid_ratio >= 0.8:
                parsed = candidate

        if parsed is None:
            continue

        expanded[f"{column}__year"] = parsed.dt.year
        expanded[f"{column}__month"] = parsed.dt.month
        expanded[f"{column}__day"] = parsed.dt.day
        expanded[f"{column}__dayofweek"] = parsed.dt.dayofweek
        expanded[f"{column}__quarter"] = parsed.dt.quarter
        expanded[f"{column}__hour"] = parsed.dt.hour
        expanded = expanded.drop(columns=[column])
        datetime_columns.append(column)

    return expanded, datetime_columns


def prepare_feature_frame(frame, drop_columns=None, datetime_columns=None):
    working = frame.copy()
    drop_columns = drop_columns or []

    resolved_drop_columns = []
    for column in drop_columns:
        resolved = _resolve_column_name(working.columns, column)
        if resolved is not None and resolved in working.columns:
            resolved_drop_columns.append(resolved)

    if resolved_drop_columns:
        working = working.drop(columns=resolved_drop_columns, errors="ignore")

    working = working.loc[:, ~working.columns.duplicated()]
    working = working.dropna(axis=1, how="all")

    if datetime_columns:
        datetime_frame = working.copy()
        for column in datetime_columns:
            try:
                resolved = _resolve_column_name(datetime_frame.columns, column)
            except ValueError:
                continue
            if resolved is None or resolved not in datetime_frame.columns:
                continue
            parsed = pd.to_datetime(datetime_frame[resolved], errors="coerce", format="mixed")
            datetime_frame[f"{resolved}__year"] = parsed.dt.year
            datetime_frame[f"{resolved}__month"] = parsed.dt.month
            datetime_frame[f"{resolved}__day"] = parsed.dt.day
            datetime_frame[f"{resolved}__dayofweek"] = parsed.dt.dayofweek
            datetime_frame[f"{resolved}__quarter"] = parsed.dt.quarter
            datetime_frame[f"{resolved}__hour"] = parsed.dt.hour
            datetime_frame = datetime_frame.drop(columns=[resolved])
        working = datetime_frame
    else:
        working, datetime_columns = _expand_datetime_features(working)

    working = working.dropna(axis=1, how="all")
    return working, datetime_columns or []


def infer_target_column(frame, target_column=None):
    if target_column:
        return _resolve_column_name(frame.columns, target_column)

    lookup = {_normalize_name(column): column for column in frame.columns}
    for candidate in DEFAULT_TARGET_CANDIDATES:
        key = _normalize_name(candidate)
        if key in lookup:
            return lookup[key]

    if frame.columns.empty:
        raise ValueError("The dataset does not contain any columns.")

    return frame.columns[-1]


def infer_task_type(target_series, requested_task="auto"):
    if requested_task is None:
        requested_task = "auto"
    requested_task = str(requested_task).strip().lower()
    if requested_task not in DEFAULT_TASK_TYPES:
        raise ValueError(
            "taskType must be one of: auto, classification, regression"
        )

    if requested_task != "auto":
        return requested_task

    if pd.api.types.is_bool_dtype(target_series):
        return "classification"

    if pd.api.types.is_numeric_dtype(target_series):
        non_null = target_series.dropna()
        unique_count = int(non_null.nunique())
        total_count = int(len(non_null))
        if unique_count <= 20 and unique_count <= max(10, total_count // 5):
            return "classification"
        if pd.api.types.is_integer_dtype(non_null) and unique_count <= 50:
            return "classification"
        return "regression"

    return "classification"


def _build_preprocessor(frame):
    numeric_columns = frame.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [column for column in frame.columns if column not in numeric_columns]

    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError("No usable feature columns remain after preprocessing.")

    return ColumnTransformer(transformers=transformers, remainder="drop"), numeric_columns, categorical_columns


def _build_model(task_type):
    if task_type == "classification":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=1,
            class_weight="balanced_subsample",
        )

    return RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=1,
    )


def _split_train_test(frame, target, task_type, test_size, random_state):
    if len(frame) < 6:
        return None

    stratify = None
    if task_type == "classification":
        counts = target.value_counts(dropna=False)
        if len(counts) > 1 and int(counts.min()) >= 2:
            stratify = target

    try:
        return train_test_split(
            frame,
            target,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(
            frame,
            target,
            test_size=test_size,
            random_state=random_state,
        )


def _evaluate(task_type, y_true, y_pred):
    if task_type == "regression":
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
            "r2": float(r2_score(y_true, y_pred)),
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precisionWeighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recallWeighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1Weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _feature_importance_summary(pipeline, top_n=10):
    model = pipeline.named_steps.get("model")
    if model is None or not hasattr(model, "feature_importances_"):
        return []

    try:
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{index}" for index in range(len(model.feature_importances_))]

    ranked = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {"feature": name, "importance": float(score)}
        for name, score in ranked[:top_n]
    ]


def save_model_artifact(artifact, model_path=DEFAULT_MODEL_PATH, metadata_path=DEFAULT_METADATA_PATH):
    ensure_model_dir()
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    if metadata_path is None:
        metadata_path = DEFAULT_METADATA_PATH
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact, model_path)
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(artifact["metadata"], fh, indent=2, default=_json_default)

    return model_path, metadata_path


def load_model_artifact(model_path=DEFAULT_MODEL_PATH):
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Train one with POST /train first."
        )

    artifact = joblib.load(model_path)
    if isinstance(artifact, dict) and "pipeline" in artifact:
        return artifact

    return {"pipeline": artifact, "metadata": {}}


def load_model_metadata(metadata_path=DEFAULT_METADATA_PATH):
    if metadata_path is None:
        metadata_path = DEFAULT_METADATA_PATH
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        return None

    with open(metadata_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def latest_model_status():
    metadata = load_model_metadata(DEFAULT_METADATA_PATH)
    model_path = Path(DEFAULT_MODEL_PATH)
    trained = metadata is not None and model_path.exists()

    if not trained:
        return {
            "trained": False,
            "modelPath": str(model_path),
            "message": "No trained model is available yet. Upload a dataset to /train.",
        }

    return {
        "trained": True,
        "modelPath": str(model_path),
        "metadataPath": str(DEFAULT_METADATA_PATH),
        "metadata": metadata,
    }


def train_tabular_model(
    frame,
    target_column=None,
    task_type="auto",
    drop_columns=None,
    test_size=0.2,
    random_state=42,
    model_path=DEFAULT_MODEL_PATH,
    metadata_path=DEFAULT_METADATA_PATH,
):
    working = frame.copy()

    if working.columns.duplicated().any():
        raise ValueError("Dataset contains duplicate column names.")

    target_name = infer_target_column(working, target_column)
    if target_name is None:
        raise ValueError("Target column could not be resolved.")

    if target_name not in working.columns:
        raise ValueError(f"Target column '{target_name}' was not found in the dataset.")

    drop_columns = parse_name_list(drop_columns)
    resolved_drops = []
    for column in drop_columns:
        resolved = _resolve_column_name(working.columns, column)
        if resolved == target_name:
            raise ValueError("The target column cannot also be dropped.")
        if resolved is not None:
            resolved_drops.append(resolved)

    if resolved_drops:
        working = working.drop(columns=resolved_drops, errors="ignore")

    working = working.dropna(subset=[target_name]).copy()
    if working.empty:
        raise ValueError("No usable rows remain after removing missing target values.")

    resolved_task_type = infer_task_type(working[target_name], task_type)
    if resolved_task_type == "regression":
        working[target_name] = pd.to_numeric(working[target_name], errors="coerce")
        working = working.dropna(subset=[target_name]).copy()
        if working.empty:
            raise ValueError("The target column could not be converted to numeric values.")

    feature_frame = working.drop(columns=[target_name]).copy()
    feature_frame, datetime_columns = prepare_feature_frame(feature_frame)

    if feature_frame.empty:
        raise ValueError("No feature columns remain after preprocessing.")

    feature_columns = feature_frame.columns.tolist()
    if resolved_task_type == "classification" and working[target_name].nunique(dropna=True) < 2:
        raise ValueError("Classification targets must contain at least two distinct values.")

    split = _split_train_test(
        feature_frame,
        working[target_name],
        resolved_task_type,
        test_size,
        random_state,
    )

    preprocessor, numeric_columns, categorical_columns = _build_preprocessor(feature_frame)
    model = _build_model(resolved_task_type)
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    if split is not None:
        x_train, x_test, y_train, y_test = split
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        evaluation = _evaluate(resolved_task_type, y_test, predictions)
        evaluation["validationRows"] = int(len(x_test))
    else:
        pipeline.fit(feature_frame, working[target_name])
        evaluation = {
            "note": "Evaluation skipped because the dataset is too small for a holdout split.",
        }

    pipeline.fit(feature_frame, working[target_name])

    classes = []
    if resolved_task_type == "classification" and hasattr(pipeline.named_steps["model"], "classes_"):
        classes = [value.item() if isinstance(value, np.generic) else value for value in pipeline.named_steps["model"].classes_]

    metadata = {
        "trainedAt": datetime.now(timezone.utc).isoformat(),
        "taskType": resolved_task_type,
        "targetColumn": target_name,
        "featureColumns": feature_columns,
        "datetimeColumns": datetime_columns,
        "numericColumns": numeric_columns,
        "categoricalColumns": categorical_columns,
        "droppedColumns": resolved_drops,
        "rowCount": int(len(working)),
        "featureCount": int(len(feature_columns)),
        "classes": classes,
        "evaluation": evaluation,
        "topFeatures": _feature_importance_summary(pipeline),
    }

    artifact = {
        "pipeline": pipeline,
        "metadata": metadata,
    }

    model_path, metadata_path = save_model_artifact(artifact, model_path, metadata_path)

    return {
        "status": "trained",
        "modelPath": str(model_path),
        "metadataPath": str(metadata_path),
        "taskType": resolved_task_type,
        "targetColumn": target_name,
        "featureColumns": feature_columns,
        "datetimeColumns": datetime_columns,
        "rowCount": int(len(working)),
        "featureCount": int(len(feature_columns)),
        "evaluation": evaluation,
        "topFeatures": metadata["topFeatures"],
    }


def predict_dataframe(frame, model_path=DEFAULT_MODEL_PATH):
    artifact = load_model_artifact(model_path)
    pipeline = artifact["pipeline"]
    metadata = artifact.get("metadata", {})
    feature_columns = metadata.get("featureColumns") or list(getattr(pipeline, "feature_names_in_", []))
    datetime_columns = metadata.get("datetimeColumns") or []

    if not feature_columns:
        raise ValueError("The trained model is missing feature metadata.")

    feature_frame, _ = prepare_feature_frame(frame, datetime_columns=datetime_columns)

    for column in feature_columns:
        if column not in feature_frame.columns:
            feature_frame[column] = np.nan

    feature_frame = feature_frame.loc[:, feature_columns]
    predictions = pipeline.predict(feature_frame)

    response = {
        "modelPath": str(Path(model_path)),
        "taskType": metadata.get("taskType", "unknown"),
        "targetColumn": metadata.get("targetColumn"),
        "inputRows": int(len(feature_frame)),
        "predictions": [
            value.item() if isinstance(value, np.generic) else value
            for value in np.asarray(predictions).tolist()
        ],
    }

    if metadata.get("taskType") == "classification" and hasattr(pipeline.named_steps["model"], "predict_proba"):
        probabilities = pipeline.predict_proba(feature_frame)
        classes = metadata.get("classes") or [
            value.item() if isinstance(value, np.generic) else value
            for value in pipeline.named_steps["model"].classes_
        ]
        response["classes"] = classes
        response["probabilities"] = [
            {str(classes[index]): float(row[index]) for index in range(len(classes))}
            for row in probabilities
        ]

    return response


def predict_records(records, model_path=DEFAULT_MODEL_PATH):
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list) or not records:
        raise ValueError("Provide one record or a non-empty list of records for prediction.")

    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("Prediction data is empty.")

    return predict_dataframe(frame, model_path=model_path)
