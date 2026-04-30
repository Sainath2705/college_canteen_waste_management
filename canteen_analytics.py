from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml_service import load_tabular_dataframe

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
CANTEEN_MODEL_PATH = MODEL_DIR / "canteen_intelligence.joblib"
CANTEEN_METADATA_PATH = MODEL_DIR / "canteen_intelligence.json"

DEFAULT_TARGETS = ("totalStudents", "riceSold", "dosaSold", "snacksSold")
DEFAULT_LAG_WINDOWS = (1, 3, 7)
DEFAULT_ROLLING_WINDOWS = (3, 7)

FOOTFALL_HINTS = (
    "totalstudents",
    "footfall",
    "attendance",
    "headcount",
    "visitor",
    "visitors",
    "customer",
    "customers",
    "walkin",
    "walkins",
    "entry",
    "entries",
)

DEMAND_HINTS = (
    "sold",
    "sale",
    "sales",
    "demand",
    "served",
    "consumed",
    "order",
    "orders",
    "qty",
    "quantity",
)

PREP_HINTS = ("prepared", "prep", "production")

EXCLUDED_TARGET_HINTS = (
    "waste",
    "profit",
    "cost",
    "revenue",
    "margin",
    "inventory",
    "stock",
    "temperature",
    "humidity",
    "weather",
    "event",
    "date",
    "day",
    "month",
    "year",
    "time",
    "slot",
    "campus",
    "location",
    "status",
    "rating",
    "score",
    "price",
)

EXACT_NUMERIC_EXCLUDES = {
    "month",
    "quarter",
    "dayofmonth",
    "dayofyear",
    "dayofweeknumber",
    "weekofyear",
    "dayssincestart",
    "isweekend",
    "hour",
    "minute",
    "second",
}


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


def _make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _normalize_name(value):
    return str(value).strip().lower().replace(" ", "").replace("_", "")


def _pretty_name(value):
    text = str(value).replace("_", " ").replace("-", " ").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(
        r"(?i)(?:\s*(sold|sales|demand|served|consumed|orders?|quantity|qty|prepared|prep))+$",
        "",
        text,
    ).strip()
    return text.title() if text else str(value).title()


def frame_fingerprint(frame):
    if frame is None or frame.empty:
        return "empty"

    sample = frame.copy()
    sample = sample.loc[:, list(sample.columns)]
    head = sample.head(100).fillna("<NA>").astype(str)
    digest = hashlib.sha256()
    digest.update(head.to_csv(index=False).encode("utf-8"))
    digest.update(str(sample.shape).encode("utf-8"))
    digest.update("|".join(sample.columns.astype(str)).encode("utf-8"))
    return digest.hexdigest()


def _resolve_column(columns, requested):
    if requested is None:
        return None
    lookup = {_normalize_name(column): column for column in columns}
    key = _normalize_name(requested)
    return lookup.get(key, requested if requested in columns else None)


def _infer_date_column(frame):
    for candidate in ("date", "day", "transaction_date", "datetime"):
        resolved = _resolve_column(frame.columns, candidate)
        if resolved is not None:
            return resolved
    return None


def _is_footfall_target(column_name):
    normalized = _normalize_name(column_name)
    return any(hint in normalized for hint in FOOTFALL_HINTS)


def _is_excluded_target(column_name):
    normalized = _normalize_name(column_name)
    if normalized in EXACT_NUMERIC_EXCLUDES:
        return True
    return any(hint in normalized for hint in EXCLUDED_TARGET_HINTS)


def _target_group_key(column_name):
    normalized = _normalize_name(column_name)
    for suffix in ("sold", "sales", "demand", "served", "consumed", "orders", "order", "qty", "quantity", "prepared", "prep"):
        if normalized.endswith(suffix):
            trimmed = normalized[: -len(suffix)]
            return trimmed or normalized
    return normalized


def _score_canteen_target_column(frame, column_name):
    if column_name not in frame.columns:
        return float("-inf")

    series = frame[column_name]
    if not pd.api.types.is_numeric_dtype(series):
        return float("-inf")

    normalized = _normalize_name(column_name)
    score = 0

    if normalized in {_normalize_name(target) for target in DEFAULT_TARGETS}:
        score += 120
    if _is_footfall_target(column_name):
        score += 100
    if any(hint in normalized for hint in DEMAND_HINTS):
        score += 80
    if any(hint in normalized for hint in PREP_HINTS):
        score -= 5

    if not _is_excluded_target(column_name):
        score += 20

    if any(hint in normalized for hint in ("sold", "sale", "demand", "order", "qty", "quantity")):
        score += 10

    sibling_prefix = normalized
    for suffix in ("sold", "sales", "demand", "served", "consumed", "orders", "order", "qty", "quantity", "prepared", "prep"):
        if sibling_prefix.endswith(suffix):
            sibling_prefix = sibling_prefix[: -len(suffix)]
            break
    if sibling_prefix:
        siblings = [candidate for candidate in frame.columns if candidate != column_name and _normalize_name(candidate).startswith(sibling_prefix)]
        if siblings:
            score += 8

    return float(score)


def _infer_canteen_target_columns(frame):
    candidates = []
    for column in frame.columns:
        score = _score_canteen_target_column(frame, column)
        if score > 0:
            candidates.append((column, score))

    if not candidates:
        return []

    best_by_group = {}
    for column, score in candidates:
        group_key = _target_group_key(column)
        existing = best_by_group.get(group_key)
        if existing is None or score > existing[1]:
            best_by_group[group_key] = (column, score)

    ranked = sorted(best_by_group.values(), key=lambda item: (item[1], item[0]), reverse=True)
    selected = []
    seen = set()
    for column, _score in ranked:
        normalized = _normalize_name(column)
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(column)

    return selected


def load_dataset(source):
    return load_tabular_dataframe(source)


def load_menu_items():
    path = DATA_DIR / "menu_items.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_supply_data():
    path = DATA_DIR / "supply_data.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def summarize_dataset(frame):
    df = frame.copy()
    summary = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missingValues": df.isna().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "preview": df.head(5).replace({np.nan: None}).to_dict(orient="records"),
    }

    date_column = _infer_date_column(df)
    if date_column is not None:
        parsed = pd.to_datetime(df[date_column], errors="coerce", format="mixed")
        valid = parsed.dropna()
        if not valid.empty:
            summary["dateRange"] = {
                "start": valid.min().date().isoformat(),
                "end": valid.max().date().isoformat(),
            }

    inferred_targets = _infer_canteen_target_columns(df)
    target_means = {}
    for target in inferred_targets:
        if pd.api.types.is_numeric_dtype(df[target]):
            target_means[target] = float(pd.to_numeric(df[target], errors="coerce").mean())
    summary["inferredTargetColumns"] = inferred_targets
    summary["targetMeans"] = target_means
    return summary


def standardize_canteen_frame(frame):
    df = frame.copy()
    alias_map = {
        "date": ["date", "day", "transactiondate", "transaction_date"],
        "dayOfWeek": ["dayofweek", "day_of_week", "dow", "weekday"],
        "weather": ["weather", "weather_main", "conditions"],
        "event": ["event", "event_type", "occasion"],
        "totalStudents": ["totalstudents", "total_students", "students", "footfall", "attendance"],
        "riceSold": ["ricesold", "rice_sold", "rice"],
        "dosaSold": ["dosasold", "dosa_sold", "dosa"],
        "snacksSold": ["snackssold", "snacks_sold", "snacks"],
    }

    rename_map = {}
    for canonical, aliases in alias_map.items():
        alias_norms = {_normalize_name(alias) for alias in aliases}
        for column in df.columns:
            normalized = _normalize_name(column)
            if normalized == _normalize_name(canonical) or normalized in alias_norms:
                rename_map[column] = canonical
                break

    if rename_map:
        df = df.rename(columns=rename_map)

    date_column = _infer_date_column(df)
    if date_column is None:
        df["date"] = pd.date_range(
            end=pd.Timestamp.now(tz=None).normalize(),
            periods=len(df),
            freq="D",
        )
    else:
        df["date"] = pd.to_datetime(df[date_column], errors="coerce", format="mixed")
        if date_column != "date":
            df = df.drop(columns=[date_column], errors="ignore")

    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    df["dayOfWeek"] = df.get("dayOfWeek", df["date"].dt.day_name())
    df["dayOfWeek"] = df["dayOfWeek"].fillna(df["date"].dt.day_name())
    if "weather" in df.columns:
        df["weather"] = df["weather"].fillna("Sunny")
    else:
        df["weather"] = pd.Series(["Sunny"] * len(df), index=df.index)
    if "event" in df.columns:
        df["event"] = df["event"].fillna("Normal")
    else:
        df["event"] = pd.Series(["Normal"] * len(df), index=df.index)

    for target in DEFAULT_TARGETS:
        resolved = _resolve_column(df.columns, target)
        if resolved is not None:
            df[resolved] = pd.to_numeric(df[resolved], errors="coerce")

    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["dayOfMonth"] = df["date"].dt.day
    df["dayOfYear"] = df["date"].dt.dayofyear
    df["dayOfWeekNumber"] = df["date"].dt.dayofweek
    df["isWeekend"] = df["dayOfWeekNumber"].isin([5, 6]).astype(int)
    iso_week = df["date"].dt.isocalendar().week
    df["weekOfYear"] = iso_week.astype(int)
    df["daysSinceStart"] = (df["date"] - df["date"].min()).dt.days
    return df


def _exogenous_columns(frame, target_columns):
    excluded = set(target_columns) | {"date"}
    return [column for column in frame.columns if column not in excluded and not column.startswith("lag__") and not column.startswith("roll__")]


def build_supervised_frame(frame, target_columns=None, lag_windows=DEFAULT_LAG_WINDOWS, rolling_windows=DEFAULT_ROLLING_WINDOWS):
    df = standardize_canteen_frame(frame)
    if target_columns is None:
        target_columns = _infer_canteen_target_columns(df)
    else:
        target_columns = [target for target in target_columns if target in df.columns]
        if not target_columns:
            target_columns = _infer_canteen_target_columns(df)

    if not target_columns:
        raise ValueError(
            "No supported canteen target columns were found in the dataset. "
            "The trainer looks for demand-like columns such as totalStudents, idli_sold, plain_dosa, vada_sold, tea_sold, or similar item sales columns."
        )

    base_columns = _exogenous_columns(df, target_columns)
    working = df.copy()

    for target in target_columns:
        target_series = pd.to_numeric(working[target], errors="coerce")
        for lag in lag_windows:
            working[f"lag__{target}__{lag}"] = target_series.shift(lag)
        for window in rolling_windows:
            working[f"roll__{target}__{window}"] = target_series.shift(1).rolling(window).mean()

    feature_columns = base_columns + [
        column
        for column in working.columns
        if column.startswith("lag__") or column.startswith("roll__")
    ]

    supervised = working.dropna(subset=feature_columns + target_columns).copy()
    return supervised, feature_columns, target_columns


def _build_pipeline(feature_frame):
    numeric_columns = feature_frame.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [column for column in feature_frame.columns if column not in numeric_columns]

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
                        ("encoder", _make_one_hot_encoder()),
                    ]
                ),
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model = RandomForestRegressor(
        n_estimators=120,
        random_state=42,
        n_jobs=1,
        min_samples_leaf=2,
        max_features="sqrt",
        max_depth=16,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def _evaluate_regression(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
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
        {"feature": str(name), "importance": float(score)}
        for name, score in ranked[:top_n]
    ]


def train_canteen_models(
    frame,
    lag_windows=DEFAULT_LAG_WINDOWS,
    rolling_windows=DEFAULT_ROLLING_WINDOWS,
    model_path=CANTEEN_MODEL_PATH,
    metadata_path=CANTEEN_METADATA_PATH,
):
    supervised, feature_columns, target_columns = build_supervised_frame(
        frame,
        lag_windows=lag_windows,
        rolling_windows=rolling_windows,
    )

    if len(supervised) < 20:
        raise ValueError("The dataset is too small to train a stable canteen model.")

    validation_rows = max(1, int(len(supervised) * 0.2))
    train_frame = supervised.iloc[:-validation_rows].copy()
    valid_frame = supervised.iloc[-validation_rows:].copy()

    metrics = {}

    pipeline = _build_pipeline(train_frame[feature_columns])
    y_train = train_frame[target_columns]
    y_valid = valid_frame[target_columns]

    pipeline.fit(train_frame[feature_columns], y_train)
    predictions = np.asarray(pipeline.predict(valid_frame[feature_columns]))
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    for index, target in enumerate(target_columns):
        metrics[target] = _evaluate_regression(y_valid[target], predictions[:, index])
        metrics[target]["validationRows"] = int(len(valid_frame))
        metrics[target]["topFeatures"] = _feature_importance_summary(pipeline)

    pipeline.fit(supervised[feature_columns], supervised[target_columns])

    history_rows = standardize_canteen_frame(frame).tail(max(lag_windows + rolling_windows) + 10)
    artifact = {
        "trainedAt": datetime.now(timezone.utc).isoformat(),
        "datasetFingerprint": frame_fingerprint(frame),
        "modelType": "multioutput_random_forest",
        "targetColumns": target_columns,
        "featureColumns": feature_columns,
        "lagWindows": list(lag_windows),
        "rollingWindows": list(rolling_windows),
        "model": pipeline,
        "metrics": metrics,
        "history": history_rows.replace({np.nan: None}).to_dict(orient="records"),
        "datasetProfile": summarize_dataset(frame),
        "menuItems": load_menu_items(),
        "supplyData": load_supply_data(),
    }

    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    metadata = {
        "trainedAt": artifact["trainedAt"],
        "datasetFingerprint": artifact["datasetFingerprint"],
        "modelPath": str(model_path),
        "metadataPath": str(metadata_path),
        "targetColumns": target_columns,
        "featureColumns": feature_columns,
        "lagWindows": list(lag_windows),
        "rollingWindows": list(rolling_windows),
        "metrics": metrics,
        "datasetProfile": artifact["datasetProfile"],
        "modelPath": str(model_path),
    }
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=_json_default)
    return artifact


def load_canteen_artifact(model_path=CANTEEN_MODEL_PATH):
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"No trained canteen model found at {model_path}. Train one from the dashboard first."
        )
    return joblib.load(model_path)


def _scenario_lookup(scenario, column_name):
    normalized = _normalize_name(column_name)
    for key, value in scenario.items():
        if _normalize_name(key) == normalized:
            return value
    return None


def _build_next_row(history_frame, scenario, feature_columns, target_columns, lag_windows, rolling_windows):
    history = standardize_canteen_frame(history_frame)
    scenario = scenario or {}

    last_date = history["date"].max() if not history.empty else pd.Timestamp.now().normalize()
    forecast_date = scenario.get("forecastDate") or scenario.get("date")
    if forecast_date:
        forecast_date = pd.to_datetime(forecast_date, errors="coerce", format="mixed")
    if pd.isna(forecast_date):
        forecast_date = last_date + timedelta(days=1)

    row = {
        "date": forecast_date,
        "dayOfWeek": _scenario_lookup(scenario, "dayOfWeek") or forecast_date.day_name(),
        "weather": _scenario_lookup(scenario, "weather") or "Sunny",
        "event": _scenario_lookup(scenario, "event") or "Normal",
        "month": int(forecast_date.month),
        "quarter": int(forecast_date.quarter),
        "dayOfMonth": int(forecast_date.day),
        "dayOfYear": int(forecast_date.dayofyear),
        "dayOfWeekNumber": int(forecast_date.dayofweek),
        "isWeekend": int(forecast_date.dayofweek in (5, 6)),
        "weekOfYear": int(forecast_date.isocalendar().week),
        "daysSinceStart": int((forecast_date.normalize() - history["date"].min().normalize()).days) if not history.empty else 0,
    }

    if "temperatureC" in feature_columns:
        row["temperatureC"] = _scenario_lookup(scenario, "temperatureC")
    if "humidity" in feature_columns:
        row["humidity"] = _scenario_lookup(scenario, "humidity")

    for column in feature_columns:
        if column in row:
            continue
        if column in history.columns and column not in target_columns:
            value = history.iloc[-1][column] if not history.empty else None
            row[column] = value

    for target in target_columns:
        series = pd.to_numeric(history[target], errors="coerce").dropna()
        fallback = float(series.mean()) if not series.empty else 0.0
        for lag in lag_windows:
            key = f"lag__{target}__{lag}"
            row[key] = float(series.iloc[-lag]) if len(series) >= lag else fallback
        for window in rolling_windows:
            key = f"roll__{target}__{window}"
            tail = series.tail(window)
            row[key] = float(tail.mean()) if not tail.empty else fallback

    for column in feature_columns:
        row.setdefault(column, np.nan)

    return pd.DataFrame([row])[feature_columns]


def _infer_shelf_life_hours(item_name):
    normalized = _normalize_name(item_name)
    if any(token in normalized for token in ("tea", "coffee", "juice", "milk")):
        return 3
    if any(token in normalized for token in ("sandwich", "samosa", "snack", "vada", "idli")):
        return 6
    if any(token in normalized for token in ("dosa", "chapati", "chappati", "biryani", "meal", "meals", "noodle", "noodles")):
        return 5
    return 6


def _infer_cost_per_unit(item_name, avg_demand):
    normalized = _normalize_name(item_name)
    if any(token in normalized for token in ("tea", "coffee")):
        return 8
    if any(token in normalized for token in ("samosa", "snack", "vada")):
        return 14
    if any(token in normalized for token in ("idli", "dosa", "chapati", "chappati", "sandwich")):
        return 20
    if any(token in normalized for token in ("meal", "meals", "biryani", "noodle", "noodles", "rice")):
        return 28
    return max(10, min(40, int(round(avg_demand * 0.25)) if avg_demand else 18))


def _infer_price_per_unit(cost_per_unit):
    return max(int(cost_per_unit + 8), int(round(cost_per_unit * 1.75)))


def _build_menu_catalog(target_columns, history_frame, menu_items=None):
    history = standardize_canteen_frame(history_frame)
    provided_items = list(menu_items or [])
    provided_lookup = {
        _normalize_name(item.get("itemName", "")): item
        for item in provided_items
        if item.get("itemName")
    }

    catalog = []
    demand_targets = [target for target in target_columns if not _is_footfall_target(target)]
    means = {}
    for target in demand_targets:
        if target in history.columns:
            series = pd.to_numeric(history[target], errors="coerce").dropna()
            means[target] = float(series.mean()) if not series.empty else 0.0
        else:
            means[target] = 0.0

    max_mean = max(means.values(), default=0.0) or 1.0

    for target in demand_targets:
        normalized_target = _normalize_name(target)
        stats_mean = means.get(target, 0.0)
        matched_item = None
        for key, candidate in provided_lookup.items():
            if normalized_target == key or normalized_target.startswith(key) or key.startswith(normalized_target):
                matched_item = candidate
                break

        item_name = (matched_item or {}).get("itemName") or _pretty_name(target)
        avg_demand = int(round((matched_item or {}).get("avgDemand", stats_mean or 0)))
        if avg_demand <= 0:
            avg_demand = int(round(stats_mean)) if stats_mean > 0 else 0

        popularity = (matched_item or {}).get("popularityScore")
        if popularity is None:
            popularity = max(0.35, min(0.98, (stats_mean / max_mean) * 0.9 if max_mean else 0.5))

        shelf_life = (matched_item or {}).get("shelfLifeHours")
        if shelf_life is None:
            shelf_life = _infer_shelf_life_hours(item_name)

        cost = (matched_item or {}).get("costPerUnit")
        if cost is None:
            cost = _infer_cost_per_unit(item_name, avg_demand)

        price = (matched_item or {}).get("pricePerUnit")
        if price is None:
            price = _infer_price_per_unit(int(cost))

        catalog.append(
            {
                "itemName": item_name,
                "sourceTarget": target,
                "avgDemand": avg_demand,
                "shelfLifeHours": int(shelf_life),
                "popularityScore": float(popularity),
                "costPerUnit": int(cost),
                "pricePerUnit": int(price),
            }
        )

    return catalog


def _menu_recommendation(item, demand, production, weather, event):
    normalized = _normalize_name(item["itemName"])
    if item["shelfLifeHours"] <= 5 and demand < 40:
        return "Cook a smaller batch and keep the remainder donation-ready."
    if demand > 100 and item["popularityScore"] >= 0.8:
        return "Increase the first batch slightly to avoid stockouts during peak traffic."
    if production - demand > 15:
        return "Trim batch size; this item is likely to become surplus."
    if event == "Festival" and any(token in normalized for token in ("snack", "samosa", "tea", "coffee")):
        return "Festival demand is strong, so keep a small buffer for snacks."
    if weather == "Rainy" and any(token in normalized for token in ("dosa", "sandwich", "tea", "coffee")):
        return "Rainy days usually soften dosa demand, so avoid overproduction."
    return "Production is aligned with forecast demand."


def _build_menu_plan(predicted_items, scenario, menu_items, supply_availability):
    plan = []
    total_demand = 0
    total_production = 0
    total_waste = 0
    total_revenue = 0.0
    total_cost = 0.0

    supply_factor = {"Low": 0.94, "Medium": 1.0, "High": 1.05}.get(supply_availability, 1.0)

    for item in menu_items:
        item_name = item["itemName"]
        demand = int(round(predicted_items.get(item_name, item.get("avgDemand", 0))))
        buffer = 0.06
        if item.get("popularityScore", 0) >= 0.85:
            buffer += 0.04
        if item.get("shelfLifeHours", 0) <= 5:
            buffer -= 0.02
        if scenario.get("event") == "Festival" and item_name.lower() == "snacks":
            buffer += 0.06
        if scenario.get("weather") == "Rainy" and item_name.lower() == "dosa":
            buffer -= 0.03
        if supply_availability == "Low":
            buffer -= 0.03
        buffer = max(0.0, min(0.18, buffer))

        production = int(math.ceil(demand * (1 + buffer) * supply_factor))
        waste_units = max(0, production - demand)
        cost = production * float(item.get("costPerUnit", 0))
        revenue = demand * float(item.get("pricePerUnit", 0))
        profit = revenue - cost

        total_demand += demand
        total_production += production
        total_waste += waste_units
        total_revenue += revenue
        total_cost += cost

        plan.append(
            {
                "itemName": item_name,
                "predictedDemand": demand,
                "suggestedProduction": production,
                "wasteEstimate": waste_units,
                "bufferPercent": round(buffer * 100, 1),
                "expectedProfit": round(profit, 2),
                "recommendation": _menu_recommendation(item, demand, production, scenario.get("weather"), scenario.get("event")),
                "pricePerUnit": item.get("pricePerUnit", 0),
                "costPerUnit": item.get("costPerUnit", 0),
                "shelfLifeHours": item.get("shelfLifeHours", 0),
            }
        )

    waste_pct = round((total_waste / max(1, total_production)) * 100, 1)
    waste_risk = "Low" if waste_pct < 10 else "Medium" if waste_pct < 18 else "High"
    return {
        "plan": plan,
        "totals": {
            "demand": total_demand,
            "production": total_production,
            "wasteUnits": total_waste,
            "wastePercent": waste_pct,
            "wasteRisk": waste_risk,
            "revenue": round(total_revenue, 2),
            "cost": round(total_cost, 2),
            "profit": round(total_revenue - total_cost, 2),
        },
    }


def _build_insights(forecasted_students, scenario, menu_plan, historical_profile):
    insights = []
    if scenario.get("weather") == "Rainy":
        insights.append("Rainy weather is suppressing footfall, so the plan leans into waste control.")
    if scenario.get("event") == "Exam":
        insights.append("Exam periods typically reduce cafeteria visits, which lowers production risk.")
    if scenario.get("event") == "Festival":
        insights.append("Festival days usually increase convenience-snack demand and high-margin item sales.")
    if not any(_is_footfall_target(target) for target in historical_profile.get("inferredTargetColumns", [])):
        insights.append("No explicit attendance column was found, so footfall is inferred from item-demand patterns.")

    baseline_footfall = historical_profile.get("targetMeans", {}).get("totalStudents")
    if baseline_footfall is None:
        for target_name, mean_value in historical_profile.get("targetMeans", {}).items():
            if _is_footfall_target(target_name):
                baseline_footfall = mean_value
                break

    if baseline_footfall is not None and forecasted_students < baseline_footfall:
        insights.append("Forecasted footfall is below the historical average, so batch sizes should stay conservative.")
    high_risk_items = [item["itemName"] for item in menu_plan["plan"] if item["wasteEstimate"] > 10]
    if high_risk_items:
        insights.append("Potential overproduction risk: " + ", ".join(high_risk_items) + ".")
    if menu_plan["totals"]["wasteRisk"] == "High":
        insights.append("Waste risk is high. Tighten tray sizes and coordinate donation pickup earlier.")
    elif menu_plan["totals"]["wasteRisk"] == "Medium":
        insights.append("Waste risk is moderate. A small buffer is still reasonable for fast-moving items.")
    else:
        insights.append("Waste risk is low. Current forecast and production plan look well balanced.")
    return insights


def predict_canteen_forecast(
    artifact,
    scenario,
    supply_availability="Medium",
    menu_items=None,
):
    history_records = artifact.get("history", [])
    history = pd.DataFrame(history_records) if history_records else pd.DataFrame()
    if history.empty:
        raise ValueError("The trained artifact does not contain enough history to forecast.")

    feature_columns = artifact["featureColumns"]
    target_columns = artifact["targetColumns"]
    lag_windows = artifact.get("lagWindows", list(DEFAULT_LAG_WINDOWS))
    rolling_windows = artifact.get("rollingWindows", list(DEFAULT_ROLLING_WINDOWS))

    feature_row = _build_next_row(
        history,
        scenario,
        feature_columns=feature_columns,
        target_columns=target_columns,
        lag_windows=lag_windows,
        rolling_windows=rolling_windows,
    )

    target_predictions = {}
    target_details = {}
    pipeline = artifact.get("model")
    if pipeline is not None:
        raw_predictions = np.asarray(pipeline.predict(feature_row))
        if raw_predictions.ndim == 1:
            raw_predictions = raw_predictions.reshape(1, -1)
        for index, target in enumerate(target_columns):
            value = float(raw_predictions[0][index])
            value = max(0, round(value))
            target_predictions[target] = value
            target_details[target] = {
                "prediction": value,
                "topFeatures": artifact["metrics"][target].get("topFeatures", []),
                "metrics": artifact["metrics"][target],
            }
    else:
        models = artifact.get("models", {})
        for target in target_columns:
            pipeline = models[target]
            value = float(pipeline.predict(feature_row)[0])
            value = max(0, round(value))
            target_predictions[target] = value
            target_details[target] = {
                "prediction": value,
                "topFeatures": artifact["metrics"][target].get("topFeatures", []),
                "metrics": artifact["metrics"][target],
            }

    yesterday_sales = scenario.get("yesterdaySales")
    if yesterday_sales is not None and "totalStudents" in target_predictions:
        try:
            yesterday_sales = float(yesterday_sales)
            blended = (target_predictions["totalStudents"] * 0.85) + (yesterday_sales * 0.15)
            target_predictions["totalStudents"] = max(0, round(blended))
            target_details["totalStudents"]["prediction"] = target_predictions["totalStudents"]
        except (TypeError, ValueError, KeyError):
            pass

    if menu_items is None:
        menu_items = artifact.get("menuItems") or load_menu_items()

    if not menu_items:
        menu_items = []

    catalog = _build_menu_catalog(target_columns, history, menu_items=menu_items)
    if not catalog:
        raise ValueError(
            "No menu-item demand columns could be identified. Add columns such as idli_sold, plain_dosa, vada_sold, tea_sold, or totalStudents."
        )

    predicted_items = {}
    for item in catalog:
        source_target = item["sourceTarget"]
        predicted_items[item["itemName"]] = int(target_predictions.get(source_target, round(item.get("avgDemand", 0))))

    footfall_targets = [target for target in target_columns if _is_footfall_target(target)]
    if footfall_targets:
        predicted_footfall = int(target_predictions.get(footfall_targets[0], 0))
    else:
        demand_values = list(predicted_items.values())
        predicted_footfall = int(round(float(np.median(demand_values)))) if demand_values else 0

    menu_plan = _build_menu_plan(predicted_items, scenario, catalog, supply_availability)
    insights = _build_insights(
        forecasted_students=predicted_footfall,
        scenario=scenario,
        menu_plan=menu_plan,
        historical_profile=artifact.get("datasetProfile", {}),
    )

    naive_production = sum(
        int(round(predicted_items[item["itemName"]] * 1.15))
        for item in catalog
    )
    naive_waste = max(0, naive_production - menu_plan["totals"]["demand"])
    optimized_waste = menu_plan["totals"]["wasteUnits"]
    waste_reduction = max(0, naive_waste - optimized_waste)
    waste_reduction_pct = round((waste_reduction / max(1, naive_waste)) * 100, 1) if naive_waste else 0.0

    donation = None
    if menu_plan["totals"]["wastePercent"] > 15 or menu_plan["totals"]["wasteUnits"] > 25:
        donation = {
            "excessFoodDetected": True,
            "excessQuantity": int(menu_plan["totals"]["wasteUnits"]),
            "message": "Surplus food is expected. Coordinate pickup with the nearest NGO or food bank.",
            "recommendedWindow": "within 2 hours of meal close",
        }

    return {
        "scenario": {
            "forecastDate": str(
                pd.to_datetime(scenario.get("forecastDate") or scenario.get("date") or (pd.Timestamp.now() + pd.Timedelta(days=1))).date()
            ),
            "dayOfWeek": scenario.get("dayOfWeek", "Unknown"),
            "weather": scenario.get("weather", "Sunny"),
            "event": scenario.get("event", "Normal"),
            "supplyAvailability": supply_availability,
            "city": scenario.get("city", ""),
            "yesterdaySales": scenario.get("yesterdaySales"),
        },
        "forecast": target_details,
        "predictedFootfall": predicted_footfall,
        "predictedItems": predicted_items,
        "menuPlan": menu_plan["plan"],
        "wasteSummary": menu_plan["totals"],
        "donationAlert": donation,
        "insights": insights,
        "benchmark": {
            "naiveWasteUnits": int(naive_waste),
            "optimizedWasteUnits": int(optimized_waste),
            "wasteReductionUnits": int(waste_reduction),
            "wasteReductionPercent": waste_reduction_pct,
        },
        "trainingMetrics": artifact.get("metrics", {}),
        "featureColumns": feature_columns,
    }


def get_canteen_status(model_path=CANTEEN_MODEL_PATH, metadata_path=CANTEEN_METADATA_PATH):
    metadata = None
    if Path(metadata_path).exists():
        with open(metadata_path, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)
    return {
        "trained": Path(model_path).exists(),
        "modelPath": str(model_path),
        "metadataPath": str(metadata_path),
        "metadata": metadata,
    }
