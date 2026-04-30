from __future__ import annotations

import argparse
import json

from canteen_analytics import CANTEEN_METADATA_PATH, CANTEEN_MODEL_PATH, train_canteen_models
from ml_service import (
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    load_tabular_dataframe,
    train_tabular_model,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train a tabular model from a user-provided CSV or JSON dataset."
    )
    parser.add_argument("dataset", help="Path to the dataset file.")
    parser.add_argument(
        "--canteen",
        action="store_true",
        help="Train the canteen-specific forecasting and menu optimization model.",
    )
    parser.add_argument(
        "--target",
        dest="target_column",
        help="Target column name. If omitted, the script tries common target names and then the last column.",
    )
    parser.add_argument(
        "--task",
        dest="task_type",
        default="auto",
        choices=["auto", "classification", "regression"],
        help="Force classification or regression, or let the script infer the task type.",
    )
    parser.add_argument(
        "--drop-columns",
        dest="drop_columns",
        default="",
        help="Comma-separated list of columns to exclude from training.",
    )
    parser.add_argument(
        "--test-size",
        dest="test_size",
        type=float,
        default=0.2,
        help="Fraction of rows reserved for evaluation.",
    )
    parser.add_argument(
        "--random-state",
        dest="random_state",
        type=int,
        default=42,
        help="Random seed used for the train/test split.",
    )
    parser.add_argument(
        "--model-path",
        dest="model_path",
        default=None,
        help="Where to save the trained model artifact.",
    )
    parser.add_argument(
        "--metadata-path",
        dest="metadata_path",
        default=None,
        help="Where to save the training metadata JSON file.",
    )
    args = parser.parse_args()

    frame = load_tabular_dataframe(args.dataset)
    if args.canteen:
        model_path = args.model_path or str(CANTEEN_MODEL_PATH)
        metadata_path = args.metadata_path or str(CANTEEN_METADATA_PATH)
        artifact = train_canteen_models(frame, model_path=model_path, metadata_path=metadata_path)
        result = {
            "status": "trained",
            "modelPath": model_path,
            "metadataPath": metadata_path,
            "datasetFingerprint": artifact.get("datasetFingerprint"),
            "datasetProfile": artifact.get("datasetProfile"),
            "metrics": artifact.get("metrics"),
        }
    else:
        model_path = args.model_path or str(DEFAULT_MODEL_PATH)
        metadata_path = args.metadata_path or str(DEFAULT_METADATA_PATH)
        result = train_tabular_model(
            frame,
            target_column=args.target_column,
            task_type=args.task_type,
            drop_columns=args.drop_columns,
            test_size=args.test_size,
            random_state=args.random_state,
            model_path=model_path,
            metadata_path=metadata_path,
        )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
