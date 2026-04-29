from pathlib import Path

from app import app

client = app.test_client()

DATASET_PATH = Path(__file__).resolve().parent / "data" / "canteen_dataset_300_rows.csv"

with open(DATASET_PATH, "rb") as fh:
    resp = client.post(
        "/train",
        data={
            "dataset": (fh, "canteen_dataset_300_rows.csv"),
            "targetColumn": "totalStudents",
            "taskType": "regression",
        },
        content_type="multipart/form-data",
    )

print("STATUS", resp.status_code)
print(resp.get_json())
