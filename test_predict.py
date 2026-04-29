from pathlib import Path

from app import app

client = app.test_client()

DATASET_PATH = Path(__file__).resolve().parent / "data" / "canteen_dataset_300_rows.csv"

with open(DATASET_PATH, "rb") as fh:
    train_resp = client.post(
        "/train",
        data={
            "dataset": (fh, "canteen_dataset_300_rows.csv"),
            "targetColumn": "totalStudents",
            "taskType": "regression",
        },
        content_type="multipart/form-data",
    )

print("TRAIN", train_resp.status_code)
print(train_resp.get_json())

predict_resp = client.post(
    "/predict",
    json={
        "records": [
            {
                "dayOfWeek": "Monday",
                "weather": "Rainy",
                "event": "Exam",
            }
        ]
    },
)

print("PREDICT", predict_resp.status_code)
print(predict_resp.get_json())
