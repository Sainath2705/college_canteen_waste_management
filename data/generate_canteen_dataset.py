import csv
import datetime
import os
import random

DATA_DIR = os.path.dirname(__file__)
OUTPUT_FILE = os.path.join(DATA_DIR, "canteen_dataset_300_rows.csv")
START_DATE = datetime.date(2025, 1, 1)
NUM_ROWS = 300
WEATHER_CHOICES = ["Sunny"] * 4 + ["Cloudy"] * 3 + ["Rainy"] * 3
EVENT_CHOICES = ["Normal"] * 7 + ["Exam"] * 2 + ["Festival"] * 1


def sample_event(date):
    if date.weekday() == 5 or date.weekday() == 6:
        return random.choices(["Normal", "Festival"], weights=[80, 20])[0]
    if date.month in [3, 4, 5, 10]:
        return random.choices(["Normal", "Exam", "Festival"], weights=[75, 15, 10])[0]
    return random.choices(EVENT_CHOICES, k=1)[0]


def sample_weather(date):
    if date.month in [6, 7, 8, 9]:
        return random.choices(["Sunny", "Rainy", "Cloudy"], weights=[40, 35, 25])[0]
    return random.choices(["Sunny", "Rainy", "Cloudy"], weights=[45, 25, 30])[0]


def build_row(date):
    day_of_week = date.strftime("%A")
    weather = sample_weather(date)
    event = sample_event(date)
    base_students = 120 + (date.month - 1) * 3

    if day_of_week in ["Saturday", "Sunday"]:
        base_students -= 15
    if weather == "Rainy":
        base_students -= 10
    if event == "Exam":
        base_students -= 18
    if event == "Festival":
        base_students += 22

    total_students = max(50, min(220, int(random.gauss(base_students, 12))))
    rice_sold = max(30, int(total_students * random.uniform(0.54, 0.65)))
    dosa_sold = max(20, int(total_students * random.uniform(0.30, 0.42)))
    snacks_sold = max(35, int(total_students * random.uniform(0.65, 0.85)))

    return {
        "date": date.isoformat(),
        "dayOfWeek": day_of_week,
        "weather": weather,
        "event": event,
        "totalStudents": total_students,
        "riceSold": rice_sold,
        "dosaSold": dosa_sold,
        "snacksSold": snacks_sold,
    }


def generate_dataset():
    today = START_DATE
    rows = []
    for index in range(NUM_ROWS):
        rows.append(build_row(today))
        today += datetime.timedelta(days=1)

    with open(OUTPUT_FILE, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["date", "dayOfWeek", "weather", "event", "totalStudents", "riceSold", "dosaSold", "snacksSold"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {NUM_ROWS} dataset rows in {OUTPUT_FILE}")


if __name__ == "__main__":
    random.seed(42)
    generate_dataset()
