import csv
import datetime
import io
import json
import math
import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

from ml_service import (
    DEFAULT_MODEL_PATH,
    latest_model_status,
    load_tabular_dataframe,
    parse_name_list,
    predict_dataframe,
    predict_records,
    train_tabular_model,
)

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
try:
    import openai
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
except ImportError:
    openai = None

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_HISTORICAL_FILE = os.path.join(DATA_DIR, "canteen_dataset_300_rows.csv")
DEFAULT_MENU_FILE = os.path.join(DATA_DIR, "menu_items.json")
DEFAULT_SUPPLY_FILE = os.path.join(DATA_DIR, "supply_data.json")

VALID_WEATHER = {"Sunny", "Rainy", "Cloudy"}
VALID_EVENTS = {"Normal", "Exam", "Festival"}
VALID_DAYS = {
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
}
VALID_SUPPLY = {"Low", "Medium", "High"}


def load_csv_dataset(file_stream):
    # Read CSV safely and validate required columns and numeric types
    text_stream = io.TextIOWrapper(file_stream, encoding="utf-8")
    reader = csv.DictReader(text_stream)
    if not reader.fieldnames:
        raise ValueError("CSV appears to be empty or missing a header row.")

    # Accept common aliases for required fields (case-insensitive)
    aliases = {
        "date": ["date", "day", "transaction_date"],
        "dayOfWeek": ["dayofweek", "day_of_week", "dow", "day"],
        "weather": ["weather", "weather_main", "conditions"],
        "event": ["event", "event_type", "occasion"],
        "totalStudents": ["totalstudents", "total_students", "students", "total"],
        "riceSold": ["ricesold", "rice_sold", "rice", "rice_units"],
        "dosaSold": ["dosasold", "dosa_sold", "dosa", "dosa_units"],
        "snacksSold": ["snackssold", "snacks_sold", "snacks", "snacks_units"]
    }

    # Normalize file headers to lowercase without spaces/underscores for matching
    file_fields = [f for f in reader.fieldnames]
    norm_map = {f: f.lower().replace(" ", "").replace("_", "") for f in file_fields}

    # Build mapping from logical name -> actual header name in file
    mapping = {}
    for logical, keys in aliases.items():
        found = None
        for file_field, norm in norm_map.items():
            if norm in keys or norm == logical.lower():
                found = file_field
                break
        if found:
            mapping[logical] = found

    missing = [k for k in aliases.keys() if k not in mapping]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    rows = []
    for lineno, raw in enumerate(reader, start=2):
        # Remap keys to expected logical names
        row = {}
        for logical, actual in mapping.items():
            row[logical] = raw.get(actual, "")
        try:
            # Convert numeric fields robustly
            row["totalStudents"] = int(float(row.get("totalStudents", 0)))
            row["riceSold"] = int(float(row.get("riceSold", 0)))
            row["dosaSold"] = int(float(row.get("dosaSold", 0)))
            row["snacksSold"] = int(float(row.get("snacksSold", 0)))
        except Exception as exc:
            raise ValueError(f"Invalid numeric value in CSV at line {lineno}: {exc}")
        rows.append(row)
    return rows
    


def load_menu_items():
    with open(DEFAULT_MENU_FILE, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_supply_data():
    with open(DEFAULT_SUPPLY_FILE, "r", encoding="utf-8") as fh:
        return json.load(fh)


def safe_load_historical(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as fh:
            return load_csv_dataset(fh)
    raise FileNotFoundError(f"Historical dataset not found: {file_path}")


def validate_string(value, valid_set, name):
    if not isinstance(value, str) or value not in valid_set:
        raise ValueError(f"{name} must be one of {sorted(valid_set)}")
    return value


def calculate_baseline(history, day_of_week, weather, event):
    if not history:
        return 120

    total = [row["totalStudents"] for row in history]
    global_avg = sum(total) / len(total)

    day_rows = [row["totalStudents"] for row in history if row["dayOfWeek"] == day_of_week]
    weather_rows = [row["totalStudents"] for row in history if row["weather"] == weather]
    event_rows = [row["totalStudents"] for row in history if row["event"] == event]

    day_avg = sum(day_rows) / len(day_rows) if day_rows else global_avg
    weather_avg = sum(weather_rows) / len(weather_rows) if weather_rows else global_avg
    event_avg = sum(event_rows) / len(event_rows) if event_rows else global_avg

    baseline = (
        global_avg * 0.35
        + day_avg * 0.30
        + weather_avg * 0.20
        + event_avg * 0.15
    )
    return baseline


def adjust_for_context(base, weather, event, supply_availability, yesterday_sales):
    score = float(base)
    if weather == "Rainy":
        score *= 0.92
    elif weather == "Cloudy":
        score *= 0.98
    else:
        score *= 1.02

    if event == "Exam":
        score *= 0.88
    elif event == "Festival":
        score *= 1.14
    else:
        score *= 1.00

    if supply_availability == "Low":
        score *= 0.90
    elif supply_availability == "Medium":
        score *= 1.00
    else:
        score *= 1.06

    if yesterday_sales is not None:
        delta = yesterday_sales - score
        score += delta * 0.06

    return max(40, round(score))


def get_item_demand(predicted_students, menu_items, event, weather):
    result = []
    for item in menu_items:
        factor = 1.0
        if item["itemName"] == "Snacks" and event == "Festival":
            factor += 0.10
        if item["itemName"] == "Rice" and weather == "Rainy":
            factor += 0.02
        if item["itemName"] == "Dosa" and weather == "Rainy":
            factor -= 0.04

        predicted = predicted_students * item["popularityScore"] * factor
        result.append({
            "itemName": item["itemName"],
            "popularityScore": item["popularityScore"],
            "avgDemand": item["avgDemand"],
            "shelfLifeHours": item["shelfLifeHours"],
            "costPerUnit": item["costPerUnit"],
            "pricePerUnit": item["pricePerUnit"],
            "predictedDemand": max(0, round(predicted))
        })
    return result


def optimize_production(item_demands, supply_availability):
    total_demand = sum(item["predictedDemand"] for item in item_demands)
    suggestions = []
    total_cost = 0
    total_sale = 0
    total_waste_units = 0

    supply_multiplier = {"Low": 0.95, "Medium": 1.0, "High": 1.08}[supply_availability]

    for item in item_demands:
        demand = item["predictedDemand"]
        if demand < 30 and item["shelfLifeHours"] <= 6:
            prepare = math.ceil(demand * 0.85)
        elif demand < 50:
            prepare = math.ceil(demand * 0.92)
        else:
            prepare = math.ceil(demand * 1.06)

        if item["popularityScore"] > 0.8:
            prepare = max(prepare, math.ceil(demand * 1.08))

        prepare = math.ceil(prepare * supply_multiplier)
        waste_units = max(0, prepare - demand)

        cost = prepare * item["costPerUnit"]
        revenue = demand * item["pricePerUnit"]
        profit = revenue - cost

        total_cost += cost
        total_sale += revenue
        total_waste_units += waste_units

        suggestions.append({
            "itemName": item["itemName"],
            "predictedDemand": demand,
            "suggestedProduction": prepare,
            "wasteEstimate": waste_units,
            "expectedProfit": round(profit, 2),
            "recommendation": generate_item_recommendation(item, demand, prepare)
        })

    waste_pct = round((total_waste_units / max(1, sum(s["suggestedProduction"] for s in suggestions))) * 100, 1)
    risk = "Low" if waste_pct < 10 else "Medium" if waste_pct < 18 else "High"
    predicted_profit = round(total_sale - total_cost, 2)

    return {
        "suggestions": suggestions,
        "totalDemand": total_demand,
        "totalSuggestedProduction": sum(s["suggestedProduction"] for s in suggestions),
        "totalWasteUnits": total_waste_units,
        "wastePercent": waste_pct,
        "wasteRisk": risk,
        "profitEstimate": predicted_profit,
        "estimatedCost": total_cost,
        "estimatedRevenue": total_sale
    }


def generate_item_recommendation(item, demand, production):
    if item["shelfLifeHours"] <= 5 and demand < 40:
        return "Reduce portion size or move to donation quickly due to short shelf life."
    if demand > 100 and item["popularityScore"] > 0.8:
        return "Increase production slightly to avoid stockouts during peak demand."
    if production - demand > 20:
        return "Production looks higher than forecast, consider leaner batch sizes to reduce waste."
    return "Production recommendations are aligned with predicted demand."


def generate_insights(predicted_students, weather, event, item_demands, waste_data):
    insights = []
    if weather == "Rainy":
        insights.append("Demand is lower due to rainy weather encouraging students to stay indoors.")
    if event == "Exam":
        insights.append("Demand is lower due to exams reducing cafeteria visits.")
    if event == "Festival":
        insights.append("Festival demand is higher as students prefer snacks and convenience items.")

    low_stock_items = [i for i in item_demands if i["predictedDemand"] < 50 and i["shelfLifeHours"] <= 6]
    if low_stock_items:
        names = ", ".join(i["itemName"] for i in low_stock_items)
        insights.append(f"High spoilage risk for {names} because short shelf life items have lower forecasted demand.")

    if waste_data["wasteRisk"] == "High":
        insights.append("Waste risk is high; consider tighter portion control and donation planning.")
    elif waste_data["wasteRisk"] == "Medium":
        insights.append("Waste risk is moderate; adjust preparation quantities for low-demand items.")
    else:
        insights.append("Waste risk is low; current demand and production alignment looks balanced.")

    if predicted_students < 90:
        insights.append("Overall footfall is below average, so cost-conscious preparation is advised.")

    return insights


def map_weather_category(weather_main):
    if not weather_main:
        return "Sunny"
    value = weather_main.lower()
    if any(key in value for key in ["rain", "drizzle", "thunder", "storm", "snow"]):
        return "Rainy"
    if any(key in value for key in ["cloud", "overcast", "mist", "fog", "haze"]):
        return "Cloudy"
    return "Sunny"


def fetch_weather_for_city(city):
    if not city or not isinstance(city, str):
        raise ValueError("city parameter must be a non-empty string")

    if OPENWEATHER_API_KEY:
        try:
            response = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"},
                timeout=6,
            )
            response.raise_for_status()
            data = response.json()
            weather_main = data["weather"][0]["main"]
            description = data["weather"][0]["description"].title()
            temp = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            category = map_weather_category(weather_main)
            return {
                "city": data.get("name", city).title(),
                "weather": category,
                "weatherMain": weather_main,
                "description": description,
                "temperatureC": round(temp, 1),
                "humidity": humidity,
                "source": "openweathermap"
            }
        except Exception as exc:
            return {
                "city": city.title(),
                "weather": "Unknown",
                "description": "Unable to fetch weather at this time.",
                "temperatureC": None,
                "humidity": None,
                "source": "fallback",
                "error": str(exc)
            }

    return {
        "city": city.title(),
        "weather": "Sunny",
        "weatherMain": "Clear",
        "description": "Simulated weather because API key is not configured.",
        "temperatureC": 28.4,
        "humidity": 52,
        "source": "fallback"
    }


def fallback_chat_response(message):
    lower = message.lower()
    if "weather" in lower:
        return "Use the weather lookup form by city or query /weather to get live conditions for your forecast."
    if "predict" in lower or "forecast" in lower:
        return "Provide the day, weather condition, event type, and yesterday’s sales to generate a next-day demand forecast."
    if "donation" in lower:
        return "When waste exceeds the threshold, the system raises a donation alert so an NGO can be notified."
    if "waste" in lower or "profit" in lower:
        return "The model optimizes production using demand, shelf life, and supply availability to help reduce waste and maximize profit."
    return "I am your canteen operations assistant. Ask about forecasting, waste optimization, menu planning, or donation coordination."


def ai_chat_response(message, context=None):
    if openai and OPENAI_API_KEY:
        try:
            system_prompt = (
                "You are an assistant for canteen demand forecasting and waste optimization. "
                "Provide concise, actionable responses for menu planning, donation coordination, and operational insights."
            )
            messages = [{"role": "system", "content": system_prompt}]
            if context:
                messages.append({"role": "system", "content": context})
            messages.append({"role": "user", "content": message})
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=220,
            )
            return completion.choices[0].message.content.strip()
        except Exception as exc:
            return f"AI service error: {str(exc)}"
    return fallback_chat_response(message)


@app.route("/weather", methods=["GET"])
def weather():
    city = request.args.get("city", "").strip()
    if not city:
        return jsonify({"error": "city query parameter is required"}), 400
    result = fetch_weather_for_city(city)
    return jsonify(result)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    message = (data.get("message") or "").strip()
    context = data.get("context")
    if not message:
        return jsonify({"error": "message is required"}), 400
    reply = ai_chat_response(message, context)
    return jsonify({"message": message, "reply": reply, "model": "openai" if openai and OPENAI_API_KEY else "fallback"})


def _get_request_payload():
    payload = request.form.to_dict(flat=True)
    if not payload:
        payload = request.get_json(silent=True) or {}
    return payload


@app.route("/", methods=["GET"])
def serve_dashboard():
    status = latest_model_status()
    return jsonify(
        {
            "message": "On-demand tabular model training API.",
            "trainedModel": status.get("trained", False),
            "modelPath": status.get("modelPath"),
            "endpoints": {
                "health": "/api/health",
                "train": "/train",
                "predict": "/predict",
                "status": "/model/status",
            },
        }
    )


@app.route("/api/health", methods=["GET"])
def health_check():
    status = latest_model_status()
    return jsonify(
        {
            "status": "ok",
            "message": "On-demand model training API is running.",
            "modelTrained": status.get("trained", False),
        }
    )


@app.route("/model/status", methods=["GET"])
def model_status():
    return jsonify(latest_model_status())


@app.route("/train", methods=["POST"])
def train_model():
    try:
        payload = _get_request_payload()
        target_column = payload.get("targetColumn") or payload.get("target")
        task_type = payload.get("taskType", "auto")
        drop_columns = parse_name_list(payload.get("dropColumns"))
        test_size = float(payload.get("testSize", 0.2))
        random_state = int(payload.get("randomState", 42))
        model_path = payload.get("modelPath") or str(DEFAULT_MODEL_PATH)
        metadata_path = payload.get("metadataPath")

        if "dataset" in request.files:
            dataset_file = request.files["dataset"]
            frame = load_tabular_dataframe(dataset_file.stream, filename=dataset_file.filename)
        elif payload.get("dataPath"):
            frame = load_tabular_dataframe(payload.get("dataPath"))
        else:
            return jsonify({"error": "Upload a dataset file or provide dataPath."}), 400

        result = train_tabular_model(
            frame,
            target_column=target_column,
            task_type=task_type,
            drop_columns=drop_columns,
            test_size=test_size,
            random_state=random_state,
            model_path=model_path,
            metadata_path=metadata_path or None,
        )
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": "Internal server error", "detail": str(exc)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = _get_request_payload()
        model_path = payload.get("modelPath") or str(DEFAULT_MODEL_PATH)

        if "dataset" in request.files:
            dataset_file = request.files["dataset"]
            frame = load_tabular_dataframe(dataset_file.stream, filename=dataset_file.filename)
            result = predict_dataframe(frame, model_path=model_path)
            return jsonify(result)

        if payload.get("dataPath"):
            frame = load_tabular_dataframe(payload.get("dataPath"))
            result = predict_dataframe(frame, model_path=model_path)
            return jsonify(result)

        records = payload.get("records")
        if records is None and payload.get("record") is not None:
            record = payload.get("record")
            if isinstance(record, str):
                try:
                    record = json.loads(record)
                except json.JSONDecodeError:
                    pass
            records = record

        result = predict_records(records, model_path=model_path)
        return jsonify(result)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 400
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": "Internal server error", "detail": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
