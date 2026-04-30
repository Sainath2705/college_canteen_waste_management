# CanteenIQ

CanteenIQ is a hackathon-ready **data science and analytics** project for solving college canteen waste.
It predicts next-day demand from historical canteen data, recommends how much food to prepare, estimates waste and profit, and triggers a donation alert when surplus food is likely.

## Problem Statement

College canteens often overproduce or underproduce because demand changes with:

- day of week
- weather
- exams and festivals
- academic calendar patterns

The result is unnecessary waste, lost profit, and missed donation opportunities.

## Solution

CanteenIQ turns your historical sales data into a decision engine with three layers:

1. **Forecasting**
   - Predict next-day student footfall
   - Estimate demand for menu items

2. **Optimization**
   - Recommend how much to cook
   - Balance waste control with profit

3. **Operations**
   - Raise a donation alert for surplus food
   - Provide a simple analytics dashboard for judging and demoing

## Why This Is Strong For A Hackathon

- Uses **real analytics** instead of a toy demo
- Works with **your own dataset** or a Kaggle-style tabular dataset
- Supports **on-demand training**, so it learns from the data you upload
- Gives both a **dashboard** and an **API**
- Includes a **donation coordination** angle, which makes the story stronger for judges

## Main Features

- Upload canteen or tabular CSV/JSON data
- Train a canteen-specific forecasting model from your own data
- Generate next-day footfall and item-level demand forecasts
- Produce a smart menu plan with production quantity, waste estimate, and profit estimate
- Show donation alerts when surplus food is expected
- Train a generic model on any Kaggle-style tabular dataset
- Export forecast results as JSON

## Tech Stack

- Python
- Flask
- HTML/CSS/JavaScript
- Chart.js
- Streamlit
- Pandas
- NumPy
- scikit-learn
- Plotly

## Project Layout

- [`app.py`](app.py) - Flask API backend and homepage
- [`static/index.html`](static/index.html) - the main browser UI
- [`streamlit_app.py`](streamlit_app.py) - optional alternate dashboard
- [`canteen_analytics.py`](canteen_analytics.py) - canteen-specific forecasting and optimization engine
- [`ml_service.py`](ml_service.py) - generic tabular model training and prediction utilities
- [`train_models.py`](train_models.py) - CLI trainer for tabular datasets
- [`data/canteen_dataset_300_rows.csv`](data/canteen_dataset_300_rows.csv) - sample demo dataset
- [`data/menu_items.json`](data/menu_items.json) - menu metadata
- [`data/supply_data.json`](data/supply_data.json) - supply availability metadata

## How To Run Locally

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Start the browser UI

```bash
python app.py
```

Then open `http://127.0.0.1:5000/`.

This is the best way to demo the project in a hackathon.

### Optional: train from the CLI

```bash
python train_models.py data/canteen_dataset_300_rows.csv --canteen
```

Use the same script without `--canteen` for generic Kaggle-style tabular datasets.

## How The Demo Works

### Canteen mode

1. Choose **Canteen Intelligence**
2. Use the demo dataset or upload your own historical canteen CSV/JSON
3. Click **Train selected mode**
4. Open **Forecast Studio**
5. Enter tomorrow's weather, event, and supply availability
6. Get:
   - next-day student forecast
   - item demand forecast
   - suggested production quantities
   - waste estimate
   - donation alert

### Custom tabular mode

If you want to use a Kaggle dataset that is not a canteen dataset:

1. Choose **Custom Tabular Trainer**
2. Upload the dataset
3. Select the target column
4. Train the model
5. Run a prediction on one sample record

## Supported Data

### Canteen training data

The canteen engine works best with columns like:

- `date`
- `dayOfWeek`
- `weather`
- `event`
- `totalStudents`
- `riceSold`
- `dosaSold`
- `snacksSold`

It also handles item-level canteen schemas such as:

- `idli_sold`
- `vada_sold`
- `plain_dosa`
- `tea_sold`
- `meals_sold`
- `samosa_sold`
- `sandwich_sold`
- `noodles_sold`

It also accepts common aliases such as:

- `day_of_week`
- `weather_main`
- `event_type`
- `footfall`
- `attendance`
- `rice_sold`
- `dosa_sold`
- `snacks_sold`

If your dataset includes `*_prep` or `*_waste` columns, the trainer ignores those for demand forecasting and focuses on the actual demand columns.

### Generic tabular data

The custom trainer can handle common Kaggle-style datasets in CSV or JSON format for:

- regression
- classification

## API Endpoints

### Generic training API

- `GET /api/health`
- `GET /model/status`
- `POST /train`
- `POST /predict`

### Canteen-specific API

- `GET /canteen/status`
- `POST /canteen/train`
- `POST /canteen/predict`

### Example: train canteen model

```bash
curl -X POST http://127.0.0.1:5000/canteen/train ^
  -F "dataset=@data/canteen_dataset_300_rows.csv"
```

### Example: predict canteen demand

```bash
curl -X POST http://127.0.0.1:5000/canteen/predict ^
  -H "Content-Type: application/json" ^
  -d "{
    \"scenario\": {
      \"forecastDate\": \"2026-05-01\",
      \"weather\": \"Rainy\",
      \"event\": \"Exam\"
    },
    \"supplyAvailability\": \"Medium\"
  }"
```

### Example: train a generic Kaggle-style dataset

```bash
curl -X POST http://127.0.0.1:5000/train ^
  -F "dataset=@your_dataset.csv" ^
  -F "targetColumn=target" ^
  -F "taskType=auto"
```

## Deployment

Use `app.py` as the main entry file. The root route serves the HTML dashboard.

### Render or Heroku-style deploys

The repo includes a [`Procfile`](Procfile) that launches:

```bash
gunicorn app:app --bind 0.0.0.0:$PORT
```

### Environment variables

- `OPENWEATHER_API_KEY` - optional, used for live weather lookup in the dashboard
- `OPENAI_API_KEY` - optional, only relevant if you want the Flask chat fallback to use OpenAI

## Suggested Hackathon Pitch

> CanteenIQ converts historical canteen sales, weather, and academic patterns into a next-day food production plan that reduces waste, protects profit, and routes surplus food to NGOs.

## Notes

- The project is designed to train from the dataset you provide.
- It does not require a pre-trained model to be present.
- If you upload a new dataset, retrain before forecasting so the model matches your data.
