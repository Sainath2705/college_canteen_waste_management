# College Canteen Waste Analytics and Smart Menu Planner

Hackathon solution for the Data Science and Analytics theme.

This project turns historical canteen sales data into a decision-support system that can:

- forecast next-day demand from historical patterns, weather, and academic events
- train on user-provided CSV or JSON datasets, including Kaggle-style tabular data
- generate per-target predictions for footfall or item demand
- help plan smarter production quantities to reduce waste
- provide a foundation for donation coordination when surplus food is expected

## Problem Statement

College canteens often waste 30 to 40 percent of food every day because demand changes with exams, festivals, weather, and day of week. The menu is usually fixed, so owners either over-produce or under-produce.

The goal is to build a data-driven system that forecasts demand, recommends menu quantities, and reduces waste while protecting profit.

## Best-Suited Solution

The most practical hackathon approach is a three-layer analytics pipeline:

1. **Forecasting**
   - Train a tabular model on historical canteen data.
   - Predict next-day footfall or item-level demand.

2. **Optimization**
   - Convert forecasts into recommended production quantities.
   - Use simple business rules or a small optimizer to balance waste and profit.

3. **Coordination**
   - If surplus food is predicted, trigger a donation workflow for nearby NGOs.

This is a strong fit for the hackathon because it is explainable, fast to train, and works well with structured datasets.

## What The Current Repo Provides

- A reusable training pipeline for tabular datasets
- Automatic handling of CSV and JSON inputs
- Automatic feature engineering for date-like columns
- Regression and classification support
- Model persistence with `joblib`
- A Flask API for training and inference
- Sample canteen data for demonstration

## Recommended Hackathon Demo

For the canteen problem, train separate models for:

- `totalStudents` for next-day footfall
- `riceSold` for rice demand
- `dosaSold` for dosa demand
- `snacksSold` for snacks demand

Then use those predictions to drive menu recommendations and waste reduction logic.

## Project Structure

- `app.py` - Flask API for training, prediction, weather, chat, and status checks
- `ml_service.py` - dataset loading, preprocessing, training, and inference helpers
- `train_models.py` - command-line trainer for user-provided datasets
- `data/canteen_dataset_300_rows.csv` - sample historical canteen data
- `data/menu_items.json` - sample menu metadata
- `data/supply_data.json` - sample supply readiness data
- `data/generate_canteen_dataset.py` - script to regenerate the sample dataset
- `models/` - saved model artifacts and metadata created after training

## Tech Stack

- Python
- Pandas
- scikit-learn
- Flask
- joblib

The original problem statement mentions Streamlit. This repo is currently Flask-based, but the analytics layer can be wrapped in Streamlit later if you want a presentation-first demo.

## Local Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the API:
   ```bash
   python app.py
   ```

3. Check the server:
   ```bash
   GET /api/health
   ```

## Training A Model

Train from the bundled sample dataset:

```bash
python train_models.py data/canteen_dataset_300_rows.csv --target totalStudents --task regression
```

Train a separate model for item demand:

```bash
python train_models.py data/canteen_dataset_300_rows.csv --target riceSold --task regression
python train_models.py data/canteen_dataset_300_rows.csv --target dosaSold --task regression
python train_models.py data/canteen_dataset_300_rows.csv --target snacksSold --task regression
```

If your dataset uses a different column name, pass it with `--target`.

You can also save different outputs to different files:

```bash
python train_models.py data/my_canteen_data.csv --target totalStudents --model-path models/footfall_model.joblib
```

## API Endpoints

- `GET /` - returns API status and available endpoints
- `GET /api/health` - health check
- `GET /model/status` - reports whether a trained model exists
- `POST /train` - trains a model from an uploaded dataset or `dataPath`
- `POST /predict` - runs inference using the trained model
- `GET /weather?city=<city>` - optional live weather lookup
- `POST /chat` - optional assistant for operational questions

## Train Through The API

Example multipart upload:

```bash
curl -X POST http://127.0.0.1:5000/train \
  -F "dataset=@data/canteen_dataset_300_rows.csv" \
  -F "targetColumn=totalStudents" \
  -F "taskType=regression"
```

Example using a server-side data path:

```bash
curl -X POST http://127.0.0.1:5000/train \
  -H "Content-Type: application/json" \
  -d '{
    "dataPath": "data/canteen_dataset_300_rows.csv",
    "targetColumn": "totalStudents",
    "taskType": "regression"
  }'
```

## Predict With The Trained Model

Send one or more feature rows to `/predict`:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "dayOfWeek": "Monday",
        "weather": "Rainy",
        "event": "Exam"
      }
    ]
  }'
```

## How To Use This For The Hackathon

1. Train a footfall model using historical canteen data.
2. Train separate demand models for each menu item.
3. Use tomorrow's day, weather, and event type to generate forecasts.
4. Convert forecasts into production quantities.
5. Flag surplus food and show a donation alert.

## Why Judges Usually Like This Approach

- It is practical and easy to explain.
- It uses real analytics, not just a static demo.
- It supports any structured Kaggle-style dataset.
- It produces measurable outputs like MAE, RMSE, and predicted demand.
- It can be extended with a dashboard or Streamlit app later.

## Notes

- The training pipeline is for structured tabular data.
- If you want image, text, or audio datasets, the preprocessing and model choice should change.
- The sample canteen CSV is included only as demo data. The model is trained only when you provide a dataset.

