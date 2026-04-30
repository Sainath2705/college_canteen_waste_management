from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from canteen_analytics import (
    get_canteen_status,
    frame_fingerprint,
    load_canteen_artifact,
    load_dataset,
    predict_canteen_forecast,
    summarize_dataset,
    train_canteen_models,
)
from ml_service import (
    DEFAULT_MODEL_PATH,
    latest_model_status,
    load_tabular_dataframe,
    predict_records,
    train_tabular_model,
)

BASE_DIR = Path(__file__).resolve().parent
DEMO_DATASET = BASE_DIR / "data" / "canteen_dataset_300_rows.csv"

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


def dataset_fingerprint(frame):
    return frame_fingerprint(frame)


def reset_cached_results():
    for key in (
        "canteen_artifact",
        "canteen_forecast_result",
        "custom_training_result",
    ):
        st.session_state.pop(key, None)


def apply_styles():
    st.markdown(
        """
        <style>
        .hero {
            padding: 2rem;
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(14, 116, 144, 0.12), rgba(15, 23, 42, 0.05));
            border: 1px solid rgba(15, 23, 42, 0.08);
            margin-bottom: 1.25rem;
        }
        .hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2.35rem;
            line-height: 1.05;
        }
        .hero p {
            margin: 0;
            font-size: 1rem;
            color: #475569;
            max-width: 68rem;
        }
        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: #0f766e;
            font-size: 0.74rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
        }
        .card {
            background: white;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 20px;
            padding: 1rem 1.15rem;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.04);
        }
        .section-label {
            color: #0f766e;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.72rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .stDataFrame, .stPlotlyChart {
            border-radius: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_demo_data():
    return load_dataset(DEMO_DATASET)


def load_uploaded_dataset(uploaded_file):
    if uploaded_file is None:
        return None
    return load_tabular_dataframe(uploaded_file, filename=uploaded_file.name)


def fetch_live_weather(city):
    city = (city or "").strip()
    if not city:
        return None

    if not OPENWEATHER_API_KEY:
        return {
            "city": city.title(),
            "weather": "Sunny",
            "description": "Fallback weather because no API key is configured.",
            "temperatureC": 28.0,
            "humidity": 50,
            "source": "fallback",
        }

    try:
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"},
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
        weather_main = data["weather"][0]["main"]
        category = "Rainy" if "rain" in weather_main.lower() else "Cloudy" if "cloud" in weather_main.lower() else "Sunny"
        return {
            "city": data.get("name", city).title(),
            "weather": category,
            "description": data["weather"][0]["description"].title(),
            "temperatureC": round(data["main"]["temp"], 1),
            "humidity": data["main"]["humidity"],
            "source": "openweathermap",
        }
    except Exception as exc:
        return {
            "city": city.title(),
            "weather": "Sunny",
            "description": "Weather lookup failed, using fallback conditions.",
            "temperatureC": 28.0,
            "humidity": 50,
            "source": "fallback",
            "error": str(exc),
        }


def metric_columns(data):
    if not data:
        return
    cols = st.columns(len(data))
    for column, item in zip(cols, data):
        with column:
            st.metric(item["label"], item["value"], item.get("delta"))
            if item.get("help"):
                st.caption(item["help"])


def show_dataset_preview(frame, max_rows=8):
    st.dataframe(frame.head(max_rows), use_container_width=True, hide_index=True)


def show_profile_cards(profile):
    metrics = []
    metrics.append({"label": "Rows", "value": f"{profile.get('rows', 0):,}", "help": "Training samples available"})
    metrics.append({"label": "Columns", "value": str(len(profile.get("columns", []))), "help": "Detected dataset fields"})
    if profile.get("dateRange"):
        metrics.append({"label": "Date range", "value": f"{profile['dateRange']['start']} to {profile['dateRange']['end']}", "help": "Historical coverage"})
    average_footfall = profile.get("targetMeans", {}).get("totalStudents")
    if average_footfall is not None and pd.notna(average_footfall):
        metrics.append({"label": "Avg footfall", "value": f"{average_footfall:.1f}", "help": "Historic student visits"})
    metric_columns(metrics[:4])


def show_forecast_cards(result):
    metrics = [
        {"label": "Predicted footfall", "value": f"{result['predictedFootfall']:,}", "help": "Next-day student demand"},
        {"label": "Waste %", "value": f"{result['wasteSummary']['wastePercent']}%", "help": result["wasteSummary"]["wasteRisk"] + " waste risk"},
        {"label": "Projected profit", "value": f"Rs. {result['wasteSummary']['profit']:,.0f}", "help": "Revenue minus cost"},
        {"label": "Donation trigger", "value": "Yes" if result["donationAlert"] else "No", "help": "Surplus food coordination"},
    ]
    metric_columns(metrics)


def plot_menu_plan(result):
    menu_df = pd.DataFrame(result["menuPlan"])
    if menu_df.empty:
        return
    fig = px.bar(
        menu_df,
        x="itemName",
        y=["predictedDemand", "suggestedProduction"],
        barmode="group",
        color_discrete_sequence=["#0ea5e9", "#0f766e"],
        title="Demand vs Suggested Production",
    )
    fig.update_layout(height=420, legend_title_text="", xaxis_title="", yaxis_title="Units")
    st.plotly_chart(fig, use_container_width=True)


def plot_profit_and_waste(result):
    menu_df = pd.DataFrame(result["menuPlan"])
    if menu_df.empty:
        return
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            menu_df,
            x="itemName",
            y="wasteEstimate",
            color="wasteEstimate",
            color_continuous_scale="OrRd",
            title="Waste Estimate by Item",
        )
        fig.update_layout(height=360, xaxis_title="", yaxis_title="Units")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(
            menu_df,
            x="itemName",
            y="expectedProfit",
            color="expectedProfit",
            color_continuous_scale="Viridis",
            title="Expected Profit by Item",
        )
        fig.update_layout(height=360, xaxis_title="", yaxis_title="Profit")
        st.plotly_chart(fig, use_container_width=True)


def plot_history(frame):
    if "totalStudents" not in frame.columns:
        return
    df = frame.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
        df = df.dropna(subset=["date"])
        if not df.empty:
            fig = px.line(df, x="date", y="totalStudents", title="Historical Footfall Trend")
            fig.update_layout(height=360, xaxis_title="", yaxis_title="Students")
            st.plotly_chart(fig, use_container_width=True)


def render_overview_tab(frame, mode, canteen_artifact, custom_status):
    st.markdown("### Overview")
    if frame is None:
        st.info("Upload a dataset or load the demo dataset to begin.")
        return

    profile = summarize_dataset(frame) if mode == "Canteen Intelligence" else {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "preview": frame.head(5).replace({pd.NA: None}).to_dict(orient="records"),
    }
    show_profile_cards(profile)

    st.markdown("#### Dataset preview")
    show_dataset_preview(frame)

    st.markdown("#### Historical trend")
    plot_history(frame)

    if mode == "Canteen Intelligence" and canteen_artifact:
        st.markdown("#### Training metrics")
        metrics_rows = []
        for target, values in canteen_artifact.get("metrics", {}).items():
            metrics_rows.append(
                {
                    "Target": target,
                    "MAE": round(values.get("mae", 0), 2),
                    "RMSE": round(values.get("rmse", 0), 2),
                    "R2": round(values.get("r2", 0), 3),
                }
            )
        if metrics_rows:
            st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Train the model to see metrics.")
    elif mode == "Custom Tabular Trainer" and custom_status:
        st.markdown("#### Generic model status")
        st.json(custom_status["metadata"] if custom_status.get("trained") else custom_status)


def render_forecast_tab(frame, canteen_artifact):
    st.markdown("### Forecast Studio")
    if not canteen_artifact:
        st.info("Train the canteen model first to enable forecasting.")
        return

    current_default = date.today() + timedelta(days=1)
    live_weather = st.session_state.get("live_weather") or {}
    weather_options = ["Sunny", "Cloudy", "Rainy"]
    default_weather = live_weather.get("weather", "Sunny")
    weather_index = weather_options.index(default_weather) if default_weather in weather_options else 0
    default_temperature = float(live_weather.get("temperatureC", 28.0))
    default_humidity = float(live_weather.get("humidity", 50.0))
    default_city = live_weather.get("city", "")

    with st.form("forecast_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            forecast_date = st.date_input("Forecast date", value=current_default)
            weather = st.selectbox("Weather", weather_options, index=weather_index)
            supply = st.selectbox("Supply availability", ["High", "Medium", "Low"], index=1)
        with col2:
            event = st.selectbox("Academic event", ["Normal", "Exam", "Festival", "Holiday"], index=0)
            city = st.text_input("City for live weather", value=default_city)
            temperature = st.number_input("Temperature (C)", value=default_temperature, step=0.5)
        with col3:
            humidity = st.number_input("Humidity (%)", value=default_humidity, min_value=0.0, max_value=100.0, step=1.0)
            submit_forecast = st.form_submit_button("Use scenario and forecast")

    if submit_forecast:
        scenario = {
            "forecastDate": forecast_date.isoformat(),
            "weather": weather,
            "event": event,
            "temperatureC": temperature,
            "humidity": humidity,
            "city": city,
        }
        result = predict_canteen_forecast(canteen_artifact, scenario, supply_availability=supply)
        st.session_state["canteen_forecast_result"] = result

    result = st.session_state.get("canteen_forecast_result")
    if not result:
        st.info("Choose a scenario and generate the next-day forecast.")
        return

    show_forecast_cards(result)

    st.markdown("#### Decision summary")
    st.write(
        f"Forecast date: **{result['scenario']['forecastDate']}** | "
        f"Weather: **{result['scenario']['weather']}** | "
        f"Event: **{result['scenario']['event']}** | "
        f"Supply: **{result['scenario']['supplyAvailability']}**"
    )

    if result["donationAlert"]:
        st.success(result["donationAlert"]["message"])
        st.write(f"Estimated surplus: **{result['donationAlert']['excessQuantity']} units**")
    else:
        st.info("No donation alert is required for this scenario.")

    col1, col2 = st.columns([1.35, 1])
    with col1:
        plot_menu_plan(result)
    with col2:
        st.markdown("#### Optimization benchmark")
        benchmark = result["benchmark"]
        st.metric("Naive waste", f"{benchmark['naiveWasteUnits']} units")
        st.metric("Optimized waste", f"{benchmark['optimizedWasteUnits']} units")
        st.metric("Waste reduction", f"{benchmark['wasteReductionPercent']}%")
        st.metric("Units saved", f"{benchmark['wasteReductionUnits']} units")

    st.markdown("#### Menu planner")
    menu_df = pd.DataFrame(result["menuPlan"])
    st.dataframe(
        menu_df[
            [
                "itemName",
                "predictedDemand",
                "suggestedProduction",
                "wasteEstimate",
                "bufferPercent",
                "expectedProfit",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Profit and waste analysis")
    plot_profit_and_waste(result)

    st.markdown("#### Insights")
    for insight in result["insights"]:
        st.write(f"- {insight}")

    st.download_button(
        "Download forecast JSON",
        data=json.dumps(result, indent=2, default=str),
        file_name="canteen_forecast_report.json",
        mime="application/json",
    )


def render_data_explorer_tab(frame, mode, canteen_artifact):
    st.markdown("### Data Explorer")
    if frame is None:
        st.info("Upload a dataset or use the demo dataset first.")
        return

    st.markdown("#### Feature statistics")
    numeric = frame.select_dtypes(include=["number"]).copy()
    if not numeric.empty:
        fig = px.imshow(
            numeric.corr(numeric_only=True).round(2),
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Teal",
            title="Numeric Correlation Matrix",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns detected for correlation analysis.")

    if mode == "Canteen Intelligence" and canteen_artifact:
        st.markdown("#### Feature importance")
        targets = list(canteen_artifact.get("metrics", {}).keys())
        if not targets:
            st.info("Train the canteen model to surface feature importance.")
            return

        chosen = st.selectbox("Select target", targets, index=0)
        if chosen:
            top_features = canteen_artifact["metrics"][chosen]["topFeatures"]
            feature_df = pd.DataFrame(top_features)
            st.dataframe(feature_df, use_container_width=True, hide_index=True)
            fig = px.bar(feature_df, x="importance", y="feature", orientation="h", title=f"Top drivers for {chosen}")
            fig.update_layout(height=380, yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)


def render_custom_trainer_tab(frame):
    st.markdown("### Custom Tabular Trainer")
    st.caption("Use this if you want to train on a Kaggle-style CSV or JSON dataset that is not the canteen schema.")
    if frame is None:
        st.info("Load a dataset to train a generic tabular model.")
        return

    columns = list(frame.columns)
    if not columns:
        st.info("The loaded dataset does not contain any usable columns.")
        return

    target_default = columns[-1] if columns else None

    with st.form("custom_train_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            target_column = st.selectbox("Target column", columns, index=columns.index(target_default) if target_default in columns else 0)
            task_type = st.selectbox("Task type", ["auto", "regression", "classification"], index=0)
        with c2:
            drop_columns = st.multiselect("Drop columns", columns, default=[])
        with c3:
            test_size = st.slider("Validation split", 0.1, 0.4, 0.2, 0.05)
        train_custom = st.form_submit_button("Train generic model")

    if train_custom:
        with st.spinner("Training custom tabular model..."):
            result = train_tabular_model(
                frame,
                target_column=target_column,
                task_type=task_type,
                drop_columns=drop_columns,
                test_size=float(test_size),
                random_state=42,
                model_path=str(DEFAULT_MODEL_PATH),
            )
        st.session_state["custom_training_result"] = result
        st.success("Custom model trained successfully.")

    result = st.session_state.get("custom_training_result")
    if not result:
        return

    st.markdown("#### Model summary")
    st.json(result)

    st.markdown("#### Predict with the custom model")
    sample_frame = frame.head(1).where(pd.notna(frame.head(1)), None)
    sample_json = json.dumps(sample_frame.to_dict(orient="records")[0], indent=2, default=str)
    payload = st.text_area("Single record JSON", value=sample_json, height=180)
    if st.button("Run custom prediction"):
        try:
            record = json.loads(payload)
            prediction = predict_records(record, model_path=result["modelPath"])
            st.success("Prediction generated.")
            st.json(prediction)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def render_deployment_tab():
    st.markdown("### Deployment Notes")
    st.write(
        "Run the dashboard with `streamlit run streamlit_app.py`. "
        "The Flask API remains available through `python app.py` if you also want endpoint access."
    )
    st.markdown(
        """
        - Main experience: `streamlit_app.py`
        - API health: `/api/health`
        - Canteen model artifact: `models/canteen_intelligence.joblib`
        - Generic model artifact: `models/latest_model.joblib`
        - Workflow: upload data, train, forecast, and export the JSON report
        """
    )


def main():
    st.set_page_config(
        page_title="CanteenIQ",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_styles()

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Data Science and Analytics Hackathon</div>
            <h1>CanteenIQ: Forecast, optimize, and donate with data</h1>
            <p>
                Upload historical canteen data, train a time-aware forecasting engine,
                generate next-day menu recommendations, reduce waste, and prepare donation-ready surplus alerts.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Workflow")
        mode = st.radio(
            "Choose a mode",
            ["Canteen Intelligence", "Custom Tabular Trainer"],
            index=0,
        )
        source = st.radio("Dataset source", ["Demo dataset", "Upload file"], index=0)
        uploaded = st.file_uploader("Upload CSV or JSON", type=["csv", "json"])

        city = st.text_input("City for weather lookup", value="Bengaluru")
        if st.button("Load live weather"):
            st.session_state["live_weather"] = fetch_live_weather(city)

        live_weather = st.session_state.get("live_weather")
        if live_weather:
            st.success(
                f"{live_weather['city']}: {live_weather['weather']} | "
                f"{live_weather['temperatureC']} C | {live_weather['humidity']}%"
            )

        st.markdown("### Training")
        train_button = st.button("Train selected mode")

    if source == "Demo dataset":
        frame = load_demo_data()
    else:
        if uploaded is None:
            st.warning("Upload a CSV or JSON file to continue.")
            return
        frame = load_uploaded_dataset(uploaded)

    if frame is None:
        st.warning("Upload a dataset or use the demo dataset to continue.")
        return

    current_signature = dataset_fingerprint(frame)
    if st.session_state.get("active_dataset_signature") != current_signature:
        reset_cached_results()
        st.session_state["active_dataset_signature"] = current_signature

    canteen_artifact = None
    canteen_status = get_canteen_status()
    custom_status = latest_model_status()

    if mode == "Canteen Intelligence":
        if train_button:
            with st.spinner("Training canteen intelligence models..."):
                canteen_artifact = train_canteen_models(frame)
                st.session_state["canteen_artifact"] = canteen_artifact
            st.success("Canteen model trained and saved successfully.")
        else:
            canteen_artifact = st.session_state.get("canteen_artifact")
            if canteen_artifact is None and canteen_status["trained"]:
                try:
                    loaded_artifact = load_canteen_artifact()
                    if loaded_artifact.get("datasetFingerprint") == current_signature:
                        canteen_artifact = loaded_artifact
                        st.session_state["canteen_artifact"] = canteen_artifact
                    else:
                        st.info("A saved canteen model exists, but it was trained on a different dataset. Train this dataset to refresh the forecast engine.")
                except Exception:
                    canteen_artifact = None
    else:
        if train_button:
            with st.spinner("Training custom tabular model..."):
                st.session_state["custom_training_result"] = train_tabular_model(
                    frame,
                    target_column=frame.columns[-1],
                    task_type="auto",
                    drop_columns=[],
                    test_size=0.2,
                    random_state=42,
                    model_path=str(DEFAULT_MODEL_PATH),
                )
            st.success("Generic model trained successfully.")

    tab_overview, tab_forecast, tab_explorer, tab_custom, tab_deployment = st.tabs(
        ["Overview", "Forecast Studio", "Data Explorer", "Custom Trainer", "Deployment"]
    )

    with tab_overview:
        render_overview_tab(frame, mode, canteen_artifact, custom_status)
    with tab_forecast:
        render_forecast_tab(frame, canteen_artifact)
    with tab_explorer:
        render_data_explorer_tab(frame, mode, canteen_artifact)
    with tab_custom:
        render_custom_trainer_tab(frame)
    with tab_deployment:
        render_deployment_tab()


if __name__ == "__main__":
    main()
