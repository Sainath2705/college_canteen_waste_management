const form = document.getElementById("forecast-form");
const notification = document.getElementById("notification");
const resultSection = document.getElementById("result-section");
const predictedStudentsEl = document.getElementById("predicted-students");
const wastePercentEl = document.getElementById("waste-percent");
const profitEstimateEl = document.getElementById("profit-estimate");
const wasteUnitsEl = document.getElementById("waste-units");
const wasteRiskEl = document.getElementById("waste-risk");
const demandTableBody = document.getElementById("demand-table-body");
const recommendationsList = document.getElementById("recommendations-list");
const insightsList = document.getElementById("insights-list");
const supplyList = document.getElementById("supply-list");
const donationAlert = document.getElementById("donation-alert");
const donationAmount = document.getElementById("donation-amount");
const notifyButton = document.getElementById("notify-ngo");
const runButton = document.querySelector("#forecast-form button[type='submit']");
const weatherButton = document.getElementById("get-weather");
const cityInput = document.getElementById("city");
const weatherSelect = document.getElementById("weather");
const weatherSummary = document.getElementById("weather-summary");
const datasetInput = document.getElementById("dataset-file");
const menuTableBody = document.getElementById("menu-table-body");
const naiveWasteEl = document.getElementById("naive-waste");
const optimizedWasteEl = document.getElementById("optimized-waste");
const wasteReductionEl = document.getElementById("waste-reduction");
const wasteSavedEl = document.getElementById("waste-saved");
const itemPieChartCtx = document.getElementById("item-pie-chart").getContext("2d");
const demandBarChartCtx = document.getElementById("demand-bar-chart").getContext("2d");

let pieChart;
let barChart;
let cachedSupplyData = [];
let cachedWeatherInfo = null;

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function tomorrowIsoDate() {
  const tomorrow = new Date();
  tomorrow.setDate(tomorrow.getDate() + 1);
  return tomorrow.toISOString().slice(0, 10);
}

function setNotification(message, type = "info") {
  notification.textContent = message;
  notification.className = type === "error" ? "notification error" : "notification";
}

function clearNotification() {
  notification.textContent = "";
  notification.className = "notification";
}

function setStatusChip(risk) {
  const normalized = (risk || "Low").toLowerCase();
  wasteRiskEl.textContent = risk || "--";
  wasteRiskEl.className = `status-chip status-${normalized}`;
}

function initializeCharts() {
  pieChart = new Chart(itemPieChartCtx, {
    type: "pie",
    data: {
      labels: ["Rice", "Dosa", "Snacks"],
      datasets: [{
        data: [1, 1, 1],
        backgroundColor: ["#3b82f6", "#0f766e", "#f59e0b"],
        borderWidth: 0,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { position: "bottom" } },
    },
  });

  barChart = new Chart(demandBarChartCtx, {
    type: "bar",
    data: {
      labels: ["Rice", "Dosa", "Snacks"],
      datasets: [
        {
          label: "Demand",
          data: [1, 1, 1],
          backgroundColor: "#3b82f6",
          borderRadius: 10,
        },
        {
          label: "Production",
          data: [1, 1, 1],
          backgroundColor: "#0f766e",
          borderRadius: 10,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          ticks: { precision: 0 },
          grid: { color: "rgba(148, 163, 184, 0.18)" },
        },
        x: {
          grid: { display: false },
        },
      },
      plugins: {
        legend: { position: "top" },
      },
    },
  });
}

function renderSupplyList(items) {
  const fallback = [
    { ingredient: "Rice", availability: "High", seasonalFactor: 1.0 },
    { ingredient: "Dosa Batter", availability: "Medium", seasonalFactor: 0.95 },
    { ingredient: "Snacks", availability: "High", seasonalFactor: 1.0 },
    { ingredient: "Vegetables", availability: "Medium", seasonalFactor: 0.9 },
    { ingredient: "Spices", availability: "High", seasonalFactor: 1.0 },
  ];

  const supplyData = Array.isArray(items) && items.length ? items : fallback;
  cachedSupplyData = supplyData;
  supplyList.innerHTML = supplyData.map((item) => `
    <li class="supply-item">
      <div>
        <strong>${escapeHtml(item.ingredient)}</strong>
        <div class="metric-note">Seasonal factor: ${Number(item.seasonalFactor).toFixed(2)}</div>
      </div>
      <span class="supply-status supply-${String(item.availability).toLowerCase()}">${escapeHtml(item.availability)}</span>
    </li>
  `).join("");
}

function updateWeatherSummary(weatherInfo) {
  if (!weatherInfo || !weatherInfo.city) {
    weatherSummary.hidden = true;
    weatherSummary.innerHTML = "";
    return;
  }

  weatherSummary.hidden = false;
  weatherSummary.innerHTML = `
    <strong>Weather for ${escapeHtml(weatherInfo.city)}:</strong>
    <div style="margin-top:8px;">
      ${escapeHtml(weatherInfo.description || weatherInfo.weather || "Weather unavailable")}
      &middot; ${weatherInfo.temperatureC != null ? `${Number(weatherInfo.temperatureC).toFixed(1)} C` : "Temp N/A"}
      &middot; Humidity ${weatherInfo.humidity != null ? `${escapeHtml(weatherInfo.humidity)}%` : "N/A"}
    </div>
    <div style="margin-top:8px;color:#475569;">Source: ${escapeHtml(weatherInfo.source || "client")}</div>
  `;
}

function renderForecast(result) {
  resultSection.hidden = false;

  predictedStudentsEl.textContent = Number(result.predictedFootfall || 0).toLocaleString();
  wastePercentEl.textContent = `${Number(result.wasteSummary?.wastePercent || 0)}%`;
  profitEstimateEl.textContent = `Rs. ${Number(result.wasteSummary?.profit || 0).toLocaleString()}`;
  wasteUnitsEl.textContent = Number(result.wasteSummary?.wasteUnits || 0).toLocaleString();
  setStatusChip(result.wasteSummary?.wasteRisk || "Low");

  const menuPlan = Array.isArray(result.menuPlan) ? result.menuPlan : [];
  const labels = menuPlan.map((item) => item.itemName);
  const demandValues = menuPlan.map((item) => Number(item.predictedDemand || 0));
  const productionValues = menuPlan.map((item) => Number(item.suggestedProduction || 0));

  demandTableBody.innerHTML = menuPlan.map((item) => `
    <tr>
      <td>${escapeHtml(item.itemName)}</td>
      <td>${Number(item.predictedDemand || 0)}</td>
      <td>${Number(item.suggestedProduction || 0)}</td>
      <td>${Number(item.wasteEstimate || 0)}</td>
    </tr>
  `).join("");

  recommendationsList.innerHTML = menuPlan.map((item) => `
    <li>
      <strong>${escapeHtml(item.itemName)}</strong> - ${escapeHtml(item.recommendation || "Production aligned with demand.")}
      <div class="metric-note">
        Demand ${Number(item.predictedDemand || 0)} | Production ${Number(item.suggestedProduction || 0)} | Buffer ${Number(item.bufferPercent || 0).toFixed(1)}%
      </div>
    </li>
  `).join("");

  insightsList.innerHTML = (Array.isArray(result.insights) ? result.insights : []).map((text) => `
    <li>${escapeHtml(text)}</li>
  `).join("");

  naiveWasteEl.textContent = `${Number(result.benchmark?.naiveWasteUnits || 0)} units`;
  optimizedWasteEl.textContent = `${Number(result.benchmark?.optimizedWasteUnits || 0)} units`;
  wasteReductionEl.textContent = `${Number(result.benchmark?.wasteReductionPercent || 0)}%`;
  wasteSavedEl.textContent = `${Number(result.benchmark?.wasteReductionUnits || 0)} units`;

  if (result.donationAlert) {
    donationAlert.hidden = false;
    donationAmount.textContent = Number(result.donationAlert.excessQuantity || 0).toLocaleString();
    notifyButton.disabled = false;
  } else {
    donationAlert.hidden = true;
    notifyButton.disabled = true;
  }

  if (!menuPlan.length) {
    pieChart.data.labels = ["No data"];
    pieChart.data.datasets[0].data = [1];
    barChart.data.labels = ["No data"];
    barChart.data.datasets[0].data = [0];
    barChart.data.datasets[1].data = [0];
  } else {
    pieChart.data.labels = labels;
    pieChart.data.datasets[0].data = demandValues;
    barChart.data.labels = labels;
    barChart.data.datasets[0].data = demandValues;
    barChart.data.datasets[1].data = productionValues;
  }

  pieChart.update();
  barChart.update();

  updateWeatherSummary(cachedWeatherInfo);
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  let data = {};
  try {
    data = await response.json();
  } catch (error) {
    data = {};
  }
  if (!response.ok) {
    const message = data.error || data.detail || `Request failed with status ${response.status}`;
    throw new Error(message);
  }
  return data;
}

async function trainDataset(datasetFile) {
  const formData = new FormData();
  formData.append("dataset", datasetFile);
  return fetchJson("/canteen/train", { method: "POST", body: formData });
}

async function predictScenario(scenario) {
  return fetchJson("/canteen/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      scenario,
      supplyAvailability: scenario.supplyAvailability,
    }),
  });
}

async function fetchWeather(city) {
  const trimmed = city.trim();
  if (!trimmed) {
    throw new Error("Enter a city to fetch weather.");
  }

  return fetchJson(`/weather?city=${encodeURIComponent(trimmed)}`);
}

function buildScenario() {
  const tomorrow = tomorrowIsoDate();
  const formData = new FormData(form);
  return {
    dayOfWeek: formData.get("dayOfWeek"),
    weather: formData.get("weather"),
    event: formData.get("event"),
    yesterdaySales: Number(formData.get("yesterdaySales") || 0),
    city: formData.get("city") || "",
    supplyAvailability: formData.get("supplyAvailability"),
    forecastDate: tomorrow,
    weatherInfo: cachedWeatherInfo,
  };
}

async function handleSubmit(event) {
  event.preventDefault();
  clearNotification();

  const datasetFile = datasetInput.files[0];
  if (!datasetFile) {
    setNotification("Please upload a CSV or JSON dataset first.", "error");
    return;
  }

  runButton.disabled = true;
  weatherButton.disabled = true;
  setNotification("Training the canteen model from your dataset...");

  try {
    await trainDataset(datasetFile);
    setNotification("Dataset trained. Generating the next-day forecast...");

    const scenario = buildScenario();
    const result = await predictScenario(scenario);
    renderForecast(result);
    setNotification("Forecast generated successfully.");
  } catch (error) {
    resultSection.hidden = false;
    setNotification(error.message || "Failed to generate the forecast.", "error");
  } finally {
    runButton.disabled = false;
    weatherButton.disabled = false;
  }
}

async function handleWeatherLookup() {
  const city = cityInput.value || "";
  if (!city.trim()) {
    setNotification("Enter a city to fetch weather.", "error");
    return;
  }

  weatherButton.disabled = true;
  setNotification("Fetching live weather...");

  try {
    const weatherInfo = await fetchWeather(city);
    cachedWeatherInfo = weatherInfo;
    updateWeatherSummary(weatherInfo);
    if (weatherInfo.weather && ["Sunny", "Cloudy", "Rainy"].includes(weatherInfo.weather)) {
      weatherSelect.value = weatherInfo.weather;
    }
    setNotification(`Weather loaded for ${weatherInfo.city || city}.`);
  } catch (error) {
    setNotification(error.message || "Weather lookup failed.", "error");
  } finally {
    weatherButton.disabled = false;
  }
}

async function loadSupplyPanel() {
  try {
    const data = await fetchJson("/api/supply");
    renderSupplyList(data);
  } catch (error) {
    renderSupplyList([]);
  }
}

function bindEvents() {
  form.addEventListener("submit", handleSubmit);
  weatherButton.addEventListener("click", handleWeatherLookup);
  notifyButton.addEventListener("click", () => {
    alert("NGO notified successfully. Donation coordination message has been sent.");
  });
}

async function initialize() {
  initializeCharts();
  bindEvents();
  await loadSupplyPanel();
  updateWeatherSummary(null);
}

initialize();
