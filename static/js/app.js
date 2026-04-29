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
const itemPieChartCtx = document.getElementById("item-pie-chart").getContext("2d");
const demandBarChartCtx = document.getElementById("demand-bar-chart").getContext("2d");
const weatherSummary = document.getElementById("weather-summary");
const cityInput = document.getElementById("city");
const weatherButton = document.getElementById("get-weather");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatLog = document.getElementById("chat-log");

let pieChart;
let barChart;

function createChartDefaults() {
  if (pieChart) pieChart.destroy();
  if (barChart) barChart.destroy();

  pieChart = new Chart(itemPieChartCtx, {
    type: "pie",
    data: { labels: [], datasets: [{ backgroundColor: ["#3b82f6", "#f59e0b", "#10b981"], data: [] }] },
    options: { responsive: true, plugins: { legend: { position: "bottom" } } }
  });

  barChart = new Chart(demandBarChartCtx, {
    type: "bar",
    data: { labels: [], datasets: [
      { label: "Demand", backgroundColor: "#3b82f6", data: [] },
      { label: "Production", backgroundColor: "#6366f1", data: [] }
    ] },
    options: {
      responsive: true,
      scales: { y: { beginAtZero: true, ticks: { precision: 0 } } },
      plugins: { legend: { position: "top" } }
    }
  });
}

function setStatusChip(risk) {
  const value = risk.toLowerCase();
  wasteRiskEl.textContent = risk;
  wasteRiskEl.className = `status-chip status-${value}`;
}

function resetDashboard() {
  resultSection.style.display = "none";
  notification.textContent = "";
  notification.className = "notification";
  demandTableBody.innerHTML = "";
  recommendationsList.innerHTML = "";
  insightsList.innerHTML = "";
  supplyList.innerHTML = "";
  donationAlert.style.display = "none";
  notifyButton.disabled = true;
  runButton.disabled = false;
}

function showNotification(message, type = "info") {
  notification.textContent = message;
  notification.className = type === "error" ? "notification error" : "notification";
}

function populateDashboard(data) {
  resultSection.style.display = "block";
  predictedStudentsEl.textContent = data.predictedStudents;
  wastePercentEl.textContent = `${data.wasteSummary.wastePercent}%`;
  profitEstimateEl.textContent = `₹ ${data.profitEstimate.toLocaleString()}`;
  wasteUnitsEl.textContent = data.wasteSummary.totalWasteUnits;
  setStatusChip(data.wasteSummary.wasteRisk);

  demandTableBody.innerHTML = data.itemDemands.map(item => `
      <tr>
        <td>${item.itemName}</td>
        <td>${item.predictedDemand}</td>
        <td>${item.popularityScore}</td>
        <td>${item.shelfLifeHours} hrs</td>
      </tr>
    `).join("");

  recommendationsList.innerHTML = data.recommendations.map(item => `
      <li>
        <strong>${item.itemName}</strong> — ${item.recommendation}
        <div class="legend-key"><span class="legend-color" style="background:#3b82f6"></span> Demand ${item.predictedDemand}</div>
        <div class="legend-key"><span class="legend-color" style="background:#6366f1"></span> Produce ${item.suggestedProduction}</div>
      </li>
    `).join("");

  insightsList.innerHTML = data.insights.map(text => `<li>${text}</li>`).join("");
  renderSupplyList(data.supplyData);
  renderWeatherSummary(data.weatherInfo);
  renderMenuTable(data.itemDemands, data.recommendations);

  if (data.donationAlert) {
    donationAlert.style.display = "grid";
    donationAmount.textContent = data.donationAlert.excessQuantity;
    notifyButton.disabled = false;
  } else {
    donationAlert.style.display = "none";
    notifyButton.disabled = true;
  }

  updateCharts(data);
}

function renderWeatherSummary(weatherInfo) {
  if (!weatherInfo || !weatherInfo.city) {
    weatherSummary.style.display = "none";
    return;
  }

  weatherSummary.style.display = "block";
  weatherSummary.innerHTML = `
    <strong>Weather data for ${weatherInfo.city}:</strong>
    <div>${weatherInfo.description} • ${weatherInfo.temperatureC !== null ? weatherInfo.temperatureC + "°C" : "N/A"} • Humidity ${weatherInfo.humidity !== null ? weatherInfo.humidity + "%" : "N/A"}</div>
    <div style="margin-top:8px;color:#334155;">Source: ${weatherInfo.source}</div>
  `;
}

function renderSupplyList(supplyData) {
  supplyList.innerHTML = supplyData.map(item => `
      <li class="supply-item">
        <div>
          <strong>${item.ingredient}</strong>
          <div class="metric-note">Seasonal factor: ${item.seasonalFactor.toFixed(2)}</div>
        </div>
        <span class="supply-status supply-${item.availability.toLowerCase()}">${item.availability}</span>
      </li>
    `).join("");
}

function renderMenuTable(itemDemands, recommendations) {
  const menuTableBody = document.getElementById("menu-table-body");
  
  menuTableBody.innerHTML = recommendations.map(rec => {
    const item = itemDemands.find(d => d.itemName === rec.itemName);
    if (!item) return "";
    
    const costPerUnit = item.costPerUnit;
    const pricePerUnit = item.pricePerUnit;
    const quantity = rec.suggestedProduction;
    const expectedDemand = item.predictedDemand;
    const profit = rec.expectedProfit;
    
    return `
      <tr>
        <td class="item-name">${rec.itemName}</td>
        <td class="quantity"><strong>${quantity} units</strong></td>
        <td class="quantity">${expectedDemand} units</td>
        <td class="cost-cell">₹ ${costPerUnit}</td>
        <td class="cost-cell">₹ ${pricePerUnit}</td>
        <td class="profit-cell">₹ ${profit.toLocaleString()}</td>
      </tr>
    `;
  }).join("");
}

function updateCharts(data) {
  const labels = data.itemDemands.map((item) => item.itemName);
  const demandData = data.itemDemands.map((item) => item.predictedDemand);
  const productionData = data.recommendations.map((item) => item.suggestedProduction);

  pieChart.data.labels = labels;
  pieChart.data.datasets[0].data = demandData;
  pieChart.update();

  barChart.data.labels = labels;
  barChart.data.datasets[0].data = demandData;
  barChart.data.datasets[1].data = productionData;
  barChart.update();
}

async function handleSubmit(event) {
  event.preventDefault();
  resetDashboard();

  const formData = new FormData(form);
  const datasetFile = document.getElementById("dataset-file").files[0];
  const city = cityInput.value.trim();
  const payload = {
    dayOfWeek: formData.get("dayOfWeek"),
    weather: formData.get("weather"),
    event: formData.get("event"),
    yesterdaySales: parseInt(formData.get("yesterdaySales"), 10) || 0,
    supplyAvailability: formData.get("supplyAvailability")
  };

  showNotification("Forecasting demand based on the selected scenario...");

  try {
    runButton.disabled = true;
    notifyButton.disabled = true;
    const options = datasetFile ? { method: "POST", body: new FormData() } : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) };
    if (datasetFile) {
      options.body.append("dataset", datasetFile);
      Object.keys(payload).forEach((key) => options.body.append(key, payload[key]));
    }

    const response = await fetch("/predict", options);
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Unable to fetch forecast");
    }

    if (city) {
      const weatherInfo = await fetchWeather(city);
      data.weatherInfo = weatherInfo;
    }

    populateDashboard(data);
    showNotification("Demand forecast generated successfully.");
  } catch (error) {
    showNotification(error.message || "Failed to generate forecast.", "error");
  } finally {
    runButton.disabled = false;
  }
}

async function fetchWeather(city) {
  try {
    const response = await fetch(`/weather?city=${encodeURIComponent(city)}`);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Unable to fetch weather data");
    }
    return data;
  } catch (error) {
    showNotification(error.message || "Weather lookup failed.", "error");
    return { city, weather: "Unknown", description: "Weather unavailable", temperatureC: null, humidity: null, source: "client" };
  }
}

form.addEventListener("submit", handleSubmit);
weatherButton.addEventListener("click", async () => {
  const city = cityInput.value.trim();
  if (!city) {
    showNotification("Enter a city to fetch weather.", "error");
    return;
  }
  showNotification("Fetching weather data...");
  const weatherInfo = await fetchWeather(city);
  renderWeatherSummary(weatherInfo);
  if (weatherInfo.weather && ["Sunny", "Rainy", "Cloudy"].includes(weatherInfo.weather)) {
    document.getElementById("weather").value = weatherInfo.weather;
  }
  showNotification("Weather data updated.");
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;
  appendChatBubble(text, "user");
  chatInput.value = "";
  showNotification("Generating assistant response...");

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text })
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Chatbot request failed.");
    }
    appendChatBubble(data.reply, "assistant");
    showNotification("Assistant replied successfully.");
  } catch (error) {
    appendChatBubble("I'm having trouble responding right now.", "assistant");
    showNotification(error.message || "Chatbot failed.", "error");
  }
});

function appendChatBubble(message, role) {
  const bubble = document.createElement("div");
  bubble.className = `chat-bubble ${role}`;
  bubble.textContent = message;
  chatLog.appendChild(bubble);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function donateConfirmation() {
  alert("NGO notified successfully. Donation coordination message has been sent.");
}

createChartDefaults();
