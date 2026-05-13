// web_interface/static/app.js
// Frontend logic for the live sign language glove dashboard.

(() => {
  const NUM_SENSORS = 5;
  const HISTORY_SIZE = 300;

  // Create sensor cards
  const sensorCardsContainer = document.getElementById("sensor-cards");
  const sensorCards = [];
  for (let i = 0; i < NUM_SENSORS; i++) {
    const card = document.createElement("div");
    card.className = "sensor-card card";
    const label = document.createElement("div");
    label.className = "label";
    label.textContent = `S${i}`;
    const value = document.createElement("div");
    value.className = "value";
    value.textContent = "0.00";
    card.appendChild(label);
    card.appendChild(value);
    sensorCardsContainer.appendChild(card);
    sensorCards.push({ card, label, value });
  }

  // Set up Chart.js for live plotting
  const ctx = document.getElementById("sensorChart").getContext("2d");
  // Colour palette for sensors
  const colours = [
    "#ff5252",
    "#448aff",
    "#69f0ae",
    "#ffb74d",
    "#b39ddb",
  ];
  const datasets = [];
  for (let i = 0; i < NUM_SENSORS; i++) {
    datasets.push({
      label: `S${i}`,
      data: [],
      borderColor: colours[i % colours.length],
      borderWidth: 2,
      fill: false,
      tension: 0.25,
    });
  }
  const chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: {
          labels: {
            color: "#e0e0e0",
          },
        },
      },
      scales: {
        x: {
          display: false,
        },
        y: {
          min: 0,
          max: 1,
          ticks: {
            color: "#e0e0e0",
          },
          grid: {
            color: "rgba(255,255,255,0.1)",
          },
        },
      },
    },
  });

  // Elements for prediction and status
  const predictionLetterEl = document.getElementById("prediction-letter");
  const predictionConfEl = document.getElementById("prediction-confidence");
  const statusEl = document.getElementById("status");
  const probabilitiesEl = document.getElementById("probabilities");

  // Helper to update probability list
  function updateProbabilities(probDict) {
    probabilitiesEl.innerHTML = "";
    if (!probDict || Object.keys(probDict).length === 0) {
      return;
    }
    // Sort probabilities descending
    const entries = Object.entries(probDict).sort((a, b) => b[1] - a[1]);
    entries.forEach(([label, prob]) => {
      const row = document.createElement("div");
      row.className = "prob-item";
      const nameSpan = document.createElement("span");
      nameSpan.textContent = label;
      const percSpan = document.createElement("span");
      const pct = (prob * 100).toFixed(1) + "%";
      percSpan.textContent = pct;
      row.appendChild(nameSpan);
      row.appendChild(percSpan);
      probabilitiesEl.appendChild(row);
    });
  }

  // Establish WebSocket connection
  function createWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const endpoint = `${protocol}://${window.location.host}/ws`;
    const ws = new WebSocket(endpoint);
    ws.onopen = () => {
      statusEl.textContent = "Connected. Waiting for samples…";
    };
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        // Update sensor numeric values
        if (Array.isArray(data.sensors)) {
          for (let i = 0; i < NUM_SENSORS; i++) {
            const v = data.sensors[i];
            sensorCards[i].value.textContent = v.toFixed(2);
            // Update chart data
            chart.data.datasets[i].data.push(v);
            if (chart.data.datasets[i].data.length > HISTORY_SIZE) {
              chart.data.datasets[i].data.shift();
            }
          }
          // Update labels array to match history length
          chart.data.labels.push("");
          if (chart.data.labels.length > HISTORY_SIZE) {
            chart.data.labels.shift();
          }
          chart.update("none");
        }
        // Update prediction and probabilities
        if (data.ready) {
          predictionLetterEl.textContent = data.prediction || "–";
          predictionConfEl.textContent = ((data.confidence || 0) * 100).toFixed(1) + "%";
          if (data.model_type === "both" && data.rf_prediction && data.cnn_prediction) {
            statusEl.textContent = `Ensemble: ${data.prediction} (RF: ${data.rf_prediction}, CNN: ${data.cnn_prediction})`;
          } else {
            statusEl.textContent = "Predicting…";
          }
          updateProbabilities(data.probabilities);
        } else {
          // Not ready yet; show buffer status
          predictionLetterEl.textContent = "–";
          predictionConfEl.textContent = "";
          const collected = data.samples_collected || 0;
          const needed = data.samples_needed || 0;
          statusEl.textContent = `Collecting buffer: ${collected}/${needed}`;
          updateProbabilities({});
        }
      } catch (err) {
        console.error("Failed to process message", err);
      }
    };
    ws.onclose = () => {
      statusEl.textContent = "Disconnected. Reconnecting…";
      setTimeout(createWebSocket, 1000);
    };
    ws.onerror = () => {
      statusEl.textContent = "Error in connection. Retrying…";
    };
    return ws;
  }

  // Kick off the WebSocket connection
  createWebSocket();
})();