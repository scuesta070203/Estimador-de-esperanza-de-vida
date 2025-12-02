const API_BASE = "http://127.0.0.1:5000";

async function loadMetadata() {
  try {
    const resp = await fetch(`${API_BASE}/metadata`);
    const meta = await resp.json();

    fillCountrySelect(meta.countries || []);
    fillHints(meta.ranges || {});
  } catch (err) {
    const msg = document.getElementById("globalMessage");
    msg.textContent =
      "No fue posible cargar la información inicial. Verifique que la API esté ejecutándose.";
  }
}

function fillCountrySelect(countries) {
  const sel = document.getElementById("countrySelect");
  sel.innerHTML = "";
  countries.forEach((c) => {
    const opt = document.createElement("option");
    opt.value = c;
    opt.textContent = c;
    sel.appendChild(opt);
  });
}

function fillHints(ranges) {
  const hints = document.querySelectorAll("[data-hint]");
  hints.forEach((el) => {
    const col = el.dataset.hint;
    const info = ranges[col];
    if (!info) return;
    const min = info.min;
    const max = info.max;
    const roundedMin = Math.round(min * 10) / 10;
    const roundedMax = Math.round(max * 10) / 10;
    el.textContent = `Rango sugerido: ${roundedMin} – ${roundedMax}`;
  });

  const yearEl = document.getElementById("yearInput");
  if (ranges["Year"]) {
    yearEl.min = 2000;
    yearEl.max = 2030;
    yearEl.value = 2025;
  }
}

function collectFeatures() {
  const elements = document.querySelectorAll("[data-col]");
  const payload = {};

  elements.forEach((el) => {
    const col = el.dataset.col;
    if (el.tagName === "SELECT") {
      payload[col] = el.value;
    } else {
      const value = el.value.trim();
      payload[col] = value === "" ? null : Number(value);
    }
  });

  // El modelo se entrenó con la columna Status.
  // Como quitamos el select, fijamos por defecto "Developing".
  if (!payload["Status"]) {
    payload["Status"] = "Developing";
  }

  return payload;
}

async function handleSubmit(event) {
  event.preventDefault();

  const regResult = document.getElementById("regResult");
  const knnResult = document.getElementById("knnResult");
  const mlpResult = document.getElementById("mlpResult");
  const globalMsg = document.getElementById("globalMessage");

  regResult.textContent = "Calculando...";
  knnResult.textContent = "Calculando...";
  mlpResult.textContent = "Calculando...";
  globalMsg.textContent = "";

  const features = collectFeatures();

  if (!features["Country"]) {
    globalMsg.textContent = "Por favor seleccione un país.";
    return;
  }

  try {
    const resp = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features }),
    });

    if (!resp.ok) {
      regResult.textContent = "No fue posible calcular.";
      knnResult.textContent = "No fue posible calcular.";
      mlpResult.textContent = "No fue posible calcular.";
      globalMsg.textContent = "Ocurrió un error al procesar la solicitud.";
      return;
    }

    const data = await resp.json();

    if (data.regression && data.regression.ok) {
      const v = data.regression.life_expectancy;
      regResult.textContent = `${v.toFixed(2)} años`;
    } else {
      regResult.textContent = "No se pudo obtener la predicción.";
    }

    if (data.knn && data.knn.ok) {
      knnResult.textContent = data.knn.category;
    } else {
      knnResult.textContent = "No se pudo obtener la clasificación.";
    }

    if (data.mlp && data.mlp.ok) {
      mlpResult.textContent = data.mlp.category;
    } else {
      mlpResult.textContent = "No se pudo obtener la clasificación.";
    }
  } catch (err) {
    regResult.textContent = "Error de conexión.";
    knnResult.textContent = "Error de conexión.";
    mlpResult.textContent = "Error de conexión.";
    globalMsg.textContent =
      "No se pudo contactar la API. Revise que el backend esté ejecutándose.";
  }
}

function resetForm() {
  document.getElementById("predictForm").reset();
  document.getElementById("regResult").textContent = "Sin cálculo aún.";
  document.getElementById("knnResult").textContent = "Sin cálculo aún.";
  document.getElementById("mlpResult").textContent = "Sin cálculo aún.";
  document.getElementById("globalMessage").textContent = "";
}

document.addEventListener("DOMContentLoaded", () => {
  loadMetadata();
  document
    .getElementById("predictForm")
    .addEventListener("submit", handleSubmit);
  document.getElementById("resetBtn").addEventListener("click", resetForm);
});
