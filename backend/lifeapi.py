from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

# Ruta del CSV original
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Life_Expectancy_Data.csv")

# Rutas modelos Regresión Lineal
LINREG_MODEL_PATH = "life_linreg_model.pkl"
LINREG_SCALER_PATH = "life_scaler.pkl"
LINREG_FEATURES_PATH = "life_features.pkl"

# Rutas modelos KNN
KNN_MODEL_PATH = "life_knn_model.pkl"
KNN_SCALER_PATH = "life_knn_scaler.pkl"
KNN_FEATURES_PATH = "life_knn_features.pkl"
KNN_CLASSES_PATH = "life_knn_classes.pkl"

# Rutas modelos MLP
MLP_MODEL_PATH = "life_mlp_model.pkl"
MLP_SCALER_PATH = "life_mlp_scaler.pkl"
MLP_FEATURES_PATH = "life_mlp_features.pkl"
MLP_CLASSES_PATH = "life_mlp_classes.pkl"

app = Flask(__name__, static_folder="../frontend")
CORS(app)

print("Cargando datos y modelos...")

# df_raw: todos los registros tal cual están en el CSV (para listar países)
df_raw = pd.read_csv(DATA_PATH)

# df: versión sin NA para consistencia con el entrenamiento
df = df_raw.dropna()

target_col = "Life expectancy "
input_columns = [c for c in df.columns if c != target_col]


def safe_load(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None


linreg_model = safe_load(LINREG_MODEL_PATH)
linreg_scaler = safe_load(LINREG_SCALER_PATH)
linreg_features = safe_load(LINREG_FEATURES_PATH)

knn_model = safe_load(KNN_MODEL_PATH)
knn_scaler = safe_load(KNN_SCALER_PATH)
knn_features = safe_load(KNN_FEATURES_PATH)
knn_classes = safe_load(KNN_CLASSES_PATH)

mlp_model = safe_load(MLP_MODEL_PATH)
mlp_scaler = safe_load(MLP_SCALER_PATH)
mlp_features = safe_load(MLP_FEATURES_PATH)
mlp_classes = safe_load(MLP_CLASSES_PATH)


def build_input_dataframe(payload: dict) -> pd.DataFrame:
    """
    Construye un DataFrame de una sola fila con todos los nombres de columnas
    usadas en el entrenamiento. Si alguna no viene en el JSON, se rellena con None.
    """
    row = {}
    for col in input_columns:
        row[col] = [payload.get(col)]
    return pd.DataFrame(row)


def prepare_for_model(base_df: pd.DataFrame, feature_names) -> pd.DataFrame:
    """
    Aplica One-Hot Encoding a Country y Status y reordena las columnas
    para que coincidan exactamente con las usadas al entrenar el modelo.
    """
    df_encoded = pd.get_dummies(base_df, columns=["Country", "Status"], drop_first=True)
    df_encoded = df_encoded.reindex(columns=feature_names, fill_value=0)
    return df_encoded


def decode_label(raw, classes):
    if classes is None:
        return str(raw)
    try:
        if isinstance(raw, (int, np.integer)):
            return str(classes[raw])
        return str(raw)
    except Exception:
        return str(raw)


@app.route("/metadata", methods=["GET"])
def metadata():
    """
    Devuelve:
      - Lista completa de países (sin perder ninguno, sin NA)
      - Rango observado de las variables numéricas
    """
    # Lista completa de países únicos desde el CSV original (sin dropna de todo el df)
    countries = sorted(df_raw["Country"].dropna().unique().tolist())

    # Status se mantiene por compatibilidad, aunque ya no haya select en el frontend
    status_values = sorted(df_raw["Status"].dropna().unique().tolist())

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    ranges = {}
    for col in numeric_cols:
        if col == target_col:
            continue

        col_min = float(df[col].min())
        col_max = float(df[col].max())

        # Ajuste especial de Year: el modelo ve 2000–2015 pero dejamos usar hasta 2030
        if col == "Year":
            col_min = 2000.0
            col_max = 2030.0

        ranges[col] = {"min": col_min, "max": col_max}

    return jsonify({
        "countries": countries,
        "status_values": status_values,
        "ranges": ranges
    })


@app.route("/predict", methods=["POST"])
def predict():
    content = request.get_json()
    if not isinstance(content, dict):
        return jsonify({"error": "JSON inválido"}), 400

    features = content.get("features")
    if features is None:
        return jsonify({"error": "Falta el bloque 'features'"}), 400

    try:
        base_df = build_input_dataframe(features)
    except Exception as e:
        return jsonify({"error": f"Error construyendo datos de entrada: {e}"}), 400

    result = {}

    # ---------------- Regresión lineal ----------------
    try:
        if linreg_model is not None and linreg_scaler is not None and linreg_features is not None:
            X_lin = prepare_for_model(base_df, linreg_features)
            X_lin_scaled = linreg_scaler.transform(X_lin)
            y_pred = float(linreg_model.predict(X_lin_scaled)[0])
            result["regression"] = {
                "ok": True,
                "life_expectancy": y_pred
            }
        else:
            result["regression"] = {"ok": False, "error": "Modelo no disponible"}
    except Exception as e:
        result["regression"] = {"ok": False, "error": str(e)}

    # ---------------- KNN ----------------
    try:
        if knn_model is not None and knn_scaler is not None and knn_features is not None:
            X_knn = prepare_for_model(base_df, knn_features)
            X_knn_scaled = knn_scaler.transform(X_knn)
            y_knn = knn_model.predict(X_knn_scaled)[0]
            label_knn = decode_label(y_knn, knn_classes)
            result["knn"] = {
                "ok": True,
                "category": label_knn
            }
        else:
            result["knn"] = {"ok": False, "error": "Modelo no disponible"}
    except Exception as e:
        result["knn"] = {"ok": False, "error": str(e)}

    # ---------------- MLP ----------------
    try:
        if mlp_model is not None and mlp_scaler is not None and mlp_features is not None:
            X_mlp = prepare_for_model(base_df, mlp_features)
            X_mlp_scaled = mlp_scaler.transform(X_mlp)
            y_mlp = mlp_model.predict(X_mlp_scaled)[0]
            label_mlp = decode_label(y_mlp, mlp_classes)
            result["mlp"] = {
                "ok": True,
                "category": label_mlp
            }
        else:
            result["mlp"] = {"ok": False, "error": "Modelo no disponible"}
    except Exception as e:
        result["mlp"] = {"ok": False, "error": str(e)}

    return jsonify(result)


@app.route("/", methods=["GET"])
def serve_frontend():
    return send_from_directory("../frontend", "panel_esperanza_vida.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)