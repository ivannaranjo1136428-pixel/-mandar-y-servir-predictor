import os
import json
from datetime import date
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Mandar y Servir | Predictor de entregas",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

ARTIFACTS = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS, "modelo_logistico.joblib")
ROUTES_PATH = os.path.join(ARTIFACTS, "catalogo_rutas.csv")
STATS_PATH = os.path.join(ARTIFACTS, "estadisticas_generales.json")
ROUTE_STATS_PATH = os.path.join(ARTIFACTS, "estadisticas_ruta.csv")
DAY_STATS_PATH = os.path.join(ARTIFACTS, "estadisticas_dia.csv")
METRICS_PATH = os.path.join(ARTIFACTS, "metricas_modelo.csv")
CONFIG_PATH = os.path.join(ARTIFACTS, "config.json")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    routes = pd.read_csv(ROUTES_PATH, dtype={"ID_ORIGEN": str, "ID_DESTINO": str})
    with open(STATS_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)
    route_stats = pd.read_csv(ROUTE_STATS_PATH) if os.path.exists(ROUTE_STATS_PATH) else pd.DataFrame()
    day_stats = pd.read_csv(DAY_STATS_PATH) if os.path.exists(DAY_STATS_PATH) else pd.DataFrame()
    metrics = pd.read_csv(METRICS_PATH) if os.path.exists(METRICS_PATH) else pd.DataFrame()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    return routes, stats, route_stats, day_stats, metrics, config

def day_name(dt):
    names = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    return names[dt.weekday()]

def pct(x):
    return f"{x:.1f}%" if pd.notna(x) else "—"

# Brand header
st.markdown("""
<style>
.main .block-container {max-width: 1180px; padding-top: 1.4rem;}
.hero {background:#10283C; padding:1.4rem 1.6rem; border-radius:16px; color:white; margin-bottom:1rem;}
.hero h1 {color:white; margin:0; font-size:2rem;}
.hero p {color:#D6E5ED; margin:.35rem 0 0 0;}
.small-note {color:#64748b; font-size:.88rem;}
.result-good {border-left:6px solid #15966f; padding:1rem; background:#eaf7f1; border-radius:10px;}
.result-risk {border-left:6px solid #d97706; padding:1rem; background:#fff7ed; border-radius:10px;}
</style>
<div class="hero">
<h1>🚚 MANDAR Y SERVIR</h1>
<p>Predictor de cumplimiento de promesas de entrega · Herramienta de apoyo a la operación</p>
</div>
""", unsafe_allow_html=True)

required = [MODEL_PATH, ROUTES_PATH, STATS_PATH, CONFIG_PATH]
missing = [p for p in required if not os.path.exists(p)]
if missing:
    st.error("La aplicación todavía no tiene los artefactos del modelo. Genéralos desde el notebook y colócalos en la carpeta artifacts/.")
    st.code("\n".join(missing))
    st.stop()

try:
    model = load_model()
    routes, stats, route_stats, day_stats, metrics, config = load_data()
except Exception as e:
    st.error(f"No fue posible cargar los archivos del modelo: {e}")
    st.stop()

tab_predict, tab_stats, tab_about = st.tabs(["🔎 Predecir pedido", "📊 Estadísticas", "ℹ️ Sobre el modelo"])

with tab_predict:
    st.subheader("Consultar probabilidad de cumplimiento")
    st.write("Selecciona una ruta disponible y la fecha en que se registra el pedido.")
    c1, c2 = st.columns(2)
    route_labels = routes["route_label"].tolist()
    with c1:
        selected_label = st.selectbox("Ruta (origen → destino)", route_labels, index=None, placeholder="Selecciona una ruta")
    with c2:
        registration_date = st.date_input("Fecha de registro", value=date.today(), min_value=date(2000,1,1), max_value=date(2100,12,31))
    selected = routes[routes["route_label"] == selected_label] if selected_label else pd.DataFrame()

    if not selected.empty:
        route = selected.iloc[0]
        st.caption(f"Estándar auditado: {route['standard_days']:g} día(s) · Servicio usado: {route['service_model']}")
        if st.button("Calcular predicción", type="primary", use_container_width=True):
            x = pd.DataFrame([{
                "Dia_Semana_Registro": day_name(registration_date),
                "Ruta_ID": f"{route['ID_ORIGEN']}_{route['ID_DESTINO']}",
                "Servicio_Modelo": str(route["service_model"])
            }])[config["feature_cols"]].astype("string").fillna("Sin dato")
            probability = float(model.predict_proba(x)[0, 1])
            threshold = float(config.get("threshold", 0.5))
            predicted = int(probability >= threshold)
            p1, p2 = st.columns(2)
            p1.metric("Probabilidad de cumplir", pct(probability*100))
            p2.metric("Probabilidad de incumplir", pct((1-probability)*100))
            if predicted == 1:
                st.markdown(f'<div class="result-good"><b>Clasificación del modelo: CUMPLE</b><br>La probabilidad estimada de cumplir es {probability:.1%}.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-risk"><b>Clasificación del modelo: POSIBLE INCUMPLIMIENTO</b><br>La probabilidad estimada de incumplir es {(1-probability):.1%}.</div>', unsafe_allow_html=True)
            st.info(f"Umbral de clasificación: {threshold:.0%}. El estándar de {route['standard_days']:g} día(s) se muestra como referencia operativa; el modelo estima la clase a partir de la ruta, el día de registro y el servicio.")
            st.caption("Esta estimación es estadística: no garantiza la entrega. El modelo actual utiliza el servicio histórico más frecuente de la ruta y no incorpora festivos como predictor.")
    else:
        st.info("Selecciona una ruta para consultar su estándar y realizar la predicción.")

with tab_stats:
    st.subheader("Indicadores históricos del conjunto analizado")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Registros analizados", f"{stats.get('total_registros', 0):,}")
    k2.metric("Registros etiquetados", f"{stats.get('registros_etiquetados', 0):,}")
    k3.metric("Cobertura de etiqueta", pct(stats.get("cobertura_etiqueta_pct", np.nan)))
    k4.metric("Cumplimiento histórico", pct(stats.get("cumplimiento_pct", np.nan)))

    k5, k6, k7 = st.columns(3)
    k5.metric("Incumplimiento histórico", pct(stats.get("incumplimiento_pct", np.nan)))
    k6.metric("Media de días observados", f"{stats.get('media_dias', np.nan):.2f}" if pd.notna(stats.get("media_dias")) else "—")
    k7.metric("Mediana de días observados", f"{stats.get('mediana_dias', np.nan):.2f}" if pd.notna(stats.get("mediana_dias")) else "—")

    st.caption("Indicadores calculados sobre la base de trabajo exportada desde el notebook. No representan datos en tiempo real.")
    if not route_stats.empty:
        st.markdown("#### Rutas con mayor volumen de incumplimientos")
        cols = [c for c in ["Origen","Destino","Entregas","Incumplidas","Porcentaje_Incumplimiento"] if c in route_stats.columns]
        rs = route_stats.sort_values("Incumplidas", ascending=False).head(10)[cols].copy()
        if "Porcentaje_Incumplimiento" in rs:
            rs["Porcentaje_Incumplimiento"] = rs["Porcentaje_Incumplimiento"].map(pct)
        st.dataframe(rs, use_container_width=True, hide_index=True)
        st.markdown("#### Rutas con mayor tasa de incumplimiento (mínimo de 30 entregas)")
        if "Entregas" in route_stats and "Porcentaje_Incumplimiento" in route_stats:
            top_rate = route_stats[route_stats["Entregas"] >= 30].sort_values("Porcentaje_Incumplimiento", ascending=False).head(10)
            show = top_rate[[c for c in ["Origen","Destino","Entregas","Porcentaje_Incumplimiento"] if c in top_rate.columns]].copy()
            if "Porcentaje_Incumplimiento" in show:
                show["Porcentaje_Incumplimiento"] = show["Porcentaje_Incumplimiento"].map(pct)
            st.dataframe(show, use_container_width=True, hide_index=True)
    if not day_stats.empty:
        st.markdown("#### Comportamiento por día de registro")
        st.dataframe(day_stats, use_container_width=True, hide_index=True)
    if not metrics.empty:
        st.markdown("#### Evaluación temporal de modelos")
        st.dataframe(metrics, use_container_width=True, hide_index=True)

with tab_about:
    st.subheader("Alcance y consideraciones")
    st.write("El modelo publicado corresponde a la regresión logística entrenada con el conjunto etiquetado disponible al generar los artefactos.")
    st.write("Variables predictoras: día de semana de registro, identificador de ruta y tipo de servicio histórico más frecuente de la ruta.")
    st.write("La clase objetivo se construyó comparando Días Entrega con el estándar auditado de la ruta.")
    st.warning("No se emite predicción para rutas sin estándar único validado. El modelo no incorpora festivos en sus predictores en esta versión del notebook.")
    st.write("Las estadísticas son históricas y estáticas hasta que se vuelva a ejecutar el proceso de entrenamiento y se actualicen los artefactos.")
    st.caption("Herramienta académica de apoyo a la toma de decisiones. No constituye garantía contractual de entrega.")
