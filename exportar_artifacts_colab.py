# EXPORTACIÓN DE ARTEFACTOS PARA EL APLICATIVO QR
# Ejecutar después de la celda 17 del notebook, cuando modelo_final y los catálogos existan.

import os, json, joblib
import numpy as np
import pandas as pd

os.makedirs("artifacts", exist_ok=True)

# 1. Modelo entrenado
joblib.dump(modelo_final, "artifacts/modelo_logistico.joblib")

# 2. Catálogo público de rutas: solo rutas con estándar único
route_catalog = (
    predictor_data.groupby(["ID_ORIGEN", "ID_DESTINO"], dropna=False)
    .agg(
        Origen=("Origen", lambda s: s.dropna().mode().iloc[0] if not s.dropna().mode().empty else "Origen"),
        Destino=("Destino", lambda s: s.dropna().mode().iloc[0] if not s.dropna().mode().empty else "Destino"),
        service_model=("Servicio_Modelo", lambda s: s.dropna().mode().iloc[0] if not s.dropna().mode().empty else "Sin dato"),
        standard_days=("Dias_Estandar_Auditado",
                       lambda s: pd.to_numeric(s, errors="coerce").dropna().unique()[0]
                       if len(pd.to_numeric(s, errors="coerce").dropna().unique()) == 1 else np.nan),
        historical_orders=("Promesa_Cumplida", "size"),
        fulfilled_pct=("Promesa_Cumplida", lambda s: 100 * (s == 1).mean())
    )
    .reset_index()
)
route_catalog = route_catalog.dropna(subset=["standard_days"])
route_catalog["ID_ORIGEN"] = route_catalog["ID_ORIGEN"].astype(str)
route_catalog["ID_DESTINO"] = route_catalog["ID_DESTINO"].astype(str)
route_catalog["route_label"] = (
    route_catalog["Origen"].astype(str) + " → " + route_catalog["Destino"].astype(str)
    + " | Ruta " + route_catalog["ID_ORIGEN"] + "-" + route_catalog["ID_DESTINO"]
)
route_catalog.to_csv("artifacts/catalogo_rutas.csv", index=False, encoding="utf-8-sig")

# 3. Estadísticas generales agregadas
total_registros = int(len(df))
n_label = int(len(etiquetados))
stats = {
    "total_registros": total_registros,
    "registros_etiquetados": n_label,
    "cobertura_etiqueta_pct": round(100*n_label/total_registros, 2) if total_registros else None,
    "cumplimiento_pct": round(100*etiquetados["Promesa_Cumplida"].mean(), 2) if n_label else None,
    "incumplimiento_pct": round(100*(1-etiquetados["Promesa_Cumplida"].mean()), 2) if n_label else None,
    "media_dias": float(pd.to_numeric(etiquetados["Dias_Observados"], errors="coerce").mean()),
    "mediana_dias": float(pd.to_numeric(etiquetados["Dias_Observados"], errors="coerce").median())
}
with open("artifacts/estadisticas_generales.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

# 4. Estadísticas agregadas por ruta y por día (sin datos personales ni remesas)
indicadores_ruta.to_csv("artifacts/estadisticas_ruta.csv", index=False, encoding="utf-8-sig")
por_dia.to_csv("artifacts/estadisticas_dia.csv", index=False, encoding="utf-8-sig")

# 5. Métricas temporales y configuración
if "metricas_temporales" in globals() and not metricas_temporales.empty:
    metricas_temporales.to_csv("artifacts/metricas_modelo.csv", index=False, encoding="utf-8-sig")
else:
    pd.DataFrame().to_csv("artifacts/metricas_modelo.csv", index=False)

config = {
    "feature_cols": feature_cols,
    "threshold": 0.5,
    "model_name": "Regresión logística",
    "calendar_predictors": ["Dia_Semana_Registro"],
    "holiday_predictor_included": False,
    "service_strategy": "Servicio histórico más frecuente de la ruta",
    "target_definition": "Promesa_Cumplida = 1 si Días Entrega <= Dias_Estandar_Auditado"
}
with open("artifacts/config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

print("Artefactos exportados en /content/artifacts")
print("Archivos:", sorted(os.listdir("artifacts")))
print("Rutas publicables:", len(route_catalog))
print("IMPORTANTE: revisa que no haya datos personales ni identificadores de remesa antes de publicar.")
