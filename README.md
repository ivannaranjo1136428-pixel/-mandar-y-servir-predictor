# Aplicativo QR — Mandar y Servir

## Qué incluye
- `app.py`: aplicación web Streamlit con predictor y estadísticas históricas.
- `requirements.txt`: dependencias.
- `exportar_artifacts_colab.py`: bloque para incorporar al final del notebook y exportar el modelo y los catálogos.
- `artifacts/`: carpeta que se genera al ejecutar el bloque de exportación.

## Importante sobre el modelo actual
El notebook fuente entrena regresión logística con estas variables:
`Dia_Semana_Registro`, `Ruta_ID`, `Servicio_Modelo`.
El servicio de predicción es el más frecuente históricamente en la ruta. Esta versión no incorpora festivos como predictor. La aplicación informa explícitamente estas limitaciones.

## Paso 1. Ejecutar el notebook
Ejecuta el notebook completo desde el inicio con los cuatro archivos fuente. Antes de la última celda de exportación, asegúrate de que existan `modelo_final`, `predictor_data`, `indicadores_ruta`, `por_dia`, `metricas_temporales`, `df`, `etiquetados`, `feature_cols`, `servicio_por_ruta` y `std_by_ids`.

## Paso 2. Exportar artefactos
Ejecuta el contenido de `exportar_artifacts_colab.py` en una celda nueva al final del notebook. Descarga la carpeta `artifacts` como ZIP y descomprímela en la raíz del proyecto.

## Paso 3. Publicar
1. Crea un repositorio GitHub privado o público, según las políticas de la institución/empresa.
2. Sube `app.py`, `requirements.txt` y la carpeta `artifacts/`.
3. En Streamlit Community Cloud, conecta el repositorio y selecciona `app.py` como archivo principal.
4. Cuando la aplicación esté publicada, copia la URL pública.

## Paso 4. Crear QR
El QR debe apuntar a la URL pública final. No se debe generar un QR definitivo antes de tener esa dirección, porque el QR necesita el destino real.

## Privacidad
La aplicación es pública y no requiere inicio de sesión. No subir nombres de clientes, números de remesa, datos personales ni filas individuales. Los artefactos exportados contienen únicamente modelo, catálogos de rutas e indicadores agregados; revisar los archivos antes de publicarlos.

## Artefactos recibidos y comprobados
El paquete recibido contiene 320 rutas, 117.116 registros históricos, 99.263 registros etiquetados y 84,76% de cobertura de etiqueta.
La estadística histórica exportada informa 81,75% de cumplimiento y 18,25% de incumplimiento.
El archivo `config.json` indica que el modelo usa día de semana, ruta y servicio; no incluye festivos como predictor.
La validación temporal exportada corresponde al corte 2026-07-14.

Antes de desplegar, se debe probar la carga real de `modelo_logistico.joblib` en el entorno de Streamlit y verificar una predicción de prueba contra el notebook. No se deben publicar archivos con registros individuales ni identificadores de remesa.
