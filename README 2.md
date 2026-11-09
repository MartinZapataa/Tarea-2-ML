# Tarea 2 AFML: probabilidad de una caída de 10% del S&P 500 en 3 meses

Grupo: Lorenzo Godoy, Renato Torres y Martín Zapata. Profesor: Joaquín Aguirre. Entrega: martes 10/11/2026.

Clasificación binaria mensual (ene-1990 a sep-2025): `caida10` = 1 si algún cierre diario de los 3 meses siguientes queda ≥10% bajo el cierre del mes. Cinco modelos con la misma base, los mismos 5 predictores y la misma partición temporal: regresión logística (referencia), KNN, SVM (RBF), Random Forest y XGBoost.

## Estructura

```
T2 Finanzas ML/
├── README.md                      este archivo
├── requirements.txt               librerías (Python 3.13.2)
├── base_tarea2_caidas_sp500.csv   base mensual final
│
├── comun.py                       funciones comunes: puntaje, métricas, umbral, tiempos, guardado
├── tarea2.ipynb                   regresión logística (+ descarga del S&P 500 e importancia)
├── knn_tarea2.py                  KNN
├── svm_tarea2.py                  SVM con kernel RBF
├── rf_tarea2.py                   Random Forest
├── xgb_tarea2.py                  XGBoost
├── comparativa.py                 tablas, curvas ROC, matrices, bootstrap
│
├── resultados/                    CSV generados por los scripts
├── figuras/                       fig1 (ROC), fig2 (matrices), fig3 (alertas)
├── salidas/                       consola de cada script (corrida de referencia)
│
├── documentos/                    informe (.docx), GUIA_TAREA2.md y revisión de código
├── respaldos/                     versiones anteriores (no se usan): codigo/, informes/, notas/
└── CLAUDE.md                      plan y convenciones para Claude Code (puede ignorarse)
```

## Cómo correr todo de cero

Desde esta carpeta (la base se lee con ruta relativa):

```bash
pip install -r requirements.txt
python -m nbconvert --to notebook --execute --inplace tarea2.ipynb   # logística (primero: es la referencia)
python svm_tarea2.py
python knn_tarea2.py
python rf_tarea2.py        # ~2 min
python xgb_tarea2.py
python comparativa.py      # tablas y figuras; se detiene si falta la logística
```

Para guardar la consola de un script: `python rf_tarea2.py > salidas/salida_rf.txt`. Los resultados se reproducen exactamente; solo los tiempos varían entre corridas. La celda 1 del notebook descarga el S&P 500 a `gspc_diario.csv` (necesita internet; ningún otro código lo usa).

## Por dónde empezar

- `documentos/GUIA_TAREA2.md`: qué hace cada modelo, resultados, formato de los archivos y dónde va cada cosa en el informe.
- `documentos/Informe Tarea 2 Machine Learning.docx`: el informe.

## Reglas comunes

Las secciones 1 y 2 (datos y partición) de cada script son idénticas y no se cambian sin avisar al grupo. La prueba (2015-01 a 2025-09) se usa una sola vez, al final; el umbral de decisión se elige solo con el entrenamiento (`comun.elegir_umbral`). No usar `ret_min_3m` como predictor: mira al futuro.

## Pendiente

Informe (Resumen, 2.1, 2.5, 4.5, conclusión, referencias), código de construcción de la base (Renato) y presentación.
