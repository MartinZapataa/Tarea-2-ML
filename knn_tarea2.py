"""
Tarea 2 AFML - KNN (K vecinos mas cercanos)
Objetivo: predecir caida10 (caida >=10% del S&P 500 en los proximos 3 meses)

Codigo de Renato (KNN.ipynb, celda 2) adaptado a la plantilla comun: misma grilla de K
que eligio Renato, pero con el umbral por cuantil de comun.py, tiempo de prediccion,
importancia por permutacion y los archivos que lee comparativa.py.

IMPORTANTE: usa la misma base, los mismos predictores y la misma particion
que la regresion logistica. Las secciones 1 y 2 son copia exacta de ese codigo;
no modificar sin avisar al grupo.
"""
import time
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from comun import (puntaje, elegir_umbral, imprimir_umbral, metricas, tabla_train_test,
                   matriz_confusion, medir_prediccion, n_combinaciones, guardar_extra,
                   guardar_puntajes, ruta)

# ---------------------------------------------------------------
# 1. Datos (igual que la logistica)
# ---------------------------------------------------------------
df = pd.read_csv("base_tarea2_caidas_sp500.csv", index_col="fecha")
df = df.dropna(subset=["caida10"])  # quita los ultimos 3 meses (objetivo desconocido)

PREDICTORES = ["cape_lag3", "mom_12m", "vix_fin", "baa10y_fin", "t10y3m_fin"]
X = df[PREDICTORES]
y = df["caida10"].astype(int)

# ---------------------------------------------------------------
# 2. Separacion entrenamiento / prueba (igual que la logistica)
# ---------------------------------------------------------------
INICIO_PRUEBA = "2015-01"
HORIZONTE = 3

es_train = X.index < INICIO_PRUEBA
idx_train = X.index[es_train][:-HORIZONTE]
idx_test = X.index[~es_train]

X_train, y_train = X.loc[idx_train], y.loc[idx_train]
X_test, y_test = X.loc[idx_test], y.loc[idx_test]

print(f"Entrenamiento: {idx_train[0]} a {idx_train[-1]} ({len(idx_train)} meses, {y_train.sum()} caidas)")
print(f"Prueba:        {idx_test[0]} a {idx_test[-1]} ({len(idx_test)} meses, {y_test.sum()} caidas)\n")

# ---------------------------------------------------------------
# 3. Modelo: estandarizar + KNN
#    El escalador va DENTRO del pipeline: en cada vuelta de validacion se
#    ajusta solo con el pasado, sin filtrar informacion del futuro.
#    KNN no tiene class_weight: el desbalance se maneja con el umbral (seccion 5).
# ---------------------------------------------------------------
pipe = Pipeline([
    ("escalar", StandardScaler()),
    ("knn", KNeighborsClassifier(metric="euclidean")),
])

# ---------------------------------------------------------------
# 4. Validacion temporal para elegir K (numero de vecinos)
#    K chico = modelo flexible (riesgo de sobreajuste)
#    K grande = modelo suave (riesgo de no captar el patron)
#    Grilla de Renato, sin cambios. OJO: el mejor K cae en el extremo (41).
# ---------------------------------------------------------------
cv = TimeSeriesSplit(n_splits=5, gap=HORIZONTE)
grilla = {"knn__n_neighbors": [1, 3, 5, 7, 11, 15, 21, 31, 41]}

busqueda = GridSearchCV(pipe, grilla, cv=cv, scoring="roc_auc", return_train_score=True)
t0 = time.time()
busqueda.fit(X_train, y_train)
tiempo = time.time() - t0  # misma definicion que en los otros modelos: busqueda completa + reajuste final

res = pd.DataFrame(busqueda.cv_results_)
sens = res[["param_knn__n_neighbors", "mean_train_score", "mean_test_score", "std_test_score"]].copy()
sens.columns = ["K", "AUC entrenamiento", "AUC validacion", "Desv. validacion"]
print("Sensibilidad a K (AUC promedio en los 5 bloques de validacion):")
print(sens.round(3).to_string(index=False))

K = busqueda.best_params_["knn__n_neighbors"]
print(f"\nMejor K: {K}  |  AUC de validacion = {busqueda.best_score_:.3f}"
      f" (+/- {res.loc[busqueda.best_index_, 'std_test_score']:.3f})"
      f"  |  tiempo de entrenamiento: {tiempo:.2f} s\n")

modelo = busqueda.best_estimator_

# ---------------------------------------------------------------
# 5. Umbral (elegido en validacion out-of-fold, SIN mirar la prueba) y metricas
#    - El puntaje es predict_proba = fraccion de los K vecinos que cayeron.
#    - Precision, recall y F1 usan el umbral comun de comun.py (por cuantil).
#      (Renato usaba la frecuencia de caidas del entrenamiento, 0,138.)
# ---------------------------------------------------------------
umbral, info = elegir_umbral(modelo, X_train, y_train, cv)
imprimir_umbral(umbral, info)

s_train = puntaje(modelo, X_train)
s_test = puntaje(modelo, X_test)
tabla = tabla_train_test(metricas(y_train, s_train, umbral),
                         metricas(y_test, s_test, umbral))
print(f"Desempeno de KNN (K = {K}):")
print(tabla.round(3).to_string())

print("\nMatriz de confusion en prueba (filas = real, columnas = predicho):")
print(matriz_confusion(y_test, s_test, umbral))

t_pred = medir_prediccion(modelo, X_test)
n_comb = n_combinaciones(busqueda)
print(f"\nTiempo de prediccion sobre la prueba ({len(X_test)} meses): {t_pred * 1000:.1f} ms"
      f"  |  combinaciones evaluadas: {n_comb}")

# ---------------------------------------------------------------
# 6. Ilustracion del sobreajuste (K = 1) e interpretacion
# ---------------------------------------------------------------
knn1 = Pipeline([("escalar", StandardScaler()),
                 ("knn", KNeighborsClassifier(n_neighbors=1))]).fit(X_train, y_train)
print("\nComparacion con K = 1 (memoriza la muestra):")
print(f"  AUC entrenamiento: {roc_auc_score(y_train, puntaje(knn1, X_train)):.3f}")
print(f"  AUC prueba:        {roc_auc_score(y_test, puntaje(knn1, X_test)):.3f}")

# KNN no tiene coeficientes: importancia por permutacion EN LA PRUEBA (igual que
# los otros modelos), con el modelo ya terminado; no se elige nada con ella.
imp = permutation_importance(modelo, X_test, y_test, scoring="roc_auc",
                             n_repeats=30, random_state=0)
imp_df = pd.DataFrame({"Caida del AUC": imp.importances_mean,
                       "Desv. est.": imp.importances_std}, index=PREDICTORES)
print("\nImportancia por permutacion (muestra de prueba):")
print(imp_df.sort_values("Caida del AUC", ascending=False).round(3).to_string())

# ---------------------------------------------------------------
# 7. Guardar resultados para la tabla comparativa final
# ---------------------------------------------------------------
tabla.to_csv(ruta("resultados", "resultados_knn.csv"))
sens.to_csv(ruta("resultados", "sensibilidad_knn.csv"), index=False)
guardar_extra("knn", {"K": K}, busqueda, umbral, info, tiempo, t_pred)
imp_df.to_csv(ruta("resultados", "importancia_knn.csv"), index_label="variable")
guardar_puntajes("knn", idx_test, y_test, s_test, umbral)
