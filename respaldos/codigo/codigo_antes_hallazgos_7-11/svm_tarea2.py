"""
Tarea 2 AFML - SVM con kernel RBF
Objetivo: predecir caida10 (caida >=10% del S&P 500 en los proximos 3 meses)

IMPORTANTE: usa la misma base, los mismos predictores y la misma particion
que la regresion logistica. Las secciones 1 y 2 son copia exacta de ese codigo;
no modificar sin avisar al grupo.
"""
import time
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.inspection import permutation_importance

from comun import (puntaje, elegir_umbral, imprimir_umbral, metricas, tabla_train_test,
                   matriz_confusion, medir_prediccion, n_combinaciones, guardar_puntajes)

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
# 3. Modelo: estandarizar + SVM
#    El escalador va DENTRO del pipeline: en cada vuelta de validacion se
#    ajusta solo con el pasado, sin filtrar informacion del futuro.
# ---------------------------------------------------------------
pipe = Pipeline([
    ("escalar", StandardScaler()),
    ("svm", SVC(kernel="rbf", class_weight="balanced")),
])

# ---------------------------------------------------------------
# 4. Validacion (solo dentro del entrenamiento) para elegir C y gamma
#    C: cuanto castiga equivocarse (alto = frontera ajustada, riesgo de sobreajuste)
#    gamma: que tan local es cada punto (alto = frontera muy curva, riesgo de sobreajuste)
# ---------------------------------------------------------------
cv = TimeSeriesSplit(n_splits=5, gap=HORIZONTE)
grilla = {
    "svm__C": [0.01, 0.1, 1, 10, 100],
    "svm__gamma": [0.001, 0.01, 0.1, 1],
}

busqueda = GridSearchCV(pipe, grilla, cv=cv, scoring="roc_auc")
t0 = time.time()
busqueda.fit(X_train, y_train)
tiempo = time.time() - t0  # misma definicion que en la logistica: busqueda completa + reajuste final

res = pd.DataFrame(busqueda.cv_results_)
res["C"] = res["param_svm__C"].astype(float)
res["gamma"] = res["param_svm__gamma"].astype(float)

print("AUC de validacion promedio por (C, gamma)  [filas = C, columnas = gamma]:")
print(res.pivot(index="C", columns="gamma", values="mean_test_score").round(3).to_string())
print("\nDesviacion estandar del AUC entre vueltas de validacion:")
print(res.pivot(index="C", columns="gamma", values="std_test_score").round(3).to_string())

mejor = busqueda.best_params_
print(f"\nMejor combinacion: C = {mejor['svm__C']}, gamma = {mejor['svm__gamma']}"
      f"  |  AUC de validacion = {busqueda.best_score_:.3f}"
      f"  |  tiempo de entrenamiento: {tiempo:.2f} s\n")

modelo = busqueda.best_estimator_

# ---------------------------------------------------------------
# 5. Umbral (elegido en validacion out-of-fold, SIN mirar la prueba) y metricas
#    - AUC se calcula con decision_function (puntaje de distancia a la frontera).
#      No se usa probability=True: su calibracion interna (Platt) usa una
#      validacion cruzada que ignora el orden del tiempo.
#    - Precision, recall y F1 usan el umbral comun de comun.py (por cuantil).
# ---------------------------------------------------------------
umbral, info = elegir_umbral(modelo, X_train, y_train, cv)
imprimir_umbral(umbral, info)

s_train = puntaje(modelo, X_train)
s_test = puntaje(modelo, X_test)
tabla = tabla_train_test(metricas(y_train, s_train, umbral),
                         metricas(y_test, s_test, umbral))
print("Desempeno de la SVM:")
print(tabla.round(3).to_string())

print("\nMatriz de confusion en prueba (filas = real, columnas = predicho):")
print(matriz_confusion(y_test, s_test, umbral))

t_pred = medir_prediccion(modelo, X_test)
n_comb = n_combinaciones(busqueda)
print(f"\nTiempo de prediccion sobre la prueba ({len(X_test)} meses): {t_pred * 1000:.1f} ms"
      f"  |  combinaciones evaluadas: {n_comb}")

# ---------------------------------------------------------------
# 6. Diagnostico de complejidad e interpretacion
# ---------------------------------------------------------------
# Vectores de soporte: los puntos que definen la frontera. Si casi todas las
# observaciones lo son, el modelo esta "memorizando" en vez de encontrar una regla.
n_sv = modelo.named_steps["svm"].n_support_.sum()
print(f"\nVectores de soporte: {n_sv} de {len(X_train)} observaciones de entrenamiento "
      f"({n_sv / len(X_train):.0%})")

# La SVM con kernel RBF no tiene coeficientes. Se usa importancia por permutacion:
# cuanto cae el AUC al desordenar una variable. Se calcula con el modelo ya terminado.
imp = permutation_importance(modelo, X_test, y_test, scoring="roc_auc",
                             n_repeats=30, random_state=0)
imp_df = pd.DataFrame({"Caida del AUC": imp.importances_mean,
                       "Desv. est.": imp.importances_std}, index=PREDICTORES)
print("\nImportancia por permutacion (muestra de prueba):")
print(imp_df.sort_values("Caida del AUC", ascending=False).round(3).to_string())

# ---------------------------------------------------------------
# 7. Guardar resultados para la tabla comparativa final
# ---------------------------------------------------------------
tabla.to_csv("resultados/resultados_svm.csv")
pd.DataFrame({"C": [mejor["svm__C"]], "gamma": [mejor["svm__gamma"]],
              "AUC_validacion": [busqueda.best_score_],
              "umbral": [umbral], "percentil_umbral": [info["percentil_corte"]],
              "tiempo_s": [tiempo], "tiempo_prediccion_s": [t_pred],
              "n_combinaciones": [n_comb], "vectores_soporte": [n_sv]}).to_csv("resultados/resultados_svm_extra.csv", index=False)
imp_df.to_csv("resultados/importancia_svm.csv", index_label="variable")
guardar_puntajes("svm", idx_test, y_test, s_test, umbral)
