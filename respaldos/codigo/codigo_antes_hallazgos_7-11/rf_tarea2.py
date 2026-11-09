"""
Tarea 2 AFML - Random Forest
Objetivo: predecir caida10 (caida >=10% del S&P 500 en los proximos 3 meses)

IMPORTANTE: usa la misma base, los mismos predictores y la misma particion
que la regresion logistica y la SVM. Las secciones 1 y 2 son copia exacta de
ese codigo; no modificar sin avisar al grupo. Las metricas, el umbral y los
tiempos vienen de comun.py.
"""
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
# 3. Modelo: Random Forest
#    No necesita escalador (los arboles no dependen de la escala).
#    class_weight="balanced" compensa el desbalance (~14% de caidas).
# ---------------------------------------------------------------
rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1)

# ---------------------------------------------------------------
# 4. Validacion (solo dentro del entrenamiento) para elegir hiperparametros
#    n_estimators: n de arboles (mas arboles = mas estable, no sobreajusta mas)
#    max_depth: profundidad maxima (None = arboles completos, riesgo de sobreajuste)
#    min_samples_leaf: minimo de meses por hoja (alto = arboles mas simples)
#    max_features: variables consideradas en cada division
# ---------------------------------------------------------------
cv = TimeSeriesSplit(n_splits=5, gap=HORIZONTE)
grilla = {
    "n_estimators": [200, 500],
    "max_depth": [2, 3, 5, None],
    "min_samples_leaf": [5, 10, 20],
    "max_features": ["sqrt", None],
}

busqueda = GridSearchCV(rf, grilla, cv=cv, scoring="roc_auc")
t0 = time.time()
busqueda.fit(X_train, y_train)
tiempo = time.time() - t0  # misma definicion que en los otros modelos: busqueda completa + reajuste final
n_comb = n_combinaciones(busqueda)

res = pd.DataFrame(busqueda.cv_results_)
mejor = busqueda.best_params_
print(f"Combinaciones evaluadas: {n_comb} ({n_comb * cv.get_n_splits()} ajustes)")
print(f"Mejor combinacion: {mejor}"
      f"\n  AUC de validacion = {busqueda.best_score_:.3f}"
      f" (+/- {res.loc[busqueda.best_index_, 'std_test_score']:.3f})"
      f"  |  tiempo de entrenamiento: {tiempo:.2f} s\n")

# Sensibilidad: AUC de validacion promedio para cada valor de cada hiperparametro
# (promediando sobre las demas dimensiones de la grilla)
print("Sensibilidad del AUC de validacion a cada hiperparametro:")
sens = []
for h in grilla:
    g = res.groupby(res[f"param_{h}"].astype(str))["mean_test_score"].agg(["mean", "min", "max"])
    for valor, fila in g.iterrows():
        sens.append({"hiperparametro": h, "valor": valor, "AUC_val_prom": fila["mean"],
                     "AUC_val_min": fila["min"], "AUC_val_max": fila["max"]})
sens = pd.DataFrame(sens)
print(sens.round(3).to_string(index=False))
print(f"\nRango del AUC de validacion en toda la grilla: "
      f"{res['mean_test_score'].min():.3f} a {res['mean_test_score'].max():.3f}\n")

modelo = busqueda.best_estimator_

# ---------------------------------------------------------------
# 5. Umbral (elegido en validacion out-of-fold, SIN mirar la prueba) y metricas
# ---------------------------------------------------------------
umbral, info = elegir_umbral(modelo, X_train, y_train, cv)
imprimir_umbral(umbral, info)

s_train = puntaje(modelo, X_train)
s_test = puntaje(modelo, X_test)
tabla = tabla_train_test(metricas(y_train, s_train, umbral),
                         metricas(y_test, s_test, umbral))
print("Desempeno del Random Forest:")
print(tabla.round(3).to_string())
if tabla.loc["AUC", "Entrenamiento"] > 0.95:
    print("  -> AUC de entrenamiento ~1: el bosque memoriza el entrenamiento (sobreajuste esperable).")

print("\nMatriz de confusion en prueba (filas = real, columnas = predicho):")
print(matriz_confusion(y_test, s_test, umbral))

t_pred = medir_prediccion(modelo, X_test)
print(f"\nTiempo de prediccion sobre la prueba ({len(X_test)} meses): {t_pred * 1000:.1f} ms")

# ---------------------------------------------------------------
# 6. Interpretacion: importancia de variables
# ---------------------------------------------------------------
# Por permutacion (caida del AUC en prueba), igual que en la SVM
imp = permutation_importance(modelo, X_test, y_test, scoring="roc_auc",
                             n_repeats=30, random_state=0)
imp_df = pd.DataFrame({"Caida del AUC": imp.importances_mean,
                       "Desv. est.": imp.importances_std,
                       "Importancia impureza (MDI)": modelo.feature_importances_},
                      index=PREDICTORES)
print("\nImportancia por permutacion (muestra de prueba) y por impureza (referencia):")
print(imp_df.sort_values("Caida del AUC", ascending=False).round(3).to_string())

# ---------------------------------------------------------------
# 7. Guardar resultados para la tabla comparativa final
# ---------------------------------------------------------------
tabla.to_csv("resultados/resultados_rf.csv")
fila_extra = {f"{h}": [mejor[h]] for h in grilla}
fila_extra.update({"AUC_validacion": [busqueda.best_score_],
                   "umbral": [umbral],
                   "percentil_umbral": [info["percentil_corte"]],
                   "tiempo_s": [tiempo],
                   "tiempo_prediccion_s": [t_pred],
                   "n_combinaciones": [n_comb]})
pd.DataFrame(fila_extra).to_csv("resultados/resultados_rf_extra.csv", index=False)
sens.to_csv("resultados/sensibilidad_rf.csv", index=False)
imp_df.to_csv("resultados/importancia_rf.csv", index_label="variable")
guardar_puntajes("rf", idx_test, y_test, s_test, umbral)
print("\nGuardados: resultados_rf.csv, resultados_rf_extra.csv, sensibilidad_rf.csv, "
      "importancia_rf.csv, puntajes_rf.csv")
