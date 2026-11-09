"""
Tarea 2 AFML - XGBoost
Objetivo: predecir caida10 (caida >=10% del S&P 500 en los proximos 3 meses)

IMPORTANTE: usa la misma base, los mismos predictores y la misma particion
que la regresion logistica y la SVM. Las secciones 1 y 2 son copia exacta de
ese codigo; no modificar sin avisar al grupo. Las metricas, el umbral y los
tiempos vienen de comun.py.
"""
import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.inspection import permutation_importance

from comun import (puntaje, elegir_umbral, imprimir_umbral, metricas, tabla_train_test,
                   matriz_confusion, medir_prediccion, n_combinaciones, sensibilidad,
                   guardar_extra, guardar_puntajes, ruta)


class XGBBalanceado(XGBClassifier):
    """XGBClassifier que calcula scale_pos_weight = n_neg / n_pos con la 'y' de CADA
    llamada a fit. Asi, en cada vuelta de GridSearchCV y de elegir_umbral el peso sale
    solo de los meses que esa vuelta tiene para entrenar (en la vuelta 1: 46/3 = 15.3),
    y no de todo el entrenamiento (256/41 = 6.24), que incluye el futuro de esa vuelta.
    En el modelo final (todo el entrenamiento) el peso vuelve a ser 256/41."""

    def fit(self, X, y, **kwargs):
        y_arr = np.asarray(y)
        self.set_params(scale_pos_weight=float((y_arr == 0).sum() / (y_arr == 1).sum()))
        return super().fit(X, y, **kwargs)

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
# 3. Modelo: XGBoost
#    No necesita escalador. scale_pos_weight = n_neg / n_pos compensa el
#    desbalance (~14% de caidas). Se recalcula en cada ajuste (XGBBalanceado),
#    o sea, solo con el entrenamiento disponible en cada vuelta.
# ---------------------------------------------------------------
n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
print(f"scale_pos_weight del modelo final = {n_neg}/{n_pos} = {n_neg / n_pos:.2f}"
      " (en cada vuelta se recalcula con su propio entrenamiento)")
for k, (i_tr, _) in enumerate(TimeSeriesSplit(n_splits=5, gap=HORIZONTE).split(X_train), start=1):
    y_k = y_train.iloc[i_tr]
    print(f"  vuelta {k}: {(y_k == 0).sum()}/{(y_k == 1).sum()} = {(y_k == 0).sum() / (y_k == 1).sum():.2f}")
print()
xgb = XGBBalanceado(random_state=42, eval_metric="logloss", n_jobs=1)

# ---------------------------------------------------------------
# 4. Validacion (solo dentro del entrenamiento) para elegir hiperparametros
#    learning_rate: paso de cada arbol (bajo = aprende lento, mas estable)
#    n_estimators: n de arboles (alto con learning_rate alto = sobreajuste)
#    max_depth: profundidad de cada arbol
#    subsample: fraccion de meses usada por arbol
#    reg_lambda: regularizacion L2 de las hojas
# ---------------------------------------------------------------
cv = TimeSeriesSplit(n_splits=5, gap=HORIZONTE)
grilla = {
    "learning_rate": [0.01, 0.1],
    "n_estimators": [100, 300],
    "max_depth": [2, 3],
    "subsample": [0.7, 1.0],
    "reg_lambda": [1, 10],
}

busqueda = GridSearchCV(xgb, grilla, cv=cv, scoring="roc_auc")
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
sens = sensibilidad(res, grilla)
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
print("Desempeno del XGBoost:")
print(tabla.round(3).to_string())
if tabla.loc["AUC", "Entrenamiento"] > 0.95:
    print("  -> AUC de entrenamiento ~1: el modelo memoriza el entrenamiento (sobreajuste).")

print("\nMatriz de confusion en prueba (filas = real, columnas = predicho):")
print(matriz_confusion(y_test, s_test, umbral))

t_pred = medir_prediccion(modelo, X_test)
print(f"\nTiempo de prediccion sobre la prueba ({len(X_test)} meses): {t_pred * 1000:.1f} ms")

# ---------------------------------------------------------------
# 6. Interpretacion: importancia de variables
# ---------------------------------------------------------------
# Por permutacion: caida del AUC al desordenar cada variable, calculada EN LA
# PRUEBA (igual en los 5 modelos), con el modelo ya terminado: no se elige nada
# con ella. Es ruidosa (21 caidas; ver la desv. est.) y mide otra cosa que los
# coeficientes de la logistica, que se estiman en entrenamiento.
imp = permutation_importance(modelo, X_test, y_test, scoring="roc_auc",
                             n_repeats=30, random_state=0)
imp_df = pd.DataFrame({"Caida del AUC": imp.importances_mean,
                       "Desv. est.": imp.importances_std,
                       "Importancia ganancia (XGB)": modelo.feature_importances_},
                      index=PREDICTORES)
print("\nImportancia por permutacion (muestra de prueba) y por ganancia (referencia):")
print(imp_df.sort_values("Caida del AUC", ascending=False).round(3).to_string())

# ---------------------------------------------------------------
# 7. Guardar resultados para la tabla comparativa final
# ---------------------------------------------------------------
tabla.to_csv(ruta("resultados", "resultados_xgb.csv"))
guardar_extra("xgb", {h: mejor[h] for h in grilla}, busqueda, umbral, info, tiempo, t_pred,
              scale_pos_weight_final=modelo.get_params()["scale_pos_weight"])
sens.to_csv(ruta("resultados", "sensibilidad_xgb.csv"), index=False)
imp_df.to_csv(ruta("resultados", "importancia_xgb.csv"), index_label="variable")
guardar_puntajes("xgb", idx_test, y_test, s_test, umbral)
print("\nGuardados: resultados_xgb.csv, resultados_xgb_extra.csv, sensibilidad_xgb.csv, "
      "importancia_xgb.csv, puntajes_xgb.csv")
