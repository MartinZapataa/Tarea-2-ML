"""
Tarea 2 AFML - Funciones comunes para los 5 modelos
(logistica, KNN, SVM, Random Forest, XGBoost)

Este archivo NO contiene los datos ni la particion: las secciones 1 y 2
(lectura de la base y separacion entrenamiento / prueba) siguen copiadas
tal cual en cada script. Aqui solo van las piezas que deben ser identicas
en todos los modelos para que la comparacion sea justa:

  - puntaje(modelo, X)               -> puntaje continuo (probabilidad o distancia)
  - elegir_umbral(est, X, y, cv)     -> umbral elegido en validacion out-of-fold
  - metricas(y, puntaje, umbral)     -> AUC, Precision, Recall, F1
  - tabla_train_test(m_tr, m_te)     -> tabla Entrenamiento / Prueba / Brecha
  - matriz_confusion(y, punt, umb)   -> matriz de confusion como DataFrame
  - medir_prediccion(modelo, X)      -> segundos para calcular el puntaje
  - n_combinaciones(busqueda)        -> n de combinaciones de la grilla
  - sensibilidad(res, grilla)        -> AUC de validacion por valor de cada hiperparametro
  - guardar_extra(nombre, ...)       -> resultados_<nombre>_extra.csv
  - guardar_puntajes(nombre, ...)    -> puntajes de prueba para la curva ROC
  - ruta(carpeta, archivo)           -> ruta de salida (resultados/, figuras/, salidas/)

Criterio de umbral (opcion 2, por cuantil): en las predicciones out-of-fold
de TimeSeriesSplit dentro del entrenamiento, cada puntaje se pasa a percentil
respecto de los puntajes de ENTRENAMIENTO (dentro de muestra) de su propio
modelo; se elige el percentil p* que MAXIMIZA F1 y el umbral final es el
cuantil p* de los puntajes de entrenamiento del modelo final. La prueba nunca
se usa. Ojo: esto NO fija la fraccion de meses marcados fuera de muestra (ver
la nota de la seccion 3).

Las salidas se guardan siempre junto a este archivo (resultados/, figuras/,
salidas/), sin importar desde que carpeta se corra. La base se sigue leyendo
con ruta relativa (seccion 1 congelada), asi que los scripts se corren desde
la carpeta del proyecto.

Uso tipico (despues de busqueda.fit):

    from comun import (puntaje, elegir_umbral, metricas, tabla_train_test,
                       matriz_confusion, medir_prediccion, n_combinaciones,
                       guardar_puntajes)

    modelo = busqueda.best_estimator_
    umbral, info = elegir_umbral(modelo, X_train, y_train, cv)   # antes de mirar la prueba
    imprimir_umbral(umbral, info)
    s_train = puntaje(modelo, X_train)
    s_test = puntaje(modelo, X_test)
    tabla = tabla_train_test(metricas(y_train, s_train, umbral),
                             metricas(y_test, s_test, umbral))
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, confusion_matrix, precision_recall_curve)

# ---------------------------------------------------------------
# 0. Rutas de salida: relativas a la carpeta de este archivo, no al cwd
# ---------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent
CARPETAS = ("resultados", "figuras", "salidas")


def ruta(carpeta, archivo):
    """Ruta de un archivo de salida dentro de RAIZ/<carpeta>; crea la carpeta si falta."""
    if carpeta not in CARPETAS:
        raise ValueError(f"Carpeta desconocida: {carpeta} (usar una de {CARPETAS})")
    d = RAIZ / carpeta
    d.mkdir(exist_ok=True)
    return d / archivo


# ---------------------------------------------------------------
# 1. Puntaje continuo del modelo
# ---------------------------------------------------------------
def puntaje(modelo, X):
    """Puntaje continuo de 'caida': mas alto = mas riesgo.

    - Si el modelo tiene predict_proba (logistica, KNN, RF, XGB): probabilidad de la clase 1.
    - Si no (SVM sin probability=True): decision_function, distancia a la frontera.
      En ese caso el umbral NO esta en escala 0-1 (el umbral 0 equivale a predict()).

    En sklearn >= 1.x un Pipeline con SVC(probability=False) NO tiene el atributo
    predict_proba (hasattr da False), asi que basta con hasattr. Si predict_proba
    existe y falla, el error se propaga: nunca se cambia de escala en silencio.
    """
    if hasattr(modelo, "predict_proba"):
        return np.asarray(modelo.predict_proba(X))[:, 1]
    return np.asarray(modelo.decision_function(X))


# ---------------------------------------------------------------
# 2. Metricas con un umbral dado
# ---------------------------------------------------------------
def metricas(y, s, umbral):
    """AUC (con el puntaje continuo) y Precision / Recall / F1 (con s >= umbral).

    Nunca accuracy: con ~15% de caidas, predecir siempre 0 da 85% de accuracy.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    pred = (s >= umbral).astype(int)
    return {
        "AUC": roc_auc_score(y, s),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
    }


def tabla_train_test(m_train, m_test):
    """Tabla con la misma forma que resultados_svm.csv:
    filas AUC/Precision/Recall/F1, columnas Entrenamiento / Prueba / Brecha (train - test)."""
    tabla = pd.DataFrame({"Entrenamiento": m_train, "Prueba": m_test})
    tabla["Brecha (train - test)"] = tabla["Entrenamiento"] - tabla["Prueba"]
    return tabla


def matriz_confusion(y, s, umbral):
    """Matriz de confusion (filas = real, columnas = predicho) con s >= umbral."""
    pred = (np.asarray(s, dtype=float) >= umbral).astype(int)
    return pd.DataFrame(confusion_matrix(np.asarray(y).astype(int), pred, labels=[0, 1]),
                        index=["Real 0", "Real 1"], columns=["Pred 0", "Pred 1"])


# ---------------------------------------------------------------
# 3. Eleccion del umbral en validacion out-of-fold, POR CUANTIL
#
#    Problema: cada vuelta de validacion usa un modelo distinto y sus puntajes
#    no estan en la misma escala (en la SVM el umbral optimo de cada vuelta iba
#    de -0.69 a 0.38). Juntar puntajes crudos de vueltas distintas no sirve.
#
#    Solucion: pasar cada puntaje a PERCENTIL respecto de los puntajes de
#    entrenamiento del mismo modelo que lo genero. El percentil si es comparable
#    entre vueltas y con el modelo final. Se elige el percentil de corte que
#    maximiza F1 out-of-fold y luego se traduce a la escala del modelo final
#    con los puntajes de entrenamiento del modelo final.
#
#    Que fija y que NO fija la regla (importante para el informe):
#    - Fija el umbral en el cuantil p* de los puntajes de entrenamiento DENTRO
#      DE MUESTRA del propio modelo. Solo en entrenamiento se marca 1 - p* de
#      los meses.
#    - Fuera de muestra (OOF o prueba) la fraccion marcada puede ser muy
#      distinta: los puntajes dentro de muestra son mas extremos que los de
#      validacion (sobre todo en RF y XGB). La regla es la misma en OOF y en
#      prueba, asi que no hay filtracion, pero no es una "tasa de alertas fija".
#    - Las vueltas entrenan con 49 a 245 meses y el modelo final con 297, asi
#      que los cuantiles no son estrictamente comparables.
#    - Con 38 caidas OOF (31 de ellas en las vueltas 2 y 4) la curva F1 es
#      plana y tiene casi empates: el corte elegido es ruidoso. Se reporta el
#      rango de percentiles con F1 casi maximo como diagnostico de estabilidad.
# ---------------------------------------------------------------
def _percentil(s_ref, s):
    """Percentil empirico de cada valor de s dentro de s_ref (fraccion de s_ref <= s)."""
    ref = np.sort(np.asarray(s_ref, dtype=float))
    return np.searchsorted(ref, np.asarray(s, dtype=float), side="right") / len(ref)


def _curva_f1(y, v):
    """F1 de cada corte candidato (valores unicos de v) con la regla v >= corte."""
    prec, rec, cortes = precision_recall_curve(y, v)
    # El ultimo punto de la curva (precision 1, recall 0) no tiene corte asociado
    prec, rec = prec[:-1], rec[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return cortes, f1


def _corte_max_f1(y, v):
    """Corte c que maximiza F1 con la regla v >= c. Candidatos = valores unicos de v.
    Si hay empate, np.argmax toma el primero = el corte mas bajo (mas recall).
    (Limitacion declarada: no hay desempate por estabilidad ni tope de meses marcados.)"""
    cortes, f1 = _curva_f1(y, v)
    k = int(np.argmax(f1))
    return float(cortes[k]), float(f1[k])


def _rango_casi_max(y, v, tolerancia=0.02):
    """Menor y mayor corte cuyo F1 queda a menos de 'tolerancia' del F1 maximo.
    Si el rango es ancho, el corte elegido depende de pocas observaciones."""
    cortes, f1 = _curva_f1(y, v)
    ok = f1 >= f1.max() - tolerancia
    return float(cortes[ok].min()), float(cortes[ok].max())


def _umbral_desde_percentil(s_ref, p):
    """Umbral u en la escala del puntaje tal que  s >= u  <=>  percentil(s) >= p,
    con percentiles calculados respecto de s_ref."""
    if p <= 0:
        # percentil(s) >= 0 se cumple para todo s (incluso bajo el minimo de s_ref)
        return -np.inf
    ref = np.sort(np.asarray(s_ref, dtype=float))
    k = int(np.ceil(p * len(ref) - 1e-9)) - 1   # menor ref[k] con (k+1)/n >= p
    return float(ref[max(k, 0)])


def elegir_umbral(estimador, X_train, y_train, cv):
    """Elige el umbral con predicciones OUT-OF-FOLD dentro del entrenamiento (por cuantil).

    Pasos:
      1. Se recorren a mano los splits de cv (TimeSeriesSplit con gap). En cada vuelta
         se clona el estimador (hiperparametros ya elegidos), se ajusta SOLO con el
         pasado de esa vuelta y se puntuan su entrenamiento y su validacion.
      2. Cada puntaje de validacion se pasa a percentil respecto del entrenamiento
         de esa vuelta.
      3. Se juntan las validaciones y se elige el percentil de corte p* que maximiza F1.
      4. El umbral final es el puntaje que deja ese mismo percentil p* en los puntajes
         de entrenamiento del modelo final (estimador ya ajustado con todo X_train).

    No se usa cross_val_predict: TimeSeriesSplit no cubre todo el entrenamiento
    (el primer bloque nunca es validacion), y cross_val_predict exige una particion.

    Parametros
    ----------
    estimador : busqueda.best_estimator_ (YA ajustado con todo el entrenamiento).
                Para las vueltas se clona, asi que el modelo final no se modifica.
    X_train, y_train : SOLO el entrenamiento. Nunca pasar la prueba.
    cv : el mismo TimeSeriesSplit(n_splits=5, gap=HORIZONTE) usado en GridSearchCV.

    Devuelve
    --------
    umbral : float, en la escala de puntaje(estimador, X) (se usa con s >= umbral).
    info : dict con diagnosticos: percentil de corte, fraccion de meses marcados,
           metricas out-of-fold y el percentil optimo de cada vuelta por separado.
    """
    X_train = pd.DataFrame(X_train)
    y_train = pd.Series(np.asarray(y_train).astype(int), index=X_train.index)

    p_oof, y_oof, por_vuelta = [], [], []
    for k, (i_tr, i_va) in enumerate(cv.split(X_train), start=1):
        y_tr, y_va = y_train.iloc[i_tr], y_train.iloc[i_va]
        if y_tr.nunique() < 2:
            warnings.warn(f"Vuelta {k}: el entrenamiento tiene una sola clase, se omite.")
            continue
        m = clone(estimador).fit(X_train.iloc[i_tr], y_tr)
        p_va = _percentil(puntaje(m, X_train.iloc[i_tr]), puntaje(m, X_train.iloc[i_va]))
        p_oof.append(p_va)
        y_oof.append(y_va.values)

        fila = {"vuelta": k,
                "validacion": f"{X_train.index[i_va[0]]} a {X_train.index[i_va[-1]]}",
                "n": len(i_va), "caidas": int(y_va.sum())}
        if y_va.nunique() == 2:
            fila["percentil_vuelta"], fila["F1_vuelta"] = _corte_max_f1(y_va.values, p_va)
        else:
            fila["percentil_vuelta"], fila["F1_vuelta"] = np.nan, np.nan
        por_vuelta.append(fila)

    p_oof = np.concatenate(p_oof)
    y_oof = np.concatenate(y_oof)
    if y_oof.sum() == 0:
        raise ValueError("No hay caidas en las validaciones out-of-fold: no se puede elegir umbral.")

    p_corte, f1_oof = _corte_max_f1(y_oof, p_oof)
    p_min_casi, p_max_casi = _rango_casi_max(y_oof, p_oof)
    m_oof = metricas(y_oof, p_oof, p_corte)

    # Traducir el percentil a la escala del modelo final (ajustado con todo el entrenamiento)
    s_final = puntaje(estimador, X_train)
    umbral = _umbral_desde_percentil(s_final, p_corte)

    info = {
        "percentil_corte": p_corte,
        "percentil_casi_max": (p_min_casi, p_max_casi),
        "frac_marcada_oof": float(np.mean(p_oof >= p_corte)),
        "frac_marcada_train": float(np.mean(s_final >= umbral)),
        "n_oof": len(y_oof),
        "caidas_oof": int(y_oof.sum()),
        "AUC_oof": m_oof["AUC"],
        "F1_oof": f1_oof,
        "Precision_oof": m_oof["Precision"],
        "Recall_oof": m_oof["Recall"],
        "por_vuelta": pd.DataFrame(por_vuelta).set_index("vuelta"),
    }
    return umbral, info


def imprimir_umbral(umbral, info):
    """Print estandar del umbral elegido (llamar ANTES de calcular metricas en prueba)."""
    print(f"Umbral elegido (cuantil, max F1 out-of-fold): {umbral:.4f}"
          f"  |  percentil de corte {info['percentil_corte']:.3f}"
          f"  |  meses marcados: {info['frac_marcada_oof']:.1%} out-of-fold,"
          f" {info['frac_marcada_train']:.1%} en entrenamiento")
    print(f"  Out-of-fold: {info['n_oof']} meses, {info['caidas_oof']} caidas | "
          f"AUC {info['AUC_oof']:.3f} | F1 {info['F1_oof']:.3f} | "
          f"Precision {info['Precision_oof']:.3f} | Recall {info['Recall_oof']:.3f}")
    lo, hi = info["percentil_casi_max"]
    print(f"  Percentiles de corte con F1 OOF a menos de 0.02 del maximo: {lo:.3f} a {hi:.3f}"
          " (rango ancho = corte poco estable)")
    print("  Percentil de corte optimo de cada vuelta por separado (estabilidad):")
    print(info["por_vuelta"].round(3).to_string())
    print()


# ---------------------------------------------------------------
# 4. Tiempos y tamano de la busqueda
# ---------------------------------------------------------------
def medir_prediccion(modelo, X, repeticiones=20):
    """Segundos para calcular el puntaje sobre X (promedio de varias repeticiones).

    Se usa time.perf_counter() en vez de time.time(): predecir 129 meses toma
    milisegundos y en Windows time.time() tiene resolucion de ~15 ms (daria 0).
    """
    t0 = time.perf_counter()
    for _ in range(repeticiones):
        puntaje(modelo, X)
    return (time.perf_counter() - t0) / repeticiones


def n_combinaciones(busqueda):
    """Numero de combinaciones de hiperparametros evaluadas por GridSearchCV."""
    return len(busqueda.cv_results_["params"])


def sensibilidad(res, grilla):
    """AUC de validacion promedio (y min / max) para cada valor de cada hiperparametro,
    promediando sobre las demas dimensiones de la grilla.

    res = pd.DataFrame(busqueda.cv_results_); grilla = el dict pasado a GridSearchCV.
    Los valores salen en el orden de la grilla (2, 3, 5, None), no en orden de
    texto (donde "10" < "5")."""
    filas = []
    for h, valores in grilla.items():
        g = res.groupby(res[f"param_{h}"].astype(str))["mean_test_score"].agg(["mean", "min", "max"])
        for v in valores:
            f = g.loc[str(v)]
            filas.append({"hiperparametro": h, "valor": str(v), "AUC_val_prom": f["mean"],
                          "AUC_val_min": f["min"], "AUC_val_max": f["max"]})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------
# 5. Guardar resultados
# ---------------------------------------------------------------
def guardar_extra(nombre, hiperparametros, busqueda, umbral, info, tiempo, t_pred, **otros):
    """Guarda resultados_<nombre>_extra.csv (una fila): hiperparametros elegidos,
    AUC de validacion, umbral, percentil, tiempos, n de combinaciones y 'otros'.
    None se guarda como el texto "None" (si no, pandas lo leeria como NaN);
    leer con keep_default_na=False."""
    fila = {h: ("None" if v is None else v) for h, v in hiperparametros.items()}
    fila.update({"AUC_validacion": busqueda.best_score_,
                 "umbral": umbral,
                 "percentil_umbral": info["percentil_corte"],
                 "tiempo_s": tiempo,
                 "tiempo_prediccion_s": t_pred,
                 "n_combinaciones": n_combinaciones(busqueda)})
    fila.update(otros)
    pd.DataFrame([fila]).to_csv(ruta("resultados", f"resultados_{nombre}_extra.csv"), index=False)


def guardar_puntajes(nombre, idx, y, s, umbral):
    """Guarda puntajes_<nombre>.csv con columnas fecha, y, puntaje, pred."""
    s = np.asarray(s, dtype=float)
    pd.DataFrame({"y": np.asarray(y).astype(int), "puntaje": s,
                  "pred": (s >= umbral).astype(int)},
                 index=pd.Index(idx, name="fecha")).to_csv(ruta("resultados", f"puntajes_{nombre}.csv"))
