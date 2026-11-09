# CLAUDE.md — Tarea 2 AFML (Aplicaciones en Finanzas de Machine Learning, PUC)

El usuario es **Lorenzo Godoy ("Lolo")**. Grupo: Lolo, **Renato Torres** y **Martín Zapata**. Profesor: Joaquín Aguirre.
**Entrega: martes 10/11/2026, antes de clase** (informe + código + presentación de 10 min).

## Tema

Clasificación binaria: **probabilidad de que el S&P 500 caiga 10% o más en los próximos 3 meses**.

- **Objetivo `caida10`** = 1 si algún cierre **diario** de los 3 meses siguientes (≈63 días hábiles) queda ≥10% bajo el cierre de fin del mes t (definición de **drawdown**, no retorno punto a punto).
- **Frecuencia mensual** (último día hábil del mes), ene-1990 a sep-2025: 429 obs. con objetivo conocido, 62 caídas (14,5%). Oct–dic 2025 quedan con `caida10` = NaN.
- **Predictores (5, uno por hipótesis):** `cape_lag3` (+), `mom_12m` (−), `vix_fin` (+), `baa10y_fin` (+), `t10y3m_fin` (−).
  Descartadas por correlación: `mom_6m`, `dist_mm10`, `vol_realizada` y los promedios `*_prom`. `ret_min_3m` mira al futuro: **nunca usarla como predictor**.
- **Modelos a comparar:** regresión logística (referencia), KNN, SVM (RBF), Random Forest, XGBoost.
- **Criterios de comparación:** sobreajuste (brecha train − test), sensibilidad a hiperparámetros, tiempo de entrenamiento e interpretabilidad. Un resultado válido es que ningún modelo supere claramente a la logística.

## Archivos

| Archivo | Contenido |
|---|---|
| `base_tarea2_caidas_sp500.csv` | Base mensual final (hecha por Renato). Índice `fecha` (AAAA-MM). |
| `tarea2.ipynb` | Celda 0: descarga ^GSPC con yfinance. Celda 1: **regresión logística** completa (con resultados). |
| `svm_tarea2.py` | **SVM RBF** (Zapata), terminado; guarda `resultados_svm.csv` y `resultados_svm_extra.csv`. |
| `documentos/Informe Tarea 2 Machine Learning.docx` | Borrador del informe: secciones 1 (Motivación) y 2 (Datos) redactadas; 3–5 y anexos con `[ ]` por llenar. |
| `Msgs.txt` | Recomendaciones metodológicas + plan y reparto de tareas del grupo. |
| `documentos/GUIA_TAREA2.md` | Guía de modelos, resultados y archivos (7/11). |

No hay en la carpeta código de construcción de la base (cruce FRED + Shiller + etiqueta). KNN, RF, XGBoost y `requirements.txt` ya existen (ver "Resultados clave").

## Convenciones obligatorias (todos los modelos)

Copiar **tal cual** las secciones 1 y 2 del código de la logística / SVM. No cambiarlas sin avisar al grupo.

```python
df = pd.read_csv("base_tarea2_caidas_sp500.csv", index_col="fecha")
df = df.dropna(subset=["caida10"])
PREDICTORES = ["cape_lag3", "mom_12m", "vix_fin", "baa10y_fin", "t10y3m_fin"]
INICIO_PRUEBA = "2015-01"; HORIZONTE = 3
idx_train = X.index[X.index < INICIO_PRUEBA][:-HORIZONTE]   # termina 2014-09 (hueco oct–dic 2014)
idx_test  = X.index[X.index >= INICIO_PRUEBA]
cv = TimeSeriesSplit(n_splits=5, gap=HORIZONTE)   # GridSearchCV con scoring="roc_auc"
```

- **Entrenamiento:** 1990-01 a 2014-09, 297 meses, 41 caídas. **Prueba:** 2015-01 a 2025-09, 129 meses, 21 caídas.
- Episodios de caída en prueba (≈6): 2015 (ago y fin de año), 4T-2018, COVID 2020, 2022, ago–oct 2023, abr 2025. Pocos eventos, así que **las métricas son ruidosas** y hay que decirlo en el informe.
- **Separación temporal**, nunca aleatoria. **La prueba se usa una sola vez**, al final.
- `StandardScaler` **dentro de un `Pipeline`** (obligatorio para KNN/SVM y también usado en la logística). RF/XGB no lo necesitan.
- **Desbalance:** `class_weight="balanced"` (logística, SVM, RF); `scale_pos_weight = n_neg/n_pos` (XGBoost, recalculado en cada vuelta con su propio entrenamiento: clase `XGBBalanceado`, desde el 7/11).
- **Métricas:** AUC, Precisión, Recall y F1 en entrenamiento y prueba, más `Brecha (train - test)`. **Nunca accuracy.** Matriz de confusión en prueba.
- **Tiempo:** `time.time()` alrededor de `busqueda.fit` (búsqueda completa + reajuste final), igual en todos los modelos.
- Guardar `resultados_<modelo>.csv` (misma forma de tabla) para armar la comparativa automáticamente.
- Fijar una semilla (`random_state`) en RF, XGB y la importancia por permutación.

## Resultados hasta ahora

**Logística, version ANTIGUA con umbral 0,5 (reemplazada; los valores vigentes estan en "Resultados clave", Bloque 4)** (mejor C = 0.001, AUC de validación 0.533 ± 0.319, 0.16 s):

| | Train | Test | Brecha |
|---|---|---|---|
| AUC | 0.773 | 0.631 | 0.142 |
| Precisión | 0.318 | 0.258 | 0.060 |
| Recall | 0.683 | 0.381 | 0.302 |
| F1 | 0.434 | 0.308 | 0.126 |

Statsmodels (sin regularizar): significativos `cape_lag3` (+0.67, p=0.002) y `baa10y_fin` (+0.81, p=0.015); `mom_12m` marginal (−0.44, p=0.085). El VIX sale con signo negativo y no significativo.

**SVM:** el código está listo pero sus resultados aún no aparecen en la carpeta (correr `svm_tarea2.py`).

## Problemas detectados (resolver o declarar)

1. **El umbral no es consistente:** la logística usa prob ≥ 0.5 y la SVM `predict()` (puntaje ≥ 0). La pauta y el informe (3.4) dicen que el **umbral se elige en validación** (por ejemplo, el que maximiza F1 o logra un recall mínimo). Falta la **función de métricas común** con elección de umbral. Era tarea de Lolo del jueves 5/11.
2. **La validación de la logística es muy ruidosa** (std ≈ 0.3) y el mejor C es el extremo de la grilla (0.001), así que los coeficientes de sklearn salen casi nulos. Conviene ampliar la grilla o comentarlo en el informe.
3. **El tiempo de predicción** lo pide la tabla del informe (4.4), pero ningún script lo mide todavía.
4. **El Anexo B del informe** nombra archivos (`datos.ipynb`, `particion_metricas.ipynb`, `modelos.ipynb`, `requirements.txt`) que no existen. Hay que ajustarlo a los archivos reales.
5. ~~Faltaban librerias en el Python del sistema~~ Resuelto: `pip install -r requirements.txt` (Python 3.13.2, incluye scipy, ipykernel y nbconvert).

## Plan del grupo (de Msgs.txt)

| Fecha | Renato | **Lolo** | Zapata |
|---|---|---|---|
| Jue 5/11 | Base de datos ✅ | Partición ✅ (en el código común), validación ✅, **función de métricas común ⏳** | Motivación + Datos del informe ✅, estructura de la presentación |
| Vie 6 – sáb 7/11 (mediodía) | Logística ✅ + KNN ⏳ | **Random Forest + XGBoost ⏳** | SVM ✅ (código) |
| Sáb 7 tarde – dom 8/11 (noche) | Interpretación económica + conclusión | **Tabla comparativa + discusión de modelos** (sobreajuste, hiperparámetros, tiempo, interpretabilidad) | Láminas |
| Lun 9/11 | Revisar, ensayar 10 min, explicar el modelo de otro | ídem | ídem; subir todo en la noche |

Se sugirió que Renato o Zapata ayuden a Lolo con interpretabilidad y tiempos, porque su parte es la más pesada.

## ✅ Checklist de Lolo

### Ya (atrasado del jueves 5/11)
- [ ] Escribir la **función de métricas común** `metricas(y, puntaje, umbral)` → AUC, Precisión, Recall, F1 + brecha, para que la usen los 5 modelos.
- [ ] Definir y acordar con el grupo el **criterio de umbral** elegido en validación (out-of-fold de `TimeSeriesSplit`, por ejemplo maximizar F1 o recall ≥ X%) y aplicarlo también a la logística y la SVM.
- [ ] Agregar la medición del **tiempo de predicción** y el **n° de combinaciones evaluadas** a la plantilla común.
- [ ] Avisar al grupo cualquier cambio a la plantilla común (secciones 1–2 congeladas).

### Modelos (listos el sáb 7/11 a mediodía)
- [ ] **Random Forest**: `RandomForestClassifier(class_weight="balanced", random_state=...)`, grilla de `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`, con el mismo `cv` y `scoring="roc_auc"`.
- [ ] **XGBoost**: `XGBClassifier(scale_pos_weight=n_neg/n_pos, random_state=...)`, grilla de `learning_rate`, `n_estimators`, `max_depth`, `subsample`, `reg_lambda` (grilla chica: hay pocas observaciones).
- [ ] Para cada uno: tabla train/test/brecha, matriz de confusión, tiempo, mejores hiperparámetros, tabla de sensibilidad del AUC de validación por hiperparámetro.
- [ ] **Importancia de variables** (por permutación, con AUC) en RF y XGB para comparar con los coeficientes de la logística.
- [ ] Guardar `resultados_rf.csv`, `resultados_xgb.csv` (+ `_extra.csv` con hiperparámetros, AUC de validación y tiempo).

### Comparativa y discusión (lista el dom 8/11 en la noche)
- [ ] Juntar los `resultados_*.csv` de los 5 modelos y llenar las **Tablas 1 y 2** (sección 4.1) y la tabla de **tiempos** (4.4).
- [ ] **Figura 1:** curvas ROC de los 5 modelos en prueba. **Figura 2:** matrices de confusión.
- [ ] Redactar **4.3** (esperado vs. observado por modelo) y **4.4** (sobreajuste, sensibilidad a hiperparámetros, tiempo, interpretabilidad).
- [ ] Llenar la sección **3.1** (tabla de partición: 297 / 41 / 13,8% en train; 129 / 21 / 16,3% en test) y **3.2** (5 vueltas, AUC, gap 3).
- [ ] Llenar el **Anexo A** (grillas y valores elegidos) y corregir el **Anexo B** (archivos reales, semilla, cómo correr).
- [ ] Revisar si alguna diferencia de AUC con la logística es mayor que el ruido esperable con ~6 episodios.
- [ ] Aportar a la **declaración de uso de IA** (sección 6): herramienta, parte de la actividad, prompts.

### Lunes 9/11
- [ ] Revisar el informe completo.
- [ ] Ensayo cronometrado (10 min) y **saber explicar los modelos de los demás** (logística, KNN, SVM).
- [ ] Crear `requirements.txt` y verificar que todo corre de cero.
- [ ] Subir informe, código y presentación en la noche.

## Notas para Claude
- Escribir en español, con código comentado en español y sin tildes en el código (como los scripts existentes).
- Mantener el estilo de los scripts actuales: secciones numeradas, `Pipeline` + `GridSearchCV`, prints con tablas `.round(3)`.
- No tocar la base ni la partición sin que Lolo lo pida explícitamente.

---

## Plan de ejecución por bloques (vie 6/11)

### Resumen de la sesión de planificación (6/11)
Lolo pidió revisar sus pendientes y armar un plan para terminarlos hoy, con el modelo y el esfuerzo recomendados para cada tarea. Estado de la carpeta al 6/11: solo están `tarea2.ipynb`, `svm_tarea2.py`, la base, el `.docx` y `Msgs.txt`. Todavía **no existen** `resultados_svm.csv`, KNN, RF, XGB ni `requirements.txt`.
Lo siguiente **no se cierra hoy**: KNN e interpretación económica (Renato), ensayo y subida (lunes 9/11). En la comparativa, esas partes quedan con espacios por llenar.

### Protocolo de uso
1. Lolo cambia el modelo y el esfuerzo: `/model <modelo>` → `/effort <nivel>`. Para Opus en bloques largos, `/fast` es opcional.
2. Lolo escribe: **"ejecuta bloque X"**.
3. Claude ejecuta **solo** ese bloque, siguiendo lo descrito abajo y las convenciones obligatorias de arriba.
4. Al terminar, Claude:
   - marca el bloque `[x]` en este archivo;
   - anota en "Resultados clave" lo esencial (umbral elegido, AUC de prueba, mejores hiperparámetros, tiempos, problemas);
   - le recuerda a Lolo qué modelo y esfuerzo usar en el bloque siguiente.
5. Si un bloque cambia la plantilla común (Bloques 1 y 4), Claude le recuerda a Lolo **avisar al grupo**.

**Dependencias:**
- 0 → 1 → (2, 3, 4).
- 5 necesita 2–4.
- 6a se puede hacer en cualquier momento.
- 6b necesita 5.
- Si falta tiempo, 5 y 6b pasan al sábado. Los Bloques 1–3 **no** se pueden atrasar.

### Resumen de modelo y esfuerzo

| Bloque | Tarea | Modelo | Esfuerzo | Por qué |
|---|---|---|---|---|
| 0 | Entorno, `requirements.txt`, correr SVM | Haiku 5.5 | low | Mecánico |
| 1 | `comun.py`: métricas + umbral OOF + tiempos | **Opus 5.5** | **high** | Riesgo de filtrar información del futuro; afecta a los 5 modelos |
| 2 | Random Forest | Sonnet 5.5 | medium | Sigue la plantilla de la SVM |
| 3 | XGBoost | Sonnet 5.5 | medium | Ídem |
| 4 | Pasar la logística y la SVM al umbral común | Sonnet 5.5 | medium | Cambio acotado en código del grupo |
| 5 | Comparativa: tablas, ROC, bootstrap | Sonnet 5.5 | medium | Código estándar |
| 6a | Informe 3.1, 3.2, Anexos A–B | Haiku 5.5 / Sonnet 5.5 | low | Copiar números |
| 6b | Informe 4.3 y 4.4 (discusión) | **Opus 5.5** | **high** | Juicio analítico con métricas ruidosas |

Tiempo total estimado: ~7–8 h.

### [x] Bloque 0 · Entorno (15 min) — Haiku 5.5 · low
- `pip install scikit-learn xgboost statsmodels yfinance pandas numpy matplotlib`
- Crear `requirements.txt` con las versiones instaladas de esos paquetes (`pip freeze` filtrado).
- Correr `python svm_tarea2.py` para generar `resultados_svm.csv` y `resultados_svm_extra.csv`.
- **Verificación:** la SVM imprime 297 meses / 41 caídas en entrenamiento y 129 / 21 en prueba, y aparecen los dos CSV.

### [x] Bloque 1 · `comun.py`: métricas + umbral + tiempos (1–1,5 h) ⚠️ crítico — Opus 5.5 · high
Archivo nuevo `comun.py`. Las secciones 1–2 congeladas **no** se mueven: siguen copiadas en cada script.
- `metricas(y, puntaje, umbral)` → dict con AUC (con el puntaje continuo), Precisión, Recall y F1 (con `puntaje >= umbral`, `zero_division=0`).
- `tabla_train_test(m_train, m_test)` → DataFrame con las columnas `Entrenamiento`, `Prueba` y `Brecha (train - test)`, con la misma forma que `resultados_svm.csv`.
- `puntaje(modelo, X)`:
  - usa `predict_proba(X)[:, 1]` si el modelo lo tiene;
  - si no, usa `decision_function(X)` (SVM sin `probability=True`).
- `elegir_umbral(estimador, X_train, y_train, cv)`:
  - Hace la predicción **out-of-fold** recorriendo a mano los splits de `TimeSeriesSplit(n_splits=5, gap=3)`: `clone` del estimador, ajuste en el train de cada vuelta y puntaje en su validación. `cross_val_predict` no sirve, porque los splits no cubren todo el entrenamiento.
  - Elige el umbral que **maximiza F1** sobre las observaciones out-of-fold, con los candidatos tomados de los puntajes únicos.
  - Usa **solo** el entrenamiento y los hiperparámetros ya elegidos (`busqueda.best_estimator_`).
- `medir_prediccion(modelo, X)` → segundos de predicción (`time.time()` alrededor del cálculo del puntaje sobre X).
- Helper o convención: `n_combinaciones = len(busqueda.cv_results_["params"])`.
- **Criterio propuesto para el grupo:** umbral que maximiza F1 out-of-fold. Lolo tiene que avisar al grupo.
- **Verificación:** probar `comun.py` con la logística (`C=0.001`) o la SVM, imprimir el umbral elegido antes de tocar la prueba y revisar que la tabla tenga la forma correcta.

### [x] Bloque 2 · `rf_tarea2.py` (1 h) — Sonnet 5.5 · medium
- Copiar la estructura de `svm_tarea2.py`: secciones 1–2 idénticas y secciones 3–7 adaptadas.
- `RandomForestClassifier(class_weight="balanced", random_state=42)`, sin escalador.
- Grilla chica:
  - `n_estimators` [200, 500]
  - `max_depth` [2, 3, 5, None]
  - `min_samples_leaf` [5, 10, 20]
  - `max_features` ["sqrt", None]
- `GridSearchCV(..., cv=cv, scoring="roc_auc")`, con `time.time()` alrededor de `busqueda.fit`.
- Con `comun.py`: umbral out-of-fold, métricas train/test/brecha, tiempo de predicción y n° de combinaciones.
- Matriz de confusión en prueba, con el umbral elegido.
- Sensibilidad: AUC de validación promedio por cada hiperparámetro (`groupby` sobre `cv_results_`).
- Importancia por permutación (`scoring="roc_auc"`, `n_repeats=30`, `random_state=0`) sobre la prueba, igual que en la SVM.
- Guardar:
  - `resultados_rf.csv`;
  - `resultados_rf_extra.csv`: hiperparámetros, AUC de validación, umbral, tiempo de entrenamiento, tiempo de predicción, n° de combinaciones;
  - `puntajes_rf.csv`: puntajes de prueba, que necesita la ROC del Bloque 5.
- **Verificación:** corre de cero. Comentar si el AUC de entrenamiento es ≈1 (sobreajuste esperable).

### [x] Bloque 3 · `xgb_tarea2.py` (1 h) — Sonnet 5.5 · medium
- Misma estructura que `rf_tarea2.py`.
- `XGBClassifier(scale_pos_weight=n_neg/n_pos, random_state=42, eval_metric="logloss")`, con `n_neg/n_pos` calculado **solo** con `y_train`.
- Grilla chica:
  - `learning_rate` [0.01, 0.1]
  - `n_estimators` [100, 300]
  - `max_depth` [2, 3]
  - `subsample` [0.7, 1.0]
  - `reg_lambda` [1, 10]
- Guardar `resultados_xgb.csv`, `resultados_xgb_extra.csv` y `puntajes_xgb.csv`.
- **Verificación:** corre de cero y la partición coincide con la de los otros modelos.

### [x] Bloque 4 · Umbral común en la logística y la SVM (30–45 min) — Sonnet 5.5 · medium
- Cambiar **solo** la sección de métricas de `svm_tarea2.py` y de la celda 1 de `tarea2.ipynb` para que usen `comun.py`, con el umbral out-of-fold y el tiempo de predicción.
- Guardar `resultados_logistica.csv`, `resultados_logistica_extra.csv` y `puntajes_logistica.csv`, más los equivalentes de la SVM.
- Opcional: ampliar la grilla de C de la logística hacia abajo, porque hoy sale en el extremo (0.001). Si no se amplía, dejarlo anotado para el informe.
- **Recordar a Lolo que avise al grupo:** cambia la plantilla común.
- **Verificación:** ambos corren de cero. Comparar con los resultados anteriores registrados más arriba en este archivo.

### [x] Bloque 5 · `comparativa.py`: tablas y figuras (1 h) — Sonnet 5.5 · medium
- Leer todos los `resultados_*.csv`, `_extra.csv` y `puntajes_*.csv` que existan. KNN entra solo si ya está.
- Armar:
  - la **Tabla 1** (métricas en prueba);
  - la **Tabla 2** (brechas train − test);
  - la **tabla de tiempos** (entrenamiento, predicción y n° de combinaciones).
- **Figura 1:** curvas ROC en prueba de todos los modelos (PNG). **Figura 2:** matrices de confusión (PNG). Usar la skill `dataviz`.
- **Ruido:** bootstrap por bloques del AUC de prueba (o prueba de DeLong) para la diferencia de cada modelo contra la logística, porque hay solo ~6 episodios.
- **Verificación:** corre de cero y sin errores aunque falte KNN.

### [x] Bloque 6a · Informe: secciones fáciles (45 min) — Haiku 5.5 / Sonnet 5.5 · low
- Editar `Informe Tarea 2 Machine Learning.docx` con la skill `docx`. Llenar:
  - **3.1** Partición: 297 / 41 / 13,8% en entrenamiento y 129 / 21 / 16,3% en prueba.
  - **3.2** Validación: TimeSeriesSplit de 5 vueltas, AUC, gap 3 y umbral out-of-fold que maximiza F1.
  - **Anexo A:** grillas y valores elegidos de cada modelo, sacados de los `_extra.csv`.
  - **Anexo B:** archivos reales (`tarea2.ipynb`, `svm_tarea2.py`, `rf_tarea2.py`, `xgb_tarea2.py`, `comun.py`, `comparativa.py`, `requirements.txt`), semilla y cómo correr todo.

### [x] Bloque 6b · Informe: discusión (salvo KNN, que falta) (1–1,5 h) — Opus 5.5 · high
- **4.1:** Tablas 1–2 y figuras del Bloque 5.
- **4.3:** lo esperado vs. lo observado en cada modelo.
- **4.4:**
  - Sobreajuste (brechas).
  - Sensibilidad a hiperparámetros (tablas de validación).
  - Tiempos de entrenamiento y predicción.
  - Interpretabilidad: coeficientes de la logística vs. importancia por permutación.
- Advertir explícitamente que las métricas son ruidosas por las pocas caídas en prueba. No sobrevender diferencias que el bootstrap no respalda.
- Ir dejando material para la **sección 6 (uso de IA)**: herramienta, parte de la actividad y prompts usados en cada bloque.

### Resultados clave (Claude los va llenando al terminar cada bloque)
- Bloque 0: requirements.txt creado (sklearn 1.9.1, xgboost 3.4.1, statsmodels 0.15.0, yfinance 1.7.0, pandas 2.3.2, numpy 2.3.2, matplotlib 3.11.2). SVM corrio OK: 297/41 train, 129/21 test. Mejor C=0.1, gamma=0.1, AUC val 0.700 (std ~0.19), 0.87 s. Test: AUC 0.824 train / 0.625 test, brecha 0.199. Umbral aun 0 (predict), pendiente de Bloque 4. Recall test 0.571, F1 0.375. Permutacion: cape_lag3 y baa10y_fin son los mas importantes; t10y3m_fin negativo.
- Bloque 1: `comun.py` creado (puntaje, metricas, tabla_train_test, matriz_confusion, elegir_umbral, imprimir_umbral, medir_prediccion [perf_counter, prom. 20 rep.], n_combinaciones, guardar_puntajes). OOF = 245 meses / 38 caidas (vueltas de 49; la 1a parte 1990-01..1994-01 nunca es validacion). Prueba: logistica C=0.001 -> umbral 0.5048, test AUC 0.631, P 0.250, R 0.286, F1 0.267 (antes con 0.5: F1 0.308). SVM C=0.1 g=0.1 -> umbral 0.0648, test AUC 0.625, P 0.263, R 0.476, F1 0.339 (antes con 0: F1 0.375). PROBLEMA: al juntar vueltas se mezclan escalas de modelos distintos (AUC OOF agrupado 0.545 logit / 0.531 SVM vs 0.533 / 0.700 promedio por vuelta) y el umbral optimo por vuelta es inestable (SVM de -0.69 a 0.38). Alternativa a decidir: umbral por cuantil (fraccion de meses marcados) o mediana de umbrales por vuelta. Criterio max F1 OOF pendiente de aprobacion del grupo.
- Bloque 1 (revision, opcion 2 elegida por Lolo): `elegir_umbral` ahora es POR CUANTIL. Cada puntaje out-of-fold se pasa a percentil respecto del entrenamiento de su propia vuelta; se elige el percentil que maximiza F1 y se traduce al modelo final con sus puntajes de entrenamiento. Arregla la mezcla de escalas: AUC OOF agrupado de la SVM sube de 0.531 a 0.649 (cerca del 0.700 por vuelta). Resultados: logistica percentil 0.801 (marca ~20% de meses en train), umbral 0.5120, test P 0.263 R 0.238 F1 0.250. SVM percentil 0.622 (marca ~38% en train), umbral -0.3786, test P 0.198 R 0.857 F1 0.321, pero marca 91 de 129 meses de prueba (los puntajes de prueba se desplazan hacia arriba). Logistica con C=0.001 tiene AUC OOF agrupado 0.492 (sin poder discriminante fuera de muestra) y percentil 0 en las vueltas 1-2. Criterio pendiente de aviso al grupo.
- Bloque 2: `rf_tarea2.py` corre de cero (90 s, 48 combinaciones = 240 ajustes). Mejor: max_depth=2, max_features=sqrt, min_samples_leaf=5, n_estimators=200; AUC val 0.676 (std 0.197). Umbral por cuantil 0.4017 (percentil 0.541), marca 80% OOF / 46% train (poco selectivo, recall OOF 0.97). Test: AUC 0.892 train / 0.568 test (brecha 0.324), P 0.208, R 0.524, F1 0.297; matriz test: 66 VN, 42 FP, 10 FN, 11 VP. Prediccion 10 ms. Sensibilidad: max_features (sqrt 0.619 vs None 0.546) y min_samples_leaf (5: 0.612 vs 20: 0.535) pesan mas que n_estimators y max_depth (~igual). Permutacion: vix_fin domina (0.060, ruidoso), resto ~0; MDI reparte entre vix, baa10y, mom_12m. AUC train 0.89 (no ~1 porque el mejor modelo es poco profundo). Genera tambien `sensibilidad_rf.csv`, `puntajes_rf.csv`.
- Bloque 3: `xgb_tarea2.py` corre de cero (22.7 s, 32 combinaciones = 160 ajustes; scale_pos_weight = 256/41 = 6.24, solo con y_train; particion 297/41 y 129/21 igual que los demas). Mejor: learning_rate=0.01, n_estimators=300, max_depth=2, subsample=0.7, reg_lambda=10; AUC val 0.635 (std 0.140). Umbral por cuantil 0.5032 (percentil 0.745), marca 46% OOF / 26% train. Test: AUC 0.930 train / 0.558 test (brecha 0.373, la mayor de todos), P 0.163, R 0.333, F1 0.219; matriz test: 72 VN, 36 FP, 14 FN, 7 VP. Prediccion 1.2 ms. Sensibilidad: learning_rate (0.01: 0.599 vs 0.1: 0.565) y reg_lambda/subsample pesan algo; rango total de la grilla 0.519-0.635 (poca sensibilidad). Permutacion: vix_fin 0.054, baa10y_fin 0.026, cape_lag3 0.014, t10y3m_fin negativo (-0.014, ruido). Genera tambien `sensibilidad_xgb.csv`, `puntajes_xgb.csv`.
- Bloque 4: `svm_tarea2.py` y la celda 1 de `tarea2.ipynb` ahora usan `comun.py` (umbral por cuantil, tiempo de prediccion, n combinaciones) y guardan `resultados_*`, `resultados_*_extra.csv` y `puntajes_*.csv` (logistica y svm). Grilla de C de la logistica NO se amplio (mejor C sigue en el extremo 0.001): comentarlo en el informe. Logistica: umbral 0.5120 (percentil 0.801), test AUC 0.631, P 0.263, R 0.238, F1 0.250 (brecha F1 0.205); matriz 94/14/16/5; pred 0.7 ms; AUC OOF 0.492. SVM: umbral -0.3786 (percentil 0.622), test AUC 0.625, P 0.198, R 0.857, F1 0.321; matriz 35/73/3/18 (marca 91 de 129 meses de prueba); pred 2.0 ms, 20 combinaciones, 236 de 297 vectores de soporte (79%). Resultados coinciden con los del Bloque 1. Las salidas guardadas en el notebook (celda 1) quedaron desactualizadas hasta que se reejecute en Jupyter. AVISAR AL GRUPO: cambia la plantilla comun (umbral por cuantil de comun.py).
- Bloque 5: `comparativa.py` corre de cero sin KNN (lo agrega solo cuando existan `resultados_knn.csv`, `_extra` y `puntajes_knn.csv`). Se agrego a rf/xgb/svm el guardado de `importancia_<m>.csv` y una celda nueva en `tarea2.ipynb` (despues de la celda 1) que guarda `importancia_logistica.csv` (permutacion + coef. sklearn + statsmodels); los numeros de modelos no cambiaron al reejecutar. Genera tabla1_prueba, tabla2_brechas, tabla_tiempos, tabla_bootstrap_auc, tabla_importancia (.csv) y fig1_roc_prueba, fig2_matrices_confusion, fig3_alertas_prueba (.png). Bootstrap circular por bloques (12 meses, 5000 rep., semilla 0; sensibilidad con 6 y 24): IC95 AUC prueba logit [0.478, 0.763]; dif. vs logit: SVM -0.005 [-0.090, +0.077], RF -0.063 [-0.156, +0.044], XGB -0.073 [-0.189, +0.069]; ninguna distinguible de 0 (p 0.21-0.92; DeLong indep. 0.15-0.87). Ranking AUC prueba: logit 0.631 > SVM 0.625 > RF 0.568 > XGB 0.558, al reves que el AUC de validacion (SVM 0.700 > RF 0.676 > XGB 0.635 > logit 0.533). Tiempos: ver la corrida de referencia del 7/11 mas abajo. Permutacion (en prueba): la SVM prioriza cape_lag3 y baa10y_fin; en la logistica cape 0.039 ~ VIX 0.038 > baa 0.031; RF y XGB priorizan vix_fin; t10y3m_fin ultimo en todos. HALLAZGO (fig3): las alertas llegan mayoritariamente DESPUES de la caida (ago-sep 2015, mar-jun 2020, dic 2018): los modelos reaccionan a VIX/spreads altos post-caida, cuando la etiqueta ya es 0, lo que explica la baja precision. (Notebook reejecutado el 7/11, ver abajo.) Las alertas tardias valen para logit/RF/XGB (13/14, 29/42, 26/36 FP en los 6 meses tras una caida), NO para la SVM (40/73; avisa 47 meses seguidos desde 2021-11).
- Bloques 6a y 6b (6/11, misma sesion): informe editado con python-docx; respaldo del original en `respaldos/informes/Informe Tarea 2 Machine Learning (antes de Claude 6-11).docx`. Llenado: 3.1 (tabla de particion + episodios), 3.2 (5 vueltas de 49 meses, caidas por vuelta 1/16/3/15/3), 3.3 (corregidos los hiperparametros ajustados de logit y SVM), 3.4 (14,5% / 85,5% y criterio de umbral por cuantil, max F1 OOF), 4.1 (Tablas 1-2 + lectura con bootstrap y referencia "avisar siempre": P 0,163, R 1, F1 0,280), 4.2 (Figuras 1-3 + texto de alertas tardias), 4.3 (columna "Que observamos"), 4.4 (sobreajuste, sensibilidad, tabla de tiempos en ms, interpretabilidad), 6 (fila de Claude como BORRADOR), Anexo A (grillas y elegidos), Anexo B (archivos reales, semillas, como correr), Anexo C (Tablas C1 importancia y C2 bootstrap), referencias FRED (codigos). Validado (XSD OK) y renderizado con Word: 17 paginas. Salidas de consola guardadas en `salida_*.txt`.

### Pendientes (actualizado el dom 8/11: KNN integrado)
Guia completa de modelos, resultados y archivos: `documentos/GUIA_TAREA2.md` (seccion 8 = KNN).

**Hecho el 8/11 (KNN, antes bloqueante):**
- `knn_tarea2.py` creado desde `respaldos/codigo/KNN_original_Renato.ipynb` (celda 3 de Renato) con la plantilla comun; NO se cambio su grilla de k. Corre de cero (`salidas/salida_knn.txt`). Mejor k = 41 (extremo de la grilla), AUC val 0.646 (std 0.156), umbral por cuantil 0.0488 (2 de 41 vecinos, percentil 0.357) que marca 83.7% OOF / 68.4% train / 107 de 129 meses de prueba (casi "avisar siempre"). Test: AUC 0.839 train / 0.619 test (brecha 0.220), P 0.187, R 0.952, F1 0.312; matriz 21/87/1/20. Entrenamiento 0.51 s (9 comb., 46 ajustes), prediccion 1.8 ms (corrida conjunta del 8/11). Con k=1: AUC 1.000 / 0.607. Permutacion: baa10y 0.067, mom_12m 0.061, cape 0.054, vix 0.043, t10y3m -0.005. Bootstrap vs logistica: -0.012 [-0.136, +0.105] (bloque 12), p 0.88. FP tardios: 41 de 87. `comparativa.py` corre con los 5 modelos (`salidas/salida_comparativa.txt`); los otros 4 no cambiaron.
- Informe actualizado (respaldo en `respaldos/informes/Informe Tarea 2 Machine Learning (antes de KNN 8-11).docx`): Tablas 1-2, tiempos, 4.1 (texto y bootstrap), 4.2 (Figuras 1-3 regeneradas, texto de alertas tardias), 4.3 (fila KNN, "cinco modelos"), 4.4 (sobreajuste, sensibilidad, tiempo, interpretabilidad), 3.4 (fraccion marcada de KNN en "Limitaciones"), Anexos A, B y C (columna KNN en C1, fila en C2), fila de IA, y Tabla 3 corregida (la grilla de Renato no incluia `weights` ni `metric`). Renderizado con Word: 19 paginas.
- `GUIA_TAREA2.md` actualizada (respaldo en `respaldos/`). Comentarios "4 modelos" -> "5 modelos" en svm/rf/xgb.
- Problemas para avisar a Renato: su umbral (frecuencia de caidas, 0.138) se reemplazo por el comun; mejor k en el extremo de la grilla (no se amplio; opcional agregar 61 y 81).

**Necesitan a Lolo o al grupo:**
5. **Avisar al grupo (y a Renato por su KNN):** umbral por cuantil (comun.py), funciones nuevas de comun.py (`guardar_extra`, `sensibilidad`, `ruta`), `XGBBalanceado` (scale_pos_weight por vuelta), celda 3 del notebook (importancia de la logistica) y los cambios hechos en el texto de Zapata (2.2-2.4 y la formula de 3.3).
6. **Revisar el texto que redacto Claude:** 3.1-3.4 (incluida la nota de KNN en 'Limitaciones') (incluidas "Limitaciones del criterio" y "Uso de la prueba durante el desarrollo"), 4.1-4.4, Anexos y la fila de IA de la seccion 6 ("[Borrador, revisar]"); agregar el uso de IA de cada integrante y la fila "[Otra, si corresponde]".
7. **Siguen con `[ ]` (no eran de Lolo):** Resumen, 2.1 [Formula], 2.5 descriptivas, 4.5 interpretacion economica (sugerencia: Figura 3 + Tabla C1), 5 conclusion (5.1, 5.2 [k] episodios ~6 y se pueden resumir ahi las limitaciones del umbral de 3.4; 5.3), fechas de consulta en Referencias, correlaciones del Anexo C y el **codigo de construccion de la base** (Anexo B; sin el, la base no es reproducible).
8. **Lunes 9/11:** ensayo de 10 min, saber explicar los modelos de los demas y subir informe + codigo + presentacion en la noche. Antes de enviar la carpeta: borrar `__pycache__/` y, si se quiere, `gspc_diario.csv` (no lo usa nada; lo regenera la celda 1).
9. Opcional: ampliar la grilla de C de la logistica hacia abajo (hoy en el extremo 0,001); cambiaria tablas y texto, y seria otra mirada a la prueba (declararla).

**Hechos el 7/11** (detalle en "Resultados clave"): hallazgos de las dos revisiones implementados, notebook reejecutado, corrida de referencia de tiempos, `requirements.txt` completo y correcciones de formato del borrador original.

## Estructura de carpetas (reorganizada el 8/11)
- Raiz (el codigo se corre desde aqui): `README.md`, `requirements.txt`, `base_tarea2_caidas_sp500.csv`, scripts (`comun.py`, `knn_tarea2.py`, `svm_tarea2.py`, `rf_tarea2.py`, `xgb_tarea2.py`, `comparativa.py`), `tarea2.ipynb` y este `CLAUDE.md`.
- `resultados/`: todos los CSV generados (`resultados_*`, `puntajes_*`, `importancia_*`, `sensibilidad_*`, `tabla*`). `figuras/`: `fig1`, `fig2`, `fig3` (PNG). `salidas/`: `salida_*.txt` (consola).
- `documentos/`: informe `.docx` actual, `GUIA_TAREA2.md`, `hallazgos_code_review.md` y `revision_hallazgos_6-11.md`.
- `respaldos/` (nada de esto se usa en la entrega): `codigo/` (codigo y corridas previas, `KNN_original_Renato.ipynb`, `gspc_diario.csv`), `informes/` (versiones anteriores del .docx) y `notas/` (CLAUDE.md y GUIA anteriores).
- La base se lee desde la raiz (seccion 1 congelada, sin cambios). `comun.py` crea las carpetas de salida si no existen. El notebook se reejecuto el 8/11 y escribe en `resultados/`; su celda 1 regenera `gspc_diario.csv` en la raiz (no lo usa nada).
- Hallazgos 7/11 (`hallazgos_code_review.md` + `revision_hallazgos_6-11.md`), decisiones de Lolo: umbral por cuantil SE MANTIENE (se arreglo el borde p=0 y se declararon las limitaciones 1 y 3 en 3.4); scale_pos_weight de XGB AHORA SE RECALCULA EN CADA VUELTA (clase `XGBBalanceado`); permutacion SE MANTIENE EN PRUEBA y se declara. Codigo: `comun.py` (p=0 -> -inf, diagnostico de casi-empates de F1, `puntaje` sin except silencioso, rutas con `ruta()` relativas a comun.py, `sensibilidad()` y `guardar_extra()` comunes con None guardado como "None"); rf/xgb/svm sin codigo duplicado ni numpy sin usar; `comparativa.py` usa `comun.matriz_confusion`/`comun.metricas` y verifica pred/metricas contra el umbral, y se detiene con mensaje si falta la logistica; notebook reejecutado (kernel python3) y escribe con `ruta()`; requirements con scipy, ipykernel, nbconvert y version de Python. Resultados: TODO IGUAL (tablas 1-2, bootstrap, figuras byte a byte) salvo XGB AUC val 0.635 -> 0.638 (std 0.137), rango grilla 0.501-0.638, sensibilidad lr 0.597 vs 0.551, OOF marca 44.1% (antes 46.1%); mismos hiperparametros y umbral. Corrida de referencia de tiempos (8/11, los 5 modelos en una misma corrida, resultados identicos a la anterior): logit 0.62 s / 1.8 ms, KNN 0.51 s / 1.8 ms, SVM 1.18 s / 3.0 ms, RF 122.6 s / 15.0 ms, XGB 34.5 s / 1.6 ms (RF ha variado 84-130 s entre corridas). Informe: A2 declarado (nota "Uso de la prueba durante el desarrollo" en 3.4 + tabla de particion + 3.2), M1-M5 y B1-B3, B7 corregidos; fila de IA ampliada. Respaldos en `respaldos/` (informe y codigo/resultados antes de los hallazgos).
