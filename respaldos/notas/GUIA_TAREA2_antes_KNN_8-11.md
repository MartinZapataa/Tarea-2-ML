# Guía de la Tarea 2 AFML: modelos, resultados y dónde está cada cosa

Grupo: Lorenzo Godoy, Renato Torres y Martín Zapata · Profesor: Joaquín Aguirre · Entrega: martes 10/11/2026, antes de clase.
Estado al sábado 7/11/2026: cuatro de los cinco modelos están terminados. **Falta el KNN de Renato** (ver la sección 8).

---

## 1. Qué predecimos

- **Objetivo `caida10`:** vale 1 si algún cierre **diario** del S&P 500 en los 3 meses siguientes (≈63 días hábiles) queda al menos 10% bajo el cierre de fin del mes t. Es un drawdown, no un retorno punto a punto.
- **Frecuencia:** mensual (último día hábil), de ene-1990 a sep-2025. Son 429 meses con objetivo conocido, de los cuales 62 (14,5%) tienen caída.
- **Predictores (uno por hipótesis, signo esperado):**

| Variable | Qué es | Signo |
|---|---|---|
| `cape_lag3` | CAPE de Shiller rezagado 3 meses (se publica con atraso) | + |
| `mom_12m` | Retorno de 12 meses del S&P 500 | − |
| `vix_fin` | VIX a fin de mes | + |
| `baa10y_fin` | Spread Baa − Tesoro a 10 años | + |
| `t10y3m_fin` | Pendiente de la curva (10 años − 3 meses) | − |

`ret_min_3m` mira al futuro (con ella se construye la etiqueta): **nunca** se usa como predictor.

## 2. Mapa de archivos

Todo se corre **desde la carpeta del proyecto**.

| Archivo / carpeta | Qué contiene | Quién |
|---|---|---|
| `base_tarea2_caidas_sp500.csv` | Base mensual final, índice `fecha` (AAAA-MM) | Renato |
| `comun.py` | Funciones comunes a todos los modelos: puntaje, métricas, umbral, tiempos, guardado | Lolo |
| `tarea2.ipynb` | Celda 1: descarga del S&P 500 (`gspc_diario.csv`, que no se usa). Celda 2: **regresión logística**. Celda 3: importancia por permutación de la logística | Renato / Lolo |
| `svm_tarea2.py` | **SVM** con kernel RBF | Zapata |
| `rf_tarea2.py` | **Random Forest** | Lolo |
| `xgb_tarea2.py` | **XGBoost** | Lolo |
| `knn_tarea2.py` | **KNN**: todavía **no existe** | Renato |
| `comparativa.py` | Junta todos los modelos: Tablas 1 y 2, tiempos, bootstrap del AUC, importancia y Figuras 1–3 | Lolo |
| `resultados/` | Todos los CSV generados (detalle en la sección 6) | — |
| `figuras/` | `fig1_roc_prueba.png`, `fig2_matrices_confusion.png`, `fig3_alertas_prueba.png` | — |
| `salidas/` | Salida de consola de cada script en la corrida de referencia (`salida_<modelo>.txt`) | — |
| `Informe Tarea 2 Machine Learning.docx` | Informe (18 páginas) | Todos |
| `respaldos/` | Versiones anteriores del informe, del código y de los resultados | — |
| `requirements.txt` | Versiones exactas (Python 3.13.2) | — |
| `hallazgos_code_review.md`, `revision_hallazgos_6-11.md` | Revisión del 6/11 y qué se corrigió el 7/11 (al final de cada archivo) | — |
| `CLAUDE.md` | Plan, convenciones, resultados clave y **pendientes** | — |
| `Msgs.txt` | Recomendaciones metodológicas y reparto de tareas | — |

## 3. Diseño común (igual en los 5 modelos)

Las **secciones 1 y 2** de cada script (datos y partición) son copia exacta y **no se cambian** sin avisar al grupo.

- **Partición temporal, nunca al azar:**

| Tramo | Período | Meses | Caídas | % |
|---|---|---|---|---|
| Entrenamiento | 1990-01 a 2014-09 | 297 | 41 | 13,8% |
| Hueco (sin usar) | 2014-10 a 2014-12 | 3 | — | — |
| Prueba | 2015-01 a 2025-09 | 129 | 21 | 16,3% |

  El hueco existe porque la etiqueta mira 3 meses hacia adelante. La prueba tiene unos 6 episodios de caída (2015, 4T-2018, COVID 2020, 2022, ago–oct 2023 y abr 2025), así que **las métricas son ruidosas**.
- **Validación:** `TimeSeriesSplit(n_splits=5, gap=3)` dentro del entrenamiento, con `GridSearchCV(scoring="roc_auc")`. Son 5 vueltas de 49 meses de validación (1994-05 a 2014-09) con 1, 16, 3, 15 y 3 caídas: muy desiguales, y por eso el AUC de validación varía mucho entre vueltas.
- **Escalado:** `StandardScaler` dentro de un `Pipeline` en la logística, KNN y SVM. RF y XGB no lo necesitan.
- **Desbalance:**
  - `class_weight="balanced"` en la logística, SVM y RF.
  - En XGBoost, `scale_pos_weight = n_neg/n_pos`, que se **recalcula en cada vuelta** (clase `XGBBalanceado` en `xgb_tarea2.py`).
- **Métricas:** AUC (con el puntaje continuo), precisión, recall y F1 (con el umbral), en entrenamiento y prueba, más la brecha (entrenamiento − prueba). **Nunca accuracy.**
- **Umbral de decisión, por cuantil** (`comun.elegir_umbral`, en `comun.py`, sección 3):
  1. Con los hiperparámetros ya elegidos, se reentrena el modelo en cada vuelta y se obtienen puntajes out-of-fold de 245 meses (38 caídas).
  2. Cada puntaje se pasa a percentil respecto de los puntajes de entrenamiento de su vuelta.
  3. Se elige el percentil que maximiza F1.
  4. El umbral final es ese percentil de los puntajes de entrenamiento del modelo final.

  La prueba no se usa para nada de esto. **Limitaciones**, declaradas en el informe (3.4):
  - Solo dentro del entrenamiento fija la fracción de meses marcados; fuera de él esa fracción cambia mucho.
  - La curva de F1 es plana.
  - En la logística el AUC out-of-fold es 0,492, así que su umbral es casi arbitrario.
- **Tiempos:**
  - Entrenamiento: `time.time()` alrededor de `busqueda.fit` (búsqueda completa + reajuste final).
  - Predicción: `comun.medir_prediccion` (promedio de 20 repeticiones sobre los 129 meses de prueba).
- **Interpretabilidad:** importancia por permutación (caída del AUC, 30 repeticiones, `random_state=0`), calculada **en la prueba** con el modelo ya terminado. Es ruidosa y no se compara uno a uno con los coeficientes.
- **Semillas:** `random_state=42` en RF y XGB; 0 en la permutación y el bootstrap.

### Funciones de `comun.py`

| Función | Para qué |
|---|---|
| `puntaje(modelo, X)` | `predict_proba[:,1]` si existe; si no, `decision_function` (SVM) |
| `elegir_umbral(est, X_train, y_train, cv)` + `imprimir_umbral` | Umbral por cuantil out-of-fold, con diagnósticos por vuelta |
| `metricas(y, s, umbral)` / `tabla_train_test` / `matriz_confusion` | AUC, P, R y F1; tabla de entrenamiento, prueba y brecha; matriz de confusión |
| `medir_prediccion`, `n_combinaciones` | Tiempo de predicción y tamaño de la grilla |
| `sensibilidad(res, grilla)` | AUC de validación por valor de cada hiperparámetro |
| `guardar_extra`, `guardar_puntajes`, `ruta` | Guardan `resultados_<m>_extra.csv` y `puntajes_<m>.csv` en `resultados/` |

## 4. Los modelos, uno por uno

Los valores de prueba usan el umbral elegido out-of-fold. La matriz de confusión de prueba se lee como VN / FP / FN / VP.

### 4.1 Regresión logística (referencia): `tarea2.ipynb`, celdas 2 y 3
- **Idea:** puntaje lineal convertido en probabilidad. Es simple, estable e interpretable.
- **Grilla:** C ∈ {0,001; 0,01; 0,1; 1; 10; 100}. **Elegido: C = 0,001**, en el extremo de la grilla (convendría ampliarla hacia abajo). AUC de validación 0,533 ± 0,319.
- **Umbral:** 0,512 (percentil 0,80).
- **Resultados:**

| | AUC | Precisión | Recall | F1 |
|---|---|---|---|---|
| Entrenamiento | 0,773 | 0,383 | 0,561 | 0,455 |
| Prueba | 0,631 | 0,263 | 0,238 | 0,250 |

  Matriz de prueba: 94 / 14 / 16 / 5. Marca 19 meses.
- **Coeficientes:** con C = 0,001 los coeficientes de sklearn son casi nulos (0,006 a 0,055). Sin regularizar (statsmodels, variables estandarizadas):
  - CAPE +0,67 (p = 0,002) y spread Baa +0,81 (p = 0,015), significativos y con el signo esperado;
  - momentum −0,44 (p = 0,085);
  - VIX −0,26 (no significativo, signo contrario al esperado);
  - pendiente −0,06.
- **Permutación:** CAPE 0,039 ≈ VIX 0,038 > Baa 0,031.
- **Salidas:** `resultados/resultados_logistica*.csv`, `puntajes_logistica.csv`, `importancia_logistica.csv` y `salidas/salida_logistica.txt`.

### 4.2 KNN: **pendiente (Renato)**
- **Idea:** mira los k meses históricos más parecidos y usa la proporción de caídas entre ellos. Es sensible a la escala (va con `StandardScaler`) y a la dimensionalidad. Con k chico sobreajusta y con k grande subajusta.
- Ver la sección 8 para lo que tiene que guardar.

### 4.3 SVM (RBF): `svm_tarea2.py`
- **Idea:** busca la frontera de máximo margen; con kernel RBF puede curvarse. El puntaje es `decision_function`, no una probabilidad: no se usa `probability=True`, porque su calibración interna ignora el orden del tiempo.
- **Grilla:** C ∈ {0,01; 0,1; 1; 10; 100} × gamma ∈ {0,001; 0,01; 0,1; 1}. **Elegido: C = 0,1 y gamma = 0,1**, con AUC de validación 0,700, el mejor de todos.
- **Umbral:** −0,379 (percentil 0,62).
- **Resultados:**

| | AUC | Precisión | Recall | F1 |
|---|---|---|---|---|
| Entrenamiento | 0,824 | 0,283 | 0,780 | 0,416 |
| Prueba | 0,625 | 0,198 | 0,857 | 0,321 |

  Matriz de prueba: 35 / 73 / 3 / 18. **Marca 91 de 129 meses**, y desde 2021-11 avisa 47 meses seguidos.
- **Diagnóstico:**
  - Es la más sensible a los hiperparámetros: con C alto el AUC de validación baja a 0,43–0,57.
  - 236 de 297 meses (79%) son vectores de soporte, así que las clases casi no se separan.
- **Permutación:** CAPE 0,081 > Baa 0,052 > momentum 0,047 > VIX 0,013.
- **Salidas:** `resultados/resultados_svm*.csv`, `puntajes_svm.csv`, `importancia_svm.csv` y `salidas/salida_svm.txt`.

### 4.4 Random Forest: `rf_tarea2.py`
- **Idea:** promedia muchos árboles entrenados con muestras y variables al azar.
- **Grilla:** n_estimators {200, 500} × max_depth {2, 3, 5, None} × min_samples_leaf {5, 10, 20} × max_features {sqrt, None}, en total 48 combinaciones.
- **Elegido:** 200 árboles, profundidad 2, hoja mínima 5 y `sqrt`. AUC de validación 0,676 ± 0,197.
- **Umbral:** 0,402 (percentil 0,54). Marca 80% de los meses out-of-fold.
- **Resultados:**

| | AUC | Precisión | Recall | F1 |
|---|---|---|---|---|
| Entrenamiento | 0,892 | 0,292 | 0,976 | 0,449 |
| Prueba | 0,568 | 0,208 | 0,524 | 0,297 |

  Matriz de prueba: 66 / 42 / 10 / 11.
- **Sensibilidad:** pesan `max_features` (sqrt 0,62 vs. None 0,55) y la hoja mínima (5: 0,61 vs. 20: 0,53). El número de árboles y la profundidad casi no influyen.
- **Permutación:** VIX 0,060 (± 0,056); el resto ≈ 0.
- **Salidas:** `resultados/resultados_rf*.csv`, `sensibilidad_rf.csv`, `puntajes_rf.csv`, `importancia_rf.csv` (incluye la importancia por impureza, MDI) y `salidas/salida_rf.txt`.

### 4.5 XGBoost: `xgb_tarea2.py`
- **Idea:** encadena árboles chicos, cada uno corrige al anterior. Tiene muchos hiperparámetros y riesgo de sobreajuste con pocos datos.
- **Grilla:** learning_rate {0,01; 0,1} × n_estimators {100, 300} × max_depth {2, 3} × subsample {0,7; 1,0} × reg_lambda {1, 10}, en total 32 combinaciones.
- **Desbalance:** `scale_pos_weight` se recalcula por vuelta (15,3 / 31,7 / 7,6 / 7,5 / 5,4 en las vueltas y 6,24 en el modelo final).
- **Elegido:** learning_rate 0,01, 300 árboles, profundidad 2, subsample 0,7 y λ = 10. AUC de validación 0,638 ± 0,137.
- **Umbral:** 0,503 (percentil 0,74).
- **Resultados:**

| | AUC | Precisión | Recall | F1 |
|---|---|---|---|---|
| Entrenamiento | 0,930 | 0,487 | 0,902 | 0,632 |
| Prueba | 0,558 | 0,163 | 0,333 | 0,219 |

  Matriz de prueba: 72 / 36 / 14 / 7. Tiene **la mayor brecha** de todos.
- **Sensibilidad:** baja; toda la grilla queda entre 0,50 y 0,64. Lo que más pesa es la tasa de aprendizaje (0,60 vs. 0,55).
- **Permutación:** VIX 0,054 > Baa 0,026 > CAPE 0,014.
- **Salidas:** `resultados/resultados_xgb*.csv`, `sensibilidad_xgb.csv`, `puntajes_xgb.csv`, `importancia_xgb.csv` (incluye la importancia por ganancia) y `salidas/salida_xgb.txt`.

## 5. Comparación (sin KNN): `comparativa.py`

| Modelo | AUC val. | AUC prueba | F1 prueba | Brecha AUC | Meses marcados | Entrenam. (s) | Predicción (ms) |
|---|---|---|---|---|---|---|---|
| Logística | 0,533 | **0,631** | 0,250 | **0,142** | 19 | 0,44 | 0,8 |
| SVM | **0,700** | 0,625 | **0,321** | 0,199 | 91 | 0,96 | 5,7 |
| Random Forest | 0,676 | 0,568 | 0,297 | 0,324 | 53 | 123,7 | 16,6 |
| XGBoost | 0,638 | 0,558 | 0,219 | 0,373 | 43 | 36,9 | 1,6 |

Mensajes principales (secciones 4.1 a 4.4 del informe):
1. **Ningún modelo supera a la logística.** Con un bootstrap circular por bloques (12 meses, 5.000 repeticiones), todas las diferencias de AUC contra la logística contienen el 0: SVM [−0,09; +0,08], RF [−0,16; +0,04] y XGB [−0,19; +0,07]. **Todos** los intervalos del AUC de prueba incluyen 0,5. Ver `resultados/tabla_bootstrap_auc.csv`.
2. **El ranking de validación se invierte en prueba:** la validación no fue buena guía para elegir entre modelos.
3. **El sobreajuste crece con la flexibilidad del modelo:** la brecha de AUC va de 0,14 en la logística a 0,37 en XGB.
4. **Alerta binaria:** avisar todos los meses da F1 0,280. Solo la SVM y RF la superan, y por poco.
5. **Alertas tardías (Figura 3):**
   - En la logística, RF y XGB, la mayoría de las falsas alarmas llegan en los 6 meses siguientes a una caída (13 de 14, 29 de 42 y 26 de 36): reaccionan a VIX y spreads altos cuando la caída ya pasó.
   - En la SVM no ocurre (40 de 73), porque avisa casi siempre.
6. **Tiempo:** no decide entre modelos (reentrenar una vez al mes es trivial). Los tiempos varían entre corridas: RF tardó entre 84 y 130 s.
7. **Interpretabilidad:**
   - La logística es la única con coeficientes.
   - Por permutación, la SVM prioriza CAPE y spread, y RF y XGB el VIX.
   - La pendiente de la curva es la última en todos los modelos.

Archivos de la comparación:
- Tablas: `resultados/tabla1_prueba.csv`, `tabla2_brechas.csv`, `tabla_tiempos.csv`, `tabla_bootstrap_auc.csv` y `tabla_importancia.csv`.
- Figuras en `figuras/`.
- Consola en `salidas/salida_comparativa.txt`.

## 6. Qué guarda cada modelo (formato que espera `comparativa.py`)

| Archivo en `resultados/` | Contenido | Cómo se genera |
|---|---|---|
| `resultados_<m>.csv` | Filas AUC, Precision, Recall y F1 × columnas Entrenamiento, Prueba y Brecha (train - test) | `tabla_train_test(...).to_csv(ruta("resultados", ...))` |
| `resultados_<m>_extra.csv` | Hiperparámetros, AUC_validacion, umbral, percentil_umbral, tiempo_s, tiempo_prediccion_s y n_combinaciones | `guardar_extra(...)` |
| `puntajes_<m>.csv` | fecha, y, puntaje y pred de los 129 meses de prueba | `guardar_puntajes(...)` |
| `importancia_<m>.csv` | Índice `variable`, columnas `"Caida del AUC"` y `"Desv. est."` | `imp_df.to_csv(..., index_label="variable")` |

`comparativa.py` comprueba que las fechas y la `y` de prueba sean iguales a las de la logística, que `pred` = (puntaje ≥ umbral) y que las métricas guardadas coincidan con `comun.metricas`. Si algo no calza, se detiene.

## 7. Cómo correr todo de cero

```bash
pip install -r requirements.txt
python -m nbconvert --to notebook --execute --inplace tarea2.ipynb   # logistica (primero: es la referencia)
python svm_tarea2.py
python rf_tarea2.py      # ~1,5-2 min
python xgb_tarea2.py
python knn_tarea2.py     # cuando exista
python comparativa.py
```

Todo se reproduce exactamente, salvo los tiempos. Para guardar la consola: `python rf_tarea2.py > salidas/salida_rf.txt`.

## 8. Lo que falta para el KNN (Renato)

1. Crear `knn_tarea2.py` copiando la estructura de `svm_tarea2.py`:
   - secciones 1 y 2 idénticas;
   - `Pipeline([("escalar", StandardScaler()), ("knn", KNeighborsClassifier())])`;
   - grilla de `knn__n_neighbors`, `knn__weights` y `knn__metric`, con el mismo `cv` y `scoring="roc_auc"`;
   - `time.time()` alrededor de `busqueda.fit`.
2. Umbral y métricas con `comun.py`: `elegir_umbral`, `imprimir_umbral`, `puntaje`, `metricas`, `tabla_train_test`, `matriz_confusion` y `medir_prediccion`.
3. Guardar con `guardar_extra("knn", {...}, busqueda, umbral, info, tiempo, t_pred)` y `guardar_puntajes("knn", idx_test, y_test, s_test, umbral)`, más `importancia_knn.csv` (permutación en prueba, `n_repeats=30`, `random_state=0`).
4. Correr `python comparativa.py`; KNN entra solo en las tablas y figuras.
5. En el informe:
   - llenar los `[ ]` y `[KNN: ...]`: Tablas 1 y 2, tabla de tiempos, fila de 4.3, Anexo A y Anexo B (fila `[knn_tarea2]`);
   - reinsertar las Figuras 1 a 3;
   - cambiar "cuatro modelos" por "cinco" en 4.3 y 4.4;
   - agregar KNN a la Tabla C1 y revisar el texto de 4.1, 4.2 y 4.4.

## 9. Dónde está cada cosa en el informe

| Sección | Contenido | Estado |
|---|---|---|
| Resumen, 1, 2 | Motivación y datos | Redactado (Zapata). Faltan Resumen, 2.1 [Fórmula] y 2.5 |
| 3.1–3.4 | Partición, validación, modelos, métricas y umbral (con limitaciones y uso de la prueba) | Listo (Claude, revisar) |
| 4.1–4.4 | Tablas, figuras, ventajas y problemas, cuatro criterios | Listo salvo KNN |
| 4.5, 5 | Interpretación económica y conclusión | Pendiente (Renato) |
| 6 | Declaración de uso de IA | Fila de Claude en borrador; falta el uso de cada integrante |
| Anexos A–C | Grillas, reproducibilidad, importancia y bootstrap | Listos salvo KNN, código de la base y correlaciones |
