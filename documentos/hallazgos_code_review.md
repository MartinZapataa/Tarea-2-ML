# Hallazgos de la revisión de código

Archivos revisados: `comun.py`, `rf_tarea2.py`, `xgb_tarea2.py`, `svm_tarea2.py`, `comparativa.py` (nivel de esfuerzo: high).
No se modificó ningún archivo. Ordenados de más a menos grave.

## 1. Percentil OOF contra puntajes de entrenamiento en muestra (`comun.py:185`)
Cada puntaje out-of-fold se convierte a percentil respecto de los puntajes de entrenamiento en muestra del mismo modelo, y ese percentil se traslada a los puntajes en muestra del modelo final (línea 207). La validación se compara con una referencia sobreajustada.
- **Escenario:** en RF y XGB los puntajes en muestra son más extremos que los de validación. El RF marca 80% de los meses OOF pero 46% en entrenamiento; la SVM marca 91 de 129 meses de prueba. El corte max-F1 no se traduce en una tasa de alertas estable.
- **Arreglo posible:** juntar los puntajes OOF de todas las vueltas como distribución de referencia, o aplicar el cuantil sobre la propia validación.

## 2. Borde p = 0 en `_umbral_desde_percentil` (`comun.py:143`)
Con p = 0 devuelve `ref[0]` (mínimo de entrenamiento), así que `s >= umbral` no equivale a `percentil(s) >= p` para puntajes menores que ese mínimo.
- **Escenario:** la logística tiene percentil 0 en las vueltas 1 y 2. Si el corte global fuera p = 0, un mes de prueba con puntaje menor que `min(s_train)` recibe pred = 0 aunque el criterio decía "marcar todo".

## 3. Empates en `_corte_max_f1` (`comun.py:126`)
Elige el primer máximo de F1 (el corte más bajo). Con vueltas chicas y pocas caídas, la curva F1 tiene muchos empates o casi empates y el corte lo decide el ruido.
- **Escenario:** en OOF hay solo 38 caídas y la vuelta 1 tiene 1. El RF queda marcando 80% de los meses OOF con recall 0,97, casi "avisar siempre". No hay desempate por estabilidad ni tope a la fracción marcada.

## 4. `scale_pos_weight` fijo con todo `y_train` (`xgb_tarea2.py:51`)
Se calcula una vez con todo `y_train` y se reutiliza en cada vuelta de `GridSearchCV` y de `elegir_umbral`. La proporción de clases de los pliegues futuros influye en el entrenamiento de cada vuelta.
- **Escenario:** en la vuelta 1 (49 meses de entrenamiento) el peso sale de 256/41 y no de lo disponible en ese momento. Fuga leve de la distribución de etiquetas; cumple "solo con el entrenamiento" a nivel global, no por vuelta.

## 5. Permutación calculada sobre la prueba (`rf_tarea2.py:123`; igual en `xgb_tarea2.py:128` y `svm_tarea2.py:125`)
La prueba debe usarse una sola vez al final, y con 21 caídas el resultado es ruidoso.
- **Escenario:** `vix_fin` 0,060 con el resto en ~0 y desviaciones del mismo orden. El informe compara estos valores con coeficientes de la logística estimados en entrenamiento, que no son comparables.
- **Arreglo posible:** calcularla en entrenamiento/validación, o declarar explícitamente que es importancia en prueba.

## 6. Hiperparámetros `None` guardados como vacío (`rf_tarea2.py:136`)
`max_depth=None` y `max_features=None` quedan como celdas vacías en el CSV y se leen como NaN.
- **Escenario:** `comparativa.py` o el Anexo A mostrarían "nan". Guardar `str(mejor[h])` o leer con `keep_default_na=False`.

## 7. `puntaje()` oculta `AttributeError` (`comun.py:63`)
El `except AttributeError` alrededor de `predict_proba` cae en silencio a `decision_function`.
- **Escenario:** un Pipeline que falla al predecir probabilidades devuelve puntajes en otra escala sin avisar. Usar `hasattr(modelo, "predict_proba")`.

## 8. Código duplicado y `numpy` sin usar (`rf_tarea2.py:82`)
El bucle de sensibilidad, el guardado de `_extra.csv` y la impresión del mejor resultado están copiados en RF y XGB (el guardado también en SVM). `import numpy as np` sin usar en RF, XGB y SVM.
- **Escenario:** cualquier cambio hay que repetirlo en 3 scripts. Además hoy el orden de valores es textual (`astype(str)`: "10" < "5"). Mover `sensibilidad()` y `guardar_extra()` a `comun.py`.

## 9. Carpetas creadas al importar (`comun.py:46`)
`comun.py` crea `resultados/`, `figuras/` y `salidas/` en el directorio de trabajo al importarse.
- **Escenario:** desde otro cwd se crean carpetas vacías y los CSV caen en el lugar equivocado; `comparativa.py` falla al leer `resultados/`. Usar `Path(__file__).parent` y crear las carpetas dentro de una función.

## 10. Matriz de confusión y AUC reimplementados (`comparativa.py:280`)
La Figura 2 rehace la matriz con un doble bucle aunque `comun.matriz_confusion` ya existe, y `comparativa.py` repite sus propias funciones de AUC en vez de `comun.metricas`.
- **Escenario:** si cambia la regla de decisión (p. ej. `s >= umbral` a `s > umbral`), la figura dejaría de coincidir con las tablas sin que nada lo detecte.

---

## Impacto en el informe
Los puntos 1 a 3 cambiarían el umbral y, por lo tanto, las Tablas 1 y 2 y el texto de 3.4 y 4.x. Si no se arreglan, declararlos como limitación. El punto 4 conviene corregirlo o mencionarlo.

---

## Estado al 7/11 (implementado por Claude Code a pedido de Lolo)
1. Se mantiene el criterio; se corrigió la descripción (comun.py y §3.4) y se declaró como limitación en el informe (§3.4, "Limitaciones del criterio"). 2. Arreglado: p ≤ 0 → umbral −∞. 3. Se mantiene el primer máximo; `imprimir_umbral` ahora muestra el rango de percentiles con F1 casi máximo y la limitación quedó declarada. 4. Arreglado: `XGBBalanceado` recalcula `scale_pos_weight` en cada fit (mismos hiperparámetros y umbral; AUC val 0,635 → 0,638). 5. Se mantiene en prueba y se declara en el código y en §4.4. 6. Arreglado (`"None"` + `keep_default_na=False`). 7. Arreglado (solo `hasattr`). 8. Arreglado (`sensibilidad()` y `guardar_extra()` en comun.py, en orden de grilla; sin `numpy` sin usar). 9. Arreglado (`ruta()` relativa a comun.py; carpetas creadas al guardar). 10. Arreglado (`comparativa.py` usa `comun.matriz_confusion`/`comun.metricas` y verifica pred y métricas contra el umbral).
