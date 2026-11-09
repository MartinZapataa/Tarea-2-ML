# Revisión del trabajo del 6/11 (solo lectura)

Revisión hecha por Claude Code (Opus 5.5) a pedido de Lolo, sin modificar archivos del proyecto.

**Resumen:** no hay filtración de la prueba en el código y todo se reproduce. Se corrió desde cero en una copia temporal: `tarea2.ipynb` (celdas 1–2 como script), SVM, XGB, RF y `comparativa.py`. Los 23 CSV salen idénticos (diferencia ≤1e-9) salvo los tiempos, y las 3 figuras coinciden byte a byte con las que tiene el `.docx`. Lo que hay que arreglar está en la documentación: el notebook, la tabla de tiempos, varias afirmaciones del informe y lo que se dice sobre cuántas veces se usó la prueba.

En el `.docx`, "P###" es el índice de párrafo en orden del documento (python-docx), porque Word no tiene líneas.

---

## 🔴 Alta

### A1. El notebook entregado contradice el informe
`tarea2.ipynb`, celda 1:
- Sus salidas guardadas son las de antes del 6/11, con umbral 0,5: F1 de prueba 0,308 y matriz 85/23/13/8. El informe dice 0,250 y 94/14/16/5.
- Las celdas 1 y 2 tienen `execution_count=None` y la celda 2 no tiene salida.
- El kernel declarado es `base` (conda). `requirements.txt` no trae `jupyter` ni `ipykernel`, así que el paso "correr las celdas" del Anexo B no se puede cumplir solo con `pip install -r requirements.txt`.

### A2. La prueba se miró más de una vez durante el desarrollo, y el informe dice que no
El Bloque 1 (`CLAUDE.md:250-251`) registra las métricas de prueba de dos criterios de umbral (máx. F1 OOF crudo y por cuantil) antes de elegir el segundo. Esto choca con:
- la Tabla de partición (§3.1): "Evaluar los modelos terminados, una sola vez";
- P082: "La muestra de prueba no se usa para ajustar nada".

Atenuantes que conviene declarar:
- la elección se justificó con un diagnóstico OOF (AUC OOF de la SVM 0,531 → 0,649), no con la prueba;
- el cambio **bajó** el F1 de prueba de ambos modelos (logística 0,267 → 0,250; SVM 0,339 → 0,321).

En el código, cada script calcula las métricas de prueba una sola vez y nada se elige con la prueba.

---

## 🟠 Media

### M1. La tabla de tiempos (§4.4, Tabla 8) y P108 no coinciden con los CSV
Los scripts se volvieron a correr a las 23:34–23:36, después de guardar el `.docx` (23:29).

| | Informe | `resultados/tabla_tiempos.csv` |
|---|---|---|
| SVM | 0,76 s / 2,0 ms | 0,77 s / 2,5 ms |
| RF | 85,10 s / 10,6 ms | 84,48 s / 10,5 ms |
| XGB | 23,77 s ("24 s" en P108) | 24,56 s |

La corrida de verificación dio otros valores (0,29 / 0,70 / 86,1 / 24,0 s). Los tiempos nunca calzan exacto entre corridas: hay que fijar una corrida de referencia.

### M2. Afirmación falsa sobre la importancia de variables en la logística
P109 (§4.4) dice que la logística y la SVM coinciden en que el CAPE y el spread son lo más importante. En la logística, por permutación, el orden es CAPE 0,039 ≈ **VIX 0,038** > Baa 0,031 (`resultados/importancia_logistica.csv`). El mismo error está en `CLAUDE.md:257`.

### M3. Todos los IC del AUC de prueba incluyen 0,5, también el de la logística [0,478; 0,763]
El informe da el intervalo, pero:
- P093 dice que RF y XGB "quedan más cerca del azar", lo que sugiere que la logística y la SVM no lo están; en realidad ningún modelo se distingue del azar.
- En la Tabla 4.3, "Tiene el mejor AUC en prueba" (logística) y "el AUC y el F1 de prueba más bajos" (XGB) aparecen sin la advertencia de ruido.
- P105 atribuye "0,18 y 0,23" al sobreajuste con un AUC de prueba de ±0,15. Además, el AUC OOF de la logística dentro del entrenamiento es 0,492, así que su brecha de 0,14 no es solo cambio de régimen.

Lo que sí está bien: las diferencias contra la logística se presentan como ruido en P093, y P094 compara bien contra "avisar siempre".

### M4. P088 y `comun.py:118` describen mal el criterio de umbral
Dicen que el criterio "decide qué fracción de los meses marcar". En realidad fija la fracción **dentro del entrenamiento**; fuera de muestra cambia mucho:

| | OOF | Entrenamiento | Prueba |
|---|---|---|---|
| Logística | 13,5% | 20,2% | 14,7% |
| SVM | 48,6% | 38,0% | 70,5% |
| RF | 80,0% | 46,1% | 41,1% |
| XGB | 46,1% | 25,6% | 33,3% |

El método no filtra la prueba, pero:
- el percentil se calcula contra puntajes **dentro de muestra** (`comun.py:185`, `comun.py:207`), con vueltas de 49 a 245 meses frente a 297 del modelo final, así que no son estrictamente comparables;
- 31 de las 38 caídas OOF vienen de las vueltas 2 y 4 (1998–2002 y 2006–2010): el umbral depende sobre todo de esos dos episodios;
- en la logística el AUC OOF es 0,492, así que su umbral es prácticamente arbitrario.

Nada de esto está declarado en el informe.

### M5. Las "alertas tardías" (P100, fig3) están respaldadas solo en parte
Falsos positivos clasificados según si caen hasta 6 meses después de un episodio de `caida10` = 1 (criterio de la revisión):

| Modelo | FP después del episodio | FP antes | Otros |
|---|---|---|---|
| Logística | 13 de 14 | 0 | 1 |
| RF | 29 de 42 | 10 | 3 |
| XGB | 26 de 36 | 7 | 3 |
| SVM | 40 de 73 | 20 | 13 |

- En la logística, RF y XGB se cumple. Los ejemplos de P100 (ago–sep 2015, dic 2018, mar–jun 2020) están bien, y también lo de los 5 aciertos de la logística (4 en 2022 más feb 2020).
- En la SVM no se cumple: solo 40 de sus 91 alertas son tardías, porque avisa **todos** los meses desde 2021-11 (47 seguidos).
- La frase general "la mayoría de las alertas llegan después" hay que acotarla a los modelos donde vale.

---

## 🟡 Baja

- **B1.** Tabla 4.3, fila logística: los coeficientes van de 0,006 a 0,055, no de "0,03 a 0,06" (`t10y3m_fin` = −0,006).
- **B2.** P106 dice que "todos los modelos prefieren las opciones más regularizadas". Pero RF eligió `min_samples_leaf=5` (la menos regularizada), XGB `n_estimators=300` y la SVM gamma 0,1 (no el mínimo).
- **B3.** P108 dice "computador con un solo núcleo". En realidad es `n_jobs=1` en RF y XGB; los hilos de BLAS en la logística y la SVM no están controlados.
- **B4.** `xgb_tarea2.py:51-54`: `scale_pos_weight` = 6,24 se calcula solo con `y_train` (cumple la convención), pero queda fijo dentro de las vueltas del CV; en la primera vuelta la proporción real es 46/3 = 15,3. Filtración menor dentro del entrenamiento; no toca la prueba.
- **B5.** `comun.py:142-143`: si el percentil de corte fuera 0, el umbral sería el mínimo del entrenamiento y no −∞. No ocurre en ningún modelo hoy.
- **B6.** A `requirements.txt` le faltan `scipy` (se importa directo en `comparativa.py:27`; instalado 1.18.1 como dependencia), `jupyter` / `ipykernel` y la versión de Python (3.13.2).
- **B7.** Al Anexo B (P141) le falta decir que las salidas van a `resultados/`, `figuras/` y `salidas/`, que hay que correr desde la raíz y que sin `resultados_logistica*` `comparativa.py` se cae (`comparativa.py:87`). Además, la celda 0 genera `gspc_diario.csv`, que ningún código usa y que no está en la carpeta.
- **B8.** `CLAUDE.md` desactualizado: `:55` sigue mostrando la tabla de la logística con umbral 0,5 sin aclarar que se reemplazó; `:256-257` dice SVM 2,0 ms, logística 0,7 ms, RF 85 s / 10,6 ms y XGB 23,8 s.

---

## ✅ Verificado sin problemas

- **`elegir_umbral`** usa solo `X_train`/`y_train`, con los mismos splits que `GridSearchCV`. La traducción del percentil al modelo final es exacta.
- **Secciones 1–2:** idénticas en los 4 modelos.
- **Partición:** 297/41 (13,8%) y 129/21 (16,3%), con hueco oct–dic 2014. Vueltas de 49 meses con 1/16/3/15/3 caídas, validación de 1994-05 a 2014-09.
- **Semillas:** 42 en RF y XGB, 0 en la permutación y el bootstrap.
- **`ret_min_3m`** nunca entra como predictor.
- **Métricas recalculadas desde `puntajes_*.csv`:** coinciden (diferencia ≤1e-16), con matrices de confusión correctas.
- **Cifras del informe:** Tablas 1 y 2, C1, C2, Anexo A, bootstrap, sensibilidad, rangos de grilla y la referencia "avisar siempre" (P 0,163, F1 0,280) coinciden con los CSV.

---

## Pendientes que pueden romper la entrega

1. **KNN.** Debe guardar los 3 archivos con el mismo índice y la misma `y` (si no, fallan los `assert` de `comparativa.py:89-91`) y `importancia_knn.csv` con la columna `"Caida del AUC"` (`comparativa.py:224`). Después: reinsertar las figuras 2 y 3 y cambiar "cuatro modelos" a cinco en P103 y P109.
2. **Volver a correr el notebook en Jupyter** (ver A1).
3. **Falta el código de construcción de la base:** sin él la base no se puede reproducir (fila pendiente del Anexo B).
4. **Placeholders abiertos:** Resumen, Fórmula 2.1, §2.5, §4.5, §5, fechas de Referencias, correlaciones del Anexo C, borrador de IA en §6 y la fila "[Otra, si corresponde]".
5. **Problemas que ya venían en el borrador original:**
   - la fórmula de §3.3 (P085) está como texto LaTeX crudo y duplicado, sin ecuación de Word;
   - P066–P067 son texto normal con estilo Heading 3, y "1.4 Hipótesis" tiene estilo Normal;
   - a P053 le falta el punto final;
   - P070 dice "umbral de KNN" (debería ser "umbral de decisión").
6. **Avisar al grupo** del umbral por cuantil y de la celda nueva del notebook.

---

## No verificable

- Si la base tiene sesgo de anticipación (look-ahead): el código de Renato no está en la carpeta.
- La ejecución en el kernel `base` de conda: la verificación corrió las celdas como script con Python 3.13 del sistema.
- Cómo se ven el `.docx` y las figuras en Word: no se renderizó.
- Que la prueba se haya mirado solo en los momentos registrados en `CLAUDE.md`: se infiere de esas notas.

---

## Estado al 7/11 (implementado por Claude Code a pedido de Lolo)
- **A1:** notebook reejecutado completo (nbconvert, kernel `python3`); salidas = informe. `requirements.txt` con ipykernel, nbconvert, scipy y la versión de Python (B6).
- **A2:** declarado en §3.4 ("Uso de la prueba durante el desarrollo"), en la tabla de partición y en §3.2.
- **M1:** nueva corrida de referencia (7/11) y Tabla 8 + texto alineados con `resultados/tabla_tiempos.csv`; se declara la variación entre corridas. **B3** corregido en el mismo párrafo.
- **M2, M3, M4, M5, B1, B2, B7:** corregidos en el informe (cifras de M5 verificadas de nuevo). **B4/B5:** arreglados en el código. **B8:** CLAUDE.md actualizado.
- Pendientes 5 (borrador original): fórmula de §3.3 como ecuación de Word, estilos de 1.4 y 2.4, punto final de P053 y "umbral de decisión" en P070, corregidos. Siguen abiertos: KNN, código de la base, placeholders de otros integrantes y avisar al grupo.
