"""
Tarea 2 AFML - Comparativa de los 5 modelos
(logistica, KNN, SVM, Random Forest, XGBoost)

Lee lo que guardo cada script de modelo:
  resultados_<m>.csv        metricas train / prueba / brecha
  resultados_<m>_extra.csv  hiperparametros, AUC de validacion, umbral, tiempos
  puntajes_<m>.csv          puntaje de prueba de cada mes (para ROC y bootstrap)
  importancia_<m>.csv       importancia por permutacion (opcional)

Un modelo entra solo si existen sus tres primeros archivos (por ejemplo, KNN
queda fuera hasta que Renato lo corra). No reentrena nada ni elige nada con
la prueba: solo resume.

Genera:
  tabla1_prueba.csv, tabla2_brechas.csv, tabla_tiempos.csv,
  tabla_bootstrap_auc.csv, tabla_importancia.csv,
  fig1_roc_prueba.png, fig2_matrices_confusion.png, fig3_alertas_prueba.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # sin ventana: solo guarda los PNG
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import rankdata, norm
from sklearn.metrics import roc_curve

# ---------------------------------------------------------------
# 0. Modelos (orden fijo) y estilo de los graficos
#    El color sigue al modelo, nunca a su posicion: si falta KNN,
#    los demas conservan su color.
# ---------------------------------------------------------------
MODELOS = {                    # clave de archivo -> (nombre, color)
    "logistica": ("Logistica", "#2a78d6"),
    "knn":       ("KNN",       "#eb6834"),
    "svm":       ("SVM (RBF)", "#1baf7a"),
    "rf":        ("Random Forest", "#eda100"),
    "xgb":       ("XGBoost",   "#e87ba4"),
}
REFERENCIA = "logistica"
TINTA, TINTA_2, TENUE = "#0b0b0b", "#52514e", "#898781"
GRILLA, EJE = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "font.size": 10, "text.color": TINTA, "axes.labelcolor": TINTA_2,
    "axes.edgecolor": EJE, "axes.linewidth": 0.8,
    "xtick.color": TENUE, "ytick.color": TENUE,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRILLA, "grid.linewidth": 0.6,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 200, "savefig.bbox": "tight",
})

SEMILLA = 0
B = 5000          # repeticiones del bootstrap
BLOQUE = 12       # largo del bloque (meses). Se reporta tambien 6 y 24 como sensibilidad.

pd.set_option("display.width", 160)


# ---------------------------------------------------------------
# 1. Leer lo que haya
# ---------------------------------------------------------------
def leer(m):
    archivos = [f"resultados/resultados_{m}.csv", f"resultados/resultados_{m}_extra.csv", f"resultados/puntajes_{m}.csv"]
    if not all(os.path.exists(a) for a in archivos):
        return None
    return {
        "tabla": pd.read_csv(archivos[0], index_col=0),
        "extra": pd.read_csv(archivos[1]).iloc[0],
        "punt": pd.read_csv(archivos[2], index_col="fecha"),
        "imp": (pd.read_csv(f"resultados/importancia_{m}.csv", index_col="variable")
                if os.path.exists(f"resultados/importancia_{m}.csv") else None),
    }


datos = {m: d for m in MODELOS if (d := leer(m)) is not None}
faltan = [MODELOS[m][0] for m in MODELOS if m not in datos]
print(f"Modelos incluidos: {', '.join(MODELOS[m][0] for m in datos)}")
if faltan:
    print(f"Sin resultados (quedan fuera): {', '.join(faltan)}")

# Todos deben tener exactamente la misma muestra de prueba
ref_idx = datos[REFERENCIA]["punt"].index
ref_y = datos[REFERENCIA]["punt"]["y"]
for m, d in datos.items():
    assert d["punt"].index.equals(ref_idx), f"{m}: meses de prueba distintos"
    assert (d["punt"]["y"] == ref_y).all(), f"{m}: objetivo de prueba distinto"
y = ref_y.values.astype(int)
print(f"Prueba: {ref_idx[0]} a {ref_idx[-1]} ({len(y)} meses, {y.sum()} caidas)\n")

nombre = {m: MODELOS[m][0] for m in datos}

# ---------------------------------------------------------------
# 2. Tablas 1 (prueba), 2 (brechas) y de tiempos
# ---------------------------------------------------------------
METRICAS = ["AUC", "Precision", "Recall", "F1"]

t1 = pd.DataFrame({nombre[m]: d["tabla"].loc[METRICAS, "Prueba"] for m, d in datos.items()}).T
t1["AUC validacion"] = [d["extra"]["AUC_validacion"] for d in datos.values()]
t1["Meses marcados"] = [int(d["punt"]["pred"].sum()) for d in datos.values()]
t1["VP"] = [int(((d["punt"]["pred"] == 1) & (d["punt"]["y"] == 1)).sum()) for d in datos.values()]
print(f"Tabla 1. Metricas en prueba (umbral elegido out-of-fold; {len(y)} meses, {y.sum()} caidas)")
print(t1.round(3).to_string(), "\n")

t2 = pd.DataFrame({nombre[m]: d["tabla"].loc[METRICAS, "Brecha (train - test)"]
                   for m, d in datos.items()}).T
t2.columns = [f"Brecha {c}" for c in t2.columns]
t2.insert(0, "AUC train", [d["tabla"].loc["AUC", "Entrenamiento"] for d in datos.values()])
print("Tabla 2. Brechas entrenamiento - prueba (positivo = peor en prueba)")
print(t2.round(3).to_string(), "\n")

tt = pd.DataFrame({
    "Entrenamiento (s)": [d["extra"]["tiempo_s"] for d in datos.values()],
    "Prediccion (ms)": [d["extra"]["tiempo_prediccion_s"] * 1000 for d in datos.values()],
    "Combinaciones": [int(d["extra"]["n_combinaciones"]) for d in datos.values()],
}, index=list(nombre.values()))
# Ajustes = combinaciones x 5 vueltas + 1 reajuste final
tt["Ajustes"] = tt["Combinaciones"] * 5 + 1
tt["s por ajuste"] = tt["Entrenamiento (s)"] / tt["Ajustes"]
print("Tabla de tiempos (entrenamiento = busqueda completa + reajuste final; "
      "prediccion = puntaje de los 129 meses de prueba)")
print(tt.round(4).to_string(), "\n")

t1.to_csv("resultados/tabla1_prueba.csv")
t2.to_csv("resultados/tabla2_brechas.csv")
tt.to_csv("resultados/tabla_tiempos.csv")

# ---------------------------------------------------------------
# 3. Ruido: bootstrap por bloques del AUC de prueba
#    Las observaciones mensuales estan autocorrelacionadas (la etiqueta mira
#    3 meses adelante y las caidas vienen en episodios), asi que no se puede
#    remuestrear mes a mes. Se usa bootstrap circular por bloques: se pegan
#    bloques de BLOQUE meses consecutivos. Los mismos indices sirven para
#    todos los modelos (comparacion pareada contra la logistica).
# ---------------------------------------------------------------
def auc_rapido(y, s):
    """AUC por rangos (Mann-Whitney), equivalente a roc_auc_score."""
    n1 = y.sum()
    n0 = len(y) - n1
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def indices_bloques(n, largo, rng):
    """Indices de una muestra bootstrap circular por bloques."""
    k = int(np.ceil(n / largo))
    inicios = rng.integers(0, n, size=k)
    return (inicios[:, None] + np.arange(largo)[None, :]).ravel()[:n] % n


def bootstrap_auc(puntajes, y, largo, B, semilla):
    rng = np.random.default_rng(semilla)
    n = len(y)
    out = {m: [] for m in puntajes}
    validas = 0
    while validas < B:
        i = indices_bloques(n, largo, rng)
        yb = y[i]
        if yb.sum() == 0 or yb.sum() == n:   # sin las dos clases no hay AUC
            continue
        for m, s in puntajes.items():
            out[m].append(auc_rapido(yb, s[i]))
        validas += 1
    return {m: np.array(v) for m, v in out.items()}


def delong_pareado(y, s1, s2):
    """Prueba de DeLong para dos AUC correlacionados (asume meses independientes:
    solo como referencia, es demasiado optimista con datos autocorrelacionados)."""
    def componentes(s):
        p, q = s[y == 1], s[y == 0]
        psi = (p[:, None] > q[None, :]) + 0.5 * (p[:, None] == q[None, :])
        return psi.mean(axis=1), psi.mean(axis=0), psi.mean()
    v1, w1, a1 = componentes(s1)
    v2, w2, a2 = componentes(s2)
    m, n = int((y == 1).sum()), int((y == 0).sum())
    s10 = np.cov(np.vstack([v1, v2]))
    s01 = np.cov(np.vstack([w1, w2]))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    z = (a1 - a2) / np.sqrt(var)
    return 2 * norm.sf(abs(z))


puntajes = {m: d["punt"]["puntaje"].values for m, d in datos.items()}
auc_obs = {m: auc_rapido(y, s) for m, s in puntajes.items()}

filas = []
sens_bloque = {}
for largo in [6, BLOQUE, 24]:
    bs = bootstrap_auc(puntajes, y, largo, B, SEMILLA)
    for m in datos:
        d = bs[m] - bs[REFERENCIA]
        p_boot = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())) if m != REFERENCIA else np.nan
        fila = {"Modelo": nombre[m], "Bloque (meses)": largo,
                "AUC prueba": auc_obs[m],
                "IC95 AUC inf": np.percentile(bs[m], 2.5),
                "IC95 AUC sup": np.percentile(bs[m], 97.5),
                "Dif. vs logistica": auc_obs[m] - auc_obs[REFERENCIA],
                "IC95 dif inf": np.percentile(d, 2.5) if m != REFERENCIA else np.nan,
                "IC95 dif sup": np.percentile(d, 97.5) if m != REFERENCIA else np.nan,
                "p bootstrap": p_boot,
                "p DeLong (indep.)": (delong_pareado(y, puntajes[m], puntajes[REFERENCIA])
                                      if m != REFERENCIA else np.nan)}
        filas.append(fila)
tb = pd.DataFrame(filas)
tb.to_csv("resultados/tabla_bootstrap_auc.csv", index=False)

print(f"Ruido del AUC de prueba: bootstrap circular por bloques ({B} repeticiones, semilla {SEMILLA})")
print(f"Bloque principal = {BLOQUE} meses:")
print(tb[tb["Bloque (meses)"] == BLOQUE].drop(columns="Bloque (meses)").round(3).to_string(index=False))
print("\nSensibilidad al largo del bloque (IC95 de la diferencia vs logistica):")
piv = tb[tb["Modelo"] != nombre[REFERENCIA]].copy()
piv["IC95 dif"] = piv.apply(lambda r: f"[{r['IC95 dif inf']:+.3f}, {r['IC95 dif sup']:+.3f}]", axis=1)
print(piv.pivot(index="Modelo", columns="Bloque (meses)", values="IC95 dif").to_string())
print("Si el intervalo contiene 0, la diferencia con la logistica no se distingue del ruido.\n")

# ---------------------------------------------------------------
# 4. Interpretabilidad: importancia por permutacion lado a lado
# ---------------------------------------------------------------
imp = pd.DataFrame({nombre[m]: d["imp"]["Caida del AUC"] for m, d in datos.items()
                    if d["imp"] is not None})
if not imp.empty:
    rango = imp.rank(ascending=False).astype(int)
    tabla_imp = imp.round(3).astype(str) + " (" + rango.astype(str) + ")"
    print("Importancia por permutacion en prueba (caida del AUC; entre parentesis el lugar)")
    print(tabla_imp.to_string(), "\n")
    imp.to_csv("resultados/tabla_importancia.csv", index_label="variable")
    if os.path.exists("resultados/importancia_logistica.csv"):
        coefs = datos[REFERENCIA]["imp"][["Coef. statsmodels", "p-valor statsmodels"]]
        print("Logistica sin regularizar (statsmodels, variables estandarizadas):")
        print(coefs.round(3).to_string(), "\n")

# ---------------------------------------------------------------
# 5. Figura 1: curvas ROC en prueba
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 5.4))
ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), lw=1, color=TENUE, zorder=1)
ax.text(0.62, 0.55, "Azar (AUC 0,5)", color=TENUE, fontsize=8, rotation=0)
for m, d in datos.items():
    s = d["punt"]["puntaje"].values
    fpr, tpr, _ = roc_curve(y, s)
    col = MODELOS[m][1]
    ax.plot(fpr, tpr, lw=2.4 if m == REFERENCIA else 2, color=col, zorder=3 if m == REFERENCIA else 2,
            label=f"{nombre[m]}  (AUC {auc_obs[m]:.3f})".replace(".", ","))
    # Punto de operacion: el umbral elegido out-of-fold
    pred = d["punt"]["pred"].values
    tpr_u = pred[y == 1].mean()
    fpr_u = pred[y == 0].mean()
    ax.scatter(fpr_u, tpr_u, s=55, color=col, edgecolor="white", linewidth=1.5, zorder=4)
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
ax.set_aspect("equal")
coma = plt.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))  # decimales con coma (informe en espanol)
ax.xaxis.set_major_formatter(coma)
ax.yaxis.set_major_formatter(coma)
ax.set_xlabel("Tasa de falsos positivos (1 - especificidad)")
ax.set_ylabel("Tasa de verdaderos positivos (recall)")
ax.set_title(f"Curvas ROC en prueba (2015-01 a 2025-09, {y.sum()} meses con caida)",
             loc="left", fontsize=11, color=TINTA)
ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=TINTA)
ax.text(0.0, -0.17, "Puntos: umbral elegido en validacion out-of-fold (por cuantil).",
        transform=ax.transAxes, fontsize=8, color=TINTA_2)
fig.savefig("figuras/fig1_roc_prueba.png")
plt.close(fig)

# ---------------------------------------------------------------
# 6. Figura 2: matrices de confusion en prueba
# ---------------------------------------------------------------
azul = LinearSegmentedColormap.from_list("azul", ["#cde2fb", "#2a78d6", "#104281"])
n = len(datos)
fig, axes = plt.subplots(1, n, figsize=(2.55 * n, 2.9))
axes = np.atleast_1d(axes)
vmax = len(y) - y.sum()
for ax, (m, d) in zip(axes, datos.items()):
    pred = d["punt"]["pred"].values
    cm = np.array([[((y == r) & (pred == c)).sum() for c in (0, 1)] for r in (0, 1)])
    ax.imshow(cm, cmap=azul, vmin=0, vmax=vmax)
    ax.grid(False)
    for r in (0, 1):
        for c in (0, 1):
            frac = cm[r, c] / max(cm[r].sum(), 1)
            ax.text(c, r, f"{cm[r, c]}\n{frac:.0%}".replace(".", ","), ha="center", va="center",
                    fontsize=10, color="white" if cm[r, c] > 0.45 * vmax else TINTA)
    ax.set_xticks([0, 1], ["Sin caida", "Caida"])
    ax.set_yticks([0, 1], ["Sin caida", "Caida"])
    ax.tick_params(length=0, labelsize=8, colors=TINTA_2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(nombre[m], fontsize=10, color=TINTA)
    ax.set_xlabel("Predicho", fontsize=8)
axes[0].set_ylabel("Real", fontsize=8)
fig.suptitle("Matrices de confusion en prueba (porcentaje sobre cada fila real)",
             x=0.01, ha="left", fontsize=11, color=TINTA)
fig.tight_layout()
fig.savefig("figuras/fig2_matrices_confusion.png")
plt.close(fig)

# ---------------------------------------------------------------
# 7. Figura 3: meses marcados como alerta vs. meses con caida (prueba)
#    Muestra en que episodios acierta cada modelo y cuan selectivo es.
# ---------------------------------------------------------------
fechas = pd.PeriodIndex(ref_idx, freq="M").to_timestamp()
filas_fig = ["Caida real"] + [nombre[m] for m in datos]
fig, ax = plt.subplots(figsize=(9.5, 0.45 * len(filas_fig) + 1.1))
ancho = pd.Timedelta(days=27)
for k, etiqueta in enumerate(filas_fig):
    fila = len(filas_fig) - 1 - k
    if k == 0:
        marcas, col = y == 1, "#52514e"
    else:
        m = list(datos)[k - 1]
        marcas, col = datos[m]["punt"]["pred"].values == 1, MODELOS[m][1]
    ax.broken_barh([(f, ancho) for f in fechas[marcas]], (fila - 0.32, 0.64), facecolors=col)
    if k > 0:
        acierto = int((marcas & (y == 1)).sum())
        ax.text(fechas[-1] + pd.Timedelta(days=60), fila,
                f"{int(marcas.sum())} marcados, {acierto}/{y.sum()} caidas",
                va="center", fontsize=8, color=TINTA_2)
ax.set_yticks(range(len(filas_fig)), filas_fig[::-1])
ax.tick_params(axis="y", length=0, colors=TINTA)
ax.grid(axis="y", visible=False)
ax.spines["left"].set_visible(False)
ax.set_ylim(-0.7, len(filas_fig) - 0.3)
ax.set_title("Meses de prueba marcados como alerta de caida (umbral out-of-fold)",
             loc="left", fontsize=11, color=TINTA)
fig.savefig("figuras/fig3_alertas_prueba.png")
plt.close(fig)

print("Guardados: tabla1_prueba.csv, tabla2_brechas.csv, tabla_tiempos.csv, tabla_bootstrap_auc.csv,")
print("           tabla_importancia.csv, fig1_roc_prueba.png, fig2_matrices_confusion.png, "
      "fig3_alertas_prueba.png")
