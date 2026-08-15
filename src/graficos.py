"""Figuras de la presentacion del TP1 (regresion).

Cada función genera UNA figura y la guarda en figuras/. Los datos de validación cruzada
salen de resultados/cv_lineal.csv y resultados/cv_lasso.csv (ya calculados por
src.experimentos, que NO se vuelve a correr aca: tarda varios minutos). Las figuras que
necesitan el dataset crudo usan src.datos.cargar(), que es instantaneo. La unica figura
que reentrena algo es la 5, y reentrena el modelo de producción final (Lasso grado 2),
que es rapido (44 features, no 494).

Correr con:  python3 -m src.graficos
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")  # backend headless: nunca se abre una ventana

import matplotlib.pyplot as plt
import numpy as np

from src.datos import OBJETIVO, cargar, outliers_iqr
from src.experimentos import preprocesar_completo
from src.modelos import Lasso
from src.preproceso import CodificadorCategoricas, quitar_duplicados
from src.validacion import rmse, separar_train_test

RUTA_FIGURAS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figuras")
RUTA_RESULTADOS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resultados")

# Lambda del modelo de producción elegido por la regla de 1 error estandar (N0, ver
# DECISIONES.md y resultados/final.json): lasso grado 2.
LAMBDA_PRODUCCION = 286.3701351700539
GRADO_PRODUCCION = 2
MAX_ITER_LASSO = 50000
TOL_LASSO = 1e-4

# --------------------------------------------------------------------------------------
# Paleta — validada, no elegida a ojo
# --------------------------------------------------------------------------------------
# Los colores NO se eligen por gusto: se verifican con un validador que mide, en espacio
# OKLab, la separación entre pares bajo simulación de daltonismo (protanopía, deuteranopía,
# tritanopía), el contraste contra la superficie y la banda de luminosidad.
#
# La paleta anterior (azul #4C72B0, terracota #C44E52, verde #55A868, ocre #C9A227) FALLABA:
#   - #C9A227 quedaba fuera de la banda de luminosidad y por debajo del piso de croma
#     (o sea: se lee como gris, no como color).
#   - El par verde ↔ terracota daba ΔE 7.3 en deuteranopía, por debajo del piso de 8: un
#     lector deuteranope no podía distinguir la curva de grado 2 de la de grado 4.
#   - Verde y ocre quedaban por debajo de 3:1 de contraste contra el fondo.
#
# Reemplazos, ambos con todos los chequeos en verde:
#   - Categóricas (identidad): azul + naranja, ΔE 24.7 en protanopía (contra el 7.3 previo).
#   - Grados del polinomio: NO son categorías nominales, son una escala ORDENADA (2 < 3 < 4).
#     Por eso van con una rampa de UN solo tono, de claro a oscuro, que codifica el orden en
#     la luminosidad en vez de gastar tres tonos distintos. Monotonía y separación de pasos
#     verificadas.
COLOR_TRAIN = "#2a78d6"       # azul (slot categórico 1): siempre "entrenamiento"
COLOR_VAL = "#eb6834"         # naranja (slot 2): siempre "validación"
COLOR_NO_FUMADOR = "#2a78d6"  # el mismo par, aplicado a la otra dicotomía del deck
COLOR_FUMADOR = "#eb6834"
COLORES_GRADO = {2: "#86b6ef", 3: "#2a78d6", 4: "#104281"}  # rampa azul claro -> oscuro

# Cromo: gris de rejilla y de ejes un escalón por encima del fondo, texto en tinta neutra.
# La rejilla nunca compite con los datos y NUNCA va punteada (el punteado significa umbral).
COLOR_REFERENCIA = "#898781"  # gris apagado: diagonales y líneas de umbral
COLOR_REJILLA = "#e1e0d9"
COLOR_EJE = "#c3c2b7"
TINTA_PRIMARIA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
SUPERFICIE = "#fcfcfb"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": COLOR_EJE,
        "axes.labelcolor": TINTA_SECUNDARIA,
        "axes.titlecolor": TINTA_PRIMARIA,
        "text.color": TINTA_PRIMARIA,
        "xtick.color": TINTA_SECUNDARIA,
        "ytick.color": TINTA_SECUNDARIA,
        "xtick.labelcolor": TINTA_SECUNDARIA,
        "ytick.labelcolor": TINTA_SECUNDARIA,
        "axes.grid": True,
        "grid.color": COLOR_REJILLA,
        "grid.linestyle": "-",  # sólida: el punteado se reserva para umbrales
        "grid.linewidth": 0.8,
        "grid.alpha": 1.0,
        "axes.axisbelow": True,  # la rejilla va DETRÁS de los datos
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.edgecolor": COLOR_EJE,
        "legend.facecolor": SUPERFICIE,
        "figure.dpi": 150,
    }
)


def _leer_csv(nombre):
    ruta = os.path.join(RUTA_RESULTADOS, nombre)
    with open(ruta, newline="") as fh:
        return list(csv.DictReader(fh))


def _guardar(fig, nombre_archivo):
    os.makedirs(RUTA_FIGURAS, exist_ok=True)
    ruta = os.path.join(RUTA_FIGURAS, nombre_archivo)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ruta


# --------------------------------------------------------------------------------------
# Figura 1 — curvas de train y validacion contra el grado del polinomio
# --------------------------------------------------------------------------------------
def figura_curvas_train_val():
    filas = _leer_csv("cv_lineal.csv")
    grados = np.array([int(f["grado"]) for f in filas])
    train_m = np.array([float(f["rmse_train_medio"]) for f in filas])
    train_s = np.array([float(f["rmse_train_desvio"]) for f in filas])
    val_m = np.array([float(f["rmse_val_medio"]) for f in filas])
    val_s = np.array([float(f["rmse_val_desvio"]) for f in filas])

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.plot(grados, train_m, "o-", color=COLOR_TRAIN, label="RMSE de entrenamiento", linewidth=2, markersize=7)
    ax.fill_between(grados, train_m - train_s, train_m + train_s, color=COLOR_TRAIN, alpha=0.18)

    ax.plot(grados, val_m, "o-", color=COLOR_VAL, label="RMSE de validación", linewidth=2, markersize=7)
    ax.fill_between(grados, val_m - val_s, val_m + val_s, color=COLOR_VAL, alpha=0.18)

    # marca el minimo de validacion
    i_min = int(np.argmin(val_m))
    ax.plot(grados[i_min], val_m[i_min], marker="*", color=COLOR_VAL, markersize=20, zorder=5,
            markeredgecolor="white", markeredgewidth=0.8)
    ax.annotate(
        f"mínimo de validación\ngrado {grados[i_min]}: ${val_m[i_min]:,.0f}",
        xy=(grados[i_min], val_m[i_min]),
        xytext=(grados[i_min] + 0.15, val_m[i_min] + 900),
        fontsize=10,
        ha="left",
        arrowprops=dict(arrowstyle="->", color="black", lw=1),
    )

    # regiones de subajuste / sobreajuste
    ax.text(1.0, ax.get_ylim()[1] * 0.97, "subajuste\n(el modelo es\ndemasiado simple)",
            fontsize=10, style="italic", color="#555555", ha="left", va="top")
    ax.text(4.0, ax.get_ylim()[1] * 0.97, "sobreajuste\n(memoriza el train,\nno generaliza)",
            fontsize=10, style="italic", color="#555555", ha="right", va="top")

    ax.set_xticks(grados)
    ax.set_xlabel("Grado del polinomio")
    ax.set_ylabel("RMSE (dólares)")
    ax.set_title("RMSE de entrenamiento y validación según el grado del polinomio\n"
                  "(banda = ±1 desvío entre los 5 folds)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return _guardar(fig, "01-curvas-train-val.png")


# --------------------------------------------------------------------------------------
# Figura 2 — camino de regularizacion de Lasso
# --------------------------------------------------------------------------------------
def figura_camino_lasso():
    filas = _leer_csv("cv_lasso.csv")

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08}
    )

    for grado in (2, 3, 4):
        subset = [f for f in filas if int(f["grado"]) == grado]
        subset.sort(key=lambda f: float(f["lambda"]))
        lam = np.array([float(f["lambda"]) for f in subset])
        val_m = np.array([float(f["rmse_val_medio"]) for f in subset])
        coefs = np.array([float(f["coefs_no_nulos_medio"]) for f in subset])
        color = COLORES_GRADO[grado]

        ax_top.plot(lam, val_m, "o-", color=color, label=f"grado {grado}", linewidth=2, markersize=6)
        ax_bot.plot(lam, coefs, "o-", color=color, linewidth=2, markersize=6)

    ax_top.set_xscale("log")
    ax_top.invert_xaxis()
    ax_top.set_ylabel("RMSE de validación (dólares)")
    ax_top.set_title("Camino de regularización de Lasso\n(más regularizado a la izquierda, menos a la derecha)")
    ax_top.legend(title="grado del\npolinomio")

    ax_bot.set_xlabel("Lambda (escala logarítmica, invertida)")
    ax_bot.set_ylabel("Coefs. no nulos\n(promedio, 5 folds)")

    return _guardar(fig, "02-camino-lasso.png")


# --------------------------------------------------------------------------------------
# Figura 3 — interaccion fumador x bmi
# --------------------------------------------------------------------------------------
def figura_interaccion_smoker_bmi():
    df = quitar_duplicados(cargar())

    fig, ax = plt.subplots(figsize=(8, 6))

    for es_fumador, color, etiqueta in ((False, COLOR_NO_FUMADOR, "no fumador"), (True, COLOR_FUMADOR, "fumador")):
        sub = df[(df["smoker"] == "yes") == es_fumador]
        ax.scatter(sub["bmi"], sub[OBJETIVO], s=16, alpha=0.5, color=color, label=etiqueta, edgecolors="none")

        # recta de minimos cuadrados ajustada solo a este grupo
        pendiente, ordenada = np.polyfit(sub["bmi"], sub[OBJETIVO], deg=1)
        x_recta = np.array([df["bmi"].min(), df["bmi"].max()])
        ax.plot(x_recta, pendiente * x_recta + ordenada, color=color, linewidth=2.5,
                label=f"ajuste {etiqueta} (pendiente {pendiente:,.0f} $/bmi)")

    ax.axvline(30, color=COLOR_REFERENCIA, linestyle="--", linewidth=1.5)
    # El rótulo del umbral va FUERA del área de datos, apoyado sobre el borde superior.
    # Adentro no hay lugar: abajo cae en la nube densa de puntos azules y arriba choca con
    # la leyenda. Colgarlo del eje lo deja legible y sin tapar un solo dato.
    ax.annotate(
        "umbral de obesidad (OMS)",
        xy=(30, 1.0), xycoords=("data", "axes fraction"),
        xytext=(0, 5), textcoords="offset points",
        fontsize=9.5, color=TINTA_SECUNDARIA, ha="center", va="bottom",
    )

    ax.set_xlabel("Índice de masa corporal (bmi)")
    ax.set_ylabel("Costo médico (charges, dólares)")
    ax.set_title("Interacción entre fumar y bmi: las pendientes son muy distintas", pad=22)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return _guardar(fig, "03-interaccion-smoker-bmi.png")


# --------------------------------------------------------------------------------------
# Figura 4 — outliers de charges
# --------------------------------------------------------------------------------------
def figura_outliers_charges():
    df = quitar_duplicados(cargar())
    _, lim_inf, lim_sup = outliers_iqr(df[OBJETIVO])

    fig, (ax_box, ax_hist) = plt.subplots(1, 2, figsize=(11, 5.5))

    # Panel izquierdo: UNA sola distribución, sin series que distinguir. Por eso va entero
    # en tinta neutra: usar el naranja aquí haría que el lector lo leyera como "fumador",
    # que es lo que ese color significa en todas las demás figuras del deck. El color sigue
    # a la entidad, y acá no hay entidad que marcar.
    ax_box.boxplot(
        df[OBJETIVO], vert=True, widths=0.5,
        boxprops=dict(color=TINTA_SECUNDARIA, linewidth=1.2),
        medianprops=dict(color=TINTA_PRIMARIA, linewidth=2),
        whiskerprops=dict(color=TINTA_SECUNDARIA, linewidth=1.2),
        capprops=dict(color=TINTA_SECUNDARIA, linewidth=1.2),
        flierprops=dict(marker="o", markerfacecolor=TINTA_SECUNDARIA, markeredgecolor="none",
                        alpha=0.35, markersize=4),
    )
    # El punteado se reserva para umbrales — que es exactamente lo que esto es.
    ax_box.axhline(lim_sup, color=COLOR_REFERENCIA, linestyle="--", linewidth=1.5)
    ax_box.text(1.32, lim_sup, f"límite superior IQR\n${lim_sup:,.0f}", fontsize=9,
                color=TINTA_SECUNDARIA, va="center", ha="left")
    ax_box.set_xticks([])  # una sola caja: la etiqueta "charges" ya está en el eje y
    ax_box.set_ylabel("Costo médico (charges, dólares)")
    ax_box.set_title("Distribución de charges\n(139 outliers por IQR, 10.4 %)")

    # Panel derecho: histograma APILADO, no superpuesto.
    # Superponer dos histogramas con transparencia crea un TERCER color en la zona de
    # solape que no está en la leyenda y que el lector no sabe interpretar. Apilados, la
    # altura total es la cantidad de personas del bin y el tramo naranja es la porción de
    # fumadores: se ve de un golpe que la cola larga es enteramente naranja.
    fumador = df["smoker"] == "yes"
    bins = np.linspace(df[OBJETIVO].min(), df[OBJETIVO].max(), 40)
    ax_hist.hist(
        [df.loc[~fumador, OBJETIVO], df.loc[fumador, OBJETIVO]],
        bins=bins, stacked=True,
        color=[COLOR_NO_FUMADOR, COLOR_FUMADOR],
        label=["no fumador", "fumador"],
    )
    ax_hist.axvline(lim_sup, color=COLOR_REFERENCIA, linestyle="--", linewidth=1.5)
    # A media altura y a la derecha de la línea: arriba chocaba con la leyenda, y a esa
    # altura el histograma ya no tiene barras, así que el rótulo no tapa datos.
    ax_hist.text(lim_sup, ax_hist.get_ylim()[1] * 0.45, "  límite IQR", fontsize=9,
                 color=TINTA_SECUNDARIA, va="center", ha="left")
    ax_hist.set_xlabel("Costo médico (charges, dólares)")
    ax_hist.set_ylabel("Cantidad de personas")
    ax_hist.set_title("La cola larga son los fumadores\n(97.8 % de los outliers fuman)")
    ax_hist.legend(loc="upper right")

    fig.tight_layout()
    return _guardar(fig, "04-outliers-charges.png")


# --------------------------------------------------------------------------------------
# La figura 5 (predicho contra real) NO vive acá.
#
# Ese gráfico se construye prediciendo sobre el conjunto de TEST, así que generarlo ES
# evaluar el test. Tenerlo en este módulo significaba que correr `python -m src.graficos`
# —algo que uno hace muchas veces mientras ajusta colores y tamaños— tocaba el test cada
# vez, en silencio.
#
# Por eso se mudó a `src/evaluar_test.py`, que corre UNA sola vez y de forma deliberada
# (decisión D-21). Ahí se genera `figuras/05-predicho-vs-real.png` junto con los números.
# --------------------------------------------------------------------------------------



# --------------------------------------------------------------------------------------
def main():
    rutas = [
        figura_curvas_train_val(),
        figura_camino_lasso(),
        figura_interaccion_smoker_bmi(),
        figura_outliers_charges(),
    ]
    for ruta in rutas:
        tamano_kb = os.path.getsize(ruta) / 1024
        print(f"{ruta}  ({tamano_kb:.1f} KB)")


if __name__ == "__main__":
    main()
