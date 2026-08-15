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
# Figura 6 — sensibilidad al numero de folds k (D-22)
# --------------------------------------------------------------------------------------
def figura_sensibilidad_k():
    """Dos paneles APILADOS sobre un eje x compartido: qué les pasa a los dos números que
    reporta el TP cuando se cambia k, con la configuración de producción FIJA (así lo
    único que varía es k). Los datos salen de resultados/sensibilidad_k.csv, que produce
    `python -m src.sensibilidad_k` (no se recalcula acá).

    Tres decisiones de forma, y ninguna es de gusto:

    1. DOS PANELES, NO UN EJE DOBLE. El nivel del RMSE vive en ~5000 dólares y el error
       estándar en ~200. Meterlos en un mismo par de ejes obliga a un segundo eje y, y en
       un gráfico de doble eje la alineación entre las dos escalas es arbitraria: inventa
       cruces y divergencias que no están en los datos. Un eje por panel.

    2. APILADOS Y COMPARTIENDO x, no lado a lado. Lo que el lector tiene que poder hacer
       es leer, para un mismo k, qué pasó con las DOS cantidades. Apilados, un k es una
       vertical y la comparación es instantánea; lado a lado hay que volver a buscar el k
       en el segundo eje. Además el eje x se dibuja y se lee una sola vez.

    3. LA GRILLA DE k SE MIDIÓ DENSA. Con sólo (5, 10, 20, 50, 1070) el tramo final es una
       recta que cruza un rango de 20x, y esa recta AFIRMA una forma intermedia que nadie
       midió. Los puntos 100, 200 y 500 están para que la curva sea un dato y no una
       interpolación.
    """
    filas = _leer_csv("sensibilidad_k.csv")
    k = np.array([int(f["k"]) for f in filas])
    media_folds = np.array([float(f["rmse_val_medio"]) for f in filas])
    agrupado = np.array([float(f["rmse_val_agrupado"]) for f in filas])
    es = np.array([float(f["error_estandar"]) for f in filas])
    k_loo = int(k[-1])

    fig, (ax_sup, ax_inf) = plt.subplots(
        2, 1, figsize=(9, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.25, 1]}
    )

    # ------------------------------------------------------------------ panel superior
    # El RMSE agrupado (una sola raíz sobre los 1070 residuos out-of-fold) contra el
    # promedio de los k RMSE de fold, que es lo que reporta el TP.
    ax_sup.plot(k, agrupado, "o-", color=COLOR_TRAIN, linewidth=2, markersize=7,
                label="RMSE agrupado (1 raíz sobre los 1070 residuos)", zorder=3)
    ax_sup.plot(k, media_folds, "o-", color=COLOR_VAL, linewidth=2, markersize=7,
                label="Media de los $k$ RMSE de fold", zorder=3)

    # La brecha ENTRE las dos curvas es el objeto del gráfico —el sesgo de Jensen—, así
    # que se sombrea y se nombra. Un wash sin etiqueta obliga al lector a deducir qué
    # significa el área, que es justo lo que el gráfico tendría que estarle diciendo.
    ax_sup.fill_between(k, media_folds, agrupado, color=COLOR_VAL, alpha=0.13, zorder=0)
    i_med = int(np.argmin(np.abs(k - 50)))
    ax_sup.annotate("la brecha es el sesgo de\npromediar raíces\n(desigualdad de Jensen)",
                    xy=(k[i_med], (media_folds[i_med] + agrupado[i_med]) / 2),
                    xytext=(-46, -74), textcoords="offset points", ha="right", va="top",
                    fontsize=11.5, color=TINTA_SECUNDARIA,
                    arrowprops=dict(arrowstyle="->", color=TINTA_SECUNDARIA, lw=1))

    # Etiquetas directas: sólo los extremos de cada curva. El eje lleva el resto.
    ax_sup.annotate(f"\\${agrupado[0]:,.0f}".replace(",", "."), xy=(k[0], agrupado[0]),
                    xytext=(4, 10), textcoords="offset points", ha="left",
                    fontsize=12, color=TINTA_SECUNDARIA)
    ax_sup.annotate(f"\\${media_folds[0]:,.0f}".replace(",", "."), xy=(k[0], media_folds[0]),
                    xytext=(4, -19), textcoords="offset points", ha="left",
                    fontsize=12, color=TINTA_SECUNDARIA)
    ax_sup.annotate(f"\\${agrupado[-1]:,.0f}".replace(",", "."),
                    xy=(k[-1], agrupado[-1]), xytext=(-8, 12), textcoords="offset points",
                    ha="right", fontsize=12, color=TINTA_SECUNDARIA)
    ax_sup.annotate(
        f"\\${media_folds[-1]:,.0f}".replace(",", ".")
        + "\ncon 1 dato por fold el promedio\nde RMSEs ES el MAE",
        xy=(k[-1], media_folds[-1]), xytext=(-24, 92), textcoords="offset points",
        ha="right", va="bottom", fontsize=12, color=TINTA_PRIMARIA,
        arrowprops=dict(arrowstyle="->", color=TINTA_PRIMARIA, lw=1),
    )

    ax_sup.set_ylim(2870, 5180)
    ax_sup.set_ylabel("RMSE de validación (dólares)")
    ax_sup.set_title("El nivel del error casi no depende de $k$: lo que se mueve es cómo se promedia",
                     fontsize=12.5, pad=10)
    ax_sup.legend(loc="lower left", fontsize=11.5)

    # ------------------------------------------------------------------ panel inferior
    # El ES fija el umbral de la regla de 1 error estándar, así que su dependencia de k es
    # lo que podría cambiar el modelo elegido.
    #
    # OJO con la lectura fácil: el ES NO se estabiliza. Sube hasta ~220 en k=10..50 y
    # despues BAJA monotonamente (197,7 / 185,3 / 152,0 / 118,0). Con solo (10, 20, 50)
    # medidos parecia una meseta; con la grilla densa se ve que es un maximo. El motivo es
    # que pasado cierto k los folds quedan tan chicos que sigma deja de medir dispersion
    # entre remuestreos y pasa a medir dispersion entre observaciones — la misma objecion
    # que se le hace a LOO, que no empieza en LOO sino mucho antes.
    RANGO_UTIL = (10, 50)  # donde un RMSE de fold todavia significa algo (>=21 puntos)
    util = (k >= RANGO_UTIL[0]) & (k <= RANGO_UTIL[1])
    nivel_util = float(np.mean(es[util]))

    # Zona degenerada: se sombrea en gris neutro en vez de marcar sólo el punto de LOO.
    # Marcar LOO como caso especial sugeriria que el problema aparece recien ahi.
    k_degenerado = float(k[k > RANGO_UTIL[1]][0]) / 1.45
    ax_inf.axvspan(k_degenerado, k[-1] * 1.5, color=COLOR_REJILLA, alpha=0.75, zorder=0)
    ax_inf.annotate("folds de ≤ 11 puntos: σ deja de medir\ndispersión entre remuestreos y pasa a\nmedirla entre observaciones",
                    xy=(0.985, 0.94), xycoords="axes fraction", ha="right", va="top",
                    fontsize=11, color=COLOR_REFERENCIA)

    # Línea de referencia sobre el rango utilizable únicamente: extenderla a todo el ancho
    # afirmaria que el ES se queda ahi, y no se queda.
    ax_inf.hlines(nivel_util, RANGO_UTIL[0] * 0.82, RANGO_UTIL[1] * 1.2,
                  color=COLOR_REFERENCIA, linestyle=":", linewidth=1.8, zorder=2)
    ax_inf.annotate(f"≈ {nivel_util:,.0f} en el rango utilizable".replace(",", "."),
                    xy=(20, nivel_util), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=12, color=COLOR_REFERENCIA)

    ax_inf.plot(k, es, "o-", color=COLOR_VAL, linewidth=2, markersize=7, zorder=3)

    ax_inf.annotate(f"{es[0]:,.1f}".replace(",", ".") + "\nel ES que reporta el TP:\nmenos de la mitad",
                    xy=(k[0], es[0]), xytext=(26, -16), textcoords="offset points",
                    ha="left", va="top", fontsize=12, color=TINTA_PRIMARIA,
                    arrowprops=dict(arrowstyle="->", color=TINTA_PRIMARIA, lw=1))

    ax_inf.set_ylim(0, max(es) * 1.42)
    ax_inf.set_ylabel("Error estándar\n$\\sigma_{\\mathrm{folds}}/\\sqrt{k}$ (dólares)")
    ax_inf.set_title("El error estándar sí depende de $k$: máximo en $k$=10–50, y el $k=5$ del TP es la mitad",
                     fontsize=12.5, pad=10)

    # ------------------------------------------- punto de operación, en los dos paneles
    # El sujeto del gráfico es el k que el TP usa. Se marca resaltando su TICK, no con una
    # vertical adentro del área de datos: k=5 es el extremo izquierdo, así que una vertical
    # ahí queda pegada al eje y encima le pisa las etiquetas a la curva.
    # ------------------------------------------------------------------ eje x compartido
    # Escala logarítmica: k crece por factores y el costo de la validación cruzada es
    # lineal en k, así que los saltos SON multiplicativos.
    etiquetas = [str(kk) for kk in k]
    etiquetas[-1] = f"{k_loo}\n(LOO)"
    ax_inf.set_xscale("log")
    ax_inf.set_xticks(k)
    ax_inf.set_xticklabels(etiquetas, fontsize=11)
    ax_inf.get_xticklabels()[0].set_color(TINTA_PRIMARIA)
    ax_inf.get_xticklabels()[0].set_fontweight("bold")
    ax_inf.minorticks_off()
    ax_inf.set_xlabel("Número de folds $k$   (escala logarítmica)")

    fig.suptitle("Sensibilidad al número de folds — configuración de producción fija "
                 "(Lasso grado 2, $\\lambda=286{,}37$)",
                 fontsize=13.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return _guardar(fig, "06-sensibilidad-k.png")


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
        figura_sensibilidad_k(),
    ]
    for ruta in rutas:
        tamano_kb = os.path.getsize(ruta) / 1024
        print(f"{ruta}  ({tamano_kb:.1f} KB)")


if __name__ == "__main__":
    main()
