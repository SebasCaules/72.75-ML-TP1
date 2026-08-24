"""Correr con: python -m src.graficos"""

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.datos import OBJETIVO, cargar_train, outliers_iqr
from src.validacion import rmse, separar_train_test

RUTA_FIGURAS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figuras")
RUTA_RESULTADOS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resultados")


COLOR_TRAIN = "#2a78d6"
COLOR_VAL = "#eb6834"
COLOR_NO_FUMADOR = "#2a78d6"
COLOR_FUMADOR = "#eb6834"
COLORES_GRADO = {2: "#86b6ef", 3: "#2a78d6", 4: "#104281"}

COLOR_REFERENCIA = "#898781"
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
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
        "grid.alpha": 1.0,
        "axes.axisbelow": True,
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


def _guardar(fig, nombre_archivo, dpi=150):
    os.makedirs(RUTA_FIGURAS, exist_ok=True)
    ruta = os.path.join(RUTA_FIGURAS, nombre_archivo)
    fig.savefig(ruta, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return ruta


def figura_curvas_train_val():
    filas = _leer_csv("cv_lineal.csv")
    grados = np.array([int(f["grado"]) for f in filas])
    train_m = np.array([float(f["rmse_train_medio"]) for f in filas])
    train_s = np.array([float(f["rmse_train_desvio"]) for f in filas])
    val_m = np.array([float(f["rmse_val_medio"]) for f in filas])
    val_s = np.array([float(f["rmse_val_desvio"]) for f in filas])

    umbral_disparado = 2.5 * float(np.median(val_m))
    disparado = val_m[val_m >= umbral_disparado]
    normales = val_m[val_m < umbral_disparado]
    partido = len(disparado) > 0 and len(normales) > 0

    if partido:
        fig, (ax_alto, ax_bajo) = plt.subplots(
            2, 1, figsize=(8, 6.2), sharex=True,
            gridspec_kw={"height_ratios": [1, 2.4], "hspace": 0.07},
        )
        ejes = (ax_alto, ax_bajo)
    else:
        fig, ax_bajo = plt.subplots(figsize=(8, 5.5))
        ax_alto = ax_bajo
        ejes = (ax_bajo,)

    for ax in ejes:
        ax.plot(grados, train_m, "o-", color=COLOR_TRAIN, label="RMSE de entrenamiento",
                linewidth=2, markersize=7)
        ax.fill_between(grados, train_m - train_s, train_m + train_s,
                        color=COLOR_TRAIN, alpha=0.18)
        ax.plot(grados, val_m, "o-", color=COLOR_VAL, label="RMSE de validación",
                linewidth=2, markersize=7)
        ax.fill_between(grados, val_m - val_s, val_m + val_s, color=COLOR_VAL, alpha=0.18)

    if partido:
        ax_bajo.set_ylim(min(train_m.min(), normales.min()) - 300, normales.max() + 400)
        ax_alto.set_ylim(disparado.min() * 0.86, float(np.max(val_m + val_s)) * 1.03)

        ax_alto.spines["bottom"].set_visible(False)
        ax_bajo.spines["top"].set_visible(False)
        ax_alto.tick_params(axis="x", bottom=False, labelbottom=False)
        kw = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=9, linestyle="none",
                  color=TINTA_SECUNDARIA, mec=TINTA_SECUNDARIA, mew=1.2, clip_on=False)
        ax_alto.plot([0, 1], [0, 0], transform=ax_alto.transAxes, **kw)
        ax_bajo.plot([0, 1], [1, 1], transform=ax_bajo.transAxes, **kw)

    i_min = int(np.argmin(val_m))
    ax_bajo.plot(grados[i_min], val_m[i_min], marker="*", color=COLOR_VAL, markersize=20,
                 zorder=5, markeredgecolor="white", markeredgewidth=0.8)
    ax_bajo.annotate(
        f"mínimo de validación\ngrado {grados[i_min]}: \\${_num(val_m[i_min], 0)}",
        xy=(grados[i_min], val_m[i_min]), xytext=(18, -6), textcoords="offset points",
        fontsize=10, ha="left", va="top",
        arrowprops=dict(arrowstyle="->", color=TINTA_PRIMARIA, lw=1),
    )

    i_max = int(np.argmax(val_m))
    if partido:
        ax_alto.annotate(
            f"grado {grados[i_max]}: \\${_num(val_m[i_max], 0)}\n"
            f"±{_num(val_s[i_max], 0)} entre folds",
            xy=(grados[i_max], val_m[i_max]), xytext=(-14, 0), textcoords="offset points",
            fontsize=10, ha="right", va="center", color=TINTA_PRIMARIA,
        )

    ax_bajo.set_xticks(grados)
    ax_bajo.set_xlabel("Grado del polinomio")
    ax_bajo.set_ylabel("RMSE (dólares)")
    ax_bajo.legend(loc="upper left", fontsize=9.5)
    ax_alto.set_title("El error de entrenamiento baja siempre; el de validación toca fondo "
                      "en grado 1\ny después se dispara  (banda = ±1 desvío entre los 5 folds)",
                      fontsize=12, pad=10)
    fig.tight_layout()
    return _guardar(fig, "01-curvas-train-val.png")


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


def figura_interaccion_smoker_bmi():
    df = cargar_train()

    fig, ax = plt.subplots(figsize=(8, 6))

    for es_fumador, color, etiqueta in ((False, COLOR_NO_FUMADOR, "no fumador"), (True, COLOR_FUMADOR, "fumador")):
        sub = df[(df["smoker"] == "yes") == es_fumador]
        ax.scatter(sub["bmi"], sub[OBJETIVO], s=16, alpha=0.5, color=color, label=etiqueta, edgecolors="none")

        pendiente, ordenada = np.polyfit(sub["bmi"], sub[OBJETIVO], deg=1)
        x_recta = np.array([df["bmi"].min(), df["bmi"].max()])
        ax.plot(x_recta, pendiente * x_recta + ordenada, color=color, linewidth=2.5,
                label=f"ajuste {etiqueta} (pendiente {pendiente:,.0f} $/bmi)")

    ax.axvline(30, color=COLOR_REFERENCIA, linestyle="--", linewidth=1.5)
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


def figura_outliers_charges():
    df = cargar_train()
    _, lim_inf, lim_sup = outliers_iqr(df[OBJETIVO])

    es_out = (df[OBJETIVO] < lim_inf) | (df[OBJETIVO] > lim_sup)
    n_out = int(es_out.sum())
    pct_out = 100 * n_out / len(df)
    pct_fuman = 100 * (df.loc[es_out, "smoker"] == "yes").mean()

    def pesos(v):
        return "$" + f"{v:,.0f}".replace(",", ".")

    q1, mediana, q3 = df[OBJETIVO].quantile([0.25, 0.50, 0.75])
    iqr = q3 - q1
    fumador = df["smoker"] == "yes"

    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=(8.2, 3.5), sharex=True, gridspec_kw={"height_ratios": [1, 1.45]}
    )

    ax_box.boxplot(
        df[OBJETIVO], vert=False, widths=0.42, showfliers=False,
        boxprops=dict(color=TINTA_SECUNDARIA, linewidth=1.3),
        medianprops=dict(color=TINTA_PRIMARIA, linewidth=2.4),
        whiskerprops=dict(color=TINTA_SECUNDARIA, linewidth=1.3),
        capprops=dict(color=TINTA_SECUNDARIA, linewidth=1.3),
    )

    y_out = 1 + np.random.default_rng(42).uniform(-0.13, 0.13, n_out)
    for marca, color, etiqueta in ((~fumador, COLOR_NO_FUMADOR, "no fumador"),
                                   (fumador, COLOR_FUMADOR, "fumador")):
        sel = (es_out & marca).values[es_out.values]
        ax_box.scatter(df.loc[es_out & marca, OBJETIVO], y_out[sel],
                       s=22, color=color, alpha=0.55, linewidths=0, zorder=3, label=etiqueta)

    ax_box.text(700, 1.66,
                f"Q1 {pesos(q1)}   ·   mediana {pesos(mediana)}   ·   Q3 {pesos(q3)}",
                ha="left", va="center", fontsize=10, color=TINTA_SECUNDARIA)

    ax_box.plot([q1, q3], [0.70, 0.70], color=TINTA_SECUNDARIA, linewidth=1)
    for x in (q1, q3):
        ax_box.plot([x, x], [0.66, 0.74], color=TINTA_SECUNDARIA, linewidth=1)
    ax_box.annotate(f"IQR = {pesos(iqr)}", xy=((q1 + q3) / 2, 0.70),
                    xytext=(0, -4), textcoords="offset points", ha="center", va="top",
                    fontsize=10, color=TINTA_SECUNDARIA)

    ax_box.annotate(f"límite = Q3 + 1,5 · IQR = {pesos(lim_sup)}",
                    xy=(lim_sup, 1.66), xytext=(7, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=10, color=COLOR_REFERENCIA)
    ax_box.annotate(f"{n_out} outliers · {pct_fuman:.1f} % fuman",
                    xy=(lim_sup, 0.62), xytext=(7, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=10, color=COLOR_FUMADOR)

    ax_box.set_ylim(0.30, 1.92)
    ax_box.set_yticks([])
    ax_box.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_box.grid(False)
    ax_box.set_title(f"El criterio: la caja es el 50 % del medio, y el corte está "
                     f"1,5 cajas más a la derecha\n({n_out} outliers, {pct_out:.1f} % de la muestra)",
                     fontsize=12, pad=12)

    bins = np.linspace(df[OBJETIVO].min(), df[OBJETIVO].max(), 46)
    ax_hist.hist(
        [df.loc[~fumador, OBJETIVO], df.loc[fumador, OBJETIVO]],
        bins=bins, stacked=True,
        color=[COLOR_NO_FUMADOR, COLOR_FUMADOR],
        label=["no fumador", "fumador"],
    )
    ax_hist.set_ylabel("Cantidad de personas")
    ax_hist.set_title("Quiénes son: pasando el corte, casi no queda azul", fontsize=12, pad=10)
    ax_hist.legend(loc="upper right")

    for ax in (ax_box, ax_hist):
        ax.axvline(lim_sup, color=COLOR_REFERENCIA, linestyle="--", linewidth=1.5, zorder=1)

    for ax in (ax_box, ax_hist):
        ax.set_xlim(0, df[OBJETIVO].max() * 1.02)
        for lado in ("top", "right", "bottom", "left"):
            ax.spines[lado].set_visible(True)
            ax.spines[lado].set_color(COLOR_EJE)
            ax.spines[lado].set_linewidth(1.0)

    ax_hist.set_xticks(np.arange(0, 70000, 10000))
    ax_hist.set_xticklabels(["0"] + [f"{v}k" for v in range(10, 70, 10)])
    ax_hist.set_xlabel("Costo médico (charges, dólares)")

    fig.tight_layout()
    return _guardar(fig, "04-outliers-charges.png")


def figura_sensibilidad_k():
    filas = _leer_csv("sensibilidad_k.csv")
    k = np.array([int(f["k"]) for f in filas])
    media_folds = np.array([float(f["rmse_val_medio"]) for f in filas])
    agrupado = np.array([float(f["rmse_val_agrupado"]) for f in filas])
    es = np.array([float(f["error_estandar"]) for f in filas])
    k_loo = int(k[-1])

    fig, (ax_sup, ax_inf) = plt.subplots(
        2, 1, figsize=(9, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.25, 1]}
    )

    ax_sup.plot(k, agrupado, "o-", color=COLOR_TRAIN, linewidth=2, markersize=7,
                label="RMSE agrupado (1 raíz sobre los 1070 residuos)", zorder=3)
    ax_sup.plot(k, media_folds, "o-", color=COLOR_VAL, linewidth=2, markersize=7,
                label="Media de los $k$ RMSE de fold", zorder=3)

    ax_sup.fill_between(k, media_folds, agrupado, color=COLOR_VAL, alpha=0.13, zorder=0)
    i_med = int(np.argmin(np.abs(k - 50)))
    ax_sup.annotate("la brecha es el sesgo de\npromediar raíces\n(desigualdad de Jensen)",
                    xy=(k[i_med], (media_folds[i_med] + agrupado[i_med]) / 2),
                    xytext=(-22, -58), textcoords="offset points", ha="right", va="top",
                    fontsize=11.5, color=TINTA_SECUNDARIA,
                    arrowprops=dict(arrowstyle="->", color=TINTA_SECUNDARIA, lw=1))

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
        xy=(k[-1], media_folds[-1]), xytext=(-52, 112), textcoords="offset points",
        ha="right", va="bottom", fontsize=12, color=TINTA_PRIMARIA,
        arrowprops=dict(arrowstyle="->", color=TINTA_PRIMARIA, lw=1),
    )

    piso = float(min(media_folds.min(), agrupado.min()))
    techo = float(max(media_folds.max(), agrupado.max()))
    margen = 0.09 * (techo - piso)
    ax_sup.set_ylim(piso - margen, techo + margen)
    ax_sup.set_ylabel("RMSE de validación (dólares)")
    ax_sup.set_title("El nivel del error casi no depende de $k$: lo que se mueve es cómo se promedia",
                     fontsize=12.5, pad=10)
    ax_sup.legend(loc="lower left", fontsize=11.5)

    RANGO_UTIL = (10, 50)
    util = (k >= RANGO_UTIL[0]) & (k <= RANGO_UTIL[1])
    nivel_util = float(np.mean(es[util]))

    k_degenerado = float(k[k > RANGO_UTIL[1]][0]) / 1.45
    ax_inf.axvspan(k_degenerado, k[-1] * 1.5, color=COLOR_REJILLA, alpha=0.75, zorder=0)
    ax_inf.annotate("folds de ≤ 11 puntos: σ deja de medir\ndispersión entre remuestreos y pasa a\nmedirla entre observaciones",
                    xy=(0.985, 0.94), xycoords="axes fraction", ha="right", va="top",
                    fontsize=11, color=COLOR_REFERENCIA)

    ax_inf.hlines(nivel_util, RANGO_UTIL[0] * 0.82, RANGO_UTIL[1] * 1.2,
                  color=COLOR_REFERENCIA, linestyle=":", linewidth=1.8, zorder=2)
    ax_inf.annotate(f"≈ {nivel_util:,.0f} en el rango utilizable".replace(",", "."),
                    xy=(20, nivel_util), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=12, color=COLOR_REFERENCIA)

    ax_inf.plot(k, es, "o-", color=COLOR_VAL, linewidth=2, markersize=7, zorder=3)

    fraccion = es[0] / nivel_util
    ax_inf.annotate(f"{es[0]:,.1f}".replace(",", ".")
                    + f"\nel ES que reporta el TP:\nel {fraccion:.0%} de ese nivel",
                    xy=(k[0], es[0]), xytext=(26, -16), textcoords="offset points",
                    ha="left", va="top", fontsize=12, color=TINTA_PRIMARIA,
                    arrowprops=dict(arrowstyle="->", color=TINTA_PRIMARIA, lw=1))

    ax_inf.set_ylim(0, max(es) * 1.42)
    ax_inf.set_ylabel("Error estándar\n$\\sigma_{\\mathrm{folds}}/\\sqrt{k}$ (dólares)")
    k_maximo = int(k[int(np.argmax(es))])
    ax_inf.set_title(f"El error estándar sí depende de $k$: máximo en $k={k_maximo}$, "
                     f"y el $k=5$ del TP queda por debajo",
                     fontsize=12.5, pad=10)

    etiquetas = [str(kk) for kk in k]
    etiquetas[-1] = f"{k_loo}\n(LOO)"
    ax_inf.set_xscale("log")
    ax_inf.set_xticks(k)
    ax_inf.set_xticklabels(etiquetas, fontsize=11)
    ax_inf.get_xticklabels()[0].set_color(TINTA_PRIMARIA)
    ax_inf.get_xticklabels()[0].set_fontweight("bold")
    ax_inf.minorticks_off()
    ax_inf.set_xlabel("Número de folds $k$   (escala logarítmica)")

    with open(os.path.join(RUTA_RESULTADOS, "modelo_elegido.json")) as fh:
        prod = json.load(fh)["produccion_1se"]
    desc = (f"{prod['modelo']} grado {prod['grado']}"
            + (f", $\\lambda={_num(prod['lambda'], 2)}$" if prod["lambda"] is not None
               else ", sin regularización"))
    fig.suptitle("Sensibilidad al número de folds — configuración de producción fija "
                 f"({desc})", fontsize=13.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return _guardar(fig, "06-sensibilidad-k.png")


COLOR_NUMERICA = COLOR_TRAIN
COLOR_CATEGORICA = "#a8447f"
COLOR_OBJETIVO = COLOR_VAL

COLOR_FUMADOR_CLARO = "#f0925a"
COLOR_FUMADOR_OSCURO = "#a83c14"

NUMERICAS_EDA = ["age", "bmi", "children", "charges"]
CATEGORICAS_EDA = ["sex", "smoker", "region"]


def _bins_freedman_diaconis(serie):
    q1, q3 = serie.quantile([0.25, 0.75])
    ancho = 2 * (q3 - q1) / len(serie) ** (1 / 3)
    return max(1, int(np.ceil((serie.max() - serie.min()) / ancho)))


def _num(valor, decimales=1, signo=False):
    crudo = f"{valor:+,.{decimales}f}" if signo else f"{valor:,.{decimales}f}"
    return crudo.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def _panel_numerica(ax, serie, etiqueta_x, color=COLOR_NUMERICA, nota=None, pesos=False):
    if serie.dtype.kind == "i":
        bordes = np.arange(serie.min() - 0.5, serie.max() + 1.5, 1.0)
    else:
        bordes = _bins_freedman_diaconis(serie)

    alturas, _, _ = ax.hist(serie, bins=bordes, color=color,
                            edgecolor=SUPERFICIE, linewidth=0.4)

    fmt = (lambda v: "\\$" + _num(v, 0)) if pesos else (lambda v: _num(v, 1))
    ax.set_title(
        f"media {fmt(serie.mean())}  ·  mediana {fmt(serie.median())}  ·  "
        f"σ {fmt(serie.std())}\nrango [{fmt(serie.min())} – {fmt(serie.max())}]  ·  "
        f"asimetría {_num(serie.skew(), 2, signo=True)}",
        fontsize=9, color=TINTA_SECUNDARIA, pad=6, loc="left",
    )
    ax.set_xlabel(etiqueta_x, fontsize=10.5, color=TINTA_PRIMARIA)

    ax.axvline(serie.median(), color=COLOR_REFERENCIA, linestyle="--", linewidth=1.3, zorder=4)

    if nota:
        ax.set_ylim(0, alturas.max() * 1.52)
        ax.annotate(nota, xy=(0.975, 0.965), xycoords="axes fraction", ha="right", va="top",
                    fontsize=9, color=TINTA_PRIMARIA, linespacing=1.35,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor=SUPERFICIE,
                              edgecolor=COLOR_EJE, linewidth=0.8))


def _panel_categorica(ax, serie, etiqueta, color=COLOR_CATEGORICA):
    conteo = serie.value_counts().sort_values()
    ax.barh(range(len(conteo)), conteo.values, color=color, height=0.62)
    ax.set_yticks(range(len(conteo)))
    ax.set_yticklabels(conteo.index, fontsize=10)
    ax.set_xlabel("Cantidad de personas", fontsize=10.5, color=TINTA_PRIMARIA)
    faltantes = int(serie.isna().sum())
    ax.set_title(f"{etiqueta} — {len(conteo)} niveles · "
                 f"{'sin faltantes' if faltantes == 0 else f'{faltantes} faltantes'}",
                 fontsize=9, color=TINTA_SECUNDARIA, pad=6, loc="left")

    total = conteo.sum()
    for i, v in enumerate(conteo.values):
        ax.annotate(f"{v}  ({_num(100 * v / total, 1)} %)", xy=(v, i), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=9.5,
                    color=TINTA_SECUNDARIA)
    ax.set_xlim(0, conteo.max() * 1.34)
    ax.grid(axis="y", visible=False)


def figura_histogramas():
    df = cargar_train()

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6))

    n_jovenes = int((df["age"] <= 19).sum())
    esperado_dos_anios = 2 * len(df) / df["age"].nunique()
    _panel_numerica(
        axes[0][0], df["age"], "Edad (años)",
        nota=f"1 bin por año.\nPico en 18–19: {n_jovenes} personas,\n"
             f"{_num(n_jovenes / esperado_dos_anios, 1)}× lo que le tocaría\na dos años cualesquiera.",
    )
    _panel_numerica(
        axes[0][1], df["bmi"], "Índice de masa corporal (bmi)",
        nota="La única aproximadamente\nsimétrica: media ≈ mediana.\nEs la que el z-score\nescala sin distorsionar.",
    )
    _panel_numerica(
        axes[0][2], df["children"], "Hijos a cargo (children)",
        nota=f"Numérica pero DISCRETA:\n{df['children'].nunique()} valores, no un continuo.\n"
             "→ tratarla como categórica\n(Clase 3, slide 35).",
    )
    _panel_numerica(
        axes[0][3], df["charges"], "Costo médico (charges, dólares)",
        color=COLOR_OBJETIVO, pesos=True,
        nota="Asimétrica a derecha y con\nlóbulos secundarios: no es\nuna distribución con cola,\nson varias poblaciones\n(slides 36–38) → figura 8.",
    )
    axes[0][3].set_xticks(np.arange(0, 70000, 20000))
    axes[0][3].set_xticklabels(["0"] + [f"{v}k" for v in range(20, 70, 20)])

    _panel_categorica(axes[1][0], df["sex"], "sex")
    _panel_categorica(axes[1][1], df["smoker"], "smoker")
    _panel_categorica(axes[1][2], df["region"], "region")

    ax_nota = axes[1][3]
    ax_nota.axis("off")
    pct_fuma = 100 * (df["smoker"] == "yes").mean()
    ax_nota.text(
        0.0, 1.0,
        "Lo que muestran los histogramas\ny las estadísticas no\n",
        transform=ax_nota.transAxes, ha="left", va="top",
        fontsize=11.5, color=TINTA_PRIMARIA, fontweight="bold", linespacing=1.3,
    )
    ax_nota.text(
        0.0, 0.80,
        "· age no es uniforme: sobran 18 y 19 años.\n"
        f"  La media ({_num(df['age'].mean(), 1)}) no lo delata.\n\n"
        f"· children está discretizada en {df['children'].nunique()} escalones.\n"
        f"  Su media ({_num(df['children'].mean(), 2)}) describe a nadie.\n\n"
        "· charges no tiene un centro: tiene tres.\n"
        f"  Media \\${_num(df[OBJETIVO].mean(), 0)} y mediana \\${_num(df[OBJETIVO].median(), 0)}\n"
        "  caen las dos dentro del primer lóbulo,\n"
        "  y ninguna describe a los otros dos.\n\n"
        f"· smoker está desbalanceada ({_num(pct_fuma, 1)} % fuma)\n"
        "  y es justo la variable que parte charges.\n\n"
        "· Ninguna variable mezcla dos unidades de\n"
        "  medida (slide 39): no hay escalas que\n"
        "  corregir, sólo que estandarizar.",
        transform=ax_nota.transAxes, ha="left", va="top",
        fontsize=9.6, color=TINTA_SECUNDARIA, linespacing=1.45,
    )

    for fila in axes[:1]:
        for ax in fila:
            ax.set_ylabel("Cantidad de personas", fontsize=10.5, color=TINTA_PRIMARIA)

    fig.suptitle(
        f"Distribución de las {df.shape[1]} variables de insurance.csv "
        f"({len(df)} filas, sin duplicados)",
        fontsize=13.5, y=0.985,
    )
    from matplotlib.patches import Patch

    fig.legend(
        handles=[
            Patch(facecolor=COLOR_NUMERICA, label="predictora numérica  (se estandariza)"),
            Patch(facecolor=COLOR_CATEGORICA, label="predictora categórica  (one-hot)"),
            Patch(facecolor=COLOR_OBJETIVO, label="objetivo: lo que el modelo predice"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, fontsize=10.5,
        frameon=False, handlelength=1.6, handleheight=1.0, columnspacing=2.4,
    )
    fig.text(0.5, 0.905, "línea punteada = mediana", ha="center", va="top",
             fontsize=10, color=TINTA_SECUNDARIA)
    fig.tight_layout(rect=(0, 0, 1, 0.895))
    return _guardar(fig, "07-histogramas.png")


def figura_histograma_poblaciones():
    df = cargar_train()
    fumador = df["smoker"] == "yes"
    obeso = df["bmi"] > 30

    grupos = [
        (~fumador, "no fumador", COLOR_NO_FUMADOR),
        (fumador & ~obeso, "fumador, bmi ≤ 30", COLOR_FUMADOR_CLARO),
        (fumador & obeso, "fumador, bmi > 30", COLOR_FUMADOR_OSCURO),
    ]

    bordes = np.histogram_bin_edges(df[OBJETIVO], bins=_bins_freedman_diaconis(df[OBJETIVO]))

    fig, (ax_todo, ax_sep) = plt.subplots(
        2, 1, figsize=(9, 7.4), sharex=True, gridspec_kw={"height_ratios": [1, 1.5]}
    )

    ax_todo.hist(df[OBJETIVO], bins=bordes, color=COLOR_OBJETIVO, edgecolor=SUPERFICIE, linewidth=0.4)
    ax_todo.set_ylabel("Cantidad de personas")
    ax_todo.set_title("Lo que se ve en la figura 7: un pico grande, y dos jorobas "
                      "más a la derecha", fontsize=12, pad=10)
    for x, texto in ((20000, "¿segundo\npico?"), (41000, "¿tercero?")):
        ax_todo.annotate(texto, xy=(x, 0.55), xycoords=("data", "axes fraction"),
                         ha="center", va="bottom", fontsize=10.5, style="italic",
                         color=TINTA_SECUNDARIA)

    etiquetas = [
        f"{etq}  —  n={int(m.sum())}, mediana \\${_num(df.loc[m, OBJETIVO].median(), 0)}"
        for m, etq, _ in grupos
    ]
    ax_sep.hist(
        [df.loc[m, OBJETIVO] for m, _, _ in grupos],
        bins=bordes, stacked=True,
        color=[c for _, _, c in grupos],
        label=etiquetas,
    )
    ax_sep.set_ylabel("Cantidad de personas")
    ax_sep.set_xlabel("Costo médico (charges, dólares)")
    ax_sep.set_title("El mismo histograma separado por población: cada joroba es un grupo",
                     fontsize=12, pad=10)
    ax_sep.legend(loc="upper right", fontsize=10)

    minimo_tercero = df.loc[fumador & obeso, OBJETIVO].min()
    pct_debajo = 100 * (df.loc[~(fumador & obeso), OBJETIVO] < minimo_tercero).mean()
    ax_sep.axvline(minimo_tercero, color=COLOR_FUMADOR_OSCURO, linestyle="--", linewidth=1.5)
    ax_sep.set_ylim(0, ax_sep.get_ylim()[1] * 1.34)
    ax_sep.annotate(
        f"desde \\${_num(minimo_tercero, 0)} para arriba\n"
        f"casi sólo hay fumadores obesos:\nel {_num(pct_debajo, 1)} % del resto queda a la izquierda",
        xy=(minimo_tercero, 0.97), xycoords=("data", "axes fraction"),
        xytext=(-10, 0), textcoords="offset points", ha="right", va="top",
        fontsize=10, color=TINTA_PRIMARIA, linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=SUPERFICIE,
                  edgecolor=COLOR_EJE, linewidth=0.8),
    )

    for ax in (ax_todo, ax_sep):
        ax.set_xlim(0, df[OBJETIVO].max() * 1.02)
    ax_sep.set_xticks(np.arange(0, 70000, 10000))
    ax_sep.set_xticklabels(["0"] + [f"{v}k" for v in range(10, 70, 10)])

    fig.suptitle("charges no es una distribución con cola: son tres poblaciones",
                 fontsize=13.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return _guardar(fig, "08-charges-poblaciones.png")


def main():
    rutas = [
        figura_curvas_train_val(),
        figura_camino_lasso(),
        figura_interaccion_smoker_bmi(),
        figura_outliers_charges(),
        figura_sensibilidad_k(),
        figura_histogramas(),
        figura_histograma_poblaciones(),
    ]
    for ruta in rutas:
        tamano_kb = os.path.getsize(ruta) / 1024
        print(f"{ruta}  ({tamano_kb:.1f} KB)")


if __name__ == "__main__":
    main()
