"""Correr con: python -m src.graficos_presentacion"""

import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.datos import OBJETIVO, cargar_train
from src.graficos import (
    COLOR_EJE,
    COLOR_FUMADOR,
    COLOR_NO_FUMADOR,
    COLOR_REFERENCIA,
    COLOR_TRAIN,
    COLOR_VAL,
    RUTA_FIGURAS,
    RUTA_RESULTADOS,
    SUPERFICIE,
    TINTA_PRIMARIA,
    TINTA_SECUNDARIA,
    _leer_csv,
)

RUTA_CUADROS = os.path.join(RUTA_FIGURAS, "presentacion")


ASPECTO_SLIDE = (10.6, 5.0)

COLOR_SIN_CLASIFICAR = "#9c9a94"


def _guardar(fig, nombre):
    os.makedirs(RUTA_CUADROS, exist_ok=True)
    ruta = os.path.join(RUTA_CUADROS, nombre)
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=SUPERFICIE)
    plt.close(fig)
    return ruta


def secuencia_curva_en_u():
    filas = _leer_csv("cv_lineal.csv")
    grados = np.array([int(f["grado"]) for f in filas])
    tr_m = np.array([float(f["rmse_train_medio"]) for f in filas])
    tr_s = np.array([float(f["rmse_train_desvio"]) for f in filas])
    va_m = np.array([float(f["rmse_val_medio"]) for f in filas])
    va_s = np.array([float(f["rmse_val_desvio"]) for f in filas])

    en_regimen = va_m < 2.5 * float(np.median(va_m))
    y_lo = min((va_m[en_regimen] - va_s[en_regimen]).min(), (tr_m - tr_s).min()) - 250
    y_hi = max((va_m[en_regimen] + va_s[en_regimen]).max(), (tr_m + tr_s).max()) + 550
    fuera = ~en_regimen

    rutas = []
    for paso in (1, 2, 3, 4):
        fig, ax = plt.subplots(figsize=ASPECTO_SLIDE)

        ax.plot(grados, tr_m, "o-", color=COLOR_TRAIN, linewidth=2.2, markersize=8,
                label="RMSE de entrenamiento", zorder=3,
                markeredgecolor=SUPERFICIE, markeredgewidth=1.5)
        if paso >= 3:
            ax.fill_between(grados, tr_m - tr_s, tr_m + tr_s, color=COLOR_TRAIN, alpha=0.16)

        if paso >= 2:
            ax.plot(grados[en_regimen], va_m[en_regimen], "o-", color=COLOR_VAL,
                    linewidth=2.2, markersize=8, label="RMSE de validación", zorder=3,
                    markeredgecolor=SUPERFICIE, markeredgewidth=1.5)
        if paso >= 3:
            ax.fill_between(grados[en_regimen], (va_m - va_s)[en_regimen],
                            (va_m + va_s)[en_regimen], color=COLOR_VAL, alpha=0.16)

        if paso >= 4:
            i = int(np.argmin(va_m))
            ax.plot(grados[i], va_m[i], marker="*", color=COLOR_VAL, markersize=24,
                    zorder=6, markeredgecolor=SUPERFICIE, markeredgewidth=1.6)
            ax.annotate(f"mínimo de validación\ngrado {grados[i]}: "
                        + f"\\${va_m[i]:,.0f}".replace(",", "."),
                        xy=(grados[i], va_m[i]), xytext=(grados[i] + 0.18, va_m[i] + 780),
                        fontsize=11, color=TINTA_PRIMARIA, ha="left",
                        arrowprops=dict(arrowstyle="->", color=TINTA_SECUNDARIA, lw=1.2))
            ax.text(1.0, y_hi - 120, "← subajuste, fuera del rango", fontsize=11,
                    style="italic", color=TINTA_SECUNDARIA, ha="left", va="top")
            x_sobreajuste = float(grados[en_regimen][-1])
            ax.text(x_sobreajuste, y_hi - 120, "sobreajuste", fontsize=11, style="italic",
                    color=TINTA_SECUNDARIA, ha="center", va="top")

        if paso >= 2:
            for g, v in zip(grados[fuera], va_m[fuera]):
                ax.annotate(
                    f"grado {g}: \\${v:,.0f}".replace(",", ".") + "\nfuera de escala",
                    xy=(g, y_hi), xytext=(-10, -8), textcoords="offset points",
                    ha="right", va="top", fontsize=11.5, color=COLOR_VAL,
                    fontweight="bold",
                )
                ax.annotate("", xy=(g, y_hi), xytext=(g, y_hi - (y_hi - y_lo) * 0.30),
                            arrowprops=dict(arrowstyle="-|>", color=COLOR_VAL, lw=2.4))

        i_min = int(np.argmin(va_m))
        titulos = {
            1: "El error de entrenamiento baja siempre",
            2: "…pero el de validación deja de acompañarlo",
            3: "Y en grado 4 además se vuelve inestable",
            4: (f"Sin regularizar, el mínimo de validación está en grado {grados[i_min]}"
                + (": el modelo más simple" if grados[i_min] == 1 else "")),
        }
        ax.set_title(titulos[paso], fontsize=14, color=TINTA_PRIMARIA, pad=12)
        ax.set_xlabel("Grado del polinomio")
        ax.set_ylabel("RMSE (dólares)")
        ax.set_xticks(grados)
        ax.set_ylim(y_lo, y_hi)
        ax.legend(loc="lower left")
        fig.tight_layout()
        rutas.append(_guardar(fig, f"curva-u-{paso}.png"))
    return rutas


def secuencia_interaccion():
    df = cargar_train()
    bmi = df["bmi"].to_numpy(float)
    y = df[OBJETIVO].to_numpy(float)
    fuma = (df["smoker"] == "yes").to_numpy()

    x_lo, x_hi = bmi.min() - 1, bmi.max() + 1
    y_lo, y_hi = -2000, y.max() * 1.06

    rutas = []
    for paso in (1, 2, 3):
        fig, ax = plt.subplots(figsize=ASPECTO_SLIDE)

        if paso == 1:
            ax.scatter(bmi, y, s=16, color=COLOR_SIN_CLASIFICAR, alpha=0.55,
                       edgecolors="none")
        else:
            ax.scatter(bmi[~fuma], y[~fuma], s=16, color=COLOR_NO_FUMADOR, alpha=0.55,
                       edgecolors="none", label="no fumador")
            ax.scatter(bmi[fuma], y[fuma], s=16, color=COLOR_FUMADOR, alpha=0.6,
                       edgecolors="none", label="fumador")

        if paso == 3:
            xs = np.linspace(x_lo, x_hi, 50)
            for mascara, color, etiqueta in ((~fuma, COLOR_NO_FUMADOR, "no fumador"),
                                             (fuma, COLOR_FUMADOR, "fumador")):
                pend, orden = np.polyfit(bmi[mascara], y[mascara], 1)
                ax.plot(xs, pend * xs + orden, color=color, linewidth=3,
                        label=f"{etiqueta}: {pend:,.0f} \\$/bmi", zorder=4)

        titulos = {
            1: "bmi contra charges: correlación de sólo 0,194",
            2: "La misma nube, separada por fumador",
            3: "Dos pendientes: 83 contra 1.473 \\$/bmi",
        }
        ax.set_title(titulos[paso], fontsize=14, color=TINTA_PRIMARIA, pad=12)
        ax.set_xlabel("Índice de masa corporal (bmi)")
        ax.set_ylabel("Costo médico (dólares)")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        if paso >= 2:
            ax.legend(loc="upper left", framealpha=1.0)
        fig.tight_layout()
        rutas.append(_guardar(fig, f"interaccion-{paso}.png"))
    return rutas


def secuencia_una_es():
    lineal = _leer_csv("cv_lineal.csv")
    lasso = _leer_csv("cv_lasso.csv")

    NO_CONVERGE = {(3, "0.003"), (4, "0.003")}

    puntos = []
    for f in lineal:
        puntos.append((f"lineal g{f['grado']}", float(f["rmse_val_medio"]), int(f["grado"]),
                       None, False))
    for f in lasso:
        if f.get("rmse_val_medio"):
            grado = int(f["grado"])
            no_convergio = (grado, f.get("frac_lambda")) in NO_CONVERGE
            puntos.append((f"lasso g{grado} λ={float(f['lambda']):.0f}",
                           float(f["rmse_val_medio"]), grado, float(f["lambda"]),
                           no_convergio))

    puntos.sort(key=lambda p: p[1])
    puntos = puntos[:10]
    etiquetas = [p[0] for p in puntos]
    valores = np.array([p[1] for p in puntos])
    descartada = np.array([p[4] for p in puntos])

    with open(os.path.join(RUTA_RESULTADOS, "modelo_elegido.json")) as fh:
        modelo_elegido = json.load(fh)
    umbral = modelo_elegido["umbral_1se"]

    rutas = []
    for paso in (1, 2):
        fig, ax = plt.subplots(figsize=ASPECTO_SLIDE)
        pos = np.arange(len(valores))[::-1]

        dentro = (valores <= umbral) & ~descartada
        if paso == 1:
            colores = [COLOR_TRAIN] * len(valores)
        else:
            colores = [COLOR_VAL if d else COLOR_TRAIN for d in dentro]
            ax.axvspan(valores.min() - 12, umbral, color=COLOR_VAL, alpha=0.10, zorder=0)
            ax.axvline(umbral, color=COLOR_REFERENCIA, linestyle="--", linewidth=1.5, zorder=1)
            ax.text(umbral, pos[0] + 0.55, "  umbral de 1 error estándar",
                    fontsize=10.5, color=TINTA_SECUNDARIA, va="center", ha="left")

        alfas = [0.35 if d else 1.0 for d in descartada]
        ax.scatter(valores, pos, s=90, c=colores, zorder=3, alpha=alfas,
                   edgecolors=SUPERFICIE, linewidths=1.5)
        for x, y, d in zip(valores, pos, descartada):
            if d:
                ax.annotate("descartada: no convergió", xy=(x, y), xytext=(10, 0),
                            textcoords="offset points", fontsize=9, style="italic",
                            color=TINTA_SECUNDARIA, va="center", ha="left")

        ax.set_yticks(pos)
        ax.set_yticklabels(etiquetas, fontsize=10)
        ax.set_xlabel("RMSE de validación (dólares)")
        ax.set_xlim(valores.min() - 12, valores.max() + 40)
        ax.set_ylim(pos.min() - 0.7, pos.max() + 1.1)
        ax.grid(axis="y", visible=False)
        titulos = {
            1: "Las 10 mejores configuraciones, ordenadas",
            2: f"{int(dentro.sum())} caen dentro de 1 error estándar: son indistinguibles",
        }
        ax.set_title(titulos[paso], fontsize=14, color=TINTA_PRIMARIA, pad=12)
        fig.tight_layout()
        rutas.append(_guardar(fig, f"una-es-{paso}.png"))
    return rutas


def camino_lasso_apaisado():
    filas = _leer_csv("cv_lasso.csv")
    por_grado = {}
    for f in filas:
        if not f.get("rmse_val_medio"):
            continue
        por_grado.setdefault(int(f["grado"]), []).append(
            (float(f["lambda"]), float(f["rmse_val_medio"]), float(f["coefs_no_nulos_medio"]))
        )

    colores = {2: "#86b6ef", 3: "#2a78d6", 4: "#104281"}

    fig, (ax_err, ax_coef) = plt.subplots(1, 2, figsize=(11.6, 4.6))

    for grado in sorted(por_grado):
        datos = sorted(por_grado[grado])
        lam = [d[0] for d in datos]
        err = [d[1] for d in datos]
        coef = [d[2] for d in datos]
        for ax, valores in ((ax_err, err), (ax_coef, coef)):
            ax.plot(lam, valores, "o-", color=colores[grado], linewidth=2.2, markersize=7,
                    label=f"grado {grado}", markeredgecolor=SUPERFICIE, markeredgewidth=1.4)

    for ax in (ax_err, ax_coef):
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("$\\lambda$  (log, invertido)")

    ax_err.set_ylabel("RMSE de validación (dólares)")
    ax_err.set_title("El error baja y vuelve a subir", fontsize=12.5, color=TINTA_PRIMARIA)
    ax_err.legend(title="grado del polinomio", loc="upper right", fontsize=9.5,
                  title_fontsize=9.5)

    ax_coef.set_ylabel("Coeficientes no nulos (promedio)")
    ax_coef.set_title("Al aflojar $\\lambda$ entran más features", fontsize=12.5,
                      color=TINTA_PRIMARIA)

    fig.tight_layout()
    return [_guardar(fig, "camino-lasso-apaisado.png")]


def main():
    rutas = (secuencia_curva_en_u() + secuencia_interaccion() + secuencia_una_es()
             + camino_lasso_apaisado())
    for ruta in rutas:
        from PIL import Image
        try:
            im = Image.open(ruta)
            aspecto = f"  aspecto {im.width / im.height:.2f}"
        except Exception:
            aspecto = ""
        print(f"{ruta}  ({os.path.getsize(ruta) / 1024:.1f} KB){aspecto}")


if __name__ == "__main__":
    main()
