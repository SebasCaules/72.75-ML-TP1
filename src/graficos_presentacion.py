"""Cuadros de animación para la presentación (Beamer).

Las figuras de `graficos.py` son para el informe: aparecen completas, y el lector tiene
todo el tiempo del mundo para leerlas. En una presentación de 10 minutos eso no funciona:
un gráfico completo que aparece de golpe obliga al que escucha a decidir solo qué mirar,
y mientras lo decide se pierde lo que uno está diciendo.

Por eso este módulo genera CUADROS: versiones progresivas de la misma figura, que en
Beamer se muestran con \\only<n> una encima de otra. El eje, la escala y el tamaño se
mantienen fijos entre cuadros para que no haya ningún salto visual — lo único que cambia
es qué capa de información está presente.

Los colores y el cromo se importan de graficos.py: la paleta ya está validada allá y no
se re-elige acá.

Correr con:  python3 -m src.graficos_presentacion
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.datos import OBJETIVO, cargar
from src.graficos import (
    COLOR_EJE,
    COLOR_FUMADOR,
    COLOR_NO_FUMADOR,
    COLOR_REFERENCIA,
    COLOR_TRAIN,
    COLOR_VAL,
    RUTA_FIGURAS,
    SUPERFICIE,
    TINTA_PRIMARIA,
    TINTA_SECUNDARIA,
    _leer_csv,
)
from src.preproceso import quitar_duplicados

RUTA_CUADROS = os.path.join(RUTA_FIGURAS, "presentacion")

# Gris neutro para los datos "todavía sin clasificar" del primer cuadro de la secuencia
# de interacción: la idea es que el público vea la nube SIN la respuesta, para que la
# separación por fumador llegue como un hallazgo y no como un dato más.
COLOR_SIN_CLASIFICAR = "#9c9a94"


def _guardar(fig, nombre):
    os.makedirs(RUTA_CUADROS, exist_ok=True)
    ruta = os.path.join(RUTA_CUADROS, nombre)
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=SUPERFICIE)
    plt.close(fig)
    return ruta


# --------------------------------------------------------------------------------------
# Secuencia A — la curva en U, construida en 4 pasos
# --------------------------------------------------------------------------------------
def secuencia_curva_en_u():
    """4 cuadros: train -> +validación -> +bandas -> +mínimo y regiones.

    El orden no es decorativo. Mostrar primero SOLO la curva de entrenamiento deja al
    público con una conclusión falsa pero natural ("más grado es mejor"), que es
    exactamente la intuición que la clase quiere romper. Recién entonces entra la curva
    de validación y la contradice. Es la misma pedagogía del enunciado.
    """
    filas = _leer_csv("cv_lineal.csv")
    grados = np.array([int(f["grado"]) for f in filas])
    tr_m = np.array([float(f["rmse_train_medio"]) for f in filas])
    tr_s = np.array([float(f["rmse_train_desvio"]) for f in filas])
    va_m = np.array([float(f["rmse_val_medio"]) for f in filas])
    va_s = np.array([float(f["rmse_val_desvio"]) for f in filas])

    # Límites FIJOS, calculados una vez con todos los datos: si cada cuadro autoescalara,
    # los ejes saltarían entre pasos y la animación se volvería mareante.
    y_lo = min((va_m - va_s).min(), (tr_m - tr_s).min()) - 250
    y_hi = max((va_m + va_s).max(), (tr_m + tr_s).max()) + 550

    rutas = []
    for paso in (1, 2, 3, 4):
        fig, ax = plt.subplots(figsize=(8.4, 5.0))

        ax.plot(grados, tr_m, "o-", color=COLOR_TRAIN, linewidth=2.2, markersize=8,
                label="RMSE de entrenamiento", zorder=3,
                markeredgecolor=SUPERFICIE, markeredgewidth=1.5)
        if paso >= 3:
            ax.fill_between(grados, tr_m - tr_s, tr_m + tr_s, color=COLOR_TRAIN, alpha=0.16)

        if paso >= 2:
            ax.plot(grados, va_m, "o-", color=COLOR_VAL, linewidth=2.2, markersize=8,
                    label="RMSE de validación", zorder=3,
                    markeredgecolor=SUPERFICIE, markeredgewidth=1.5)
        if paso >= 3:
            ax.fill_between(grados, va_m - va_s, va_m + va_s, color=COLOR_VAL, alpha=0.16)

        if paso >= 4:
            i = int(np.argmin(va_m))
            ax.plot(grados[i], va_m[i], marker="*", color=COLOR_VAL, markersize=24,
                    zorder=6, markeredgecolor=SUPERFICIE, markeredgewidth=1.6)
            ax.annotate(f"mínimo de validación\ngrado {grados[i]}: \\${va_m[i]:,.0f}",
                        xy=(grados[i], va_m[i]), xytext=(grados[i] + 0.18, va_m[i] + 780),
                        fontsize=11, color=TINTA_PRIMARIA, ha="left",
                        arrowprops=dict(arrowstyle="->", color=TINTA_SECUNDARIA, lw=1.2))
            ax.text(1.0, y_hi - 120, "subajuste", fontsize=11, style="italic",
                    color=TINTA_SECUNDARIA, ha="left", va="top")
            ax.text(4.0, y_hi - 120, "sobreajuste", fontsize=11, style="italic",
                    color=TINTA_SECUNDARIA, ha="right", va="top")

        titulos = {
            1: "El error de entrenamiento baja siempre",
            2: "…pero el de validación deja de acompañarlo",
            3: "Y en grado 4 además se vuelve inestable",
            4: "El mejor compromiso está en grado 2",
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


# --------------------------------------------------------------------------------------
# Secuencia B — la interacción, construida en 3 pasos
# --------------------------------------------------------------------------------------
def secuencia_interaccion():
    """3 cuadros: nube sin clasificar -> coloreada por fumador -> con las dos pendientes.

    El primer cuadro es el importante: con los puntos todos del mismo gris, la relación
    entre bmi y charges parece débil y sin forma, que es justamente lo que dice la
    correlación de Pearson (0,198). Colorear por fumador revela que había DOS
    poblaciones superpuestas. Es el argumento de por qué una correlación baja no
    autoriza a descartar una variable.
    """
    df = quitar_duplicados(cargar())
    bmi = df["bmi"].to_numpy(float)
    y = df[OBJETIVO].to_numpy(float)
    fuma = (df["smoker"] == "yes").to_numpy()

    x_lo, x_hi = bmi.min() - 1, bmi.max() + 1
    y_lo, y_hi = -2000, y.max() * 1.06

    rutas = []
    for paso in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(8.4, 5.2))

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
            1: "bmi contra charges: correlación de sólo 0,198",
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


# --------------------------------------------------------------------------------------
# Secuencia C — la regla de 1 error estándar
# --------------------------------------------------------------------------------------
def secuencia_una_es():
    """2 cuadros: los RMSE de validación ordenados -> con la banda de 1 ES encima.

    Es un gráfico de puntos con barra de error, no de barras: los valores no arrancan en
    cero (van de 4900 a 5400) y una barra que no arranca en cero exagera las diferencias,
    que es exactamente lo contrario de lo que este gráfico quiere mostrar.
    """
    lineal = _leer_csv("cv_lineal.csv")
    lasso = _leer_csv("cv_lasso.csv")

    puntos = []
    for f in lineal:
        puntos.append((f"lineal g{f['grado']}", float(f["rmse_val_medio"]), int(f["grado"]), None))
    for f in lasso:
        if f.get("rmse_val_medio"):
            puntos.append((f"lasso g{f['grado']} λ={float(f['lambda']):.0f}",
                           float(f["rmse_val_medio"]), int(f["grado"]), float(f["lambda"])))

    puntos.sort(key=lambda p: p[1])
    puntos = puntos[:10]  # sólo la parte de arriba de la tabla: el resto está lejos
    etiquetas = [p[0] for p in puntos]
    valores = np.array([p[1] for p in puntos])

    mejor = valores[0]
    umbral = mejor + 100.3  # ES = 224,3 / sqrt(5)

    rutas = []
    for paso in (1, 2):
        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        pos = np.arange(len(valores))[::-1]

        dentro = valores <= umbral
        if paso == 1:
            colores = [COLOR_TRAIN] * len(valores)
        else:
            colores = [COLOR_VAL if d else COLOR_TRAIN for d in dentro]
            ax.axvspan(valores.min() - 12, umbral, color=COLOR_VAL, alpha=0.10, zorder=0)
            ax.axvline(umbral, color=COLOR_REFERENCIA, linestyle="--", linewidth=1.5, zorder=1)
            ax.text(umbral, pos[0] + 0.55, "  umbral de 1 error estándar",
                    fontsize=10.5, color=TINTA_SECUNDARIA, va="center", ha="left")

        ax.scatter(valores, pos, s=90, c=colores, zorder=3,
                   edgecolors=SUPERFICIE, linewidths=1.5)

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


def main():
    for ruta in secuencia_curva_en_u() + secuencia_interaccion() + secuencia_una_es():
        print(f"{ruta}  ({os.path.getsize(ruta) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
