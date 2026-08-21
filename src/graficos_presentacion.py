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

import json
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
    RUTA_RESULTADOS,
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

# El area util de una slide 16:9, descontando titulo y pie, tiene una relacion de
# aspecto de ~2,1. Una figura mas angosta que eso queda atada por el ALTO y deja
# franjas vacias a los costados: se ve chica aunque el archivo sea grande. Por eso
# los cuadros de presentacion se generan apaisados, a la medida de la pantalla, y no
# con las proporciones de las figuras del informe (que van en una pagina vertical).
ASPECTO_SLIDE = (10.6, 5.0)   # -> 2,12

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
    #
    # PERO desde D-27/D-28 el grado 4 se va a 87.917 con una banda de ±47.000, y encajarlo en el
    # eje aplasta los grados 1 a 3 —los tres que deciden el modelo— en una línea plana. En
    # el informe eso se resuelve con un eje partido; en una slide no: un eje partido pide
    # una explicación de diez segundos que no hay, y la defensa dura diez minutos.
    #
    # Acá el eje se calcula con los grados que estan EN régimen y el que se disparó se
    # rotula como fuera de escala, con una flecha y su valor. Para hablarlo es mejor: "el
    # grado 4 literalmente se sale del gráfico" es una frase que se entiende sola.
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
            # La curva termina en el último grado que entra en el eje: dibujar el tramo
            # 3->4 lo hacía salir por arriba y "volver" como flecha en x=4 — dos trazos
            # verticales desconectados. El grado 4 vive solo en el cartel con su flecha.
            ax.plot(grados[en_regimen], va_m[en_regimen], "o-", color=COLOR_VAL,
                    linewidth=2.2, markersize=8, label="RMSE de validación", zorder=3,
                    markeredgecolor=SUPERFICIE, markeredgewidth=1.5)
        if paso >= 3:
            # Sólo los grados en régimen: la banda del grado 4 (±47.000) no entra en el eje,
            # y el polígono clippeado entre 3 y 4 quedaba como una mancha vertical que parecía
            # un error de render. El desvío lo dice el pie de la slide; acá basta el cartel.
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
            # El grado 1 es el MINIMO de validacion con una brecha train-val de $62: zona de
            # buen ajuste (Clase 2, slides 53-55), no de subajuste. El subajuste real
            # quedaria en modelos con menos capacidad que el barrido no incluye, asi que
            # la etiqueta apunta fuera del rango en vez de pisar el punto optimo.
            ax.text(1.0, y_hi - 120, "← subajuste, fuera del rango", fontsize=11,
                    style="italic", color=TINTA_SECUNDARIA, ha="left", va="top")
            # "sobreajuste" se apoya en el ultimo grado que TODAVIA entra en el eje: en el
            # que se salio esta el cartel de fuera de escala y los dos textos se pisaban.
            x_sobreajuste = float(grados[en_regimen][-1])
            ax.text(x_sobreajuste, y_hi - 120, "sobreajuste", fontsize=11, style="italic",
                    color=TINTA_SECUNDARIA, ha="center", va="top")

        # Los grados que se salen del eje se rotulan con su valor y una flecha hacia
        # arriba, desde el paso en que su curva ya está dibujada.
        if paso >= 2:
            for g, v in zip(grados[fuera], va_m[fuera]):
                # El cartel va a la IZQUIERDA de la flecha (ha="right"): centrado en x=g la
                # flecha, que sube por x=g, atravesaba el texto.
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

    El umbral se LEE de resultados/modelo_elegido.json (mismo patrón que usa
    figura_sensibilidad_k en graficos.py): estaba hardcodeado a la corrida anterior a
    D-23 y quedó falso en cuanto el pipeline cambió de 8 a 9 columnas.
    """
    lineal = _leer_csv("cv_lineal.csv")
    lasso = _leer_csv("cv_lasso.csv")

    # D-20: dos configuraciones Lasso no convergieron en algún fold (informe/salida-
    # seleccion.txt, "PUNTO 5"): grado 3 y grado 4, ambas en frac_lambda=0.003 (el lambda
    # más chico probado). El CSV no guarda esa marca por fila, así que se identifican por
    # esa combinación — es la misma que usó `src/experimentos.py` para descartarlas.
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
    puntos = puntos[:10]  # sólo la parte de arriba de la tabla: el resto está lejos
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

        # Las descartadas por no converger (D-20) se atenúan y se marcan aparte: su RMSE
        # no es el del modelo Lasso, es dónde quedó la optimización al cortarla, y no
        # participan de la regla de 1 ES aunque su valor caiga dentro de la banda.
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


# --------------------------------------------------------------------------------------
# Camino de Lasso, versión apaisada
# --------------------------------------------------------------------------------------
def camino_lasso_apaisado():
    """La misma información que la figura del informe, pero en 1x2 en vez de 2x1.

    En el informe los dos paneles van apilados y comparten el eje x, que es lo correcto
    en una página vertical. En una slide 16:9 esa figura queda casi cuadrada (relación
    0,98 contra 1,78 de la pantalla): al ajustarla al ancho se desborda por abajo, y al
    ajustarla al alto queda diminuta con dos franjas vacías a los costados.

    Lado a lado la relación pasa a 2,6 y llena la slide. Se pierde el eje x compartido,
    asi que cada panel lleva el suyo rotulado.
    """
    filas = _leer_csv("cv_lasso.csv")
    por_grado = {}
    for f in filas:
        if not f.get("rmse_val_medio"):
            continue
        por_grado.setdefault(int(f["grado"]), []).append(
            (float(f["lambda"]), float(f["rmse_val_medio"]), float(f["coefs_no_nulos_medio"]))
        )

    # Rampa ordinal: el grado es una escala ordenada, no categorías nominales.
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
        ax.invert_xaxis()  # más regularizado a la izquierda
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
        from PIL import Image  # sólo para reportar la relación de aspecto
        try:
            im = Image.open(ruta)
            aspecto = f"  aspecto {im.width / im.height:.2f}"
        except Exception:
            aspecto = ""
        print(f"{ruta}  ({os.path.getsize(ruta) / 1024:.1f} KB){aspecto}")


if __name__ == "__main__":
    main()
