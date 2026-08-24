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


def _leer_cv_lineal():
    filas = _leer_csv("cv_lineal.csv")
    return (
        np.array([int(f["grado"]) for f in filas]),
        np.array([float(f["rmse_train_medio"]) for f in filas]),
        np.array([float(f["rmse_train_desvio"]) for f in filas]),
        np.array([float(f["rmse_val_medio"]) for f in filas]),
        np.array([float(f["rmse_val_desvio"]) for f in filas]),
    )


def _leyenda_arriba(ax):
    """Leyenda en una fila, ARRIBA y afuera del area de datos.

    Adentro no hay lugar honesto: en el grafico completo el unico hueco libre
    esta justo donde van los rotulos de subajuste/sobreajuste, y en el zoom es
    el mismo hueco que pide la anotacion del minimo. Afuera no compite con
    nada, y ademas los dos graficos quedan con la leyenda en el mismo lado,
    que es lo que permite leer al segundo como continuacion del primero.
    """
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
              frameon=False, fontsize=11)


def secuencia_curva_en_u():
    """Los CUATRO grados en escala, en tres pasos.

    Antes el grado 4 (87.917, veinte veces el del grado 1) quedaba fuera del
    rango del eje y se lo reemplazaba por una flecha con el rotulo "fuera de
    escala". Eso obligaba a explicar el grafico antes de poder leerlo, y dejaba
    la unica linea que importa —la de validacion— cortada justo donde el
    argumento se vuelve concluyente.

    La escala logaritmica en y los mete a los cuatro sin aplastar a los tres
    primeros: 3.379 a 87.917 son 1,4 decadas. Ademas la banda de +-1 desvio del
    grado 4 (+-47.000) pasa a verse como lo que es, una franja enorme al lado
    de las otras tres, que es exactamente el argumento de la inestabilidad.

    Lo que este grafico ya NO hace es elegir: la estrella del minimo y su
    anotacion se mudaron a secuencia_zoom_sin_grado_4(). Aca la log es la
    escala correcta para mostrar el desastre del grado 4, pero es la incorrecta
    para comparar 4.413 contra 4.517 —los aplasta al mismo punto—, y elegir es
    justamente comparar esos dos. Un grafico muestra el rango; el otro elige.
    """
    grados, tr_m, tr_s, va_m, va_s = _leer_cv_lineal()

    # El piso y el techo salen de los datos con sus bandas, con un margen
    # multiplicativo (no aditivo) porque el eje es logaritmico.
    y_lo = min((tr_m - tr_s).min(), (va_m - va_s).min()) / 1.35
    y_hi = (va_m + va_s).max() * 1.5

    rutas = []
    for paso in (1, 2, 3):
        fig, ax = plt.subplots(figsize=ASPECTO_SLIDE)

        ax.plot(grados, tr_m, "o-", color=COLOR_TRAIN, linewidth=2.2, markersize=8,
                label="RMSE de entrenamiento", zorder=3,
                markeredgecolor=SUPERFICIE, markeredgewidth=1.5)
        if paso >= 3:
            ax.fill_between(grados, tr_m - tr_s, tr_m + tr_s, color=COLOR_TRAIN, alpha=0.16)

        if paso >= 2:
            ax.plot(grados, va_m, "o-", color=COLOR_VAL,
                    linewidth=2.2, markersize=8, label="RMSE de validación", zorder=3,
                    markeredgecolor=SUPERFICIE, markeredgewidth=1.5)
        if paso >= 3:
            ax.fill_between(grados, va_m - va_s, va_m + va_s, color=COLOR_VAL, alpha=0.16)
            # En coordenadas de ejes: el eje es logaritmico y estos dos rotulos
            # marcan ZONAS del eje x, no valores de y.
            ax.text(0.02, 0.97, "← subajuste, fuera del rango", fontsize=11,
                    style="italic", color=TINTA_SECUNDARIA, ha="left", va="top",
                    transform=ax.transAxes)
            ax.text(0.98, 0.97, "sobreajuste", fontsize=11, style="italic",
                    color=TINTA_SECUNDARIA, ha="right", va="top",
                    transform=ax.transAxes)

        ax.set_xlabel("Grado del polinomio")
        ax.set_ylabel("RMSE (dólares, escala log)")
        ax.set_xticks(grados)
        ax.set_yscale("log")
        ax.set_ylim(y_lo, y_hi)
        # Ticks elegidos a mano: los que pone matplotlib en log son 10^4 y nada
        # mas, y los cuatro numeros que importan viven entre 3.000 y 90.000.
        marcas = [4000, 6000, 10000, 20000, 40000, 90000]
        ax.set_yticks([m for m in marcas if y_lo <= m <= y_hi])
        ax.set_yticklabels([f"{m:,.0f}".replace(",", ".")
                            for m in marcas if y_lo <= m <= y_hi])
        ax.minorticks_off()
        _leyenda_arriba(ax)
        fig.tight_layout()
        rutas.append(_guardar(fig, f"curva-u-{paso}.png"))
    return rutas


def secuencia_zoom_sin_grado_4():
    """El mismo grafico SIN el grado 4, en escala lineal, y ahi se elige.

    El grafico anterior tiene que ser logaritmico para que el grado 4 entre, y
    esa escala miente sobre lo unico que despues hay que decidir: en log, 4.413
    (grado 1) y 4.517 (grado 2) son el mismo punto. Sacando el grado 4 el rango
    baja a 3.900-7.600, la lineal vuelve a servir, y recien ahi se ve que entre
    los dos primeros grados hay 103 dolares y que el 3 ya se fue a 6.758.

    Por eso la eleccion del grado 1 —la estrella y su anotacion— se anima aca y
    no alla: se elige sobre el grafico donde la diferencia se ve.

    Dos pasos: la curva sola, y despues el minimo marcado.
    """
    grados, tr_m, tr_s, va_m, va_s = _leer_cv_lineal()

    dentro = grados <= 3
    grados, tr_m, tr_s, va_m, va_s = (a[dentro] for a in (grados, tr_m, tr_s, va_m, va_s))

    y_lo = (tr_m - tr_s).min() - 300
    y_hi = (va_m + va_s).max() + 500

    rutas = []
    for paso in (1, 2):
        fig, ax = plt.subplots(figsize=ASPECTO_SLIDE)

        ax.plot(grados, tr_m, "o-", color=COLOR_TRAIN, linewidth=2.2, markersize=8,
                label="RMSE de entrenamiento", zorder=3,
                markeredgecolor=SUPERFICIE, markeredgewidth=1.5)
        ax.fill_between(grados, tr_m - tr_s, tr_m + tr_s, color=COLOR_TRAIN, alpha=0.16)

        ax.plot(grados, va_m, "o-", color=COLOR_VAL, linewidth=2.2, markersize=8,
                label="RMSE de validación", zorder=3,
                markeredgecolor=SUPERFICIE, markeredgewidth=1.5)
        ax.fill_between(grados, va_m - va_s, va_m + va_s, color=COLOR_VAL, alpha=0.16)

        if paso >= 2:
            i = int(np.argmin(va_m))
            ax.plot(grados[i], va_m[i], marker="*", color=COLOR_VAL, markersize=26,
                    zorder=6, markeredgecolor=SUPERFICIE, markeredgewidth=1.6)
            ax.annotate("mínimo de validación\n"
                        + f"grado {grados[i]}: \\${va_m[i]:,.0f}".replace(",", "."),
                        xy=(grados[i], va_m[i]), xytext=(30, 46),
                        textcoords="offset points",
                        fontsize=11.5, color=TINTA_PRIMARIA, ha="left", va="bottom",
                        arrowprops=dict(arrowstyle="->", color=TINTA_SECUNDARIA, lw=1.2))

        ax.set_xlabel("Grado del polinomio")
        ax.set_ylabel("RMSE (dólares)")
        ax.set_xticks(grados)
        ax.set_ylim(y_lo, y_hi)
        ax.set_yticklabels([f"{v:,.0f}".replace(",", ".") for v in ax.get_yticks()])
        _leyenda_arriba(ax)
        fig.tight_layout()
        rutas.append(_guardar(fig, f"curva-u-zoom-{paso}.png"))
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
        # Sin titulo: la frase narrativa ("N caen dentro de 1 error estandar")
        # la dice el presentador. Lo que el grafico tiene que mostrar solo es la
        # banda sombreada y el umbral, que ya estan dibujados.
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

    # Los dos paneles van SIN titulo: el rotulo del eje y de cada uno ya dice
    # que mide ("RMSE de validacion" contra "Coeficientes no nulos"), asi que el
    # titulo solo agregaba la lectura en prosa, que es la que se dice hablando.
    ax_err.set_ylabel("RMSE de validación (dólares)")
    ax_err.legend(title="grado del polinomio", loc="upper right", fontsize=9.5,
                  title_fontsize=9.5)

    ax_coef.set_ylabel("Coeficientes no nulos (promedio)")

    fig.tight_layout()
    return [_guardar(fig, "camino-lasso-apaisado.png")]


def variantes_sin_titulo():
    """Las dos figuras que el deck COMPARTE con el informe, sin titulo narrativo.

    Se generan aca y no en src/graficos.py porque son artefactos de la
    presentacion: el informe sigue usando `04-outliers-charges.png` y
    `08-charges-poblaciones.png` con sus titulos intactos. Las variantes van a
    figuras/presentacion/ con sufijo `-slide`, que es donde el deck busca todo
    lo suyo.
    """
    from src.graficos import figura_outliers_charges

    rutas = []
    for figura, nombre in ((figura_outliers_charges, "04-outliers-charges-slide.png"),):
        ruta_informe = figura(titulos=False, nombre=nombre)
        destino = os.path.join(RUTA_CUADROS, nombre)
        os.makedirs(RUTA_CUADROS, exist_ok=True)
        os.replace(ruta_informe, destino)
        rutas.append(destino)
    return rutas


def histogramas_charges_por_slide():
    """La figura 08, partida en DOS graficos apaisados, uno por slide.

    En el informe los dos paneles van apilados y funciona: la pagina es alta y
    el lector compara moviendo los ojos. Proyectado no. Apilados, cada panel se
    queda con media pantalla de 16:9 y las barras del histograma bajan a unos
    pocos milimetros: el hallazgo del deck —que charges son tres poblaciones—
    quedaba ilegible justo en la slide que lo anuncia.

    Separados, cada uno usa la pantalla entera con el aspecto que le conviene.
    Y el orden se vuelve el del relato: primero la pregunta (un histograma con
    dos jorobas raras), despues la respuesta (separado por poblacion).

    Los dos comparten eje x por construccion, porque los dibuja la misma
    funcion (ver _eje_charges en src/graficos.py). Eso importa mas aca que en
    el informe: en pantalla los graficos se suceden en el mismo lugar, asi que
    cualquier corrimiento del eje se leeria como un cambio en los datos.
    """
    from src.graficos import (
        bordes_charges,
        cargar_train,
        panel_charges_por_poblacion,
        panel_charges_total,
    )

    df = cargar_train()
    bordes = bordes_charges(df)

    rutas = []
    for panel, nombre in ((panel_charges_total, "08-charges-total-slide.png"),
                          (panel_charges_por_poblacion, "08-charges-separado-slide.png")):
        fig, ax = plt.subplots(figsize=ASPECTO_SLIDE)
        panel(ax, df, bordes)
        # El panel de arriba no trae xlabel (en el informe lo comparte con el de
        # abajo via sharex). Suelto lo necesita.
        if not ax.get_xlabel():
            ax.set_xlabel("Costo médico (charges, dólares)")
        fig.tight_layout()
        rutas.append(_guardar(fig, nombre))
    return rutas


def main():
    rutas = (secuencia_curva_en_u() + secuencia_zoom_sin_grado_4()
             + secuencia_interaccion() + secuencia_una_es()
             + camino_lasso_apaisado() + variantes_sin_titulo()
             + histogramas_charges_por_slide())
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
