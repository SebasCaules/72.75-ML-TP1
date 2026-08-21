"""Figuras de la presentacion del TP1 (regresion).

Cada función genera UNA figura y la guarda en figuras/. Los datos de validación cruzada
salen de resultados/cv_lineal.csv y resultados/cv_lasso.csv (ya calculados por
src.experimentos, que NO se vuelve a correr aca: tarda varios minutos). Las figuras que
necesitan el dataset crudo usan src.datos.cargar(), que es instantaneo. La unica figura
que reentrena algo es la 5, que vive en evaluar_test.py (D-21),
que es rapido (44 features, no 494).

Correr con:  python3 -m src.graficos
"""

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")  # backend headless: nunca se abre una ventana

import matplotlib.pyplot as plt
import numpy as np

from src.datos import OBJETIVO, cargar, outliers_iqr
from src.preproceso import quitar_duplicados
from src.validacion import rmse, separar_train_test

RUTA_FIGURAS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figuras")
RUTA_RESULTADOS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resultados")

# Este modulo NO entrena nada: dibuja a partir de los CSV de resultados/ y del dataset
# crudo. Hasta D-21 tenia importados el Lasso, el preprocesamiento y las constantes del
# modelo de produccion, porque la figura 5 vivia aca; cuando esa figura se mudo a
# `evaluar_test.py` quedaron sin uso, y ademas quedaron congeladas en el lambda y el grado
# de la corrida vieja. Una constante muerta que ademas esta desactualizada es peor que
# ninguna: parece autoridad. Se borraron.

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


def _guardar(fig, nombre_archivo, dpi=150):
    """Guarda en figuras/. El dpi es aparte del tamaño: las figuras que van a una slide se
    dibujan CHICAS (para que su cuerpo de letra no se achique al escalarlas) y necesitan
    dpi alto para no verse blandas proyectadas."""
    os.makedirs(RUTA_FIGURAS, exist_ok=True)
    ruta = os.path.join(RUTA_FIGURAS, nombre_archivo)
    fig.savefig(ruta, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return ruta


# --------------------------------------------------------------------------------------
# Figura 1 — curvas de train y validacion contra el grado del polinomio
# --------------------------------------------------------------------------------------
def figura_curvas_train_val():
    """La curva del punto 5, con EJE PARTIDO.

    Desde D-27/D-28 el grado 4 sin regularizar se va a 87.917 dolares de RMSE de validacion
    —con 1364 columnas, rango 436 y 856 filas por fold, el ajuste es una extrapolacion
    salvaje— mientras los grados 1 a 3 viven entre 4.400 y 5.000. En un solo par de ejes
    ese punto se come toda la escala: los tres grados que importan quedan aplastados en una
    linea plana y la figura deja de mostrar lo unico que tiene que mostrar, que es donde
    esta el minimo de validacion.

    La alternativa era escala logaritmica, pero un eje log en dolares hace ilegible una
    diferencia de 500 dolares justo en la zona donde se decide el modelo. El eje partido
    conserva la escala lineal en las dos regiones y no miente sobre ninguna: las marcas de
    corte avisan explicitamente que el eje esta interrumpido.
    """
    filas = _leer_csv("cv_lineal.csv")
    grados = np.array([int(f["grado"]) for f in filas])
    train_m = np.array([float(f["rmse_train_medio"]) for f in filas])
    train_s = np.array([float(f["rmse_train_desvio"]) for f in filas])
    val_m = np.array([float(f["rmse_val_medio"]) for f in filas])
    val_s = np.array([float(f["rmse_val_desvio"]) for f in filas])

    # Que un grado se dispare no es una constante del problema: depende de la corrida. El
    # corte se decide con los datos —mas de 2,5 veces la mediana de los RMSE de
    # validacion— y si NINGUNO lo supera, la figura vuelve a ser de un solo panel. Fijar
    # "el grado 4 explota" seria escribir en el codigo un resultado que el codigo tiene
    # que descubrir.
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

        # Marcas de eje partido: dos diagonales en el borde interior de cada panel. Sin
        # ellas el lector lee dos graficos, no un eje interrumpido.
        ax_alto.spines["bottom"].set_visible(False)
        ax_bajo.spines["top"].set_visible(False)
        ax_alto.tick_params(axis="x", bottom=False, labelbottom=False)
        kw = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=9, linestyle="none",
                  color=TINTA_SECUNDARIA, mec=TINTA_SECUNDARIA, mew=1.2, clip_on=False)
        ax_alto.plot([0, 1], [0, 0], transform=ax_alto.transAxes, **kw)
        ax_bajo.plot([0, 1], [1, 1], transform=ax_bajo.transAxes, **kw)

    # El minimo de validacion, que es la respuesta al punto 5.1 del enunciado.
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
    """Boxplot y histograma APILADOS sobre un eje x compartido.

    Antes iban lado a lado, y era la causa de que el panel izquierdo no se entendiera: las
    dos mitades miden LA MISMA VARIABLE en los mismos dolares, pero al estar una al lado de
    la otra —una vertical y la otra horizontal— el lector tenia que reconstruir a mano que
    el eje y de la izquierda era el eje x de la derecha. Apilados y compartiendo el eje, el
    umbral es UNA sola vertical que cruza los dos paneles y se lee de arriba abajo: aca esta
    la caja, aca el corte, y aca quienes quedaron del otro lado.

    El boxplot va horizontal por eso mismo (para compartir el eje), y ademas porque asi hay
    ancho para rotular Q1, la mediana y Q3 sin encimarlos.
    """
    df = quitar_duplicados(cargar())
    _, lim_inf, lim_sup = outliers_iqr(df[OBJETIVO])

    # Los tres numeros de los titulos se CALCULAN. Estaban escritos a mano ("139", "10.4 %",
    # "97.8 %") y coincidian, pero si cambiara el criterio IQR, el dedup o el dataset, la
    # figura seguiria afirmandolos sin que nadie se entere. Un numero hardcodeado en un
    # grafico es una mentira esperando el momento.
    es_out = (df[OBJETIVO] < lim_inf) | (df[OBJETIVO] > lim_sup)
    n_out = int(es_out.sum())
    pct_out = 100 * n_out / len(df)
    pct_fuman = 100 * (df.loc[es_out, "smoker"] == "yes").mean()

    # Formato de moneda a la española: punto para los miles. Se hace con una funcion y no
    # con un .replace(",", ".") sobre la frase entera, que era lo que habia: ese reemplazo
    # ciego tambien convertia la coma DECIMAL de "1,5" en un punto.
    def pesos(v):
        return "$" + f"{v:,.0f}".replace(",", ".")

    q1, mediana, q3 = df[OBJETIVO].quantile([0.25, 0.50, 0.75])
    iqr = q3 - q1
    fumador = df["smoker"] == "yes"

    fig, (ax_box, ax_hist) = plt.subplots(
        # Apaisada a proposito: la figura tiene que entrar en una slide 16:9 con una linea
        # de texto debajo. Mas alta que esto y la cota que ata pasa a ser el alto, con lo
        # cual sobra ancho a los costados y la figura se lee diminuta proyectada.
        # Aspecto ~2,5 a proposito. En una slide 16:9, descontados titulo, regla, una linea
        # de pie y el pie de pagina, el hueco util tiene mas o menos esa proporcion: con una
        # figura mas alta la cota que ata pasa a ser el alto, sobran margenes a los costados
        # y la figura se lee diminuta proyectada.
        # Dibujada CHICA a proposito. La figura entra en la slide a ~5,2 pulgadas de ancho:
        # dibujarla a 12" la reduce al 43 % y su tipografia termina en 5 pt proyectada, que
        # es ilegible desde la tercera fila. A 7,8" la reduccion es del 67 % y el mismo
        # cuerpo de letra queda en ~7,5 pt. El aspecto (~2,5) no cambia.
        # Tamaño de INFORME: se muestra a 6,3" en una A4 y el lector tiene todo el tiempo
        # del mundo. La version para proyectar es otra —mas grande de letra y con la mitad
        # de los rotulos— y vive en graficos_presentacion.figura_outliers_slide().
        2, 1, figsize=(8.2, 3.5), sharex=True, gridspec_kw={"height_ratios": [1, 1.45]}
    )

    # ------------------------------------------------------------------ panel superior
    # La caja va en tinta neutra A PROPOSITO. En este deck el azul significa "no fumador"
    # y el naranja "fumador"; pintar la caja de cualquiera de los dos le haria decir algo
    # sobre fumar que la caja no dice. El color sigue a la entidad, y la caja no es una
    # entidad: es el 50 % del medio de TODA la muestra.
    ax_box.boxplot(
        df[OBJETIVO], vert=False, widths=0.42, showfliers=False,
        boxprops=dict(color=TINTA_SECUNDARIA, linewidth=1.3),
        medianprops=dict(color=TINTA_PRIMARIA, linewidth=2.4),
        whiskerprops=dict(color=TINTA_SECUNDARIA, linewidth=1.3),
        capprops=dict(color=TINTA_SECUNDARIA, linewidth=1.3),
    )

    # Los outliers SI van coloreados, y por condicion de fumador — no por "ser outlier".
    # Asi el color mantiene el unico significado que tiene en todo el deck, y de paso el
    # panel muestra su propia conclusion: casi todos los puntos de la derecha son naranjas.
    y_out = 1 + np.random.default_rng(42).uniform(-0.13, 0.13, n_out)
    for marca, color, etiqueta in ((~fumador, COLOR_NO_FUMADOR, "no fumador"),
                                   (fumador, COLOR_FUMADOR, "fumador")):
        sel = (es_out & marca).values[es_out.values]
        ax_box.scatter(df.loc[es_out & marca, OBJETIVO], y_out[sel],
                       s=22, color=color, alpha=0.55, linewidths=0, zorder=3, label=etiqueta)

    # Los tres cuartiles van en UNA SOLA LINEA de resumen, no en tres rotulos colgados de
    # sus posiciones. En el eje x la mediana cae entre Q1 y Q3 y los tres textos se pisaban;
    # apilarlos en niveles distintos tampoco entraba, porque el panel es deliberadamente
    # bajo. Una linea corrida los ordena sin ambiguedad —el orden del texto ES el orden en
    # el eje— y deja el panel despejado.
    ax_box.text(700, 1.66,
                f"Q1 {pesos(q1)}   ·   mediana {pesos(mediana)}   ·   Q3 {pesos(q3)}",
                ha="left", va="center", fontsize=10, color=TINTA_SECUNDARIA)

    # Llave del IQR: es el unico rotulo que SI tiene que estar anclado a la geometria, porque
    # lo que dice es justamente que ese numero es el ancho de la caja.
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

    # ------------------------------------------------------------------ panel inferior
    # Histograma APILADO, no superpuesto. Superponer dos histogramas con transparencia crea
    # un TERCER color en la zona de solape que no esta en la leyenda y que el lector no sabe
    # interpretar. Apilados, la altura total es la cantidad de personas del bin y el tramo
    # naranja es la porcion de fumadores.
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

    # ------------------------------------------- el umbral cruza los dos paneles
    # Es el mismo numero en los dos, asi que se dibuja igual en los dos y queda leyendose
    # como una sola vertical continua. Punteado: en este deck eso significa umbral.
    for ax in (ax_box, ax_hist):
        ax.axvline(lim_sup, color=COLOR_REFERENCIA, linestyle="--", linewidth=1.5, zorder=1)

    # Eje compartido: marcas cada 10 000 rotuladas "10k" en vez de "10000". Las etiquetas
    # largas eran ilegibles proyectadas, y el orden de magnitud es lo unico que importa.
    # El eje ARRANCA EN 0 y no en el margen que matplotlib pone por defecto. Por defecto
    # agrega un 5 % del rango a cada lado; como el minimo es 1122, ese margen caia en
    # NEGATIVO, y un costo medico negativo no existe. Se recorta a [0, max] y de paso el
    # hueco entre 0 y el primer dato queda visible, que tambien es informacion.
    for ax in (ax_box, ax_hist):
        ax.set_xlim(0, df[OBJETIVO].max() * 1.02)
        # Marco completo en los dos paneles: sin el, con las espinas de arriba y derecha
        # apagadas por defecto, los dos graficos flotan y no se ve donde termina uno y
        # empieza el otro. Cada panel dice una cosa distinta y tiene que leerse como una
        # unidad cerrada.
        for lado in ("top", "right", "bottom", "left"):
            ax.spines[lado].set_visible(True)
            ax.spines[lado].set_color(COLOR_EJE)
            ax.spines[lado].set_linewidth(1.0)

    ax_hist.set_xticks(np.arange(0, 70000, 10000))
    ax_hist.set_xticklabels(["0"] + [f"{v}k" for v in range(10, 70, 10)])
    ax_hist.set_xlabel("Costo médico (charges, dólares)")

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
                    xytext=(-22, -58), textcoords="offset points", ha="right", va="top",
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
        xy=(k[-1], media_folds[-1]), xytext=(-52, 112), textcoords="offset points",
        ha="right", va="bottom", fontsize=12, color=TINTA_PRIMARIA,
        arrowprops=dict(arrowstyle="->", color=TINTA_PRIMARIA, lw=1),
    )

    # Limites DERIVADOS de los datos. Estaban fijados a mano en (2870, 5180), calzados a
    # la corrida vieja; con el modelo de produccion de D-23 la media de folds cae hasta
    # 2492 en LOO y ese punto quedaba recortado FUERA del eje, sin ningun aviso. Un eje
    # con limites hardcodeados es un grafico que en algun momento va a mentir en silencio.
    piso = float(min(media_folds.min(), agrupado.min()))
    techo = float(max(media_folds.max(), agrupado.max()))
    margen = 0.09 * (techo - piso)
    ax_sup.set_ylim(piso - margen, techo + margen)
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

    # La comparacion se CALCULA. Decia "menos de la mitad", que era cierto con los numeros
    # de la corrida anterior a D-23 (100,3 contra ~220) y dejo de serlo con los actuales
    # (164,1 contra el pico de 293,9). Una frase cualitativa hardcodeada envejece
    # igual que un numero hardcodeado, y encima es mas dificil de detectar.
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

    # La descripcion del modelo se LEE de resultados/modelo_elegido.json. Estaba escrita a
    # mano ("Lasso grado 2, lambda=286,37") y quedo falsa en cuanto D-23 cambio el modelo
    # elegido: la figura habria seguido rotulando un modelo que el TP ya no entrega.
    with open(os.path.join(RUTA_RESULTADOS, "modelo_elegido.json")) as fh:
        prod = json.load(fh)["produccion_1se"]
    desc = (f"{prod['modelo']} grado {prod['grado']}"
            + (f", $\\lambda={_num(prod['lambda'], 2)}$" if prod["lambda"] is not None
               else ", sin regularización"))
    fig.suptitle("Sensibilidad al número de folds — configuración de producción fija "
                 f"({desc})", fontsize=13.5, y=0.985)
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
# Figura 7 — histogramas de TODAS las variables (EDA, Clase 3)
# --------------------------------------------------------------------------------------
# La Clase 3 pide, antes de entrenar nada, "explorar y resumir el dataset" con dos
# herramientas juntas: estadisticas descriptivas (media, mediana, desvio, min, max) e
# histogramas, que "nos permiten entender mejor la distribucion de los datos" y detectar
# outliers y valores erroneos (Clase 3, slide 34). Por eso cada panel numerico lleva su
# linea de estadisticas ARRIBA del histograma: el deck las presenta como una sola lectura,
# no como dos laminas distintas.
#
# El deck da tres ejemplos de cosas que SOLO se ven en el histograma, y las tres tienen su
# analogo exacto en este dataset:
#   - slide 35: una variable que parecia continua y el histograma revela discretizada
#     -> aca es `children` (seis valores enteros), y la receta del deck es la que ya
#        aplica el TP: tratarla como discreta / one-hot en vez de como continua.
#   - slides 36-38: un segundo pico => DOS POBLACIONES distintas, y la instruccion es
#     "ver el histograma separado para cada poblacion" -> aca es `charges`, y ese
#     histograma separado es la figura 8.
#   - slide 39: dos sistemas de medida mezclados -> no ocurre en insurance (ningun panel
#     muestra dos escalas superpuestas), y decirlo tambien es un resultado del EDA.
#
# EL COLOR CODIFICA EL ROL DE LA VARIABLE EN EL TP, que es la unica particion que la
# figura puede afirmar sin inventar nada: los siete paneles no son siete series de una
# misma escala (no hay nada que comparar entre ellos), pero si son tres cosas distintas
# —lo que se predice, y los dos tipos de variable con los que se lo predice— y esa
# distincion es la que decide todo el preproceso: las numericas se estandarizan, las
# categoricas se codifican one-hot, y el objetivo no se toca.
#
# El tercer color se VALIDO con el mismo criterio que declara la paleta de arriba, no se
# eligio a ojo. Contra el par ya establecido (azul #2a78d6 / naranja #eb6834), midiendo
# separacion en OKLab bajo simulacion de daltonismo (matrices de Vienot 1999):
#
#   ciruela #a8447f   peor par: ΔE 14,4   contraste 5,4:1   croma 14,7   -> pasa
#   verde   #2e8b57   colapsa contra el naranja en protanopia (ΔE 3,7)   -> NO
#   violeta #7a4fd0   colapsa contra el azul en deuteranopia (ΔE 3,2)    -> NO
#   teal    #0e8a8a   pasa (ΔE 10,0) pero con croma 9,6: se lee lavado
#
# Los verdes y los violetas son la eleccion intuitiva para "un tercer color" y son
# justamente los dos que fallan: el verde se confunde con el naranja para un protanope y
# el violeta con el azul para un deuteranope. Por eso hay validador y no criterio propio.
#
# NOTA sobre el naranja: en las figuras 3, 4 y 8 significa "fumador". Aca significa "la
# variable objetivo". No hay ambiguedad porque el color sigue a la entidad DENTRO de cada
# figura, y en esta las entidades son las variables, no las personas — es el mismo reuso
# que ya hace el azul, que en la figura 1 es "entrenamiento" y en la 3 es "no fumador".
COLOR_NUMERICA = COLOR_TRAIN       # azul     — predictoras numericas: se estandarizan
COLOR_CATEGORICA = "#a8447f"       # ciruela  — predictoras categoricas: se codifican one-hot
COLOR_OBJETIVO = COLOR_VAL         # naranja  — `charges`: es lo que el modelo predice

# Rampa de un solo tono para el sub-split ORDENADO de fumadores por bmi (figura 8). Mismo
# criterio que COLORES_GRADO: un corte ordenado (bmi<=30 < bmi>30) se codifica en la
# luminosidad de un unico tono, no gastando dos tonos nominales distintos. El tono es el
# naranja del deck, que ya significa "fumador", asi que los dos escalones se leen como
# "las dos mitades del grupo naranja". Contraste contra la superficie 2,3 y 6,2; el paso
# entre ambos es 2,7:1, muy por encima de lo que hace falta para ordenarlos de un vistazo.
COLOR_FUMADOR_CLARO = "#f0925a"
COLOR_FUMADOR_OSCURO = "#a83c14"

NUMERICAS_EDA = ["age", "bmi", "children", "charges"]
CATEGORICAS_EDA = ["sex", "smoker", "region"]


def _bins_freedman_diaconis(serie):
    """Ancho de bin = 2·IQR/n^(1/3), la regla de Freedman-Diaconis.

    El numero de bins NO se elige a ojo, porque es la unica decision del histograma y
    cambia lo que el grafico afirma: con pocos bins se borran los picos (que es justo lo
    que la Clase 3 manda buscar) y con demasiados el ruido de muestreo se lee como
    estructura. FD deriva el ancho de la dispersion robusta y del tamano de muestra, y es
    la regla estandar para variables asimetricas: usa el IQR, no el desvio, asi que la cola
    de `charges` no le infla el ancho.
    """
    q1, q3 = serie.quantile([0.25, 0.75])
    ancho = 2 * (q3 - q1) / len(serie) ** (1 / 3)
    return max(1, int(np.ceil((serie.max() - serie.min()) / ancho)))


def _num(valor, decimales=1, signo=False):
    """Numero a la espanola: punto para los miles, coma para los decimales.

    Se hace con un centinela y no con dos .replace() encadenados, que es la trampa
    clasica: reemplazar la coma por punto y despues el punto por coma deja los dos
    separadores iguales. Es el mismo cuidado que ya se toma en la figura 4.
    """
    crudo = f"{valor:+,.{decimales}f}" if signo else f"{valor:,.{decimales}f}"
    return crudo.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def _panel_numerica(ax, serie, etiqueta_x, color=COLOR_NUMERICA, nota=None, pesos=False):
    """Un histograma con su linea de estadisticas descriptivas y su hallazgo anotado."""
    # `age` y `children` son ENTERAS: no hay decision de binning que tomar, porque los
    # datos ya vienen en cajones. Un bin por valor muestra la distribucion real; aplicarles
    # FD las suavizaria (a `age` le daria 11 bins) y taparia el pico de 18-19 anios, que es
    # exactamente el tipo de hallazgo que la Clase 3 pide buscar en el histograma.
    if serie.dtype.kind == "i":
        bordes = np.arange(serie.min() - 0.5, serie.max() + 1.5, 1.0)
    else:
        bordes = _bins_freedman_diaconis(serie)

    # El borde va del color de la superficie y no negro: separa barra de barra sin
    # agregar una reja oscura que compita con el relleno, que es lo que importa leer.
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

    # La mediana va marcada porque es la referencia contra la que se lee la asimetria: si
    # coincide con el centro de la masa, la variable es simetrica; si queda a la izquierda
    # del grueso del area, hay cola a derecha.
    ax.axvline(serie.median(), color=COLOR_REFERENCIA, linestyle="--", linewidth=1.3, zorder=4)

    if nota:
        # Headroom explicito ANTES de colgar la nota. Sin esto el cartel se apoya sobre las
        # barras mas altas y tapa datos: matplotlib ajusta el eje al maximo del histograma y
        # no sabe que hay un texto ocupando el cuarto superior. El 1,52 es lo que mide el
        # cartel mas alto (4 lineas a 9 pt) en estos paneles.
        ax.set_ylim(0, alturas.max() * 1.52)
        ax.annotate(nota, xy=(0.975, 0.965), xycoords="axes fraction", ha="right", va="top",
                    fontsize=9, color=TINTA_PRIMARIA, linespacing=1.35,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor=SUPERFICIE,
                              edgecolor=COLOR_EJE, linewidth=0.8))


def _panel_categorica(ax, serie, etiqueta, color=COLOR_CATEGORICA):
    """Barras de frecuencia: el histograma de una variable sin orden numerico.

    Van HORIZONTALES para que los niveles ('northeast', 'southwest'…) se lean derechos, y
    ordenadas por frecuencia salvo que el orden natural diga otra cosa.
    """
    conteo = serie.value_counts().sort_values()
    ax.barh(range(len(conteo)), conteo.values, color=color, height=0.62)
    ax.set_yticks(range(len(conteo)))
    ax.set_yticklabels(conteo.index, fontsize=10)
    ax.set_xlabel("Cantidad de personas", fontsize=10.5, color=TINTA_PRIMARIA)
    # El recuento de faltantes se CALCULA. Estaba escrito "sin faltantes" a mano: si el
    # dataset cambiara, la figura seguiria afirmandolo sin que nadie se entere.
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
    """Los 7 histogramas del dataset, en una sola lamina.

    Dos filas con significado: arriba las cuatro numericas (histograma propiamente dicho),
    abajo las tres categoricas (barras de frecuencia, que es el histograma de una variable
    sin orden). El octavo panel no es relleno: lleva el resumen de lo que las cuatro
    distribuciones revelan y que la tabla de estadisticas descriptivas NO muestra, que es
    el argumento entero de por que la Clase 3 pide el histograma ademas de la media.

    Se dibuja sobre el dataset ya deduplicado, como el resto de las figuras: la Clase 3
    ubica el EDA DESPUES de la limpieza ("EDA es el proceso siguiente", slide 32).
    """
    df = quitar_duplicados(cargar())

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6))

    # ------------------------------------------------------------------ fila 1: numericas
    # Los numeros de las notas se calculan, no se escriben: un numero hardcodeado en un
    # grafico es una afirmacion que nadie vuelve a verificar (mismo criterio que la fig. 4).
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
    # Marcas cada 10.000 rotuladas "10k": las etiquetas completas no entran en un panel de
    # este ancho y se pisan entre si. Mismo formato que la figura 4, que grafica lo mismo.
    axes[0][3].set_xticks(np.arange(0, 70000, 20000))
    axes[0][3].set_xticklabels(["0"] + [f"{v}k" for v in range(20, 70, 20)])

    # ------------------------------------------------------------------ fila 2: categoricas
    _panel_categorica(axes[1][0], df["sex"], "sex")
    _panel_categorica(axes[1][1], df["smoker"], "smoker")
    _panel_categorica(axes[1][2], df["region"], "region")

    # ------------------------------------------------------- panel 8: lo que agrega el EDA
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
    # La leyenda del codigo de color va en el encabezado y no en un recuadro adentro de un
    # panel: aplica a los siete, no a uno. Cada entrada lleva su parche del color real, asi
    # que la equivalencia se ve en vez de leerse.
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


# --------------------------------------------------------------------------------------
# Figura 8 — el histograma de charges separado por poblacion
# --------------------------------------------------------------------------------------
def figura_histograma_poblaciones():
    """La receta literal de la Clase 3 cuando el histograma muestra un segundo pico.

    El deck (slides 36-38) no se queda en "hay dos poblaciones": prescribe que hacer.
    a) dar esa informacion al sistema como variable binaria, o b) partir el dataset y
    entrenar dos modelos; y antes de decidir, "ver el histograma separado para cada
    poblacion para ver si realmente son distintas". Esta figura es ese paso.

    Y la respuesta no es la misma para las dos separaciones, que es justamente lo que la
    figura tiene que dejar ver:
      - fumar SI parte la variable en dos regimenes con medianas 2,8x distintas, aunque
        el grupo de fumadores delgados se solapa con la cola de los no fumadores;
      - dentro de los fumadores, el corte de obesidad recorta un tercer grupo que arranca
        por encima del 99 % de todos los demas: ahi si los histogramas son disjuntos.

    Se eligio la opcion (a) del deck: `smoker` ya es una columna del dataset y el termino
    de interaccion con bmi aparece solo al expandir a grado 2. Partir en dos modelos
    habria dejado 274 filas para el de fumadores.
    """
    df = quitar_duplicados(cargar())
    fumador = df["smoker"] == "yes"
    obeso = df["bmi"] > 30

    grupos = [
        (~fumador, "no fumador", COLOR_NO_FUMADOR),
        (fumador & ~obeso, "fumador, bmi ≤ 30", COLOR_FUMADOR_CLARO),
        (fumador & obeso, "fumador, bmi > 30", COLOR_FUMADOR_OSCURO),
    ]

    # Mismos bordes de bin en los dos paneles y en los tres grupos: si cada histograma
    # eligiera los suyos, las alturas dejarian de ser comparables y el solape entre lobulos
    # seria un artefacto del binning. FD sobre la variable completa.
    bordes = np.histogram_bin_edges(df[OBJETIVO], bins=_bins_freedman_diaconis(df[OBJETIVO]))

    fig, (ax_todo, ax_sep) = plt.subplots(
        2, 1, figsize=(9, 7.4), sharex=True, gridspec_kw={"height_ratios": [1, 1.5]}
    )

    # ------------------------------------------------------------------ panel superior
    # Naranja, el mismo color con el que `charges` aparece en la figura 7: este panel es
    # literalmente ese histograma, y el titulo lo dice. El panel de abajo cambia de paleta
    # porque cambia la pregunta — deja de colorear POR VARIABLE y pasa a colorear POR
    # POBLACION, que es el par azul/naranja de las figuras 3 y 4.
    ax_todo.hist(df[OBJETIVO], bins=bordes, color=COLOR_OBJETIVO, edgecolor=SUPERFICIE, linewidth=0.4)
    ax_todo.set_ylabel("Cantidad de personas")
    ax_todo.set_title("Lo que se ve en la figura 7: un pico grande, y dos jorobas "
                      "más a la derecha", fontsize=12, pad=10)
    for x, texto in ((20000, "¿segundo\npico?"), (41000, "¿tercero?")):
        ax_todo.annotate(texto, xy=(x, 0.55), xycoords=("data", "axes fraction"),
                         ha="center", va="bottom", fontsize=10.5, style="italic",
                         color=TINTA_SECUNDARIA)

    # ------------------------------------------------------------------ panel inferior
    # APILADO, no superpuesto: la altura total sigue siendo la cantidad de personas del
    # bin, y cada tramo es la porcion de cada grupo. Superponerlos con transparencia
    # inventaria colores de solape que no estan en la leyenda (mismo criterio que fig. 4).
    #
    # La mediana de cada grupo va EN LA LEYENDA y no colgada del eje. Colgada del eje se
    # pisaba con el rotulo del eje x, y ademas obligaba a saltar entre el grafico y el pie
    # para saber de que grupo era cada numero. En la leyenda, el color, el nombre, el n y
    # la mediana se leen juntos.
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

    # El hallazgo que decide el TP, con su numero calculado: el tercer grupo esta
    # practicamente separado del resto, y el segundo NO — se solapa con la cola de los no
    # fumadores. Decir "no se pisan" de los tres seria falso, y es el tipo de afirmacion
    # que un grafico hace sin que nadie la revise.
    minimo_tercero = df.loc[fumador & obeso, OBJETIVO].min()
    pct_debajo = 100 * (df.loc[~(fumador & obeso), OBJETIVO] < minimo_tercero).mean()
    ax_sep.axvline(minimo_tercero, color=COLOR_FUMADOR_OSCURO, linestyle="--", linewidth=1.5)
    # Headroom explicito para que el cartel no se apoye sobre las barras: sin el, el unico
    # hueco libre queda en la mitad del histograma y el texto tapa datos.
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


# --------------------------------------------------------------------------------------
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
