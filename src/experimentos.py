"""Experimentos de regresion: validacion cruzada y seleccion de modelo.

ESTE MODULO NO TOCA EL CONJUNTO DE TEST. Llega hasta la eleccion y se detiene; la
evaluacion de test es un paso separado y de una sola vez (src/evaluar_test.py, D-21).

Cubre los puntos 2, 3, 4 y 5 del enunciado. Es el modulo que produce los numeros que se
presentan y se defienden, asi que cada paso queda comentado con el PORQUE, no solo el que.

El pipeline completo (codificar -> split -> CV con doble escalado -> eleccion -> test) esta
descripto con el detalle exacto en el docstring de main(). Las decisiones metodologicas
(D-01 a D-20) estan documentadas en DECISIONES.md y no se re-discuten aca.

Correr con:  python3 -m src.experimentos
"""

import json
import os
import sys
import time

import numpy as np

from src.datos import OBJETIVO, agregar_derivadas, cargar
from src.modelos import Lasso, RegresionLineal, lambda_maximo
from src.preproceso import (
    CodificadorCategoricas,
    Estandarizador,
    expandir_polinomica,
    nombres_polinomicos,
    quitar_duplicados,
)
from src.validacion import k_fold, resumen_folds, rmse, separar_train_test

RUTA_RESULTADOS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resultados")

GRADOS = (1, 2, 3, 4)
K_FOLDS = 5
SEMILLA = 42
FRACCIONES_LAMBDA = (0.3, 0.1, 0.03, 0.01, 0.003)

# Tolerancia y tope de barridas del Lasso (decisiones D-19 y D-20).
#
# El criterio de corte del descenso por coordenadas es max_j |delta w_j| < TOL_LASSO, y esa
# comparacion esta en UNIDADES ABSOLUTAS: los coeficientes de este problema estan en dolares
# (charges tiene media 13 447 y desvio 12 289). Pedir 1e-7 seria exigir convergencia a once
# ordenes de magnitud por debajo de la senal — una centesima de centavo en un coeficiente —,
# precision que ningun numero que reportamos usa y que en grado 4 cuesta decenas de miles de
# barridas por fold. 1e-4 ya esta muy por debajo de cualquier digito significativo.
#
# Esto importa porque las features polinomicas de este dataset son EXACTAMENTE colineales, no
# solo "casi". El numero de condicion completo (sigma_max / sigma_min) de la matriz de diseno
# estandarizada, medido sobre las 1070 filas de train, supera la precision de float64 (~1e16)
# YA DESDE GRADO 2 (5.8e16) y llega a un maximo de 1.5e18 en grado 3 (grado 4 baja a 3.8e17):
# desde D-27/D-28 ya NO crece monotono con el grado (ver informe/salida-seleccion.txt,
# diagnostico de rango). De grado 2 en adelante la matriz es NUMERICAMENTE SINGULAR. Por eso
# D-12 exige resolver OLS con lstsq (SVD) y no con inv() ni solve().
#
# Restringido al subespacio de rango completo (descartando los valores singulares nulos) el
# condicionamiento es benigno y crece mucho mas lento con el grado — de 17.5 en grado 1 a
# 401.6 en grado 2, 31 398.9 en grado 3 y unos 1.5e6 en grado 4 — ver la columna `cond` de
# `diagnostico_de_rango` en informe/salida-seleccion.txt —, asi que las PREDICCIONES son
# estables. Lo que no es estable es el reparto de coeficientes entre columnas colineales.
#
# CAUSA NUEVA de degeneracion desde D-27/D-28: la expansion polinomica duplica EXACTAMENTE a
# las dos features derivadas a partir de grado 2 (age*age = edad_al_cuadrado, y
# bmi*smoker=yes = bmi_si_fuma), lo que sube la redundancia (68.0 % de columnas en grado 4,
# contra 59.7 % antes de D-27/D-28) y hace mas lento al descenso por coordenadas — de ahi que
# 4 configuraciones no converjan en vez de 2. En grado 1, el de produccion, no hay
# duplicacion: ahi esas dos columnas son la unica via de esas dos estructuras.
#
# Con esa geometria el descenso por coordenadas avanza muy despacio, y de ahi el tope alto de
# barridas de abajo.
TOL_LASSO = 1e-4
MAX_ITER_LASSO = 50000


# --------------------------------------------------------------------------------------
# Punto 2 y 3 — funcion central de validacion cruzada, reusada por lineal y Lasso
# --------------------------------------------------------------------------------------
def diagnostico_de_rango(X_train):
    """Cuantas de las columnas polinomicas son linealmente redundantes, y por que.

    Este diagnostico existe porque la expansion polinomica sobre variables BINARIAS genera
    columnas que son combinaciones lineales EXACTAS de otras, no aproximadas:

      1. Una dummy elevada a una potencia sigue teniendo dos valores distintos, y cualquier
         vector de dos valores es una funcion afin de cualquier otro con el mismo patron.
         Concretamente, con smoker=yes estandarizada a {-0.5087, 1.9656}:
             smoker=yes^2 = 1.456866 * smoker=yes + 1.0     (residuo maximo 1.5e-14)
         O sea que smoker^2, smoker^3 y smoker^4 no aportan NADA que smoker no aporte ya.

      2. El producto de dos dummies de un mismo one-hot cae en el ESPACIO GENERADO por las
         dummies y la constante. Con `region`, las 3 dummies mas la constante generan TODAS
         las funciones posibles sobre las 4 regiones (un espacio de dimension 4), y el
         producto region=southeast * region=southwest es una funcion sobre esas 4 regiones:
         por lo tanto ya estaba ahi. Verificado: proyectarlo sobre [1, nw, se, sw] deja un
         residuo de 4.4e-15.

         OJO con un atajo tentador y FALSO: en la codificacion cruda 0/1 ese producto es
         literalmente la columna nula (una fila no puede estar en dos regiones). Pero D-06
         estandariza ANTES de expandir, y una vez estandarizadas las dummies valen dos
         numeros distintos de 0 y 1, con lo cual el producto NO es cero — toma tres valores
         distintos. Sigue siendo redundante, pero por la razon de arriba, no por ser nulo.

    Consecuencia practica, y hay que decirla en la defensa: en grado 4 mas de la mitad de
    las columnas son redundantes, y dentro de un grupo colineal el reparto de los
    coeficientes es ARBITRARIO — `lstsq` elige la solucion de norma minima, pero cualquier
    otro reparto da las mismas predicciones. Por eso los coeficientes individuales de un
    grupo colineal no se pueden interpretar de a uno; el grupo si.

    Esto NO invalida los resultados: `lstsq` (SVD) resuelve sistemas de rango deficiente sin
    problemas, y el Lasso sigue siendo un problema convexo. Lo que invalida es la lectura
    ingenua de un coeficiente aislado.
    """
    filas = []
    for grado in GRADOS:
        P_esc, _, _ = preprocesar_completo(X_train, grado)
        valores_singulares = np.linalg.svd(P_esc, compute_uv=False)
        # rango numerico: valores singulares por encima de una tolerancia relativa al mayor
        rango = int(np.sum(valores_singulares > valores_singulares[0] * 1e-10))
        n_cols = P_esc.shape[1]
        filas.append(
            {
                "grado": grado,
                "columnas": n_cols,
                "rango": rango,
                "redundantes": n_cols - rango,
                "cond": valores_singulares[0] / valores_singulares[rango - 1],
            }
        )
    return filas


def evaluar_con_cv(X_train, y_train, grado, modelo_factory, k=K_FOLDS, semilla=SEMILLA):
    """Corre k-fold CV sobre X_train/y_train para un grado polinomico y una familia de modelo.

    Por cada fold se repite el pipeline completo (escalar -> expandir -> escalar) ajustando
    SIEMPRE con el sub-train del fold, nunca con el fold de validacion ni con X_train entero:
    eso es lo que evita la fuga de datos que describe D-05/D-06 en DECISIONES.md.

    modelo_factory es una funcion sin argumentos que devuelve una instancia nueva del modelo
    (RegresionLineal() o Lasso(lam=...)) para cada fold: los modelos no se reusan entre folds
    porque cada uno tiene que ajustarse desde cero sobre datos distintos.

    Devuelve el dict de resumen_folds(), enriquecido con:
      - "grado": el grado polinomico evaluado
      - "n_features": cantidad de columnas tras la expansion (igual en todos los folds)
      - "coefs_no_nulos_medio": promedio, entre folds, de coeficientes con |coef| > 0
        (para Lasso es el argumento de seleccion de variables; para RegresionLineal es
        simplemente p, porque OLS/Ridge no anulan coeficientes)
      - "n_no_convergio": cuantos de los k folds NO convergieron (solo relevante en Lasso)
    """
    folds = k_fold(len(y_train), k=k, semilla=semilla)

    errores_train = []
    errores_val = []
    coefs_no_nulos = []
    n_no_convergio = 0
    n_features = None

    for i_tr, i_va in folds:
        Xtr, ytr = X_train[i_tr], y_train[i_tr]
        Xva, yva = X_train[i_va], y_train[i_va]

        # primer escalado: ajustado SOLO con el sub-train del fold (D-05)
        e1 = Estandarizador().ajustar(Xtr)
        Xtr_s, Xva_s = e1.transformar(Xtr), e1.transformar(Xva)

        # expansion polinomica de ambos, ya en la escala aprendida de train
        Ptr, Pva = expandir_polinomica(Xtr_s, grado), expandir_polinomica(Xva_s, grado)

        # segundo escalado, sobre las features expandidas, tambien ajustado SOLO con Ptr
        e2 = Estandarizador().ajustar(Ptr)
        Ptr_s, Pva_s = e2.transformar(Ptr), e2.transformar(Pva)

        modelo = modelo_factory()
        modelo.ajustar(Ptr_s, ytr)

        errores_train.append(rmse(ytr, modelo.predecir(Ptr_s)))
        errores_val.append(rmse(yva, modelo.predecir(Pva_s)))
        coefs_no_nulos.append(int(np.sum(np.abs(modelo.coef_) > 0)))
        if hasattr(modelo, "convergio_") and not modelo.convergio_:
            n_no_convergio += 1
        n_features = Ptr_s.shape[1]

    resumen = resumen_folds(errores_train, errores_val)
    resumen["grado"] = grado
    resumen["n_features"] = n_features
    resumen["coefs_no_nulos_medio"] = float(np.mean(coefs_no_nulos))
    resumen["n_no_convergio"] = n_no_convergio
    return resumen


# --------------------------------------------------------------------------------------
# Pipeline completo (escalar -> expandir -> escalar) SIN folds: ajustado con TODO X.
#
# Se reusa en dos lugares del punto 3.3/5 donde ajustar con el train completo es licito
# porque no hay ninguna medicion de error de por medio:
#   1. Punto 3.3: fijar lambda_maximo (y por lo tanto la grilla de lambda) UNA sola vez
#      sobre el train completo, con el mismo pipeline que ve cada fold (D-15).
#   2. Punto 5: el reentrenamiento final de la configuracion elegida, donde ya no hay
#      folds y los dos Estandarizadores se ajustan con los 1070 de train.
# --------------------------------------------------------------------------------------
def preprocesar_completo(X, grado, e1=None, e2=None):
    """Aplica escalar->expandir->escalar. Si e1/e2 son None, los AJUSTA con X (fit+transform).

    Se reusa tanto para ajustar sobre el train completo (e1=e2=None) como para transformar
    el test con los estandarizadores ya ajustados en train (e1, e2 dados): el test nunca
    puede ajustar sus propios parametros de escalado, por definicion de "no tocar test".
    """
    ajustar_e1 = e1 is None
    if ajustar_e1:
        e1 = Estandarizador().ajustar(X)
    X_esc = e1.transformar(X)

    P = expandir_polinomica(X_esc, grado)

    ajustar_e2 = e2 is None
    if ajustar_e2:
        e2 = Estandarizador().ajustar(P)
    P_esc = e2.transformar(P)

    return P_esc, e1, e2




# --------------------------------------------------------------------------------------
# Reporte: tabla en terminal
# --------------------------------------------------------------------------------------
def texto_lambda(config):
    """`lambda=296.36` para Lasso, `sin regularizacion` para el lineal.

    Existe porque el lineal no tiene lambda —es None— y todos los f-strings de reporte lo
    formateaban con :.2f dando por sentado que el ganador siempre seria un Lasso. Mientras
    lo fue, nadie se entero; en cuanto la regla de 1 ES eligio `lineal grado 1` (D-23), el
    script murio con TypeError DESPUES de veinte minutos de calculo y antes de escribir
    resultados/modelo_elegido.json.
    """
    return "sin regularizacion" if config["lambda"] is None else f"lambda={config['lambda']:.2f}"


def imprimir_tabla(filas):
    """Imprime una tabla alineada con las columnas que pide el punto 4 del enunciado."""
    encabezado = (
        f"{'modelo':<12} {'grado':>5} {'lambda':>12} "
        f"{'RMSE train (media+-desvio)':>28} {'RMSE val (media+-desvio)':>28} "
        f"{'coefs!=0':>9}"
    )
    print(encabezado)
    print("-" * len(encabezado))
    for f in filas:
        lam_str = f"{f['lambda']:.4g}" if f["lambda"] is not None else "-"
        tr_str = f"{f['rmse_train_medio']:.1f} +- {f['rmse_train_desvio']:.1f}"
        va_str = f"{f['rmse_val_medio']:.1f} +- {f['rmse_val_desvio']:.1f}"
        print(
            f"{f['modelo']:<12} {f['grado']:>5} {lam_str:>12} "
            f"{tr_str:>28} {va_str:>28} {f['coefs_no_nulos_medio']:>9.1f}"
        )


def guardar_csv_lineal(filas, ruta):
    """Guarda cv_lineal.csv: grado,n_features,rmse_train_medio,rmse_train_desvio,rmse_val_medio,rmse_val_desvio"""
    with open(ruta, "w") as fh:
        fh.write("grado,n_features,rmse_train_medio,rmse_train_desvio,rmse_val_medio,rmse_val_desvio\n")
        for f in filas:
            fh.write(
                f"{f['grado']},{f['n_features']},{f['rmse_train_medio']:.6f},"
                f"{f['rmse_train_desvio']:.6f},{f['rmse_val_medio']:.6f},{f['rmse_val_desvio']:.6f}\n"
            )


def guardar_csv_lasso(filas, ruta):
    """Guarda cv_lasso.csv: grado,frac_lambda,lambda,n_features,rmse_train_medio,rmse_val_medio,coefs_no_nulos_medio"""
    with open(ruta, "w") as fh:
        fh.write("grado,frac_lambda,lambda,n_features,rmse_train_medio,rmse_val_medio,coefs_no_nulos_medio\n")
        for f in filas:
            fh.write(
                f"{f['grado']},{f['frac_lambda']},{f['lambda']:.8f},{f['n_features']},"
                f"{f['rmse_train_medio']:.6f},{f['rmse_val_medio']:.6f},{f['coefs_no_nulos_medio']:.4f}\n"
            )


# --------------------------------------------------------------------------------------
def main():
    # linea a linea aunque la salida vaya a un archivo (via `| tee`, por ejemplo): sin esto
    # python bufferea por bloques cuando stdout no es una terminal, y la salida no aparece
    # hasta que el proceso termina, lo que hace imposible seguir el progreso de una corrida
    # larga (el punto 3.3 con grado 4 puede tardar varios minutos).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    os.makedirs(RUTA_RESULTADOS, exist_ok=True)

    t0 = time.time()

    # ------------------------------------------------------------------------------
    # Pasos 1-4 del enunciado del pipeline: cargar, quitar duplicados, split, codificar
    # ------------------------------------------------------------------------------
    df = agregar_derivadas(quitar_duplicados(cargar()))
    print("=" * 100)
    print(f"Filas tras quitar_duplicados: {len(df)}")

    idx_train, idx_test = separar_train_test(len(df), prop_test=0.2, semilla=SEMILLA)
    # De idx_test solo se usa la CANTIDAD, para poder reportarla. Nunca se lo usa para
    # indexar el DataFrame: las 267 filas reservadas no se cargan en ningun momento de
    # este modulo. La evaluacion de test es un paso aparte (D-21).
    N_TEST_RESERVADAS = len(idx_test)
    del idx_test  # que no quede ni la tentacion
    print(f"Split: {len(idx_train)} train / {N_TEST_RESERVADAS} reservadas para test")

    cod = CodificadorCategoricas().ajustar(df.iloc[idx_train])
    X_train = cod.transformar(df.iloc[idx_train])

    y_train = df[OBJETIVO].values[idx_train]

    print(f"Columnas codificadas ({len(cod.nombres_)}): {cod.nombres_}")
    print("EL TEST NO SE VUELVE A TOCAR HASTA EL PUNTO 5.")

    # ------------------------------------------------------------------------------
    # Diagnostico previo: cuantas columnas polinomicas son realmente independientes
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("DIAGNOSTICO — RANGO DE LA MATRIZ DE DISENO POLINOMICA")
    print("=" * 100)
    print("Las variables binarias rompen la expansion polinomica, por dos motivos distintos:")
    print("  1. Una dummy al cuadrado sigue teniendo DOS valores, asi que es una funcion afin")
    print("     EXACTA de la dummy original: smoker^2, smoker^3 y smoker^4 no aportan nada.")
    print("  2. El producto de dos dummies del mismo one-hot ya vive en el espacio generado")
    print("     por las dummies y la constante (3 dummies + 1 generan todas las funciones")
    print("     sobre las 4 regiones, y el producto es una de ellas).")
    print()
    print(f"{'grado':>6} {'columnas':>9} {'rango':>7} {'redundantes':>12} {'% redund.':>10} {'cond':>9}")
    print("-" * 60)
    for d in diagnostico_de_rango(X_train):
        print(
            f"{d['grado']:>6} {d['columnas']:>9} {d['rango']:>7} {d['redundantes']:>12} "
            f"{100 * d['redundantes'] / d['columnas']:>9.1f}% {d['cond']:>9.1f}"
        )
    print()
    print("Los coeficientes de un grupo colineal NO son interpretables de a uno: cualquier")
    print("reparto entre ellos da las mismas predicciones. El grupo si es interpretable.")

    # ------------------------------------------------------------------------------
    # PUNTOS 2 y 3 — CV para regresion lineal, grados 1..4 (grado 1 = punto 2)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("PUNTO 2 y 3 — VALIDACION CRUZADA, REGRESION LINEAL (grado 1 = punto 2)")
    print("=" * 100)

    filas_lineal = []
    resultados_lineal = {}
    for grado in GRADOS:
        resumen = evaluar_con_cv(X_train, y_train, grado, lambda: RegresionLineal(), k=K_FOLDS, semilla=SEMILLA)
        resultados_lineal[grado] = resumen
        filas_lineal.append(
            {
                "modelo": "lineal",
                "grado": grado,
                "lambda": None,
                "rmse_train_medio": resumen["rmse_train_medio"],
                "rmse_train_desvio": resumen["rmse_train_desvio"],
                "rmse_val_medio": resumen["rmse_val_medio"],
                "rmse_val_desvio": resumen["rmse_val_desvio"],
                "coefs_no_nulos_medio": resumen["coefs_no_nulos_medio"],
            }
        )

    imprimir_tabla(filas_lineal)
    guardar_csv_lineal(
        [{**resultados_lineal[g], "grado": g} for g in GRADOS],
        os.path.join(RUTA_RESULTADOS, "cv_lineal.csv"),
    )

    # ------------------------------------------------------------------------------
    # PUNTO 3.3 — Lasso, grados 2..4, grilla de lambda relativa a lambda_maximo
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("PUNTO 3.3 — VALIDACION CRUZADA, LASSO (grados 2, 3, 4)")
    print("=" * 100)

    filas_lasso = []
    resultados_lasso = {}  # (grado, frac) -> resumen
    for grado in (2, 3, 4):
        # lambda_max se calcula UNA vez sobre las 1070 filas de train completas, con el mismo
        # pipeline de preprocesamiento (D-15).
        #
        # HONESTIDAD SOBRE ESTO: para un fold dado, esas 1070 filas incluyen las ~214 que en
        # ESE fold hacen de validacion. O sea que la escala de la grilla de lambda se fijo
        # mirando, entre otras, filas que despues se usan para validar. Es una fuga, aunque
        # de las mas leves que existen, y conviene decirla en vez de que la encuentre otro:
        #
        #   - Lo que se filtra es UN escalar por grado (el maximo de |x_j^T (y - y_medio)|/n),
        #     no informacion fila por fila. Ningun parametro del modelo se ajusta con esto.
        #   - La alternativa —recalcular lambda_max dentro de cada fold— tendria un costo
        #     peor: los lambdas absolutos dejarian de ser los mismos entre folds y los cinco
        #     RMSE que se promedian corresponderian a modelos con regularizaciones distintas.
        #     El promedio dejaria de significar algo.
        #   - Se elige entonces comparabilidad entre folds por sobre pureza en la definicion
        #     de la grilla, que es la practica habitual (glmnet hace lo mismo).
        P_esc, _, _ = preprocesar_completo(X_train, grado)
        lam_max = lambda_maximo(P_esc, y_train)
        print(f"\ngrado={grado}: lambda_maximo (sobre train completo) = {lam_max:.6f}")

        # Todos los grados usan el MISMO tope de barridas y la MISMA tolerancia. Bajar
        # max_iter para los grados caros —que era lo que hacia la primera version— produce
        # un RMSE que no es el del modelo Lasso sino el del punto donde se corto la
        # optimizacion: no es reproducible, no es interpretable, y elegir esa configuracion
        # como ganadora invalidaria la respuesta del punto 5. Ver D-19 y D-20.
        max_iter_usado = MAX_ITER_LASSO

        for frac in FRACCIONES_LAMBDA:
            lam = lam_max * frac
            resumen = evaluar_con_cv(
                X_train,
                y_train,
                grado,
                lambda lam=lam: Lasso(lam=lam, max_iter=max_iter_usado, tol=TOL_LASSO),
                k=K_FOLDS,
                semilla=SEMILLA,
            )
            resultados_lasso[(grado, frac)] = {**resumen, "lambda": lam, "frac_lambda": frac}
            filas_lasso.append(
                {
                    "modelo": "lasso",
                    "grado": grado,
                    "lambda": lam,
                    "rmse_train_medio": resumen["rmse_train_medio"],
                    "rmse_train_desvio": resumen["rmse_train_desvio"],
                    "rmse_val_medio": resumen["rmse_val_medio"],
                    "rmse_val_desvio": resumen["rmse_val_desvio"],
                    "coefs_no_nulos_medio": resumen["coefs_no_nulos_medio"],
                }
            )
            if resumen["n_no_convergio"] > 0:
                print(
                    f"  AVISO: grado={grado} frac={frac} lambda={lam:.6f}: "
                    f"{resumen['n_no_convergio']}/{K_FOLDS} folds NO convergieron "
                    f"(max_iter={max_iter_usado})"
                )

    print()
    imprimir_tabla(filas_lasso)
    guardar_csv_lasso(
        [{**resultados_lasso[k]} for k in resultados_lasso],
        os.path.join(RUTA_RESULTADOS, "cv_lasso.csv"),
    )

    # ------------------------------------------------------------------------------
    # PUNTO 4 — tabla completa (lineal + lasso juntos), ya impresa arriba por bloques;
    # se repite unificada para que quede una sola tabla de referencia.
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("PUNTO 4 — TABLA COMPLETA (lineal + lasso)")
    print("=" * 100)
    imprimir_tabla(filas_lineal + filas_lasso)

    # ------------------------------------------------------------------------------
    # PUNTO 5 — eleccion del modelo, reentrenamiento y test
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("PUNTO 5 — ELECCION, REENTRENAMIENTO Y TEST")
    print("=" * 100)

    # candidatos: cada uno con su rmse_val_medio y como reconstruir su modelo
    candidatos = []
    for grado in GRADOS:
        r = resultados_lineal[grado]
        candidatos.append(
            {
                "modelo": "lineal",
                "grado": grado,
                "lambda": None,
                "rmse_val_medio": r["rmse_val_medio"],
                "rmse_val_desvio": r["rmse_val_desvio"],
                # OLS se resuelve de forma cerrada (lstsq/SVD): no es iterativo, asi que no
                # existe la nocion de "no convergio". Siempre es elegible.
                "n_no_convergio": 0,
                "coefs": None,
                "modelo_factory": (lambda: RegresionLineal()),
            }
        )
    for (grado, frac), r in resultados_lasso.items():
        lam = r["lambda"]
        candidatos.append(
            {
                "modelo": "lasso",
                "grado": grado,
                "lambda": lam,
                "rmse_val_medio": r["rmse_val_medio"],
                "rmse_val_desvio": r["rmse_val_desvio"],
                "n_no_convergio": r["n_no_convergio"],
                "coefs": r.get("coefs_no_nulos_medio"),
                "modelo_factory": (
                    lambda lam=lam: Lasso(lam=lam, max_iter=MAX_ITER_LASSO, tol=TOL_LASSO)
                ),
            }
        )

    # ---- D-20: una configuracion que no convergio no puede ser elegida ----
    # Su RMSE no es el del modelo Lasso: es donde quedo la optimizacion al cortarla. No es
    # reproducible ni interpretable, asi que se descarta de la seleccion, pero se DECLARA
    # cuales fueron en vez de hacerlas desaparecer de la tabla.
    descartados = [c for c in candidatos if c.get("n_no_convergio", 0) > 0]
    elegibles = [c for c in candidatos if c.get("n_no_convergio", 0) == 0]
    if descartados:
        print(
            f"\nDescartadas de la seleccion por no converger en algun fold "
            f"({len(descartados)} de {len(candidatos)} configuraciones):"
        )
        for c in descartados:
            print(
                f"  {c['modelo']} grado={c['grado']} {texto_lambda(c)} "
                f"-> {c['n_no_convergio']}/{K_FOLDS} folds sin converger "
                f"(rmse_val aparente {c['rmse_val_medio']:.1f}, NO comparable)"
            )
    else:
        print(f"\nTodas las {len(candidatos)} configuraciones convergieron.")
    if not elegibles:
        raise RuntimeError(
            "Ninguna configuracion convergio: subi MAX_ITER_LASSO o revisa TOL_LASSO."
        )

    # ---- Pregunta 5.1: cual obtuvo MENOR error ----
    mejor = min(elegibles, key=lambda c: c["rmse_val_medio"])
    print(
        f"\n[5.1] Menor rmse_val_medio entre las {len(elegibles)} configuraciones elegibles:\n"
        f"  modelo={mejor['modelo']}  grado={mejor['grado']}  "
        f"{texto_lambda(mejor)}  rmse_val_medio={mejor['rmse_val_medio']:.4f} "
        f"+- {mejor['rmse_val_desvio']:.4f}"
    )

    # ---- Pregunta 5.2: regla de 1 error estandar ----
    # El minimo de un conjunto de estimaciones ruidosas es un blanco movil: si dos
    # configuraciones difieren en 8 unidades de RMSE y el desvio entre folds es de 230,
    # cual salio "mejor" lo decide el ruido de la particion, no el modelo. La regla de 1
    # error estandar (Hastie, Tibshirani & Friedman, cap. 7) toma el modelo MAS SIMPLE cuyo
    # error de validacion cae dentro de un error estandar del mejor: entre modelos
    # estadisticamente indistinguibles, elige el mas parsimonioso.
    #
    # Error estandar de la media sobre k folds = desvio / sqrt(k).
    error_estandar = mejor["rmse_val_desvio"] / np.sqrt(K_FOLDS)
    umbral_1se = mejor["rmse_val_medio"] + error_estandar
    dentro_1se = [c for c in elegibles if c["rmse_val_medio"] <= umbral_1se]
    # "mas simple" = menor grado; a igual grado, mayor lambda (mas regularizado = menos
    # coeficientes vivos); a igualdad de ambos, menor error.
    parsimonioso = min(
        dentro_1se,
        key=lambda c: (c["grado"], -(c["lambda"] or 0.0), c["rmse_val_medio"]),
    )
    print(
        f"\n[5.2] Regla de 1 error estandar (ES = {mejor['rmse_val_desvio']:.1f}/sqrt({K_FOLDS})"
        f" = {error_estandar:.1f}, umbral = {umbral_1se:.1f}):\n"
        f"  {len(dentro_1se)} configuraciones son estadisticamente indistinguibles del mejor.\n"
        f"  La mas simple de ellas: modelo={parsimonioso['modelo']} grado={parsimonioso['grado']} "
        f"{texto_lambda(parsimonioso)} rmse_val_medio={parsimonioso['rmse_val_medio']:.4f}"
    )

    # ------------------------------------------------------------------------------
    # Los coeficientes del modelo de produccion.
    #
    # Esto NO toca test: reentrena la configuracion elegida sobre el train completo y
    # mira sus coeficientes. Es informacion sobre el modelo, no sobre su desempeno.
    #
    # OJO CON EL CASO SIN LASSO. Hasta D-23 el modelo de produccion habia salido Lasso en
    # todas las corridas, y este bloque estaba escrito dando eso por sentado: titulaba
    # "que features selecciono el Lasso", formateaba el lambda con :.2f y cerraba contando
    # cuantas features apago la penalizacion L1. Con la feature nueva la regla de 1 ES
    # elige `lineal grado 1`, que no tiene lambda (es None) ni penalizacion, y el script
    # se caia con TypeError justo despues de haber hecho los 20 minutos de calculo.
    #
    # Que un cambio de resultado rompa el REPORTE es una senal de que el reporte estaba
    # afirmando algo que no habia verificado. Ahora los dos casos estan contemplados y
    # cada uno dice lo suyo: con Lasso, cuantas features sobrevivieron; sin Lasso, que no
    # hay ninguna seleccion que reportar porque no hubo penalizacion.
    # ------------------------------------------------------------------------------
    hay_lasso = parsimonioso["lambda"] is not None
    print("\n" + "-" * 100)
    print("COEFICIENTES DEL MODELO DE PRODUCCION (entrenado con train)")
    print("-" * 100)
    Ptr_s, _, _ = preprocesar_completo(X_train, parsimonioso["grado"])
    modelo_prod = parsimonioso["modelo_factory"]().ajustar(Ptr_s, y_train)
    nombres_prod = nombres_polinomicos(cod.nombres_, parsimonioso["grado"])
    coefs_prod = modelo_prod.coef_
    vivos = np.flatnonzero(coefs_prod != 0)
    orden = vivos[np.argsort(-np.abs(coefs_prod[vivos]))]

    descripcion = texto_lambda(parsimonioso)
    if hay_lasso:
        cierre = (f"{len(vivos)} de {len(coefs_prod)} features sobreviven a la "
                  f"penalizacion L1")
    else:
        cierre = (f"{len(coefs_prod)} features, todas con coeficiente no nulo: "
                  f"sin penalizacion no hay seleccion")
    print(f"{parsimonioso['modelo']} grado={parsimonioso['grado']} {descripcion}"
          f"  ->  {cierre}\n")
    print(f"{'nombre':<34} {'coeficiente':>14}")
    for i in orden:
        print(f"{nombres_prod[i]:<34} {coefs_prod[i]:>14.2f}")
    if hay_lasso:
        print(f"\nL1 apago {len(coefs_prod) - len(vivos)} de {len(coefs_prod)} features "
              f"poniendolas EXACTAMENTE en cero, no en un valor chico.")
    else:
        print("\nNo hay features apagadas: el modelo elegido no lleva penalizacion. La "
              "parsimonia\nde este modelo no viene de anular coeficientes sino de no haber "
              "expandido el\nespacio de features (grado 1).")

    # ------------------------------------------------------------------------------
    # FIN DE ESTE MODULO. El test NO se toca aca.
    #
    # Este script llega hasta la ELECCION del modelo y se detiene. La evaluacion sobre
    # el conjunto de test es un paso SEPARADO, manual y de una sola vez, que corre
    # `src/evaluar_test.py`. Ver DECISIONES.md, decision D-21 y el protocolo del test.
    #
    # La garantia no es una promesa: este modulo NUNCA CONSTRUYE X_test ni y_test. Las
    # 267 filas reservadas existen solo como una lista de indices que no se usa para
    # indexar nada. Se puede verificar mecanicamente:
    #     grep -n "X_test\\|y_test" src/experimentos.py     -> sin resultados
    # ------------------------------------------------------------------------------
    eleccion = {
        "ganador_cv": {
            "modelo": mejor["modelo"], "grado": mejor["grado"], "lambda": mejor["lambda"],
            "rmse_val_medio": mejor["rmse_val_medio"],
            "rmse_val_desvio": mejor["rmse_val_desvio"],
        },
        "produccion_1se": {
            "modelo": parsimonioso["modelo"], "grado": parsimonioso["grado"],
            "lambda": parsimonioso["lambda"],
            "rmse_val_medio": parsimonioso["rmse_val_medio"],
        },
        "error_estandar": error_estandar,
        "umbral_1se": umbral_1se,
        "n_dentro_1se": len(dentro_1se),
        "n_elegibles": len(elegibles),
        "n_descartadas_sin_converger": len(descartados),
        "coeficientes_produccion": {nombres_prod[i]: float(coefs_prod[i]) for i in orden},
        "n_train": int(len(y_train)),
        "n_test_reservadas": int(N_TEST_RESERVADAS),
        "semilla": SEMILLA,
        "k_folds": K_FOLDS,
    }
    with open(os.path.join(RUTA_RESULTADOS, "modelo_elegido.json"), "w") as fh:
        json.dump(eleccion, fh, indent=2, ensure_ascii=False)

    print("\n" + "=" * 100)
    print("SELECCION CERRADA — EL TEST NO SE TOCO")
    print("=" * 100)
    print(f"  Filas de train usadas:            {len(y_train)}")
    print(f"  Filas reservadas para test:       {N_TEST_RESERVADAS}  (nunca cargadas)")
    print(f"  Configuraciones evaluadas:        {len(candidatos)}")
    print(f"  Descartadas por no converger:     {len(descartados)}")
    print()
    print(f"  [5.1] Menor RMSE de validacion:   {mejor['modelo']} grado {mejor['grado']}, "
          f"{texto_lambda(mejor)}  ->  {mejor['rmse_val_medio']:.4f}")
    print(f"  [5.2] Modelo de produccion (1 ES): {parsimonioso['modelo']} grado "
          f"{parsimonioso['grado']}, {texto_lambda(parsimonioso)}  ->  "
          f"{parsimonioso['rmse_val_medio']:.4f}")
    print()
    print("  Guardado en resultados/modelo_elegido.json")
    print()
    print("  SIGUIENTE PASO, y se hace UNA SOLA VEZ:")
    print("      python3 -m src.evaluar_test")
    print(f"\nTiempo total: {time.time() - t0:.1f} s")



if __name__ == "__main__":
    main()
