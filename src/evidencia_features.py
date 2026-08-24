"""Correr con: python -m src.evidencia_features"""

import csv
import os

import numpy as np

from src.datos import (
    CATEGORICAS,
    OBJETIVO,
    UMBRAL_OBESIDAD,
    agregar_derivadas,
    cargar,
)
from src.experimentos import (
    MAX_ITER_LASSO,
    RUTA_RESULTADOS,
    SEMILLA,
    TOL_LASSO,
    preprocesar_completo,
)
from src.modelos import Lasso, RegresionLineal
from src.preproceso import CodificadorCategoricas, quitar_duplicados
from src.validacion import k_fold, rmse, separar_train_test

SEMILLAS_FOLDS = (SEMILLA, 1, 2, 3, 4, 5, 6, 7)
K = 5

LAMBDA_REFERENCIA = 286.3701351700539


def _cv(X, y, grado, fabrica, folds):
    errores = []
    for idx_tr, idx_va in folds:
        P_tr, e1, e2 = preprocesar_completo(X[idx_tr], grado)
        P_va, _, _ = preprocesar_completo(X[idx_va], grado, e1, e2)
        modelo = fabrica().ajustar(P_tr, y[idx_tr])
        errores.append(rmse(y[idx_va], modelo.predecir(P_va)))
    return float(np.mean(errores)), float(np.std(errores, ddof=1))


def _sobre_particiones(X, y, grado, fabrica, generador_folds):
    medias = [_cv(X, y, grado, fabrica, generador_folds(s))[0] for s in SEMILLAS_FOLDS]
    return float(np.mean(medias)), float(np.std(medias))


def _folds_aleatorios(n):
    return lambda semilla: list(k_fold(n, k=K, semilla=semilla))


def _folds_estratificados(poblacion):
    def generar(semilla):
        rng = np.random.default_rng(semilla)
        asignacion = np.empty(len(poblacion), dtype=int)
        for grupo in np.unique(poblacion):
            indices = rng.permutation(np.where(poblacion == grupo)[0])
            asignacion[indices] = np.arange(len(indices)) % K
        folds = []
        for f in range(K):
            val = np.where(asignacion == f)[0]
            folds.append((np.where(asignacion != f)[0], val))
        return folds
    return generar


OLS = ("OLS", lambda: RegresionLineal())
LASSO = ("Lasso", lambda: Lasso(lam=LAMBDA_REFERENCIA, max_iter=MAX_ITER_LASSO, tol=TOL_LASSO))


def main():
    df = agregar_derivadas(quitar_duplicados(cargar()))
    idx_train, _ = separar_train_test(len(df), prop_test=0.2, semilla=SEMILLA)
    train = df.iloc[idx_train].reset_index(drop=True)
    y = train[OBJETIVO].to_numpy()
    n = len(train)
    aleatorios = _folds_aleatorios(n)

    X_sin = CodificadorCategoricas(
        numericas=["age", "bmi", "children"]
    ).ajustar_transformar(train)
    X_escalon = CodificadorCategoricas(
        numericas=["age", "bmi", "children", "fumador_obeso"]
    ).ajustar_transformar(train)
    X_con = CodificadorCategoricas().ajustar_transformar(train)

    filas = []

    print("=" * 86)
    print(f"EVIDENCIA DE LAS ELECCIONES DE FEATURES   ·   {n} filas de train   ·   "
          f"{len(SEMILLAS_FOLDS)} particiones de {K} folds")
    print("=" * 86)

    print("\n¿Aporta `fumador_obeso`?  (RMSE de validación, ± desvío entre particiones)\n")
    print(f"{'grado':>5s} {'modelo':>7s} {'sin la feature':>20s} {'con la feature':>20s} {'diferencia':>14s}")
    for grado in (1, 2, 3):
        for nombre, fabrica in (OLS, LASSO):
            sin_m, sin_s = _sobre_particiones(X_sin, y, grado, fabrica, aleatorios)
            con_m, con_s = _sobre_particiones(X_escalon, y, grado, fabrica, aleatorios)
            print(f"{grado:5d} {nombre:>7s} {sin_m:13,.0f} ± {sin_s:4,.0f} "
                  f"{con_m:13,.0f} ± {con_s:4,.0f} {con_m - sin_m:+13,.0f}")
            filas.append({"bloque": "fumador_obeso", "caso": f"grado {grado} {nombre}",
                          "rmse_sin": round(sin_m, 2), "rmse_con": round(con_m, 2),
                          "diferencia": round(con_m - sin_m, 2)})

    print("\n¿El umbral 30 es el que sirve, o cualquier corte da lo mismo?")
    print("       (OLS grado 1; si fuera ruido la curva sería plana)\n")
    for umbral in (24, 26, 28, 29, 30, 31, 32, 34, 36):
        binaria = ((train["smoker"] == "yes") & (train["bmi"] > umbral)).to_numpy(float)
        X_u = np.column_stack([X_sin, binaria])
        media, desvio = _sobre_particiones(X_u, y, 1, OLS[1], aleatorios)
        marca = "   <-- umbral clínico de la OMS (el que usa el pipeline)" if umbral == UMBRAL_OBESIDAD else ""
        print(f"       bmi > {umbral:2d}   RMSE {media:8,.0f} ± {desvio:4,.0f}{marca}")
        filas.append({"bloque": "umbral", "caso": f"bmi>{umbral}",
                      "rmse_sin": "", "rmse_con": round(media, 2), "diferencia": ""})

    print("\nControl: ¿es la interacción, o alcanza con alguna de las dos partes?")
    print("       (OLS grado 1)\n")
    controles = (
        ("nada (línea de base)", None),
        ("sólo bmi > 30", (train["bmi"] > UMBRAL_OBESIDAD).to_numpy(float)),
        ("sólo fuma (ya está en X)", (train["smoker"] == "yes").to_numpy(float)),
        ("fuma Y bmi > 30", train["fumador_obeso"].to_numpy(float)),
    )
    for etiqueta, columna in controles:
        X_c = X_sin if columna is None else np.column_stack([X_sin, columna])
        media, desvio = _sobre_particiones(X_c, y, 1, OLS[1], aleatorios)
        print(f"       {etiqueta:28s} RMSE {media:8,.0f} ± {desvio:4,.0f}")
        filas.append({"bloque": "control", "caso": etiqueta,
                      "rmse_sin": "", "rmse_con": round(media, 2), "diferencia": ""})

    print("\n¿Estratificar los folds por población reduce la varianza entre folds?\n")
    fuma = (train["smoker"] == "yes").to_numpy()
    obeso = (train["bmi"] > UMBRAL_OBESIDAD).to_numpy()
    poblacion = np.where(~fuma, 0, np.where(~obeso, 1, 2))
    estratificados = _folds_estratificados(poblacion)

    print(f"       {'partición':>10s} {'aleatorio: σ entre folds':>26s} {'estratificado: σ':>19s}")
    sigmas = {"aleatorio": [], "estratificado": []}
    for semilla in SEMILLAS_FOLDS:
        _, s_a = _cv(X_con, y, 2, LASSO[1], aleatorios(semilla))
        _, s_e = _cv(X_con, y, 2, LASSO[1], estratificados(semilla))
        sigmas["aleatorio"].append(s_a)
        sigmas["estratificado"].append(s_e)
        print(f"       {semilla:10d} {s_a:26,.0f} {s_e:19,.0f}")
    for etiqueta, valores in sigmas.items():
        print(f"       promedio {etiqueta:16s} σ = {np.mean(valores):6,.0f}   "
              f"(error estándar = σ/√{K} = {np.mean(valores) / np.sqrt(K):5,.0f})")
        filas.append({"bloque": "estratificar", "caso": f"σ entre folds, {etiqueta}",
                      "rmse_sin": "", "rmse_con": round(float(np.mean(valores)), 2),
                      "diferencia": ""})
    print("\n       Estratificar NO reduce σ: balancear los conteos de cada población no")
    print("       balancea las MAGNITUDES, y lo que mueve el RMSE de un fold es qué")
    print("       individuos extremos concretos le tocaron, no cuántos.")

    print("\n¿Aportan `edad_al_cuadrado` y `bmi_si_fuma`?")
    print("       (partiendo del pipeline con `fumador_obeso`; OLS grado 1, el de producción)\n")
    edad2 = train["edad_al_cuadrado"].to_numpy(float)
    bmifuma = train["bmi_si_fuma"].to_numpy(float)
    variantes = (
        ("base con fumador_obeso (9 features)", X_escalon),
        ("+ edad_al_cuadrado", np.column_stack([X_escalon, edad2])),
        ("+ bmi_si_fuma", np.column_stack([X_escalon, bmifuma])),
        ("+ las dos (pipeline actual)", X_con),
    )
    referencia = None
    for etiqueta, matriz in variantes:
        media, desvio = _sobre_particiones(matriz, y, 1, OLS[1], aleatorios)
        if referencia is None:
            referencia = media
        print(f"       {etiqueta:30s} RMSE {media:8,.0f} ± {desvio:4,.0f}   "
              f"{media - referencia:+8,.0f}")
        filas.append({"bloque": "derivadas", "caso": etiqueta,
                      "rmse_sin": round(referencia, 2), "rmse_con": round(media, 2),
                      "diferencia": round(media - referencia, 2)})
    print("\n       Las dos aportan, y su efecto es aditivo: modelan estructuras distintas")
    print("       (curvatura en edad, y pendiente de bmi entre fumadores). Aun asi la mejora")
    print("       conjunta es menor que un error estandar de la seleccion.")

    print("\n`children`: ¿numérica (actual) o one-hot?\n")
    X_oh = CodificadorCategoricas(
        categoricas=list(CATEGORICAS) + ["children"],
        numericas=["age", "bmi", "fumador_obeso", "edad_al_cuadrado", "bmi_si_fuma"],
    ).ajustar_transformar(train)
    print(f"       features base: numérica {X_con.shape[1]}  ·  one-hot {X_oh.shape[1]}\n")
    print(f"       {'grado':>5s} {'modelo':>7s} {'numérica':>18s} {'one-hot':>18s} {'diferencia':>14s}")
    for grado in (1, 2):
        for nombre, fabrica in (OLS, LASSO):
            num_m, num_s = _sobre_particiones(X_con, y, grado, fabrica, aleatorios)
            oh_m, oh_s = _sobre_particiones(X_oh, y, grado, fabrica, aleatorios)
            print(f"       {grado:5d} {nombre:>7s} {num_m:11,.0f} ± {num_s:4,.0f} "
                  f"{oh_m:11,.0f} ± {oh_s:4,.0f} {oh_m - num_m:+13,.0f}")
            filas.append({"bloque": "children", "caso": f"grado {grado} {nombre}",
                          "rmse_sin": round(num_m, 2), "rmse_con": round(oh_m, 2),
                          "diferencia": round(oh_m - num_m, 2)})

    os.makedirs(RUTA_RESULTADOS, exist_ok=True)
    ruta = os.path.join(RUTA_RESULTADOS, "evidencia_features.csv")
    with open(ruta, "w", newline="") as fh:
        escritor = csv.DictWriter(fh, fieldnames=["bloque", "caso", "rmse_sin", "rmse_con", "diferencia"])
        escritor.writeheader()
        escritor.writerows(filas)
    print(f"\n[ok] {ruta}")


if __name__ == "__main__":
    main()
