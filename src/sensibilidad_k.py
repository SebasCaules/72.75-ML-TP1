"""Correr con: python -m src.sensibilidad_k"""

import json
import os
import time

import numpy as np

from src.datos import agregar_derivadas, cargar
from src.experimentos import (
    FRACCIONES_LAMBDA,
    GRADOS,
    MAX_ITER_LASSO,
    OBJETIVO,
    SEMILLA,
    TOL_LASSO,
    evaluar_con_cv,
    preprocesar_completo,
)
from src.modelos import Lasso, RegresionLineal, lambda_maximo
from src.preproceso import CodificadorCategoricas, Estandarizador, expandir_polinomica, quitar_duplicados
from src.validacion import k_fold, separar_train_test

RUTA_RESULTADOS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resultados")

K_SELECCION = (5, 10, 20)

K_CONTROLADO = (5, 10, 20, 50, 100, 200, 500, 1070)

def config_produccion():
    ruta = os.path.join(RUTA_RESULTADOS, "modelo_elegido.json")
    if not os.path.exists(ruta):
        raise SystemExit(
            "Falta resultados/modelo_elegido.json: corre antes `python3 -m src.experimentos`."
        )
    with open(ruta) as fh:
        produccion = json.load(fh)["produccion_1se"]
    return produccion["modelo"], produccion["grado"], produccion["lambda"]


def fabrica_produccion(lam):
    if lam is None:
        return lambda: RegresionLineal()
    return lambda: Lasso(lam=lam, max_iter=MAX_ITER_LASSO, tol=TOL_LASSO)


def preparar_train():
    df = agregar_derivadas(quitar_duplicados(cargar()))
    idx_train, idx_test = separar_train_test(len(df), prop_test=0.2, semilla=SEMILLA)
    del idx_test

    cod = CodificadorCategoricas().ajustar(df.iloc[idx_train])
    return cod.transformar(df.iloc[idx_train]), df[OBJETIVO].values[idx_train]


def grilla_completa(X_train, y_train, k, lam_max):
    candidatos = []
    for grado in GRADOS:
        r = evaluar_con_cv(X_train, y_train, grado, lambda: RegresionLineal(), k=k, semilla=SEMILLA)
        candidatos.append({**r, "modelo": "lineal", "lambda": None})

    for grado in (2, 3, 4):
        for frac in FRACCIONES_LAMBDA:
            lam = lam_max[grado] * frac
            r = evaluar_con_cv(
                X_train,
                y_train,
                grado,
                lambda lam=lam: Lasso(lam=lam, max_iter=MAX_ITER_LASSO, tol=TOL_LASSO),
                k=k,
                semilla=SEMILLA,
            )
            candidatos.append({**r, "modelo": "lasso", "lambda": lam, "frac_lambda": frac})
    return candidatos


def seleccionar(candidatos, k):
    elegibles = [c for c in candidatos if c.get("n_no_convergio", 0) == 0]
    mejor = min(elegibles, key=lambda c: c["rmse_val_medio"])

    error_estandar = mejor["rmse_val_desvio"] / np.sqrt(k)
    umbral = mejor["rmse_val_medio"] + error_estandar
    dentro = [c for c in elegibles if c["rmse_val_medio"] <= umbral]
    produccion = min(dentro, key=lambda c: (c["grado"], -(c["lambda"] or 0.0), c["rmse_val_medio"]))

    campos = ("modelo", "grado", "lambda", "rmse_val_medio", "rmse_val_desvio", "coefs_no_nulos_medio")
    return {
        "k": k,
        "n_elegibles": len(elegibles),
        "n_descartadas_sin_converger": len(candidatos) - len(elegibles),
        "ganador": {c: mejor[c] for c in campos},
        "error_estandar": error_estandar,
        "umbral_1se": umbral,
        "n_dentro_1se": len(dentro),
        "produccion": {c: produccion[c] for c in campos},
    }


def rmse_agrupado(X_train, y_train, k):
    _, grado, lam = config_produccion()
    fabrica = fabrica_produccion(lam)
    residuos = np.zeros(len(y_train))
    for i_tr, i_va in k_fold(len(y_train), k=k, semilla=SEMILLA):
        e1 = Estandarizador().ajustar(X_train[i_tr])
        Ptr = expandir_polinomica(e1.transformar(X_train[i_tr]), grado)
        Pva = expandir_polinomica(e1.transformar(X_train[i_va]), grado)
        e2 = Estandarizador().ajustar(Ptr)
        modelo = fabrica().ajustar(e2.transformar(Ptr), y_train[i_tr])
        residuos[i_va] = y_train[i_va] - modelo.predecir(e2.transformar(Pva))
    return float(np.sqrt(np.mean(residuos**2)))


def barrido_controlado(X_train, y_train):
    _, grado, lam = config_produccion()
    filas = []
    for k in K_CONTROLADO:
        r = evaluar_con_cv(
            X_train,
            y_train,
            grado,
            fabrica_produccion(lam),
            k=k,
            semilla=SEMILLA,
        )
        n = len(y_train)
        filas.append(
            {
                "k": k,
                "n_val_por_fold": round(n / k),
                "n_train_por_fold": n - round(n / k),
                "rmse_val_medio": r["rmse_val_medio"],
                "rmse_val_agrupado": rmse_agrupado(X_train, y_train, k),
                "rmse_train_medio": r["rmse_train_medio"],
                "sigma_folds": r["rmse_val_desvio"],
                "error_estandar": r["rmse_val_desvio"] / np.sqrt(k),
            }
        )
        print(
            f"  k={k:>3}  n_val/fold={filas[-1]['n_val_por_fold']:>4}  "
            f"rmse_val(media de folds)={r['rmse_val_medio']:>8.1f}  "
            f"rmse_val(agrupado)={filas[-1]['rmse_val_agrupado']:>8.1f}  "
            f"sigma={r['rmse_val_desvio']:>8.1f}  ES={filas[-1]['error_estandar']:>7.1f}",
            flush=True,
        )
    return filas


def guardar_csv(filas_b, ruta):
    with open(ruta, "w", newline="") as fh:
        fh.write(
            "k,n_val_por_fold,n_train_por_fold,rmse_val_medio,rmse_val_agrupado,"
            "rmse_train_medio,sigma_folds,error_estandar\n"
        )
        for f in filas_b:
            fh.write(
                f"{f['k']},{f['n_val_por_fold']},{f['n_train_por_fold']},"
                f"{f['rmse_val_medio']:.6f},{f['rmse_val_agrupado']:.6f},"
                f"{f['rmse_train_medio']:.6f},"
                f"{f['sigma_folds']:.6f},{f['error_estandar']:.6f}\n"
            )


def main():
    import sys

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    os.makedirs(RUTA_RESULTADOS, exist_ok=True)
    t0 = time.time()

    X_train, y_train = preparar_train()
    print("=" * 100)
    print(f"SENSIBILIDAD AL NUMERO DE FOLDS — n_train = {len(y_train)}")
    print("EL TEST NO SE TOCA EN NINGUN MOMENTO DE ESTE MODULO.")
    print("=" * 100)

    lam_max = {}
    for grado in (2, 3, 4):
        P, _, _ = preprocesar_completo(X_train, grado)
        lam_max[grado] = lambda_maximo(P, y_train)

    print("\n" + "-" * 100)
    modelo_p, grado_p, lam_p = config_produccion()
    descripcion_p = (f"{modelo_p} grado {grado_p}"
                     + (f", lambda={lam_p:.2f}" if lam_p is not None else ", sin regularizacion"))
    print(f"B — BARRIDO CONTROLADO (configuracion de produccion fija: {descripcion_p})")
    print("-" * 100)
    filas_b = barrido_controlado(X_train, y_train)
    guardar_csv(filas_b, os.path.join(RUTA_RESULTADOS, "sensibilidad_k.csv"))

    print("\n" + "-" * 100)
    print("A — BARRIDO DE SELECCION (grilla completa, 19 configuraciones por k)")
    print("-" * 100)
    seleccion = {}
    for k in K_SELECCION:
        tk = time.time()
        candidatos = grilla_completa(X_train, y_train, k, lam_max)
        seleccion[k] = {**seleccionar(candidatos, k), "segundos": round(time.time() - tk, 1)}
        s = seleccion[k]
        def lam_txt(cfg):
            return "sin reg." if cfg["lambda"] is None else f"lam={cfg['lambda']:.1f}"

        print(
            f"  k={k:>3}  ganador: {s['ganador']['modelo']} g{s['ganador']['grado']} "
            f"{lam_txt(s['ganador'])} rmse={s['ganador']['rmse_val_medio']:.1f}  |  "
            f"ES={s['error_estandar']:.1f}  dentro_1ES={s['n_dentro_1se']}  |  "
            f"PRODUCCION: {s['produccion']['modelo']} g{s['produccion']['grado']} "
            f"{lam_txt(s['produccion'])}  [{s['segundos']}s]",
            flush=True,
        )

    modelos_produccion = {
        (s["produccion"]["modelo"], s["produccion"]["grado"],
         None if s["produccion"]["lambda"] is None else round(s["produccion"]["lambda"], 4))
        for s in seleccion.values()
    }
    estable = len(modelos_produccion) == 1
    print(f"\n  Modelo de produccion identico en los {len(K_SELECCION)} valores de k: "
          f"{'SI' if estable else 'NO'}")

    salida = {
        "n_train": len(y_train),
        "semilla": SEMILLA,
        "seleccion": {str(k): v for k, v in seleccion.items()},
        "controlado": filas_b,
        "produccion_estable_en_k": estable,
        "segundos_totales": round(time.time() - t0, 1),
    }
    with open(os.path.join(RUTA_RESULTADOS, "sensibilidad_k.json"), "w") as fh:
        json.dump(salida, fh, indent=2)
    guardar_csv(filas_b, os.path.join(RUTA_RESULTADOS, "sensibilidad_k.csv"))

    print(f"\nListo en {salida['segundos_totales']}s. "
          f"Escrito: resultados/sensibilidad_k.json y resultados/sensibilidad_k.csv")


if __name__ == "__main__":
    main()
