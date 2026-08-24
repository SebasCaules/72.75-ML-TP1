"""Correr con: python -m src.experimentos"""

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

TOL_LASSO = 1e-4
MAX_ITER_LASSO = 50000


def diagnostico_de_rango(X_train):
    filas = []
    for grado in GRADOS:
        P_esc, _, _ = preprocesar_completo(X_train, grado)
        valores_singulares = np.linalg.svd(P_esc, compute_uv=False)
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
    folds = k_fold(len(y_train), k=k, semilla=semilla)

    errores_train = []
    errores_val = []
    coefs_no_nulos = []
    n_no_convergio = 0
    n_features = None

    for i_tr, i_va in folds:
        Xtr, ytr = X_train[i_tr], y_train[i_tr]
        Xva, yva = X_train[i_va], y_train[i_va]

        e1 = Estandarizador().ajustar(Xtr)
        Xtr_s, Xva_s = e1.transformar(Xtr), e1.transformar(Xva)

        Ptr, Pva = expandir_polinomica(Xtr_s, grado), expandir_polinomica(Xva_s, grado)

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


def preprocesar_completo(X, grado, e1=None, e2=None):
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


def texto_lambda(config):
    return "sin regularizacion" if config["lambda"] is None else f"lambda={config['lambda']:.2f}"


def imprimir_tabla(filas):
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
    with open(ruta, "w") as fh:
        fh.write("grado,n_features,rmse_train_medio,rmse_train_desvio,rmse_val_medio,rmse_val_desvio\n")
        for f in filas:
            fh.write(
                f"{f['grado']},{f['n_features']},{f['rmse_train_medio']:.6f},"
                f"{f['rmse_train_desvio']:.6f},{f['rmse_val_medio']:.6f},{f['rmse_val_desvio']:.6f}\n"
            )


def guardar_csv_lasso(filas, ruta):
    with open(ruta, "w") as fh:
        fh.write("grado,frac_lambda,lambda,n_features,rmse_train_medio,rmse_val_medio,coefs_no_nulos_medio\n")
        for f in filas:
            fh.write(
                f"{f['grado']},{f['frac_lambda']},{f['lambda']:.8f},{f['n_features']},"
                f"{f['rmse_train_medio']:.6f},{f['rmse_val_medio']:.6f},{f['coefs_no_nulos_medio']:.4f}\n"
            )


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    os.makedirs(RUTA_RESULTADOS, exist_ok=True)

    t0 = time.time()

    df = agregar_derivadas(quitar_duplicados(cargar()))
    print("=" * 100)
    print(f"Filas tras quitar_duplicados: {len(df)}")

    idx_train, idx_test = separar_train_test(len(df), prop_test=0.2, semilla=SEMILLA)
    N_TEST_RESERVADAS = len(idx_test)
    del idx_test
    print(f"Split: {len(idx_train)} train / {N_TEST_RESERVADAS} reservadas para test")

    cod = CodificadorCategoricas().ajustar(df.iloc[idx_train])
    X_train = cod.transformar(df.iloc[idx_train])

    y_train = df[OBJETIVO].values[idx_train]

    print(f"Columnas codificadas ({len(cod.nombres_)}): {cod.nombres_}")
    print("EL TEST NO SE VUELVE A TOCAR HASTA EL PUNTO 5.")

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

    print("\n" + "=" * 100)
    print("PUNTO 3.3 — VALIDACION CRUZADA, LASSO (grados 2, 3, 4)")
    print("=" * 100)

    filas_lasso = []
    resultados_lasso = {}
    for grado in (2, 3, 4):
        P_esc, _, _ = preprocesar_completo(X_train, grado)
        lam_max = lambda_maximo(P_esc, y_train)
        print(f"\ngrado={grado}: lambda_maximo (sobre train completo) = {lam_max:.6f}")

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

    print("\n" + "=" * 100)
    print("PUNTO 4 — TABLA COMPLETA (lineal + lasso)")
    print("=" * 100)
    imprimir_tabla(filas_lineal + filas_lasso)

    print("\n" + "=" * 100)
    print("PUNTO 5 — ELECCION, REENTRENAMIENTO Y TEST")
    print("=" * 100)

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

    mejor = min(elegibles, key=lambda c: c["rmse_val_medio"])
    print(
        f"\n[5.1] Menor rmse_val_medio entre las {len(elegibles)} configuraciones elegibles:\n"
        f"  modelo={mejor['modelo']}  grado={mejor['grado']}  "
        f"{texto_lambda(mejor)}  rmse_val_medio={mejor['rmse_val_medio']:.4f} "
        f"+- {mejor['rmse_val_desvio']:.4f}"
    )

    error_estandar = mejor["rmse_val_desvio"] / np.sqrt(K_FOLDS)
    umbral_1se = mejor["rmse_val_medio"] + error_estandar
    dentro_1se = [c for c in elegibles if c["rmse_val_medio"] <= umbral_1se]
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
