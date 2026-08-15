"""Analisis de sensibilidad al numero de folds k (decision D-22).

Responde dos preguntas distintas que conviene no mezclar:

  A. BARRIDO DE SELECCION — ¿el modelo que elige el punto 5 depende de k?
     Corre la grilla COMPLETA (19 configuraciones: lineal grados 1-4 + Lasso grados 2-4
     por 5 lambdas) y repite la seleccion del punto 5 —ganador crudo, regla de 1 error
     estandar, criterio de parsimonia— para cada k. Si el modelo de produccion sale el
     mismo, la eleccion es robusta a k y no un artefacto de haber puesto 5.

  B. BARRIDO CONTROLADO — ¿como afecta k a los numeros que se reportan?
     En A el ganador crudo cambia de configuracion entre k=5 y k=10, asi que comparar su
     RMSE o su desvio mezcla dos efectos (mas datos por fold, y otro modelo). Acá se FIJA
     la configuracion de produccion (Lasso grado 2, lambda=286,37) y se varia solo k, con
     lo cual la unica causa de las diferencias es k. Es el experimento limpio.

NO TOCA EL CONJUNTO DE TEST. Se separa el split con la misma semilla que el resto del TP y
se descarta idx_test sin usarlo: todo lo de acá vive dentro de train. Eso es lo que permite
correrlo DESPUES de la evaluacion de test sin violar D-09 —no re-evalua test, mide cuan
estable es un procedimiento de seleccion que ya se ejecuto—.

Costo: ~50 minutos. El grado 4 con lambda chico domina (Lasso con p=494 necesita decenas de
miles de barridas) y el costo es lineal en k. Por eso NO forma parte de `src.experimentos`:
se corre aparte, una vez, y deja sus resultados en resultados/ para que las figuras y el
informe los lean sin recalcular.

Correr con:  python3 -m src.sensibilidad_k
"""

import json
import os
import time

import numpy as np

from src.datos import cargar
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

# Los k del barrido de seleccion (experimento A). Se para en 20 porque el costo es lineal
# en k y el barrido controlado (B) ya muestra que de 10 en adelante nada se mueve.
K_SELECCION = (5, 10, 20)

# Los k del barrido controlado (experimento B). Llega hasta n=1070 —o sea leave-one-out—
# porque es barato: grado 2 son 44 columnas y cada ajuste tarda milisegundos. El extremo
# LOO no es una alternativa que se proponga usar; esta para mostrar adonde lleva la curva.
#
# Los valores intermedios (100, 200, 500) NO son decorativos: sin ellos la figura une
# k=50 con k=1070 por una recta que cruza un rango de 20x, y esa recta AFIRMA una forma
# intermedia que nadie midio. Como cada punto extra cuesta segundos, se mide en vez de
# interpolar.
K_CONTROLADO = (5, 10, 20, 50, 100, 200, 500, 1070)

# La configuracion de produccion elegida en el punto 5, que es la que se fija en B.
LAMBDA_PRODUCCION = 286.3701351700539
GRADO_PRODUCCION = 2


# --------------------------------------------------------------------------------------
def preparar_train():
    """Carga, deduplica, separa y codifica — exactamente como src.experimentos.main().

    Devuelve (X_train, y_train). El idx_test se genera (hace falta para que el split de
    train sea el mismo) pero se descarta de inmediato: este modulo no lo usa nunca.
    """
    df = quitar_duplicados(cargar())
    idx_train, idx_test = separar_train_test(len(df), prop_test=0.2, semilla=SEMILLA)
    del idx_test  # que no quede ni la tentacion (misma linea que experimentos.py)

    cod = CodificadorCategoricas().ajustar(df.iloc[idx_train])
    return cod.transformar(df.iloc[idx_train]), df[OBJETIVO].values[idx_train]


# --------------------------------------------------------------------------------------
# Experimento A — ¿cambia el modelo elegido?
# --------------------------------------------------------------------------------------
def grilla_completa(X_train, y_train, k, lam_max):
    """Evalua las 19 configuraciones del punto 5 con k folds. Devuelve la lista de dicts."""
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
    """Repite la seleccion del punto 5 sobre `candidatos`: ganador, 1 ES y parsimonia.

    Es la MISMA logica que src/experimentos.py (D-20 para elegibles, ES = desvio/sqrt(k),
    y desempate por grado -> mayor lambda -> menor error). Se reimplementa acá en vez de
    importarse porque en experimentos.py vive dentro de main(), no en una funcion aparte.
    """
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


# --------------------------------------------------------------------------------------
# Experimento B — ¿como afecta k a los numeros, a configuracion fija?
# --------------------------------------------------------------------------------------
def rmse_agrupado(X_train, y_train, k):
    """RMSE calculado sobre los residuos out-of-fold AGRUPADOS, no promediando RMSEs.

    Por qué hace falta, y por qué no alcanza con `evaluar_con_cv`. El pipeline del TP
    reporta la MEDIA de los k RMSE de fold, y esa media tiene un sesgo que crece cuando
    los folds se achican, por desigualdad de Jensen: la raiz es concava, asi que

        media_i sqrt(ECM_i)  <  sqrt(media_i ECM_i)

    y la brecha crece con la varianza de ECM_i entre folds, que a su vez crece cuando cada
    fold tiene menos puntos. O sea: al subir k, el numero reportado baja SOLO POR LA FORMA
    DE PROMEDIAR, aunque el modelo no haya mejorado nada.

    Ese sesgo se confunde con el efecto que si nos interesa medir (con mas folds, cada
    modelo entrena con mas filas y la CV deja de ser tan pesimista). Para separarlos se
    calcula tambien esta version: se juntan los residuos de validacion de LOS k FOLDS en un
    solo vector de n=1070 y se toma una unica raiz. Al no promediar raices, no hay sesgo de
    Jensen, y lo que quede de tendencia contra k es efecto real de la curva de aprendizaje.

    El bucle de folds repite el pipeline de `evaluar_con_cv` (escalar -> expandir ->
    escalar, siempre ajustando con el sub-train del fold, D-05/D-06). Se reimplementa acá
    en vez de tocar `experimentos.py` porque ese modulo ya produjo los resultados que estan
    en el informe y no se lo modifica por una metrica de diagnostico.
    """
    residuos = np.zeros(len(y_train))
    for i_tr, i_va in k_fold(len(y_train), k=k, semilla=SEMILLA):
        e1 = Estandarizador().ajustar(X_train[i_tr])
        Ptr = expandir_polinomica(e1.transformar(X_train[i_tr]), GRADO_PRODUCCION)
        Pva = expandir_polinomica(e1.transformar(X_train[i_va]), GRADO_PRODUCCION)
        e2 = Estandarizador().ajustar(Ptr)
        modelo = Lasso(lam=LAMBDA_PRODUCCION, max_iter=MAX_ITER_LASSO, tol=TOL_LASSO)
        modelo.ajustar(e2.transformar(Ptr), y_train[i_tr])
        residuos[i_va] = y_train[i_va] - modelo.predecir(e2.transformar(Pva))
    return float(np.sqrt(np.mean(residuos**2)))


def barrido_controlado(X_train, y_train):
    """Varia k con la configuracion de produccion FIJA. Aisla el efecto de k."""
    filas = []
    for k in K_CONTROLADO:
        r = evaluar_con_cv(
            X_train,
            y_train,
            GRADO_PRODUCCION,
            lambda: Lasso(lam=LAMBDA_PRODUCCION, max_iter=MAX_ITER_LASSO, tol=TOL_LASSO),
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


# --------------------------------------------------------------------------------------
def guardar_csv(filas_b, ruta):
    """CSV plano para que src.graficos dibuje la figura 6 sin recalcular nada."""
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
    print(f"SENSIBILIDAD AL NUMERO DE FOLDS (D-22) — n_train = {len(y_train)}")
    print("EL TEST NO SE TOCA EN NINGUN MOMENTO DE ESTE MODULO.")
    print("=" * 100)

    # lambda_maximo se calcula UNA vez sobre el train completo y se reusa para TODO k
    # (D-15). Si se recalculara por k, las grillas de lambda dejarian de ser comparables
    # entre barridos y la comparacion perderia sentido.
    lam_max = {}
    for grado in (2, 3, 4):
        P, _, _ = preprocesar_completo(X_train, grado)
        lam_max[grado] = lambda_maximo(P, y_train)

    print("\n" + "-" * 100)
    print("B — BARRIDO CONTROLADO (configuracion de produccion fija: Lasso grado 2, lambda=286,37)")
    print("-" * 100)
    filas_b = barrido_controlado(X_train, y_train)
    # Se escribe ya: B tarda segundos y A tarda ~45 min. Si A se interrumpe, el CSV que
    # alimenta la figura 6 igual quedo en disco.
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
        print(
            f"  k={k:>3}  ganador: {s['ganador']['modelo']} g{s['ganador']['grado']} "
            f"lam={s['ganador']['lambda']:.1f} rmse={s['ganador']['rmse_val_medio']:.1f}  |  "
            f"ES={s['error_estandar']:.1f}  dentro_1ES={s['n_dentro_1se']}  |  "
            f"PRODUCCION: {s['produccion']['modelo']} g{s['produccion']['grado']} "
            f"lam={s['produccion']['lambda']:.1f}  [{s['segundos']}s]",
            flush=True,
        )

    modelos_produccion = {(s["produccion"]["modelo"], s["produccion"]["grado"], round(s["produccion"]["lambda"], 4))
                          for s in seleccion.values()}
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
