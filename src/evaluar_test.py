"""Correr con: python -m src.evaluar_test"""

import json
import os
import sys

import numpy as np

from src.datos import OBJETIVO, agregar_derivadas, cargar
from src.experimentos import (
    MAX_ITER_LASSO,
    RUTA_RESULTADOS,
    SEMILLA,
    TOL_LASSO,
    preprocesar_completo,
)
from src.modelos import Lasso, RegresionLineal
from src.preproceso import CodificadorCategoricas, quitar_duplicados
from src.validacion import r2, rmse, separar_train_test

RUTA_INFORME = os.path.join(os.path.dirname(os.path.dirname(__file__)), "informe")


def construir_modelo(config):
    if config["modelo"] == "lasso":
        return Lasso(lam=config["lambda"], max_iter=MAX_ITER_LASSO, tol=TOL_LASSO)
    return RegresionLineal()


def entrenar_y_evaluar(X_train, y_train, X_test, y_test, grado, config):
    Ptr_s, e1, e2 = preprocesar_completo(X_train, grado)
    Pte_s, _, _ = preprocesar_completo(X_test, grado, e1=e1, e2=e2)
    modelo = construir_modelo(config).ajustar(Ptr_s, y_train)
    pred_test = modelo.predecir(Pte_s)
    return {
        "rmse_train": rmse(y_train, modelo.predecir(Ptr_s)),
        "rmse_test": rmse(y_test, pred_test),
        "r2_test": r2(y_test, pred_test),
        "n_features": int(Ptr_s.shape[1]),
        "coefs_no_nulos": int(np.sum(modelo.coef_ != 0)),
    }


def figura_predicho_vs_real(y_test, y_pred, fumador_test, error_test, ruta_figuras):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    AZUL, NARANJA, GRIS, TINTA = "#2a78d6", "#eb6834", "#898781", "#0b0b0b"
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    lim = [min(y_test.min(), y_pred.min()) * 0.95, max(y_test.max(), y_pred.max()) * 1.03]
    ax.plot(lim, lim, "--", color=GRIS, linewidth=1.5, label="predicción perfecta (y=x)")
    ax.scatter(y_test[~fumador_test], y_pred[~fumador_test], s=22, alpha=0.6,
               color=AZUL, edgecolors="none", label="no fumador")
    ax.scatter(y_test[fumador_test], y_pred[fumador_test], s=22, alpha=0.6,
               color=NARANJA, edgecolors="none", label="fumador")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Costo real (charges, dólares)")
    ax.set_ylabel("Costo predicho (dólares)")
    error_es = f"{error_test:,.2f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    ax.set_title(f"Modelo de producción sobre test\nRMSE = ${error_es}  (n={len(y_test)})",
                 color=TINTA)
    ax.legend(loc="upper left", framealpha=1.0)
    ax.grid(color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    os.makedirs(ruta_figuras, exist_ok=True)
    ruta = os.path.join(ruta_figuras, "05-predicho-vs-real.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    return ruta


def main():
    ruta_eleccion = os.path.join(RUTA_RESULTADOS, "modelo_elegido.json")
    if not os.path.exists(ruta_eleccion):
        print("ERROR: falta resultados/modelo_elegido.json.")
        print("Corre primero la seleccion:  python3 -m src.experimentos")
        sys.exit(1)

    with open(ruta_eleccion) as fh:
        eleccion = json.load(fh)

    ruta_previa = os.path.join(RUTA_RESULTADOS, "evaluacion_test.json")
    if os.path.exists(ruta_previa):
        print("=" * 78)
        print("AVISO: el test YA fue evaluado antes.")
        print("=" * 78)
        print("Existe resultados/evaluacion_test.json de una corrida anterior.")
        print()
        print("Volver a evaluarlo no es un error tecnico —da el mismo numero, porque todo")
        print("es determinista con semilla fija— pero SI importa si en el medio se cambio")
        print("algo del modelo mirando el resultado anterior. En ese caso el numero nuevo")
        print("ya no es una estimacion honesta del error de generalizacion.")
        print()
        respuesta = input("Escribi 'si' para evaluar de nuevo: ").strip().lower()
        if respuesta != "si":
            print("Cancelado. No se toco el test.")
            return

    df = agregar_derivadas(quitar_duplicados(cargar()))
    idx_train, idx_test = separar_train_test(len(df), prop_test=0.2, semilla=SEMILLA)

    cod = CodificadorCategoricas().ajustar(df.iloc[idx_train])
    X_train = cod.transformar(df.iloc[idx_train])
    y_train = df[OBJETIVO].values[idx_train]
    X_test = cod.transformar(df.iloc[idx_test])
    y_test = df[OBJETIVO].values[idx_test]

    assert len(idx_train) == eleccion["n_train"], "la particion no coincide con la seleccion"
    assert len(idx_test) == eleccion["n_test_reservadas"], "cambio la cantidad de test"

    print("=" * 78)
    print("EVALUACION SOBRE TEST — UNA SOLA VEZ")
    print("=" * 78)
    print(f"  Train: {len(y_train)} filas    Test: {len(y_test)} filas    semilla: {SEMILLA}")
    print(f"  El codificador y los dos estandarizadores se ajustan SOLO con train.")
    print()

    ganador = eleccion["ganador_cv"]
    produccion = eleccion["produccion_1se"]

    r_ganador = entrenar_y_evaluar(X_train, y_train, X_test, y_test, ganador["grado"], ganador)
    mismo = (ganador["modelo"] == produccion["modelo"]
             and ganador["grado"] == produccion["grado"]
             and ganador["lambda"] == produccion["lambda"])
    r_prod = r_ganador if mismo else entrenar_y_evaluar(
        X_train, y_train, X_test, y_test, produccion["grado"], produccion)

    r_lineal = entrenar_y_evaluar(X_train, y_train, X_test, y_test, 1, {"modelo": "lineal"})
    media_train = float(np.mean(y_train))
    rmse_baseline = rmse(y_test, np.full_like(y_test, media_train, dtype=float))

    def mostrar(titulo, cfg, r):
        lam = f", lambda={cfg['lambda']:.2f}" if cfg.get("lambda") else ""
        print(f"  {titulo}")
        print(f"     modelo={cfg['modelo']} grado={cfg['grado']}{lam}")
        print(f"     RMSE train (reentrenado) = {r['rmse_train']:.4f}")
        print(f"     RMSE test                = {r['rmse_test']:.4f}")
        print(f"     R2 test                  = {r['r2_test']:.4f}")
        print(f"     coeficientes != 0        = {r['coefs_no_nulos']} / {r['n_features']}")
        print()

    mostrar("[5.1] Ganador de la validacion cruzada", ganador, r_ganador)
    if mismo:
        print("  [5.2] El modelo de produccion coincide con el ganador.\n")
    else:
        mostrar("[5.2] Modelo de PRODUCCION (regla de 1 error estandar)", produccion, r_prod)
        print(f"     costo de la simplicidad  = "
              f"{r_prod['rmse_test'] - r_ganador['rmse_test']:+.2f} dolares de RMSE, "
              f"a cambio de {r_prod['n_features']} features en vez de {r_ganador['n_features']}\n")

    es_la_referencia = (produccion["modelo"] == "lineal" and produccion["grado"] == 1)
    if es_la_referencia:
        print("  Referencia, lineal simple (grado 1): ES el modelo de produccion "
              f"(RMSE test = {r_lineal['rmse_test']:.4f}).")
        print("  La regla de 1 error estandar eligio el modelo mas simple del espacio.")
    else:
        print(f"  Referencia, lineal simple (grado 1): RMSE test = {r_lineal['rmse_test']:.4f}")
    print(f"  Baseline, predecir siempre la media ({media_train:.2f}): "
          f"RMSE test = {rmse_baseline:.4f}")
    print()

    sesgo = r_prod["rmse_test"] - produccion["rmse_val_medio"]
    print("=" * 78)
    print("PUNTO 5.3 — QUE RMSE SE ESPERA EN DATOS NUEVOS")
    print("=" * 78)
    print(f"  RMSE de validacion del modelo elegido : {produccion['rmse_val_medio']:.4f}")
    print(f"  RMSE de test                          : {r_prod['rmse_test']:.4f}")
    print(f"  R2 de test                            : {r_prod['r2_test']:.4f}")
    print(f"  diferencia (test - validacion)        : {sesgo:+.4f}")
    print()
    print(f"  La respuesta es {r_prod['rmse_test']:.2f}, el numero de TEST.")
    print("  El de validacion esta sesgado a la baja: esa configuracion se eligio por ser")
    print("  el minimo de varias estimaciones ruidosas, y el minimo de un conjunto de")
    print("  estimaciones ruidosas esta sesgado hacia abajo aunque cada una sea insesgada.")

    salida = {
        "ADVERTENCIA": "Evaluacion unica del test. No re-evaluar tras cambiar el modelo.",
        "ganador_cv": {**ganador, **r_ganador},
        "produccion_1se": {**produccion, **r_prod},
        "referencia_lineal_grado1": r_lineal,
        "baseline_media": {"media_y_train": media_train, "rmse_test": rmse_baseline},
        "diferencia_test_menos_val_produccion": sesgo,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "semilla": SEMILLA,
    }
    os.makedirs(RUTA_RESULTADOS, exist_ok=True)
    with open(os.path.join(RUTA_RESULTADOS, "evaluacion_test.json"), "w") as fh:
        json.dump(salida, fh, indent=2, ensure_ascii=False)

    def es(x, dec=2):
        s = f"{x:,.{dec}f}"
        return s.replace(",", "").replace(".", ",").replace("", ".")

    macros = f"""% GENERADO POR src/evaluar_test.py — no editar a mano.
% Producto de la evaluacion unica del conjunto de test.
\\newcommand{{\\rmsetestproduccion}}{{{es(r_prod['rmse_test'])}}}
\\newcommand{{\\rmsetestganador}}{{{es(r_ganador['rmse_test'])}}}
\\newcommand{{\\rmsetestlineal}}{{{es(r_lineal['rmse_test'])}}}
\\newcommand{{\\rmsetestbaseline}}{{{es(rmse_baseline)}}}
\\newcommand{{\\costosimplicidad}}{{{es(r_prod['rmse_test'] - r_ganador['rmse_test'])}}}
\\newcommand{{\\featuresproduccion}}{{{r_prod['n_features']}}}
\\newcommand{{\\coefsproduccion}}{{{r_prod['coefs_no_nulos']}}}
\\newcommand{{\\featuresganador}}{{{r_ganador['n_features']}}}
\\newcommand{{\\coefsganador}}{{{r_ganador['coefs_no_nulos']}}}
\\newcommand{{\\rmsevalproduccion}}{{{es(produccion['rmse_val_medio'])}}}
\\newcommand{{\\rmsevalganador}}{{{es(ganador['rmse_val_medio'])}}}
\\newcommand{{\\rmsetestredondeado}}{{{r_prod['rmse_test']:,.0f}}}
\\newcommand{{\\rdostest}}{{{es(r_prod['r2_test'], 3)}}}
\\newcommand{{\\testevaluado}}{{si}}
"""
    with open(os.path.join(RUTA_INFORME, "resultados-test.tex"), "w") as fh:
        fh.write(macros)

    Ptr_s, e1, e2 = preprocesar_completo(X_train, produccion["grado"])
    Pte_s, _, _ = preprocesar_completo(X_test, produccion["grado"], e1=e1, e2=e2)
    modelo_prod = construir_modelo(produccion).ajustar(Ptr_s, y_train)
    ruta_fig = figura_predicho_vs_real(
        y_test, modelo_prod.predecir(Pte_s),
        (df.iloc[idx_test]["smoker"] == "yes").to_numpy(),
        r_prod["rmse_test"],
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "figuras"))

    print()
    print(f"  Figura de diagnóstico: {ruta_fig}")
    print("  Guardado en resultados/evaluacion_test.json")
    print("  Macros de LaTeX en informe/resultados-test.tex")
    print("  Recompilar el informe y la presentacion para que tomen los numeros.")


if __name__ == "__main__":
    main()
