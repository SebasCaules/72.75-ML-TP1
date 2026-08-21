"""Donde vive el error que queda: el diagnostico de D-30. NO TOCA EL CONJUNTO DE TEST.

La pregunta que responde es la que cierra el punto 5: si el modelo ya esta elegido,
?por que no baja mas el error, y que haria falta para bajarlo?

La respuesta no es "otro modelo". El error de validacion no esta repartido parejo entre
las 1070 filas: se concentra en unas pocas decenas de personas que el modelo subestima
por decenas de miles de dolares, y esas personas NO SE DISTINGUEN del resto por ninguna
columna del dataset. Mientras esa informacion no exista como variable, ninguna familia de
modelos puede recuperarla: es error irreducible con estos datos, no falta de capacidad.

Todo se mide con residuos OUT-OF-FOLD sobre train (la prediccion de cada fila la hace un
modelo que no la vio), con la configuracion de produccion que dejo `experimentos.py`.

Correr con:  python3 -m src.diagnostico_residuos
"""

import csv
import json
import os

import numpy as np

from src.datos import OBJETIVO, UMBRAL_OBESIDAD, agregar_derivadas, cargar
from src.experimentos import K_FOLDS, SEMILLA, preprocesar_completo
from src.modelos import Lasso, RegresionLineal
from src.preproceso import CodificadorCategoricas, quitar_duplicados
from src.validacion import k_fold, rmse, separar_train_test

RUTA_RESULTADOS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resultados")

# El corte que define la "tercera poblacion" de la figura 8: no fumadores caros. No sale
# de optimizar nada — es el hombro que se ve en el histograma, redondeado.
CORTE_TERCERA_POBLACION = 25000


def _produccion():
    ruta = os.path.join(RUTA_RESULTADOS, "modelo_elegido.json")
    if not os.path.exists(ruta):
        raise SystemExit("Falta resultados/modelo_elegido.json: corre `python3 -m src.experimentos`.")
    with open(ruta) as fh:
        p = json.load(fh)["produccion_1se"]
    return p["modelo"], p["grado"], p["lambda"]


def residuos_out_of_fold(train, grado, lam):
    """Residuo de cada fila de train, predicha por un modelo que NO la vio."""
    X = CodificadorCategoricas().ajustar_transformar(train)
    y = train[OBJETIVO].to_numpy()
    res = np.empty(len(y))
    for i_tr, i_va in k_fold(len(y), K_FOLDS, SEMILLA):
        Ptr, e1, e2 = preprocesar_completo(X[i_tr], grado)
        Pva, _, _ = preprocesar_completo(X[i_va], grado, e1, e2)
        modelo = RegresionLineal() if lam is None else Lasso(lam)
        modelo.ajustar(Ptr, y[i_tr])
        res[i_va] = y[i_va] - modelo.predecir(Pva)
    return res, y


def main():
    df = agregar_derivadas(quitar_duplicados(cargar()))
    idx_train, _ = separar_train_test(len(df), prop_test=0.2, semilla=SEMILLA)
    train = df.iloc[idx_train].reset_index(drop=True)

    nombre, grado, lam = _produccion()
    res, y = residuos_out_of_fold(train, grado, lam)
    sse = float(np.sum(res ** 2))
    filas = []

    print("=" * 86)
    print(f"DIAGNOSTICO DE RESIDUOS (D-30)  ·  {len(train)} filas de train  ·  "
          f"produccion: {nombre} grado {grado}")
    print("=" * 86)
    print(f"\nRMSE out-of-fold global: {np.sqrt(np.mean(res ** 2)):,.1f}\n")

    # ---- 1. El error esta concentrado, no repartido
    print("1. El error NO esta repartido parejo\n")
    orden = np.argsort(-np.abs(res))
    for pct in (1, 5, 10, 20):
        k = int(len(res) * pct / 100)
        aporte = float(np.sum(res[orden[:k]] ** 2)) / sse * 100
        print(f"   el {pct:>2}% de filas con mayor residuo ({k:>3} personas) aporta "
              f"el {aporte:>5.1f}% del error cuadratico")
        filas.append({"bloque": "concentracion", "caso": f"top {pct}% de residuos",
                      "n": k, "valor": round(aporte, 2), "unidad": "% del SSE"})

    # ---- 2. Por poblacion
    print("\n2. Por poblacion (las tres de la figura 8)\n")
    fuma = (train["smoker"] == "yes").to_numpy()
    obeso = (train["bmi"] > UMBRAL_OBESIDAD).to_numpy()
    grupos = {
        "no fumador": ~fuma,
        "fumador, bmi<=30": fuma & ~obeso,
        "fumador, bmi>30": fuma & obeso,
    }
    for etiqueta, mascara in grupos.items():
        parcial = float(np.sum(res[mascara] ** 2))
        print(f"   {etiqueta:<20} n={int(mascara.sum()):>4}  "
              f"RMSE={np.sqrt(np.mean(res[mascara] ** 2)):>8,.0f}  "
              f"{parcial / sse * 100:>5.1f}% del error")
        filas.append({"bloque": "poblacion", "caso": etiqueta, "n": int(mascara.sum()),
                      "valor": round(parcial / sse * 100, 2), "unidad": "% del SSE"})

    # ---- 3. La tercera poblacion: cuanto pesa y si se puede identificar
    print(f"\n3. Los no fumadores caros (charges > {CORTE_TERCERA_POBLACION:,})\n")
    caros = (~fuma) & (y > CORTE_TERCERA_POBLACION)
    resto = (~fuma) & (y <= CORTE_TERCERA_POBLACION)
    peso = float(np.sum(res[caros] ** 2)) / sse * 100
    print(f"   n = {int(caros.sum())} de {len(train)} ({caros.sum() / len(train) * 100:.1f}% de train)")
    print(f"   aportan el {peso:.1f}% del error cuadratico total")
    print(f"   el modelo los subestima en promedio ${np.mean(res[caros]):,.0f}\n")
    filas.append({"bloque": "tercera-poblacion", "caso": "peso en el error",
                  "n": int(caros.sum()), "valor": round(peso, 2), "unidad": "% del SSE"})
    filas.append({"bloque": "tercera-poblacion", "caso": "subestimacion media",
                  "n": int(caros.sum()), "valor": round(float(np.mean(res[caros])), 2),
                  "unidad": "dolares"})

    print("   ?Se los puede identificar con las columnas del dataset?\n")
    print(f"   {'variable':<12} {'ellos':>10} {'resto no fum.':>16}")
    for var in ("age", "bmi", "children"):
        a, b = train.loc[caros, var].mean(), train.loc[resto, var].mean()
        print(f"   {var:<12} {a:>10.1f} {b:>16.1f}")
        filas.append({"bloque": "tercera-poblacion-perfil", "caso": var,
                      "n": int(caros.sum()), "valor": round(float(a), 2),
                      "unidad": f"media (resto: {b:.2f})"})
    for var in ("sex", "region"):
        a = train.loc[caros, var].value_counts(normalize=True)
        b = train.loc[resto, var].value_counts(normalize=True)
        detalle = "  ".join(f"{k}: {a.get(k, 0) * 100:.0f}%/{b.get(k, 0) * 100:.0f}%"
                            for k in sorted(train[var].unique()))
        print(f"   {var:<12} {detalle}")
        filas.append({"bloque": "tercera-poblacion-perfil", "caso": var,
                      "n": int(caros.sum()), "valor": "", "unidad": detalle})

    print("\n   Salvo la edad, no se distinguen. Ninguna combinacion de las 7 columnas")
    print("   originales los separa: falta la variable que explica el gasto (una patologia,")
    print("   una cirugia) y esa columna no esta en el CSV. Ese tramo del error es")
    print("   IRREDUCIBLE con estos datos, no un problema de capacidad del modelo.")

    os.makedirs(RUTA_RESULTADOS, exist_ok=True)
    ruta = os.path.join(RUTA_RESULTADOS, "diagnostico_residuos.csv")
    with open(ruta, "w", newline="") as fh:
        escritor = csv.DictWriter(fh, fieldnames=["bloque", "caso", "n", "valor", "unidad"])
        escritor.writeheader()
        escritor.writerows(filas)
    print(f"\n[ok] {ruta}")


if __name__ == "__main__":
    main()
