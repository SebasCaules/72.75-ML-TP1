"""Genera las tablas de resultados del informe a partir de los CSV de `resultados/`.

POR QUE EXISTE
--------------
Las dos tablas del punto 4 estaban escritas a mano en `informe/informe.tex`. Mientras el
pipeline no cambiara, funcionaba; en cuanto cambio (D-23), quedaron afirmando numeros de
una corrida que ya no existe, y nada en el repo lo hubiera detectado: LaTeX compila igual
con numeros viejos.

`evaluar_test.py` ya resolvia esto para los numeros de test, generando macros que el
informe lee con \\input. Este modulo hace lo mismo para las tablas de validacion cruzada:
la fuente de verdad son los CSV que produce `experimentos.py`, y el .tex se regenera.

No calcula nada: solo lee, formatea y escribe. Si los CSV no estan, hay que correr antes
`python3 -m src.experimentos`.

Correr con:  python3 -m src.tablas
"""

import csv
import json
import os

RUTA_RESULTADOS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resultados")
RUTA_INFORME = os.path.join(os.path.dirname(os.path.dirname(__file__)), "informe")


def _leer(nombre):
    with open(os.path.join(RUTA_RESULTADOS, nombre), newline="") as fh:
        return list(csv.DictReader(fh))


def num(valor, decimales=1):
    """Numero en notacion espanola dentro de modo matematico: 6034{,}7."""
    return f"{float(valor):,.{decimales}f}".replace(",", "").replace(".", "{,}")


def md(valor, decimales=1):
    """El mismo numero, para las tablas markdown del README."""
    return f"{float(valor):,.{decimales}f}".replace(",", "")


def tabla_lineal(filas, destacar):
    lineas = [
        r"\begin{tabular}{crrr}", r"\toprule",
        r"\textbf{Grado} & \textbf{RMSE train} & \textbf{RMSE validación} & "
        r"\textbf{Características} \\", r"\midrule",
    ]
    for f in filas:
        val = (f"{num(f['rmse_val_medio'])} \\pm {num(f['rmse_val_desvio'])}")
        celda = f"$\\mathbf{{{val}}}$" if int(f["grado"]) == destacar else f"${val}$"
        lineas.append(
            f"{f['grado']} & ${num(f['rmse_train_medio'])} \\pm "
            f"{num(f['rmse_train_desvio'])}$ & {celda} & {f['n_features']} \\\\"
        )
    lineas += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lineas)


def tabla_lasso(filas, mejor):
    lineas = [
        r"\begin{tabular}{ccrrr}", r"\toprule",
        r"\textbf{Grado} & $\boldsymbol{\lambda}$ & \textbf{RMSE train} & "
        r"\textbf{RMSE validación} & \textbf{Coefs. $\neq 0$} \\", r"\midrule",
    ]
    grado_previo = None
    for f in filas:
        if grado_previo is not None and f["grado"] != grado_previo:
            lineas.append(r"\midrule")
        grado_previo = f["grado"]
        es_mejor = (int(f["grado"]) == mejor["grado"]
                    and abs(float(f["lambda"]) - mejor["lambda"]) < 1e-6)
        val = num(f["rmse_val_medio"])
        celda = (f"$\\mathbf{{{val} \\pm {num(mejor['rmse_val_desvio'])}}}$"
                 if es_mejor else val)
        lineas.append(
            f"{f['grado']} & {num(f['lambda'], 2)} & {num(f['rmse_train_medio'])} & "
            f"{celda} & {num(f['coefs_no_nulos_medio'])} \\\\"
        )
    lineas += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lineas)


def main():
    lineal = _leer("cv_lineal.csv")
    lasso = _leer("cv_lasso.csv")
    with open(os.path.join(RUTA_RESULTADOS, "modelo_elegido.json")) as fh:
        elegido = json.load(fh)
    ganador = elegido["ganador_cv"]

    mejor_lineal = min(lineal, key=lambda f: float(f["rmse_val_medio"]))
    contenido = "\n\n".join([
        "% Generado por `python3 -m src.tablas` a partir de resultados/*.csv.",
        "% NO editar a mano: se sobrescribe. Si un numero esta mal, esta mal en el CSV.",
        r"\newcommand{\tablacvlineal}{%", tabla_lineal(lineal, int(mejor_lineal["grado"])), "}",
        r"\newcommand{\tablacvlasso}{%", tabla_lasso(lasso, ganador), "}",
    ])
    destino = os.path.join(RUTA_INFORME, "tablas-cv.tex")
    with open(destino, "w") as fh:
        fh.write(contenido + "\n")
    print(f"[ok] {destino}")

    # La misma tabla en markdown, para pegar en el README.
    print("\n--- markdown para el README ---\n")
    print("| Modelo | Grado | λ | RMSE train | RMSE validación | Features |")
    print("|---|---:|---:|---:|---:|---:|")
    for f in lineal:
        print(f"| lineal | {f['grado']} | — | {md(f['rmse_train_medio'])} ± "
              f"{md(f['rmse_train_desvio'])} | {md(f['rmse_val_medio'])} ± "
              f"{md(f['rmse_val_desvio'])} | {f['n_features']} |")
    for f in lasso:
        if abs(float(f["lambda"]) - ganador["lambda"]) < 1e-6:
            print(f"| lasso | {f['grado']} | {md(f['lambda'], 1)} | "
                  f"{md(f['rmse_train_medio'])} | {md(f['rmse_val_medio'])} | "
                  f"{md(f['coefs_no_nulos_medio'], 0)} |")


if __name__ == "__main__":
    main()
