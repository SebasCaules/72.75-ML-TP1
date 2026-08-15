"""Carga del dataset y análisis exploratorio (punto 1 del enunciado).

Este módulo NO modifica el dataset: sólo lo lee y lo describe. Todas las decisiones de
limpieza viven en `limpieza.py`, para que el análisis que las justifica quede separado
de la transformación que las aplica.

Correr con:  python -m src.datos
"""

from pathlib import Path

import numpy as np
import pandas as pd

# El repo es autocontenido: el CSV vive adentro, no se referencia el vault de apuntes.
RUTA_CSV = Path(__file__).resolve().parent.parent / "data" / "raw" / "insurance.csv"

OBJETIVO = "charges"
CATEGORICAS = ["sex", "smoker", "region"]
NUMERICAS = ["age", "bmi", "children"]


def cargar():
    """Devuelve el dataset crudo, sin ninguna transformación."""
    return pd.read_csv(RUTA_CSV)


# --------------------------------------------------------------------------------------
# Punto 1.1 — variables categóricas
# --------------------------------------------------------------------------------------
def analizar_categoricas(df):
    """Identifica qué columnas son categóricas y con cuántos niveles.

    Distinguimos dos casos, porque piden estrategias distintas:
      - binarias (2 niveles)  -> alcanza UNA columna 0/1, no hacen falta dos
      - nominales (>2 niveles, sin orden) -> one-hot
    """
    filas = []
    for col in df.columns:
        if df[col].dtype == object:
            niveles = sorted(df[col].unique())
            filas.append(
                {
                    "columna": col,
                    "niveles": len(niveles),
                    "valores": ", ".join(map(str, niveles)),
                    "tipo": "binaria" if len(niveles) == 2 else "nominal",
                }
            )
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------------------
# Punto 1.2 — valores faltantes
# --------------------------------------------------------------------------------------
def analizar_faltantes(df):
    """Cuenta nulos por columna y filas duplicadas exactas.

    Un duplicado exacto no es un 'faltante', pero se analiza acá porque es el otro
    defecto de integridad que hay que mirar antes de partir los datos: si una fila
    duplicada cae una copia en train y otra en test, el test deja de ser independiente.
    """
    nulos = df.isna().sum()
    return {
        "nulos_por_columna": nulos[nulos > 0].to_dict(),
        "nulos_totales": int(nulos.sum()),
        "filas": len(df),
        "duplicados_exactos": int(df.duplicated().sum()),
        "duplicados_indices": df.index[df.duplicated(keep=False)].tolist(),
    }


# --------------------------------------------------------------------------------------
# Punto 1.3 — outliers
# --------------------------------------------------------------------------------------
def outliers_iqr(serie, k=1.5):
    """Criterio de Tukey: fuera de [Q1 - k·IQR, Q3 + k·IQR].

    Es el criterio robusto: los cuartiles no se mueven aunque haya valores extremos.
    Con k=1.5 marca 'atípicos'; con k=3.0, 'extremos'.
    """
    q1, q3 = serie.quantile([0.25, 0.75])
    iqr = q3 - q1
    lim_inf, lim_sup = q1 - k * iqr, q3 + k * iqr
    mascara = (serie < lim_inf) | (serie > lim_sup)
    return mascara, lim_inf, lim_sup


def outliers_zscore(serie, umbral=3.0):
    """Criterio del z-score: |x - media| / desvío > umbral.

    OJO: no es robusto. La media y el desvío los corren los propios outliers, así que
    en distribuciones asimétricas (como `charges`) detecta MENOS de los que hay.
    Lo reportamos justamente para mostrar esa diferencia.
    """
    z = (serie - serie.mean()) / serie.std()
    return z.abs() > umbral


def analizar_outliers(df, columnas=None):
    """Compara los dos criterios sobre cada variable numérica."""
    columnas = columnas or NUMERICAS + [OBJETIVO]
    filas = []
    for col in columnas:
        mascara_iqr, lim_inf, lim_sup = outliers_iqr(df[col])
        mascara_z = outliers_zscore(df[col])
        filas.append(
            {
                "variable": col,
                "min": df[col].min(),
                "max": df[col].max(),
                "asimetria": df[col].skew(),
                "lim_inf_iqr": lim_inf,
                "lim_sup_iqr": lim_sup,
                "n_iqr": int(mascara_iqr.sum()),
                "pct_iqr": 100 * mascara_iqr.mean(),
                "n_zscore": int(mascara_z.sum()),
                "pct_zscore": 100 * mascara_z.mean(),
            }
        )
    return pd.DataFrame(filas)


def perfil_de_outliers(df):
    """¿QUIÉNES son los outliers de `charges`? Esta es la tabla que decide el punto 1.3.

    Si los atípicos fueran errores de carga, estarían repartidos al azar entre fumadores
    y no fumadores. Si se concentran en un subgrupo identificable, no son ruido: son
    señal, y eliminarlos sería borrar justo la subpoblación que hay que predecir.
    """
    mascara, _, _ = outliers_iqr(df[OBJETIVO])
    tabla = (
        df.assign(es_outlier=mascara)
        .groupby("es_outlier")
        .agg(
            n=("charges", "size"),
            pct_fumadores=("smoker", lambda s: 100 * (s == "yes").mean()),
            edad_media=("age", "mean"),
            bmi_medio=("bmi", "mean"),
            charges_medio=("charges", "mean"),
        )
    )
    return tabla


# --------------------------------------------------------------------------------------
# Punto 1.4 — características: ¿qué relación tiene cada una con el objetivo?
# --------------------------------------------------------------------------------------
def correlaciones(df):
    """Correlación de Pearson de cada numérica con el objetivo.

    Pearson mide relación LINEAL. Una correlación baja no significa 'no sirve': puede
    haber una relación fuerte pero no lineal, o que sólo aparezca en interacción con
    otra variable. Es exactamente lo que pasa acá con `bmi`.
    """
    num = df.copy()
    num["smoker"] = (num["smoker"] == "yes").astype(int)
    cols = NUMERICAS + ["smoker"]
    return num[cols + [OBJETIVO]].corr()[OBJETIVO].drop(OBJETIVO).sort_values(
        key=abs, ascending=False
    )


def interaccion_fumador_bmi(df, corte_bmi=30):
    """La tabla que explica por qué un modelo lineal se va a quedar corto.

    Cruza fumador × obesidad (bmi > 30, el umbral clínico de la OMS). Si el efecto de
    la obesidad fuera el mismo para fumadores y no fumadores, las dos diferencias serían
    iguales y un modelo aditivo (lineal) alcanzaría. Spoiler: no lo son.
    """
    grupo = np.where(df["bmi"] > corte_bmi, f"bmi>{corte_bmi}", f"bmi<={corte_bmi}")
    return (
        df.groupby([df["smoker"], grupo])["charges"]
        .agg(n="size", charges_medio="mean")
        .unstack()
    )


def escalas(df):
    """Rangos y desvíos de las numéricas: la evidencia para decidir si hay que escalar."""
    return df[NUMERICAS].agg(["min", "max", "mean", "std"]).T.assign(
        rango=lambda t: t["max"] - t["min"]
    )


# --------------------------------------------------------------------------------------
def main():
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    df = cargar()

    print("=" * 78)
    print(f"DATASET: insurance.csv    {df.shape[0]} filas × {df.shape[1]} columnas")
    print("=" * 78)
    print(df.head(5).to_string(index=False))
    print()
    print("Tipos de dato:")
    print(df.dtypes.to_string())

    print("\n" + "-" * 78)
    print("PUNTO 1.1 — VARIABLES CATEGÓRICAS")
    print("-" * 78)
    print(analizar_categoricas(df).to_string(index=False))

    print("\n" + "-" * 78)
    print("PUNTO 1.2 — VALORES FALTANTES")
    print("-" * 78)
    f = analizar_faltantes(df)
    print(f"Nulos totales en el dataset: {f['nulos_totales']}  (sobre {f['filas']} filas)")
    print(f"Columnas con algún nulo:     {f['nulos_por_columna'] or 'ninguna'}")
    print(f"Filas duplicadas exactas:    {f['duplicados_exactos']}")
    if f["duplicados_indices"]:
        print("\nLas filas involucradas:")
        print(df.loc[f["duplicados_indices"]].to_string())

    print("\n" + "-" * 78)
    print("PUNTO 1.3 — OUTLIERS")
    print("-" * 78)
    print(analizar_outliers(df).to_string(index=False))
    print("\n¿Quiénes son los outliers de charges?")
    print(perfil_de_outliers(df).to_string())

    print("\n" + "-" * 78)
    print("PUNTO 1.4 — CARACTERÍSTICAS Y ESCALAS")
    print("-" * 78)
    print("Correlación de Pearson con charges:")
    print(correlaciones(df).to_string())
    print("\nEscalas de las numéricas:")
    print(escalas(df).to_string())
    print("\nInteracción fumador × obesidad (charges medio):")
    print(interaccion_fumador_bmi(df).to_string())


if __name__ == "__main__":
    main()
