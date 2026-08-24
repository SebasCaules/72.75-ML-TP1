"""Correr con: python -m src.datos"""

from pathlib import Path

import numpy as np
import pandas as pd

RUTA_CSV = Path(__file__).resolve().parent.parent / "data" / "raw" / "insurance.csv"

OBJETIVO = "charges"
CATEGORICAS = ["sex", "smoker", "region"]
NUMERICAS = ["age", "bmi", "children"]

UMBRAL_OBESIDAD = 30

DERIVADAS = ["fumador_obeso", "edad_al_cuadrado", "bmi_si_fuma"]


def agregar_derivadas(df):
    fuma = (df["smoker"] == "yes")
    return df.assign(
        fumador_obeso=(fuma & (df["bmi"] > UMBRAL_OBESIDAD)).astype(float),
        edad_al_cuadrado=(df["age"] ** 2).astype(float),
        bmi_si_fuma=(fuma * df["bmi"]).astype(float),
    )


def cargar():
    return pd.read_csv(RUTA_CSV)


def limpiar(df):
    return df.drop_duplicates(keep="first").reset_index(drop=True)


def cargar_train():
    from src.validacion import separar_train_test

    df = limpiar(cargar())
    idx_train, _ = separar_train_test(len(df), prop_test=0.2, semilla=42)
    return df.iloc[idx_train].reset_index(drop=True)


def analizar_categoricas(df):
    filas = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
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


def analizar_faltantes(df):
    nulos = df.isna().sum()
    return {
        "nulos_por_columna": nulos[nulos > 0].to_dict(),
        "nulos_totales": int(nulos.sum()),
        "filas": len(df),
        "duplicados_exactos": int(df.duplicated().sum()),
        "duplicados_indices": df.index[df.duplicated(keep=False)].tolist(),
    }


def outliers_iqr(serie, k=1.5):
    q1, q3 = serie.quantile([0.25, 0.75])
    iqr = q3 - q1
    lim_inf, lim_sup = q1 - k * iqr, q3 + k * iqr
    mascara = (serie < lim_inf) | (serie > lim_sup)
    return mascara, lim_inf, lim_sup


def outliers_zscore(serie, umbral=3.0):
    z = (serie - serie.mean()) / serie.std()
    return z.abs() > umbral


def analizar_outliers(df, columnas=None):
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


def correlaciones(df):
    num = df.copy()
    num["smoker"] = (num["smoker"] == "yes").astype(int)
    cols = NUMERICAS + ["smoker"]
    return num[cols + [OBJETIVO]].corr()[OBJETIVO].drop(OBJETIVO).sort_values(
        key=abs, ascending=False
    )


def interaccion_fumador_bmi(df, corte_bmi=30):
    grupo = np.where(df["bmi"] > corte_bmi, f"bmi>{corte_bmi}", f"bmi<={corte_bmi}")
    return (
        df.groupby([df["smoker"], grupo])["charges"]
        .agg(n="size", charges_medio="mean")
        .unstack()
    )


def escalon_fumador_obeso(df, tramos=((15, 25), (25, 28), (28, 29), (29, 30),
                                     (30, 31), (31, 32), (32, 35), (35, 55))):
    filas = []
    for lo, hi in tramos:
        en_tramo = (df["bmi"] >= lo) & (df["bmi"] < hi)
        fila = {"tramo_bmi": f"[{lo}, {hi})"}
        for etiqueta, fuma in (("fumadores", True), ("no_fumadores", False)):
            sub = df.loc[en_tramo & ((df["smoker"] == "yes") == fuma), OBJETIVO]
            fila[f"n_{etiqueta}"] = len(sub)
            fila[f"medio_{etiqueta}"] = sub.mean() if len(sub) else float("nan")
        filas.append(fila)
    return pd.DataFrame(filas)


def escalas(df):
    return df[NUMERICAS].agg(["min", "max", "mean", "std"]).T.assign(
        rango=lambda t: t["max"] - t["min"]
    )


def main():
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

    crudo = cargar()
    df = cargar_train()

    print("=" * 78)
    print(f"DATASET CRUDO: insurance.csv    {crudo.shape[0]} filas x {crudo.shape[1]} columnas")
    print(f"ANALISIS SOBRE TRAIN:           {df.shape[0]} filas (80 %, semilla 42)")
    print("=" * 78)
    print(crudo.head(5).to_string(index=False))
    print()
    print("Tipos de dato:")
    print(crudo.dtypes.to_string())

    print("\n" + "-" * 78)
    print("PUNTO 1.2 - VALORES FALTANTES E INTEGRIDAD (sobre el dataset completo)")
    print("-" * 78)
    print("Se mira antes de partir: no requiere ninguna decision estadistica y un")
    print("duplicado repartido entre train y test romperia la independencia del test.")
    f = analizar_faltantes(crudo)
    print(f"Nulos totales en el dataset: {f['nulos_totales']}  (sobre {f['filas']} filas)")
    print(f"Columnas con algun nulo:     {f['nulos_por_columna'] or 'ninguna'}")
    print(f"Filas duplicadas exactas:    {f['duplicados_exactos']}")
    if f["duplicados_indices"]:
        print("\nLas filas involucradas:")
        print(crudo.loc[f["duplicados_indices"]].to_string())
    print(f"\nTras limpiar: {len(limpiar(crudo))} filas -> train {len(df)}, test "
          f"{len(limpiar(crudo)) - len(df)}")

    print("\n" + "=" * 78)
    print("DE ACA EN ADELANTE, TODO SE MIDE SOLO SOBRE LAS %d FILAS DE TRAIN" % len(df))
    print("=" * 78)

    print("\n" + "-" * 78)
    print("PUNTO 1.1 - VARIABLES CATEGORICAS")
    print("-" * 78)
    print(analizar_categoricas(df).to_string(index=False))

    print("\n" + "-" * 78)
    print("PUNTO 1.3 - OUTLIERS")
    print("-" * 78)
    print(analizar_outliers(df).to_string(index=False))
    print("\n Quienes son los outliers de charges?")
    print(perfil_de_outliers(df).to_string())

    print("\n" + "-" * 78)
    print("PUNTO 1.4 - CARACTERISTICAS Y ESCALAS")
    print("-" * 78)
    print("Correlacion de Pearson con charges:")
    print(correlaciones(df).to_string())
    print("\nEscalas de las numericas:")
    print(escalas(df).to_string())
    print("\nInteraccion fumador x obesidad (charges medio):")
    print(interaccion_fumador_bmi(df).to_string())

    print("\n" + "-" * 78)
    print("PUNTO 1.4bis - ES UN ESCALON O UNA PENDIENTE? (justifica `fumador_obeso`)")
    print("-" * 78)
    print(escalon_fumador_obeso(df).to_string(index=False))
    print(
        "\nEntre fumadores el salto [29,30) -> [30,31) es de otro orden que cualquier otro\n"
        "par contiguo, y entre no fumadores el mismo corte casi no mueve nada. Eso es un\n"
        "escalon, no una pendiente: un termino `bmi*smoker` no puede representarlo."
    )

    print("\n" + "-" * 78)
    print("PUNTO 1.4ter - LAS TRES POBLACIONES DE charges")
    print("-" * 78)
    fuma = df["smoker"] == "yes"
    obeso = df["bmi"] > UMBRAL_OBESIDAD
    grupos = {
        "no fumadores": df.loc[~fuma, OBJETIVO],
        "fumadores bmi<=30": df.loc[fuma & ~obeso, OBJETIVO],
        "fumadores bmi>30": df.loc[fuma & obeso, OBJETIVO],
    }
    for nombre, serie in grupos.items():
        print(f"  {nombre:20s} n={len(serie):4d}  mediana={serie.median():10,.0f}  "
              f"min={serie.min():10,.0f}")
    tercero = grupos["fumadores bmi>30"]
    resto = df.loc[~(fuma & obeso), OBJETIVO]
    print(f"\n  El tercer grupo arranca en {tercero.min():,.0f}, por encima del "
          f"{100 * (resto < tercero.min()).mean():.1f} % de todos los demas.")


if __name__ == "__main__":
    main()
