"""Correr con: python -m tests.test_preproceso"""

from itertools import combinations_with_replacement

import numpy as np
import pandas as pd

from src.datos import agregar_derivadas, cargar
from src.preproceso import (
    CodificadorCategoricas,
    Estandarizador,
    expandir_polinomica,
    nombres_polinomicos,
    quitar_duplicados,
)


def test_quitar_duplicados_csv_real():
    df = cargar()
    assert len(df) == 1338, f"se esperaban 1338 filas crudas, salieron {len(df)}"
    assert df.duplicated().sum() == 1, "se esperaba exactamente 1 fila duplicada"

    limpio = quitar_duplicados(df)
    assert len(limpio) == 1337, f"se esperaban 1337 filas tras deduplicar, salieron {len(limpio)}"
    assert list(limpio.index) == list(range(1337)), "quitar_duplicados no reindexó correctamente"
    fila_195_original = df.loc[195]
    coincidencias = (limpio == fila_195_original.values).all(axis=1)
    assert coincidencias.sum() == 1, "la fila 195 original debería sobrevivir una sola vez"
    print("ok  quitar_duplicados sobre el CSV real deja 1337 filas y reindexa 0..1336")


def test_agregar_derivadas_cubre_las_cuatro_combinaciones():
    df = agregar_derivadas(pd.DataFrame({
        "smoker": ["yes", "yes", "no", "no"],
        "bmi": [35.0, 25.0, 35.0, 25.0],
        "age": [40, 40, 40, 40],
    }))
    assert df["fumador_obeso"].tolist() == [1.0, 0.0, 0.0, 0.0]
    print("ok  agregar_derivadas cubre la tabla de verdad del AND fuma × obeso")


def test_agregar_derivadas_el_umbral_es_estricto():
    df = agregar_derivadas(pd.DataFrame({
        "smoker": ["yes", "yes", "yes"],
        "bmi": [29.999, 30.0, 30.001],
        "age": [40, 40, 40],
    }))
    assert df["fumador_obeso"].tolist() == [0.0, 0.0, 1.0]
    print("ok  el umbral de fumador_obeso es estricto (> 30, no >= 30)")


def test_agregar_derivadas_no_toca_el_dataframe_original():
    original = pd.DataFrame({"smoker": ["yes"], "bmi": [35.0], "age": [40]})
    agregar_derivadas(original)
    for derivada in ("fumador_obeso", "edad_al_cuadrado", "bmi_si_fuma"):
        assert derivada not in original.columns
    print("ok  agregar_derivadas no muta el DataFrame que recibe")


def test_agregar_derivadas_edad_al_cuadrado_valores_literales():
    df = agregar_derivadas(pd.DataFrame({
        "age": [0, 1, 18, 40, 64],
        "bmi": [25.0] * 5,
        "smoker": ["no"] * 5,
    }))
    assert df["edad_al_cuadrado"].tolist() == [0.0, 1.0, 324.0, 1600.0, 4096.0]
    print("ok  edad_al_cuadrado = age^2 en valores calculados a mano")


def test_agregar_derivadas_bmi_si_fuma_es_bmi_o_cero():
    df = agregar_derivadas(pd.DataFrame({
        "smoker": ["yes", "no", "yes", "no"],
        "bmi": [35.0, 35.0, 22.5, 22.5],
        "age": [40, 40, 40, 40],
    }))
    assert df["bmi_si_fuma"].tolist() == [35.0, 0.0, 22.5, 0.0]
    print("ok  bmi_si_fuma vale bmi entre fumadores y 0 entre no fumadores")


def test_derivadas_no_son_redundantes_entre_si():
    df = agregar_derivadas(quitar_duplicados(cargar()))
    columnas = ["age", "bmi", "children", "fumador_obeso", "edad_al_cuadrado", "bmi_si_fuma"]
    for i, a in enumerate(columnas):
        for b in columnas[i + 1:]:
            assert not np.allclose(df[a].to_numpy(float), df[b].to_numpy(float)), (
                f"{a} y {b} son la misma columna")
    print(f"ok  las {len(columnas)} columnas numericas del pipeline son distintas dos a dos")


def test_agregar_derivadas_sobre_el_csv_real_cuenta_lo_que_dice_la_figura_8():
    df = agregar_derivadas(quitar_duplicados(cargar()))
    assert int(df["fumador_obeso"].sum()) == 144
    print("ok  fumador_obeso marca 144 personas en el CSV real, como la figura 8")


def test_codificador_csv_real_nombres_y_forma():
    df = agregar_derivadas(quitar_duplicados(cargar()))
    codificador = CodificadorCategoricas()
    X = codificador.ajustar_transformar(df)

    nombres_esperados = [
        "age", "bmi", "children",
        "fumador_obeso", "edad_al_cuadrado", "bmi_si_fuma",
        "sex=male", "smoker=yes",
        "region=northwest", "region=southeast", "region=southwest",
    ]
    assert codificador.nombres_ == nombres_esperados, (
        f"nombres inesperados: {codificador.nombres_}"
    )
    assert X.shape == (1337, 11), f"forma inesperada: {X.shape}"
    assert "region=northeast" not in codificador.nombres_, "la categoría de referencia no debe aparecer"
    assert X.dtype == np.float64
    print("ok  CodificadorCategoricas sobre el CSV real produce las 11 columnas esperadas, en orden")


def test_codificador_dataframe_chico_valor_literal():
    df_chico = agregar_derivadas(pd.DataFrame({
        "age": [25, 40, 60, 33],
        "bmi": [20.0, 30.5, 45.0, 27.5],
        "children": [0, 2, 1, 3],
        "sex": ["female", "male", "male", "female"],
        "smoker": ["no", "yes", "no", "yes"],
        "region": ["northeast", "southwest", "southeast", "northwest"],
    }))
    codificador = CodificadorCategoricas()
    X = codificador.ajustar_transformar(df_chico)

    esperado = np.array([
        [25.0, 20.0, 0.0,      0.0,        625.0,   0.0,         0.0,      0.0,        0.0, 0.0, 0.0],
        [40.0, 30.5, 2.0,      1.0,       1600.0,  30.5,         1.0,      1.0,        0.0, 0.0, 1.0],
        [60.0, 45.0, 1.0,      0.0,       3600.0,   0.0,         1.0,      0.0,        0.0, 1.0, 0.0],
        [33.0, 27.5, 3.0,      0.0,       1089.0,  27.5,         0.0,      1.0,        1.0, 0.0, 0.0],
    ])
    assert np.allclose(X, esperado), f"salida distinta de la esperada:\n{X}\nesperado:\n{esperado}"

    nombres_esperados = [
        "age", "bmi", "children", "fumador_obeso", "edad_al_cuadrado", "bmi_si_fuma",
        "sex=male", "smoker=yes",
        "region=northwest", "region=southeast", "region=southwest",
    ]
    assert codificador.nombres_ == nombres_esperados
    print("ok  CodificadorCategoricas sobre DataFrame chico coincide con la matriz literal esperada")


def test_codificador_nivel_no_visto_lanza_error():
    df_ajuste = agregar_derivadas(pd.DataFrame({
        "age": [25, 40], "bmi": [20.0, 30.0], "children": [0, 1],
        "sex": ["female", "male"], "smoker": ["no", "yes"],
        "region": ["northeast", "southwest"],
    }))
    df_nuevo = agregar_derivadas(pd.DataFrame({
        "age": [30], "bmi": [22.0], "children": [0],
        "sex": ["female"], "smoker": ["no"],
        "region": ["northwest"],
    }))
    codificador = CodificadorCategoricas()
    codificador.ajustar(df_ajuste)
    try:
        codificador.transformar(df_nuevo)
        lanzo = False
    except ValueError:
        lanzo = True
    assert lanzo, "transformar() con un nivel no visto debería lanzar ValueError"
    print("ok  CodificadorCategoricas.transformar lanza ValueError ante un nivel no visto en ajustar")


def test_estandarizador_media_cero_desvio_uno():
    X = np.array([
        [1.0, 100.0],
        [3.0, 200.0],
        [5.0, 300.0],
    ])
    estandarizador = Estandarizador()
    X_esc = estandarizador.ajustar_transformar(X)

    assert np.allclose(estandarizador.media_, [3.0, 200.0]), f"media inesperada: {estandarizador.media_}"
    assert np.allclose(X_esc.mean(axis=0), [0.0, 0.0], atol=1e-12), "la media tras estandarizar debe ser 0"
    assert np.allclose(X_esc.std(axis=0), [1.0, 1.0], atol=1e-12), "el desvío tras estandarizar debe ser 1"

    desvio_col0 = np.sqrt(8.0 / 3.0)
    esperado_col0 = np.array([1.0, 3.0, 5.0]) - 3.0
    esperado_col0 = esperado_col0 / desvio_col0
    assert np.allclose(X_esc[:, 0], esperado_col0), f"columna 0 estandarizada no coincide: {X_esc[:, 0]}"
    print("ok  Estandarizador produce media 0 y desvío 1 exactos sobre datos con media/desvío conocidos")


def test_estandarizador_ajustar_y_transformar_distintos_no_da_media_cero():
    X_train = np.array([
        [1.0],
        [2.0],
        [3.0],
    ])
    X_val = np.array([
        [10.0],
        [20.0],
        [30.0],
    ])
    estandarizador = Estandarizador()
    estandarizador.ajustar(X_train)
    X_val_esc = estandarizador.transformar(X_val)

    assert not np.allclose(X_val_esc.mean(axis=0), [0.0]), (
        "transformar un fold distinto del ajustado no debería dar media 0 "
        "(si diera 0, se estaría reajustando en vez de reutilizar los parámetros de train)"
    )

    desvio_train = np.sqrt(2.0 / 3.0)
    esperado = (np.array([10.0, 20.0, 30.0]) - 2.0) / desvio_train
    assert np.allclose(X_val_esc[:, 0], esperado), f"transformar no aplicó los parámetros de train: {X_val_esc[:, 0]}"
    print("ok  Estandarizador: ajustar en train y transformar en validación no reajusta (sin fuga)")


def test_estandarizador_columna_constante_sin_nan():
    X = np.array([
        [1.0, 5.0],
        [2.0, 5.0],
        [3.0, 5.0],
    ])
    estandarizador = Estandarizador()
    X_esc = estandarizador.ajustar_transformar(X)

    assert estandarizador.desvio_[1] == 1.0, "el desvío guardado para la columna constante debe ser 1.0"
    assert not np.any(np.isnan(X_esc)), "no debe haber NaN"
    assert not np.any(np.isinf(X_esc)), "no debe haber inf"
    assert np.allclose(X_esc[:, 1], 0.0), "la columna constante debe quedar en ceros tras centrar"
    print("ok  Estandarizador con columna constante no produce NaN/inf y queda en ceros")


def test_expandir_polinomica_grado1_copia_identica():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    X_poly = expandir_polinomica(X, 1)
    assert np.allclose(X_poly, X)
    assert X_poly is not X, "grado=1 debe devolver una copia, no la misma matriz"
    print("ok  expandir_polinomica con grado=1 devuelve una copia idéntica de X")


def test_expandir_polinomica_grado2_valor_literal():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    X = np.column_stack([a, b])

    X_poly = expandir_polinomica(X, 2)
    esperado = np.column_stack([a, b, a * a, a * b, b * b])

    assert X_poly.shape == (3, 5), f"forma inesperada: {X_poly.shape}"
    assert np.allclose(X_poly, esperado), f"salida distinta de la esperada:\n{X_poly}\nesperado:\n{esperado}"
    print("ok  expandir_polinomica grado 2 sobre [a, b] da exactamente [a, b, a^2, a*b, b^2]")


def test_expandir_polinomica_grado3_valor_literal():
    x = np.array([2.0, 3.0])
    X = x.reshape(-1, 1)
    X_poly = expandir_polinomica(X, 3)
    esperado = np.column_stack([x, x ** 2, x ** 3])
    assert np.allclose(X_poly, esperado), f"salida distinta de la esperada:\n{X_poly}\nesperado:\n{esperado}"
    print("ok  expandir_polinomica grado 3 sobre una columna da [x, x^2, x^3]")


def test_expandir_polinomica_conteo_columnas():
    for p, conteos_esperados in ((8, {1: 8, 2: 44, 3: 164, 4: 494}),
                                 (9, {1: 9, 2: 54, 3: 219, 4: 714})):
        X = np.zeros((5, p))
        for grado, esperado in conteos_esperados.items():
            X_poly = expandir_polinomica(X, grado)
            assert X_poly.shape[1] == esperado, (
                f"p={p}, grado {grado}: se esperaban {esperado} columnas, "
                f"salieron {X_poly.shape[1]}"
            )
            conteo_formula = sum(
                len(list(combinations_with_replacement(range(p), g))) for g in range(1, grado + 1)
            )
            assert conteo_formula == esperado, "la fórmula de combinatoria no coincide con el número documentado"
    print("ok  expandir_polinomica da 8/44/164/494 (p=8) y 9/54/219/714 (p=9) columnas, g=1..4")


def test_nombres_polinomicos_coincide_en_cantidad_y_orden():
    nombres = ["age", "bmi"]
    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    for grado in range(1, 5):
        X_poly = expandir_polinomica(X, grado)
        nombres_g = nombres_polinomicos(nombres, grado)
        assert len(nombres_g) == X_poly.shape[1], (
            f"grado {grado}: {len(nombres_g)} nombres vs {X_poly.shape[1]} columnas"
        )

    nombres_grado2 = nombres_polinomicos(nombres, 2)
    assert nombres_grado2 == ["age", "bmi", "age^2", "age*bmi", "bmi^2"], (
        f"nombres de grado 2 inesperados: {nombres_grado2}"
    )
    print("ok  nombres_polinomicos coincide en cantidad con expandir_polinomica para g=1..4 y en texto para g=2")


def test_nombres_polinomicos_features_reales_coincide_con_expandir():
    df = agregar_derivadas(quitar_duplicados(cargar()))
    nombres = CodificadorCategoricas().ajustar(df).nombres_
    X = np.zeros((3, len(nombres)))
    for grado in range(1, 5):
        X_poly = expandir_polinomica(X, grado)
        nombres_g = nombres_polinomicos(nombres, grado)
        assert len(nombres_g) == X_poly.shape[1]
        assert len(set(nombres_g)) == len(nombres_g), "no debería haber nombres de monomio repetidos"
    print(f"ok  nombres_polinomicos con las {len(nombres)} features reales coincide en cantidad, "
          "sin repetidos, g=1..4")


def main():
    test_quitar_duplicados_csv_real()
    test_agregar_derivadas_cubre_las_cuatro_combinaciones()
    test_agregar_derivadas_el_umbral_es_estricto()
    test_agregar_derivadas_no_toca_el_dataframe_original()
    test_agregar_derivadas_edad_al_cuadrado_valores_literales()
    test_agregar_derivadas_bmi_si_fuma_es_bmi_o_cero()
    test_derivadas_no_son_redundantes_entre_si()
    test_agregar_derivadas_sobre_el_csv_real_cuenta_lo_que_dice_la_figura_8()

    test_codificador_csv_real_nombres_y_forma()
    test_codificador_dataframe_chico_valor_literal()
    test_codificador_nivel_no_visto_lanza_error()
    test_estandarizador_media_cero_desvio_uno()
    test_estandarizador_ajustar_y_transformar_distintos_no_da_media_cero()
    test_estandarizador_columna_constante_sin_nan()
    test_expandir_polinomica_grado1_copia_identica()
    test_expandir_polinomica_grado2_valor_literal()
    test_expandir_polinomica_grado3_valor_literal()
    test_expandir_polinomica_conteo_columnas()
    test_nombres_polinomicos_coincide_en_cantidad_y_orden()
    test_nombres_polinomicos_features_reales_coincide_con_expandir()
    print("TODOS LOS TESTS OK")


if __name__ == "__main__":
    main()
