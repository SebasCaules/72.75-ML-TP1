"""Tests de src/preproceso.py con casos de respuesta conocida, calculados a mano.

No hay pytest ni sklearn: cada caso compara contra un resultado que se puede verificar
con lápiz y papel (o, como mucho, con una cuenta de combinatoria), no contra la propia
función ni contra "no explota".

Correr con:  python -m tests.test_preproceso
"""

from itertools import combinations_with_replacement

import numpy as np
import pandas as pd

from src.datos import cargar
from src.preproceso import (
    CodificadorCategoricas,
    Estandarizador,
    expandir_polinomica,
    nombres_polinomicos,
    quitar_duplicados,
)


def test_quitar_duplicados_csv_real():
    """Sobre insurance.csv hay exactamente un par duplicado (índices 195 y 581)."""
    df = cargar()
    assert len(df) == 1338, f"se esperaban 1338 filas crudas, salieron {len(df)}"
    assert df.duplicated().sum() == 1, "se esperaba exactamente 1 fila duplicada"

    limpio = quitar_duplicados(df)
    assert len(limpio) == 1337, f"se esperaban 1337 filas tras deduplicar, salieron {len(limpio)}"
    # reindexado: los índices tienen que ser 0..n-1 contiguos, sin huecos
    assert list(limpio.index) == list(range(1337)), "quitar_duplicados no reindexó correctamente"
    # el índice 195 (primera aparición) se conserva; el 581 (duplicado posterior) se descarta
    fila_195_original = df.loc[195]
    coincidencias = (limpio == fila_195_original.values).all(axis=1)
    assert coincidencias.sum() == 1, "la fila 195 original debería sobrevivir una sola vez"
    print("ok  quitar_duplicados sobre el CSV real deja 1337 filas y reindexa 0..1336")


def test_codificador_csv_real_nombres_y_forma():
    """Sobre el CSV real: 8 columnas, con esos nombres exactos, en ese orden, sin region=northeast."""
    df = quitar_duplicados(cargar())
    codificador = CodificadorCategoricas()
    X = codificador.ajustar_transformar(df)

    nombres_esperados = [
        "age", "bmi", "children",
        "sex=male", "smoker=yes",
        "region=northwest", "region=southeast", "region=southwest",
    ]
    assert codificador.nombres_ == nombres_esperados, (
        f"nombres inesperados: {codificador.nombres_}"
    )
    assert X.shape == (1337, 8), f"forma inesperada: {X.shape}"
    assert "region=northeast" not in codificador.nombres_, "la categoría de referencia no debe aparecer"
    assert X.dtype == np.float64
    print("ok  CodificadorCategoricas sobre el CSV real produce las 8 columnas esperadas, en orden")


def test_codificador_dataframe_chico_valor_literal():
    """Matriz chica armada a mano: se escribe el resultado esperado literal y se compara."""
    df_chico = pd.DataFrame({
        "age": [25, 40, 60, 33],
        "bmi": [20.0, 30.5, 45.0, 27.5],
        "children": [0, 2, 1, 3],
        "sex": ["female", "male", "male", "female"],
        "smoker": ["no", "yes", "no", "yes"],
        "region": ["northeast", "southwest", "southeast", "northwest"],
    })
    codificador = CodificadorCategoricas()
    X = codificador.ajustar_transformar(df_chico)

    # niveles de region ordenados alfabéticamente: northeast(ref), northwest, southeast, southwest
    # niveles de sex: female, male (1 para male) -- niveles de smoker: no, yes (1 para yes)
    esperado = np.array([
        # age,  bmi,  children, sex=male, smoker=yes, region=nw, region=se, region=sw
        [25.0, 20.0, 0.0,      0.0,      0.0,        0.0,       0.0,       0.0],  # northeast
        [40.0, 30.5, 2.0,      1.0,      1.0,        0.0,       0.0,       1.0],  # southwest
        [60.0, 45.0, 1.0,      1.0,      0.0,        0.0,       1.0,       0.0],  # southeast
        [33.0, 27.5, 3.0,      0.0,      1.0,        1.0,       0.0,       0.0],  # northwest
    ])
    assert np.allclose(X, esperado), f"salida distinta de la esperada:\n{X}\nesperado:\n{esperado}"

    nombres_esperados = [
        "age", "bmi", "children", "sex=male", "smoker=yes",
        "region=northwest", "region=southeast", "region=southwest",
    ]
    assert codificador.nombres_ == nombres_esperados
    print("ok  CodificadorCategoricas sobre DataFrame chico coincide con la matriz literal esperada")


def test_codificador_nivel_no_visto_lanza_error():
    """Un nivel que ajustar() no vio tiene que frenar con ValueError, no codificarse en silencio."""
    df_ajuste = pd.DataFrame({
        "age": [25, 40], "bmi": [20.0, 30.0], "children": [0, 1],
        "sex": ["female", "male"], "smoker": ["no", "yes"],
        "region": ["northeast", "southwest"],
    })
    df_nuevo = pd.DataFrame({
        "age": [30], "bmi": [22.0], "children": [0],
        "sex": ["female"], "smoker": ["no"],
        "region": ["northwest"],  # nivel no visto en el ajuste
    })
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
    """Sobre una matriz con media y desvío conocidos, la salida estandarizada es exacta."""
    # columna 0: media 3, desvío poblacional 2 (valores 1,3,5) -- columna 1: media 10, desvío 0 hasta escalar
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

    # valor literal esperado para la columna 0: desvío poblacional = sqrt(((1-3)^2+(3-3)^2+(5-3)^2)/3) = sqrt(8/3)
    desvio_col0 = np.sqrt(8.0 / 3.0)
    esperado_col0 = np.array([1.0, 3.0, 5.0]) - 3.0
    esperado_col0 = esperado_col0 / desvio_col0
    assert np.allclose(X_esc[:, 0], esperado_col0), f"columna 0 estandarizada no coincide: {X_esc[:, 0]}"
    print("ok  Estandarizador produce media 0 y desvío 1 exactos sobre datos con media/desvío conocidos")


def test_estandarizador_ajustar_y_transformar_distintos_no_da_media_cero():
    """Ajustar con una matriz y transformar OTRA no debería dar media 0: es el punto de la separación."""
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

    # valor literal: media_train=2, desvio_train=sqrt(2/3)
    desvio_train = np.sqrt(2.0 / 3.0)
    esperado = (np.array([10.0, 20.0, 30.0]) - 2.0) / desvio_train
    assert np.allclose(X_val_esc[:, 0], esperado), f"transformar no aplicó los parámetros de train: {X_val_esc[:, 0]}"
    print("ok  Estandarizador: ajustar en train y transformar en validación no reajusta (sin fuga)")


def test_estandarizador_columna_constante_sin_nan():
    """Una columna constante (desvío 0) no debe producir NaN ni inf; debe quedar en ceros."""
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
    """grado=1 devuelve X sin cambios (pero una copia, no la misma referencia)."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    X_poly = expandir_polinomica(X, 1)
    assert np.allclose(X_poly, X)
    assert X_poly is not X, "grado=1 debe devolver una copia, no la misma matriz"
    print("ok  expandir_polinomica con grado=1 devuelve una copia idéntica de X")


def test_expandir_polinomica_grado2_valor_literal():
    """X de 2 columnas [a, b], grado 2: [a, b, a^2, a*b, b^2] en ese orden exacto."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    X = np.column_stack([a, b])

    X_poly = expandir_polinomica(X, 2)
    esperado = np.column_stack([a, b, a * a, a * b, b * b])

    assert X_poly.shape == (3, 5), f"forma inesperada: {X_poly.shape}"
    assert np.allclose(X_poly, esperado), f"salida distinta de la esperada:\n{X_poly}\nesperado:\n{esperado}"
    print("ok  expandir_polinomica grado 2 sobre [a, b] da exactamente [a, b, a^2, a*b, b^2]")


def test_expandir_polinomica_grado3_valor_literal():
    """Chequeo adicional a mano: X de 1 columna [x], grado 3 -> [x, x^2, x^3]."""
    x = np.array([2.0, 3.0])
    X = x.reshape(-1, 1)
    X_poly = expandir_polinomica(X, 3)
    esperado = np.column_stack([x, x ** 2, x ** 3])
    assert np.allclose(X_poly, esperado), f"salida distinta de la esperada:\n{X_poly}\nesperado:\n{esperado}"
    print("ok  expandir_polinomica grado 3 sobre una columna da [x, x^2, x^3]")


def test_expandir_polinomica_conteo_columnas_p8():
    """Para p=8 columnas (el caso real del dataset), las cantidades documentadas en la tarea."""
    X = np.zeros((5, 8))
    conteos_esperados = {1: 8, 2: 44, 3: 164, 4: 494}
    for grado, esperado in conteos_esperados.items():
        X_poly = expandir_polinomica(X, grado)
        assert X_poly.shape[1] == esperado, (
            f"grado {grado}: se esperaban {esperado} columnas, salieron {X_poly.shape[1]}"
        )
        # la misma cantidad, calculada independientemente con la fórmula de combinatoria
        p = 8
        conteo_formula = sum(
            len(list(combinations_with_replacement(range(p), g))) for g in range(1, grado + 1)
        )
        assert conteo_formula == esperado, "la fórmula de combinatoria no coincide con el número documentado"
    print("ok  expandir_polinomica con p=8 da 8/44/164/494 columnas para grado 1/2/3/4")


def test_nombres_polinomicos_coincide_en_cantidad_y_orden():
    """len(nombres) == X_poly.shape[1] para g en 1..4, y los nombres de grado 2 coinciden a mano."""
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


def test_nombres_polinomicos_p8_coincide_con_expandir():
    """Sobre el caso real de 8 features, la cantidad de nombres coincide para cada grado."""
    nombres = [
        "age", "bmi", "children", "sex=male", "smoker=yes",
        "region=northwest", "region=southeast", "region=southwest",
    ]
    X = np.zeros((3, 8))
    for grado in range(1, 5):
        X_poly = expandir_polinomica(X, grado)
        nombres_g = nombres_polinomicos(nombres, grado)
        assert len(nombres_g) == X_poly.shape[1]
        assert len(set(nombres_g)) == len(nombres_g), "no debería haber nombres de monomio repetidos"
    print("ok  nombres_polinomicos con las 8 features reales coincide en cantidad, sin repetidos, g=1..4")


def main():
    test_quitar_duplicados_csv_real()
    test_codificador_csv_real_nombres_y_forma()
    test_codificador_dataframe_chico_valor_literal()
    test_codificador_nivel_no_visto_lanza_error()
    test_estandarizador_media_cero_desvio_uno()
    test_estandarizador_ajustar_y_transformar_distintos_no_da_media_cero()
    test_estandarizador_columna_constante_sin_nan()
    test_expandir_polinomica_grado1_copia_identica()
    test_expandir_polinomica_grado2_valor_literal()
    test_expandir_polinomica_grado3_valor_literal()
    test_expandir_polinomica_conteo_columnas_p8()
    test_nombres_polinomicos_coincide_en_cantidad_y_orden()
    test_nombres_polinomicos_p8_coincide_con_expandir()
    print("TODOS LOS TESTS OK")


if __name__ == "__main__":
    main()
