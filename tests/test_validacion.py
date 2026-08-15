"""Tests de src/validacion.py — sin pytest, asserts planos.

Cada caso compara contra un valor calculado A MANO (no contra la propia función
llamada de otra forma), siguiendo la regla del TP: sin sklearn no hay contra qué
contrastar, así que la única validación honesta es la aritmética.

Correr con:  python -m tests.test_validacion
"""

import numpy as np

from src.validacion import k_fold, resumen_folds, rmse, separar_train_test


# ----------------------------------------------------------------------------------
# rmse
# ----------------------------------------------------------------------------------
def test_rmse_prediccion_perfecta():
    """Si y_pred == y_real en todos los puntos, el error es exactamente 0."""
    y_real = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    assert rmse(y_real, y_pred) == 0.0
    print("ok  rmse da 0 con predicción perfecta")


def test_rmse_error_constante():
    """Si todos los errores valen exactamente 1, RMSE = 1 (MSE=1, sqrt(1)=1).

    Caso trivial pero determinante: distingue una implementación que calcula RMSE de
    una que por error deja el MSE sin la raíz (ahí daría 1 igual, pero...) o que
    promedia mal; se complementa con el caso de errores mixtos de abajo, donde MSE y
    RMSE sí difieren y una implementación sin la raíz fallaría.
    """
    y_real = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 1.0, 1.0, 1.0])
    assert rmse(y_real, y_pred) == 1.0
    print("ok  rmse con error constante de 1 da 1.0")


def test_rmse_errores_mixtos_calculado_a_mano():
    """Caso con dos errores distintos, calculado a mano para separar RMSE de MSE.

    y_real=[0, 10], y_pred=[3, 4]
    errores = [-3, 6]
    cuadrados = [9, 36]
    mse = (9 + 36) / 2 = 22.5
    rmse = sqrt(22.5) = 4.743416490252569...

    Si la implementación se olvidara de la raíz devolvería 22.5 (el MSE), un valor
    bien distinto que este assert detecta.
    """
    y_real = [0.0, 10.0]
    y_pred = [3.0, 4.0]
    esperado = 4.743416490252569
    assert abs(rmse(y_real, y_pred) - esperado) < 1e-9
    print("ok  rmse con errores mixtos coincide con el cálculo a mano")


def test_rmse_acepta_listas_python():
    """rmse tiene que funcionar recibiendo listas, no sólo arrays de numpy."""
    valor = rmse([5.0, 5.0], [5.0, 5.0])
    assert valor == 0.0
    print("ok  rmse acepta listas de python, no sólo arrays")


# ----------------------------------------------------------------------------------
# separar_train_test
# ----------------------------------------------------------------------------------
def test_separar_train_test_tamanos_n1337():
    """Con n=1337 y prop_test=0.2: round(1337*0.2)=round(267.4)=267 en test, 1070 en train.

    1337 - 267 = 1070: se verifica que no se pierde ni se duplica ningún índice al
    hacer el corte.
    """
    n = 1337
    idx_train, idx_test = separar_train_test(n, prop_test=0.2, semilla=42)
    assert len(idx_test) == 267
    assert len(idx_train) == 1070
    assert len(idx_train) + len(idx_test) == n
    print("ok  separar_train_test(n=1337) da 1070 train / 267 test")


def test_separar_train_test_cubre_todo_sin_solapar():
    """train y test tienen que ser disjuntos y su unión, exactamente 0..n-1."""
    n = 200
    idx_train, idx_test = separar_train_test(n, prop_test=0.2, semilla=42)

    conjunto_train = set(idx_train.tolist())
    conjunto_test = set(idx_test.tolist())

    assert len(conjunto_train & conjunto_test) == 0
    assert conjunto_train | conjunto_test == set(range(n))
    # tampoco puede haber índices repetidos DENTRO de cada partición
    assert len(conjunto_train) == len(idx_train)
    assert len(conjunto_test) == len(idx_test)
    print("ok  separar_train_test cubre 0..n-1 sin solapar ni repetir")


def test_separar_train_test_es_reproducible():
    """La misma semilla tiene que dar EXACTAMENTE la misma partición dos veces.

    Es la propiedad que permite que el docente corra el código y vea los mismos
    números que en el informe.
    """
    n = 500
    idx_train_a, idx_test_a = separar_train_test(n, prop_test=0.2, semilla=42)
    idx_train_b, idx_test_b = separar_train_test(n, prop_test=0.2, semilla=42)

    assert np.array_equal(idx_train_a, idx_train_b)
    assert np.array_equal(idx_test_a, idx_test_b)
    print("ok  separar_train_test es reproducible con la misma semilla")


def test_separar_train_test_semillas_distintas_dan_particiones_distintas():
    """Dos semillas distintas tienen que dar (con altísima probabilidad) particiones
    distintas: si no, el barajado no está usando la semilla en absoluto."""
    n = 500
    idx_train_a, _ = separar_train_test(n, prop_test=0.2, semilla=42)
    idx_train_b, _ = separar_train_test(n, prop_test=0.2, semilla=7)

    assert not np.array_equal(idx_train_a, idx_train_b)
    print("ok  semillas distintas producen particiones distintas")


def test_separar_train_test_efectivamente_baraja():
    """El resultado no puede ser simplemente 0..n_train-1 en orden: eso indicaría
    que no se barajó antes de cortar."""
    n = 1337
    idx_train, _ = separar_train_test(n, prop_test=0.2, semilla=42)
    assert not np.array_equal(idx_train, np.arange(len(idx_train)))
    print("ok  separar_train_test efectivamente baraja (no deja el orden original)")


# ----------------------------------------------------------------------------------
# k_fold
# ----------------------------------------------------------------------------------
def test_k_fold_tamanos_n1337_k5():
    """Con n=1337, k=5: 1337 % 5 = 2, así que 2 folds de validación tienen 268
    elementos y 3 folds tienen 267. La suma tiene que dar 1337."""
    folds = k_fold(1337, k=5, semilla=42)
    tamanos = sorted(len(idx_val) for _, idx_val in folds)
    assert tamanos == [267, 267, 267, 268, 268]
    assert sum(tamanos) == 1337
    print("ok  k_fold(n=1337, k=5) reparte 267,267,267,268,268")


def test_k_fold_conjuntos_de_validacion_disjuntos():
    """Los k conjuntos de validación no pueden compartir ningún índice entre sí:
    si compartieran, algún dato validaría más de una vez y otro ninguna."""
    n = 1337
    folds = k_fold(n, k=5, semilla=42)
    vistos = set()
    total = 0
    for _, idx_val in folds:
        conjunto = set(idx_val.tolist())
        assert conjunto.isdisjoint(vistos), "los folds de validación se solapan"
        vistos |= conjunto
        total += len(idx_val)
    assert total == n
    print("ok  los k conjuntos de validación son disjuntos dos a dos")


def test_k_fold_union_de_validaciones_es_todo_el_rango():
    """La unión de los k conjuntos de validación tiene que ser exactamente 0..n-1:
    cada dato valida en algún fold, ninguno queda afuera."""
    n = 1337
    folds = k_fold(n, k=5, semilla=42)
    union = set()
    for _, idx_val in folds:
        union |= set(idx_val.tolist())
    assert union == set(range(n))
    print("ok  la unión de los conjuntos de validación cubre 0..n-1")


def test_k_fold_train_val_disjuntos_y_completos_por_fold():
    """En cada fold individual, idx_train e idx_val tienen que ser disjuntos entre sí
    y su unión tiene que reconstruir el dataset completo (nadie se pierde)."""
    n = 1337
    folds = k_fold(n, k=5, semilla=42)
    for i, (idx_train, idx_val) in enumerate(folds):
        conjunto_train = set(idx_train.tolist())
        conjunto_val = set(idx_val.tolist())
        assert conjunto_train.isdisjoint(conjunto_val), f"fold {i}: train y val se solapan"
        assert conjunto_train | conjunto_val == set(range(n)), f"fold {i}: falta algún índice"
        # tampoco puede haber índices repetidos dentro de cada lado
        assert len(conjunto_train) == len(idx_train)
        assert len(conjunto_val) == len(idx_val)
    print("ok  en cada fold, train y val son disjuntos y cubren 0..n-1 juntos")


def test_k_fold_diferencia_de_tamanos_a_lo_sumo_uno():
    """Los tamaños de validación de los k folds no pueden diferir en más de 1
    elemento entre sí: es la definición de 'lo más parejo posible'."""
    folds = k_fold(1337, k=5, semilla=42)
    tamanos = [len(idx_val) for _, idx_val in folds]
    assert max(tamanos) - min(tamanos) <= 1
    print("ok  los tamaños de validación difieren en a lo sumo 1")


def test_k_fold_caso_chico_calculado_a_mano():
    """n=7, k=3, semilla=42: se verifica el reparto exacto de tamaños (3, 2, 2) y
    que cada bloque de validación es el que corresponde a la permutación de esa
    semilla, calculado a mano corriendo la misma permutación.

    permutación de 0..6 con semilla 42 = [3, 2, 6, 4, 1, 5, 0]
    bloques (np.array_split en 3 partes) = [3,2,6], [4,1], [5,0]
    """
    folds = k_fold(7, k=3, semilla=42)
    assert len(folds) == 3

    tamanos = [len(idx_val) for _, idx_val in folds]
    assert tamanos == [3, 2, 2]

    idx_val_0 = folds[0][1]
    idx_val_1 = folds[1][1]
    idx_val_2 = folds[2][1]
    assert list(idx_val_0) == [3, 2, 6]
    assert list(idx_val_1) == [4, 1]
    assert list(idx_val_2) == [5, 0]

    # y el train del fold 0 tiene que ser exactamente los otros dos bloques concatenados
    idx_train_0 = folds[0][0]
    assert list(idx_train_0) == [4, 1, 5, 0]
    print("ok  k_fold(n=7, k=3) coincide con el reparto calculado a mano")


def test_k_fold_es_reproducible():
    """La misma semilla tiene que producir exactamente los mismos folds dos veces."""
    folds_a = k_fold(1337, k=5, semilla=42)
    folds_b = k_fold(1337, k=5, semilla=42)
    for (train_a, val_a), (train_b, val_b) in zip(folds_a, folds_b):
        assert np.array_equal(train_a, train_b)
        assert np.array_equal(val_a, val_b)
    print("ok  k_fold es reproducible con la misma semilla")


def test_k_fold_cantidad_de_folds():
    """k_fold(n, k) tiene que devolver exactamente k pares, ni más ni menos."""
    folds = k_fold(100, k=5, semilla=42)
    assert len(folds) == 5
    folds_10 = k_fold(100, k=10, semilla=42)
    assert len(folds_10) == 10
    print("ok  k_fold devuelve exactamente k pares")


# ----------------------------------------------------------------------------------
# resumen_folds
# ----------------------------------------------------------------------------------
def test_resumen_folds_calculado_a_mano():
    """errores_train=[1,2,3], errores_val=[4,6,8], calculado a mano:

    media train = (1+2+3)/3 = 2
    desvío train (poblacional, ddof=0) = sqrt( ((1-2)^2+(2-2)^2+(3-2)^2) / 3 )
                                        = sqrt( (1+0+1)/3 ) = sqrt(2/3) = 0.816496580927726

    media val = (4+6+8)/3 = 6
    desvío val (poblacional) = sqrt( ((4-6)^2+(6-6)^2+(8-6)^2) / 3 )
                              = sqrt( (4+0+4)/3 ) = sqrt(8/3) = 1.632993161855452
    """
    resumen = resumen_folds([1.0, 2.0, 3.0], [4.0, 6.0, 8.0])

    assert abs(resumen["rmse_train_medio"] - 2.0) < 1e-9
    assert abs(resumen["rmse_train_desvio"] - 0.816496580927726) < 1e-9
    assert abs(resumen["rmse_val_medio"] - 6.0) < 1e-9
    assert abs(resumen["rmse_val_desvio"] - 1.632993161855452) < 1e-9
    print("ok  resumen_folds coincide con el cálculo a mano (media y desvío poblacional)")


def test_resumen_folds_lista_constante_tiene_desvio_cero():
    """Si todos los folds dieron exactamente el mismo error, el desvío tiene que
    ser 0 (no hay dispersión que medir)."""
    resumen = resumen_folds([5.0, 5.0, 5.0], [10.0, 10.0, 10.0])
    assert resumen["rmse_train_desvio"] == 0.0
    assert resumen["rmse_val_desvio"] == 0.0
    assert resumen["rmse_train_medio"] == 5.0
    assert resumen["rmse_val_medio"] == 10.0
    print("ok  resumen_folds da desvío 0 cuando todos los folds coinciden")


def test_resumen_folds_devuelve_las_cuatro_claves():
    """El diccionario devuelto tiene que tener exactamente las cuatro claves
    documentadas, ni de más ni de menos, para que quien lo consuma no falle por
    un nombre inesperado."""
    resumen = resumen_folds([1.0, 2.0], [3.0, 4.0])
    claves_esperadas = {
        "rmse_train_medio",
        "rmse_train_desvio",
        "rmse_val_medio",
        "rmse_val_desvio",
    }
    assert set(resumen.keys()) == claves_esperadas
    print("ok  resumen_folds devuelve exactamente las cuatro claves esperadas")


def main():
    test_rmse_prediccion_perfecta()
    test_rmse_error_constante()
    test_rmse_errores_mixtos_calculado_a_mano()
    test_rmse_acepta_listas_python()

    test_separar_train_test_tamanos_n1337()
    test_separar_train_test_cubre_todo_sin_solapar()
    test_separar_train_test_es_reproducible()
    test_separar_train_test_semillas_distintas_dan_particiones_distintas()
    test_separar_train_test_efectivamente_baraja()

    test_k_fold_tamanos_n1337_k5()
    test_k_fold_conjuntos_de_validacion_disjuntos()
    test_k_fold_union_de_validaciones_es_todo_el_rango()
    test_k_fold_train_val_disjuntos_y_completos_por_fold()
    test_k_fold_diferencia_de_tamanos_a_lo_sumo_uno()
    test_k_fold_caso_chico_calculado_a_mano()
    test_k_fold_es_reproducible()
    test_k_fold_cantidad_de_folds()

    test_resumen_folds_calculado_a_mano()
    test_resumen_folds_lista_constante_tiene_desvio_cero()
    test_resumen_folds_devuelve_las_cuatro_claves()

    print("TODOS LOS TESTS OK")


if __name__ == "__main__":
    main()
