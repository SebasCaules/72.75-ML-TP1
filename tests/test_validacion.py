"""Correr con: python -m tests.test_validacion"""

import numpy as np

from src.validacion import k_fold, r2, resumen_folds, rmse, separar_train_test


def test_rmse_prediccion_perfecta():
    y_real = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    assert rmse(y_real, y_pred) == 0.0
    print("ok  rmse da 0 con predicción perfecta")


def test_rmse_error_constante():
    y_real = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 1.0, 1.0, 1.0])
    assert rmse(y_real, y_pred) == 1.0
    print("ok  rmse con error constante de 1 da 1.0")


def test_rmse_errores_mixtos_calculado_a_mano():
    y_real = [0.0, 10.0]
    y_pred = [3.0, 4.0]
    esperado = 4.743416490252569
    assert abs(rmse(y_real, y_pred) - esperado) < 1e-9
    print("ok  rmse con errores mixtos coincide con el cálculo a mano")


def test_rmse_acepta_listas_python():
    valor = rmse([5.0, 5.0], [5.0, 5.0])
    assert valor == 0.0
    print("ok  rmse acepta listas de python, no sólo arrays")


def test_separar_train_test_tamanos_n1337():
    n = 1337
    idx_train, idx_test = separar_train_test(n, prop_test=0.2, semilla=42)
    assert len(idx_test) == 267
    assert len(idx_train) == 1070
    assert len(idx_train) + len(idx_test) == n
    print("ok  separar_train_test(n=1337) da 1070 train / 267 test")


def test_separar_train_test_cubre_todo_sin_solapar():
    n = 200
    idx_train, idx_test = separar_train_test(n, prop_test=0.2, semilla=42)

    conjunto_train = set(idx_train.tolist())
    conjunto_test = set(idx_test.tolist())

    assert len(conjunto_train & conjunto_test) == 0
    assert conjunto_train | conjunto_test == set(range(n))
    assert len(conjunto_train) == len(idx_train)
    assert len(conjunto_test) == len(idx_test)
    print("ok  separar_train_test cubre 0..n-1 sin solapar ni repetir")


def test_separar_train_test_es_reproducible():
    n = 500
    idx_train_a, idx_test_a = separar_train_test(n, prop_test=0.2, semilla=42)
    idx_train_b, idx_test_b = separar_train_test(n, prop_test=0.2, semilla=42)

    assert np.array_equal(idx_train_a, idx_train_b)
    assert np.array_equal(idx_test_a, idx_test_b)
    print("ok  separar_train_test es reproducible con la misma semilla")


def test_separar_train_test_semillas_distintas_dan_particiones_distintas():
    n = 500
    idx_train_a, _ = separar_train_test(n, prop_test=0.2, semilla=42)
    idx_train_b, _ = separar_train_test(n, prop_test=0.2, semilla=7)

    assert not np.array_equal(idx_train_a, idx_train_b)
    print("ok  semillas distintas producen particiones distintas")


def test_separar_train_test_efectivamente_baraja():
    n = 1337
    idx_train, _ = separar_train_test(n, prop_test=0.2, semilla=42)
    assert not np.array_equal(idx_train, np.arange(len(idx_train)))
    print("ok  separar_train_test efectivamente baraja (no deja el orden original)")


def test_k_fold_tamanos_n1337_k5():
    folds = k_fold(1337, k=5, semilla=42)
    tamanos = sorted(len(idx_val) for _, idx_val in folds)
    assert tamanos == [267, 267, 267, 268, 268]
    assert sum(tamanos) == 1337
    print("ok  k_fold(n=1337, k=5) reparte 267,267,267,268,268")


def test_k_fold_conjuntos_de_validacion_disjuntos():
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
    n = 1337
    folds = k_fold(n, k=5, semilla=42)
    union = set()
    for _, idx_val in folds:
        union |= set(idx_val.tolist())
    assert union == set(range(n))
    print("ok  la unión de los conjuntos de validación cubre 0..n-1")


def test_k_fold_train_val_disjuntos_y_completos_por_fold():
    n = 1337
    folds = k_fold(n, k=5, semilla=42)
    for i, (idx_train, idx_val) in enumerate(folds):
        conjunto_train = set(idx_train.tolist())
        conjunto_val = set(idx_val.tolist())
        assert conjunto_train.isdisjoint(conjunto_val), f"fold {i}: train y val se solapan"
        assert conjunto_train | conjunto_val == set(range(n)), f"fold {i}: falta algún índice"
        assert len(conjunto_train) == len(idx_train)
        assert len(conjunto_val) == len(idx_val)
    print("ok  en cada fold, train y val son disjuntos y cubren 0..n-1 juntos")


def test_k_fold_diferencia_de_tamanos_a_lo_sumo_uno():
    folds = k_fold(1337, k=5, semilla=42)
    tamanos = [len(idx_val) for _, idx_val in folds]
    assert max(tamanos) - min(tamanos) <= 1
    print("ok  los tamaños de validación difieren en a lo sumo 1")


def test_k_fold_caso_chico_calculado_a_mano():
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

    idx_train_0 = folds[0][0]
    assert list(idx_train_0) == [4, 1, 5, 0]
    print("ok  k_fold(n=7, k=3) coincide con el reparto calculado a mano")


def test_k_fold_es_reproducible():
    folds_a = k_fold(1337, k=5, semilla=42)
    folds_b = k_fold(1337, k=5, semilla=42)
    for (train_a, val_a), (train_b, val_b) in zip(folds_a, folds_b):
        assert np.array_equal(train_a, train_b)
        assert np.array_equal(val_a, val_b)
    print("ok  k_fold es reproducible con la misma semilla")


def test_k_fold_cantidad_de_folds():
    folds = k_fold(100, k=5, semilla=42)
    assert len(folds) == 5
    folds_10 = k_fold(100, k=10, semilla=42)
    assert len(folds_10) == 10
    print("ok  k_fold devuelve exactamente k pares")


def test_resumen_folds_calculado_a_mano():
    resumen = resumen_folds([1.0, 2.0, 3.0], [4.0, 6.0, 8.0])

    assert abs(resumen["rmse_train_medio"] - 2.0) < 1e-9
    assert abs(resumen["rmse_train_desvio"] - 0.816496580927726) < 1e-9
    assert abs(resumen["rmse_val_medio"] - 6.0) < 1e-9
    assert abs(resumen["rmse_val_desvio"] - 1.632993161855452) < 1e-9
    print("ok  resumen_folds coincide con el cálculo a mano (media y desvío poblacional)")


def test_resumen_folds_lista_constante_tiene_desvio_cero():
    resumen = resumen_folds([5.0, 5.0, 5.0], [10.0, 10.0, 10.0])
    assert resumen["rmse_train_desvio"] == 0.0
    assert resumen["rmse_val_desvio"] == 0.0
    assert resumen["rmse_train_medio"] == 5.0
    assert resumen["rmse_val_medio"] == 10.0
    print("ok  resumen_folds da desvío 0 cuando todos los folds coinciden")


def test_resumen_folds_devuelve_las_cuatro_claves():
    resumen = resumen_folds([1.0, 2.0], [3.0, 4.0])
    claves_esperadas = {
        "rmse_train_medio",
        "rmse_train_desvio",
        "rmse_val_medio",
        "rmse_val_desvio",
    }
    assert set(resumen.keys()) == claves_esperadas
    print("ok  resumen_folds devuelve exactamente las cuatro claves esperadas")


def test_r2_prediccion_perfecta_vale_uno():
    y = np.array([100.0, 250.0, 3000.0, 42.0])
    assert r2(y, y) == 1.0
    print("ok  r2 de una prediccion perfecta vale exactamente 1")


def test_r2_predecir_la_media_vale_cero():
    y = np.array([1.0, 2.0, 3.0, 10.0])
    pred = np.full_like(y, y.mean())
    assert abs(r2(y, pred)) < 1e-12
    print("ok  r2 de predecir siempre la media vale 0")


def test_r2_peor_que_la_media_es_negativo():
    y = np.array([1.0, 2.0, 3.0])
    assert r2(y, np.array([10.0, -5.0, 40.0])) < 0
    print("ok  r2 es negativo si el modelo predice peor que la media")


def test_r2_calculado_a_mano():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.0, 2.0, 3.0, 5.0])
    assert abs(r2(y, pred) - 0.8) < 1e-12
    print("ok  r2 coincide con el valor calculado a mano (0,8)")


def test_r2_es_coherente_con_rmse_sobre_el_mismo_vector():
    rng = np.random.default_rng(7)
    y = rng.normal(10_000, 3_000, size=200)
    pred = y + rng.normal(0, 800, size=200)
    esperado = 1 - (rmse(y, pred) / y.std()) ** 2
    assert abs(r2(y, pred) - esperado) < 1e-12
    print("ok  r2 y rmse son coherentes entre si sobre el mismo vector")


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

    test_r2_prediccion_perfecta_vale_uno()
    test_r2_predecir_la_media_vale_cero()
    test_r2_peor_que_la_media_es_negativo()
    test_r2_calculado_a_mano()
    test_r2_es_coherente_con_rmse_sobre_el_mismo_vector()

    print("TODOS LOS TESTS OK")


if __name__ == "__main__":
    main()
