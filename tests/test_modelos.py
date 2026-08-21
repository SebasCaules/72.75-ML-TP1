"""Tests de src/modelos.py contra casos de respuesta conocida, calculados a mano.

No hay pytest ni sklearn en este entorno: cada test compara contra un resultado que se
puede derivar analiticamente (una recta exacta, un lambda_maximo que por definicion
anula todo, una norma que tiene que decrecer), nunca contra la funcion consigo misma.

Correr con:  python3 -m tests.test_modelos
"""

import numpy as np

from src.modelos import RegresionLineal, Lasso, lambda_maximo, _soft_threshold


# ----------------------------------------------------------------------------------
# OLS
# ----------------------------------------------------------------------------------
def test_ols_lineal_exacto_multivariado():
    """X aleatorio (semilla fija), y = X@w + b sin ruido: OLS debe recuperar w y b exactos.

    Esto es una respuesta conocida real: no hay ruido, asi que el minimo del error
    cuadratico es CERO y se alcanza exactamente en (w_verdadero, b_verdadero), no en una
    aproximacion.
    """
    rng = np.random.default_rng(42)
    n, p = 200, 5
    X = rng.normal(size=(n, p)) * rng.uniform(1, 10, size=p)  # escalas distintas por columna
    w_verdadero = np.array([3.0, -2.0, 0.5, 10.0, -7.5])
    b_verdadero = 15.0
    y = X @ w_verdadero + b_verdadero

    modelo = RegresionLineal(alfa=0.0).ajustar(X, y)

    assert np.allclose(modelo.coef_, w_verdadero, atol=1e-8), (
        f"coef_ {modelo.coef_} != w_verdadero {w_verdadero}"
    )
    assert np.isclose(modelo.intercepto_, b_verdadero, atol=1e-8), (
        f"intercepto_ {modelo.intercepto_} != b_verdadero {b_verdadero}"
    )
    print("ok  OLS recupera w y b exactos sobre datos lineales sin ruido (multivariado)")


def test_ols_1d_a_mano():
    """x = [1,2,3,4], y = [2,4,6,8]: recta y = 2x, pendiente 2, intercepto 0.

    Caso trivial verificable a mano: la pendiente es exactamente 2 (y = 2x para los
    cuatro puntos) y la recta pasa por el origen.
    """
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0])

    modelo = RegresionLineal(alfa=0.0).ajustar(X, y)

    assert np.allclose(modelo.coef_, [2.0], atol=1e-10), f"coef_ {modelo.coef_} != [2.0]"
    assert np.isclose(modelo.intercepto_, 0.0, atol=1e-10), (
        f"intercepto_ {modelo.intercepto_} != 0.0"
    )
    print("ok  OLS 1-D a mano: x=[1,2,3,4], y=[2,4,6,8] -> pendiente 2, intercepto 0")


def test_ols_rmse_train_sin_ruido_es_cero():
    """Sobre datos exactamente lineales, el RMSE de train de OLS debe ser ~0."""
    rng = np.random.default_rng(7)
    n, p = 150, 4
    X = rng.normal(size=(n, p))
    w_verdadero = rng.normal(size=p) * 5
    b_verdadero = -3.0
    y = X @ w_verdadero + b_verdadero

    modelo = RegresionLineal(alfa=0.0).ajustar(X, y)
    pred = modelo.predecir(X)
    rmse = np.sqrt(np.mean((y - pred) ** 2))

    assert rmse < 1e-8, f"rmse de train {rmse} no es ~0 sobre datos sin ruido"
    print("ok  RMSE de train de OLS sobre datos sin ruido es ~0")


# ----------------------------------------------------------------------------------
# Ridge
# ----------------------------------------------------------------------------------
def _datos_con_ruido(seed, n=300, p=6):
    """Datos con ruido real (no perfectamente lineales), para que OLS ya de coeficientes
    no triviales y Ridge tenga margen para encogerlos sin llegar a anularlos.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    w_verdadero = rng.normal(size=p) * 4
    b_verdadero = 2.0
    ruido = rng.normal(scale=1.0, size=n)
    y = X @ w_verdadero + b_verdadero + ruido
    return X, y


def test_ridge_encoge_pero_no_anula():
    """Ridge con alfa grande da coeficientes de norma L2 menor que OLS (alfa=0), pero
    ninguno queda exactamente en cero: la penalizacion L2 encoge todo parejo, no
    selecciona variables (a diferencia de Lasso).
    """
    X, y = _datos_con_ruido(seed=123)

    ols = RegresionLineal(alfa=0.0).ajustar(X, y)
    ridge = RegresionLineal(alfa=500.0).ajustar(X, y)

    norma_ols = np.linalg.norm(ols.coef_)
    norma_ridge = np.linalg.norm(ridge.coef_)

    assert norma_ridge < norma_ols, (
        f"norma ridge {norma_ridge} no es menor que norma ols {norma_ols}"
    )
    assert np.all(ridge.coef_ != 0.0), f"ridge anulo algun coeficiente: {ridge.coef_}"
    print("ok  Ridge con alfa grande encoge la norma de los coeficientes sin anular ninguno")


def test_ridge_forma_cerrada_caso_chico():
    """Caso chico determinista (6x3): compara RegresionLineal(alfa=lam) contra la
    solucion cerrada de Ridge, calculada aparte con numpy sin reusar el modulo:

        w = (Xc'Xc + lam*I)^-1 Xc'yc         intercepto = y_barra - x_barra @ w

    Es la misma cuenta que la ecuacion normal del docstring de RegresionLineal, pero
    escrita independientemente para no poder compartir un bug con la implementacion.
    """
    X = np.array(
        [
            [1.0, 2.0, 0.0],
            [2.0, 0.0, 1.0],
            [3.0, 1.0, 2.0],
            [0.0, 4.0, 1.0],
            [4.0, 2.0, 3.0],
            [1.0, 1.0, 4.0],
        ]
    )
    y = np.array([5.0, 3.0, 8.0, 6.0, 11.0, 7.0])
    lam = 2.5

    media_X = X.mean(axis=0)
    media_y = y.mean()
    Xc = X - media_X
    yc = y - media_y
    p = X.shape[1]
    w_esperado = np.linalg.inv(Xc.T @ Xc + lam * np.eye(p)) @ (Xc.T @ yc)
    b_esperado = media_y - media_X @ w_esperado

    modelo = RegresionLineal(alfa=lam).ajustar(X, y)

    assert np.allclose(modelo.coef_, w_esperado), (
        f"coef_ {modelo.coef_} != solucion cerrada {w_esperado}"
    )
    assert np.isclose(modelo.intercepto_, b_esperado), (
        f"intercepto_ {modelo.intercepto_} != solucion cerrada {b_esperado}"
    )
    print("ok  Ridge (caso chico 6x3) coincide con la solucion cerrada centrada calculada a mano")


def test_ridge_no_penaliza_intercepto():
    """El test que atrapa el error de implementacion mas comun: si el intercepto se
    penalizara junto con los coeficientes, un alfa enorme lo empujaria hacia 0. Como no
    se penaliza (se obtiene centrando, D-11), un alfa enorme lleva w -> 0 pero el
    intercepto se mantiene en y.mean(), porque intercepto_ = y.mean() - X.mean(0)@w y
    el segundo termino desaparece cuando w -> 0.
    """
    X, y = _datos_con_ruido(seed=99)

    ridge_enorme = RegresionLineal(alfa=1e8).ajustar(X, y)

    assert np.isclose(ridge_enorme.intercepto_, y.mean(), atol=1e-3), (
        f"intercepto_ {ridge_enorme.intercepto_} no esta cerca de y.mean() {y.mean()}"
    )
    assert not np.isclose(ridge_enorme.intercepto_, 0.0, atol=1e-2) or np.isclose(
        y.mean(), 0.0, atol=1e-2
    ), "intercepto_ colapso a 0 en vez de mantenerse en y.mean() (salvo que y.mean() sea 0)"
    print("ok  Ridge con alfa enorme deja intercepto_ ~ y.mean(), no ~ 0 (intercepto no penalizado)")


# ----------------------------------------------------------------------------------
# Lasso
# ----------------------------------------------------------------------------------
def test_lambda_maximo_anula_todos_los_coeficientes():
    """Por definicion, lam >= lambda_maximo(X, y) hace que la condicion de optimalidad
    del subgradiente en w=0 se cumpla para toda coordenada: TODOS los coeficientes
    deben quedar exactamente en 0, y el intercepto en y.mean() (porque
    intercepto_ = y.mean() - X.mean(0)@w y w es el vector nulo).
    """
    rng = np.random.default_rng(11)
    n, p = 100, 8
    X = rng.normal(size=(n, p)) * rng.uniform(1, 5, size=p)
    y = X @ (rng.normal(size=p) * 3) + 7.0 + rng.normal(scale=2.0, size=n)

    lam_max = lambda_maximo(X, y)
    modelo = Lasso(lam=lam_max * 1.05, max_iter=2000).ajustar(X, y)

    assert np.all(modelo.coef_ == 0.0), f"coef_ no son todos cero: {modelo.coef_}"
    assert np.isclose(modelo.intercepto_, y.mean(), atol=1e-10), (
        f"intercepto_ {modelo.intercepto_} != y.mean() {y.mean()}"
    )
    print("ok  Lasso con lam >= lambda_maximo anula TODOS los coeficientes e intercepto_ = y.mean()")


def test_lasso_lam_chico_se_acerca_a_ols():
    """Con un lam casi nulo (lambda_maximo * 1e-6), la penalizacion es despreciable y el
    Lasso debe converger a algo muy cercano a la solucion de OLS.
    """
    X, y = _datos_con_ruido(seed=55, n=400, p=5)

    ols = RegresionLineal(alfa=0.0).ajustar(X, y)
    lam_max = lambda_maximo(X, y)
    lasso = Lasso(lam=lam_max * 1e-6, max_iter=5000, tol=1e-10).ajustar(X, y)

    escala = np.linalg.norm(ols.coef_)
    assert np.allclose(lasso.coef_, ols.coef_, atol=escala * 1e-2), (
        f"lasso.coef_ {lasso.coef_} lejos de ols.coef_ {ols.coef_}"
    )
    print("ok  Lasso con lam ~0 (lambda_maximo * 1e-6) da coeficientes casi iguales a OLS")


def test_lasso_selecciona_variables_con_ceros_exactos():
    """X con 5 columnas donde y depende SOLO de las dos primeras: con un lam intermedio,
    Lasso debe poner en cero exactamente los coeficientes de columnas irrelevantes
    (o al menos una parte sustancial de ellos), que es la propiedad que Ridge no tiene.
    """
    rng = np.random.default_rng(2026)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # columnas irrelevantes, sin relacion con y (ruido puro, no combinacion de x1/x2)
    irrelevantes = rng.normal(size=(n, 3))
    X = np.column_stack([x1, x2, irrelevantes])
    y = 5.0 * x1 - 3.0 * x2 + 1.0 + rng.normal(scale=0.3, size=n)

    lam_max = lambda_maximo(X, y)
    modelo = Lasso(lam=lam_max * 0.3, max_iter=3000).ajustar(X, y)

    coefs_irrelevantes = modelo.coef_[2:]
    assert np.any(coefs_irrelevantes == 0.0), (
        f"ningun coeficiente irrelevante quedo en cero: {coefs_irrelevantes}"
    )
    # las relevantes (x1, x2) deben sobrevivir, con el signo correcto
    assert modelo.coef_[0] > 0, f"coef_[0] (x1) deberia ser positivo: {modelo.coef_[0]}"
    assert modelo.coef_[1] < 0, f"coef_[1] (x2) deberia ser negativo: {modelo.coef_[1]}"
    print("ok  Lasso anula exactamente coeficientes de features irrelevantes con lam intermedio")


def test_lasso_sparsidad_no_creciente_en_lambda():
    """Ejercita Lasso.ajustar directamente (no reimplementa el algoritmo): sobre un caso
    chico fijo, con una grilla de lambda creciente, la cantidad de coeficientes no nulos
    tiene que ser no-creciente en lambda (mas penalizacion nunca puede sumar variables al
    modelo), y con lam >= lambda_maximo tienen que quedar todos exactamente en 0.
    """
    rng = np.random.default_rng(2020)
    n, p = 80, 6
    X = rng.normal(size=(n, p)) * rng.uniform(1, 4, size=p)
    y = X @ (rng.normal(size=p) * 3) + 2.0 + rng.normal(scale=1.0, size=n)

    lam_max = lambda_maximo(X, y)
    grilla = lam_max * np.array([0.01, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.05])

    no_nulos = [
        int(np.sum(Lasso(lam=lam, max_iter=3000, tol=1e-9).ajustar(X, y).coef_ != 0.0))
        for lam in grilla
    ]

    diferencias = np.diff(no_nulos)
    assert np.all(diferencias <= 0), (
        f"la cantidad de coeficientes no nulos aumento al crecer lambda: {no_nulos}"
    )
    assert no_nulos[-1] == 0, (
        f"con lam >= lambda_maximo deberian quedar todos en cero, quedaron {no_nulos[-1]}"
    )
    print(
        f"ok  Lasso.ajustar: cantidad de coeficientes no nulos no-creciente en lambda "
        f"(grilla -> {no_nulos})"
    )


def _objetivo_lasso(Xc, yc, w, lam, n):
    """J(w) = (1/(2n)) ||yc - Xc w||^2 + lam ||w||_1, calculado independientemente del
    modulo (no se reusa ningun metodo interno de Lasso mas alla del soft-threshold, que
    es una formula cerrada y no un lugar donde pueda esconderse un bug de convergencia).
    """
    residuo = yc - Xc @ w
    return 0.5 / n * np.sum(residuo ** 2) + lam * np.sum(np.abs(w))


def test_lasso_objetivo_no_aumenta_entre_barridas():
    """El descenso por coordenadas resuelve, en cada coordenada, el minimo EXACTO del
    subproblema 1-D dejando las demas fijas. Eso garantiza que el objetivo J(w) es
    monotono no creciente barrida a barrida (nunca puede empeorar, porque cada paso es
    un minimo exacto de una funcion convexa en esa coordenada). Este test reimplementa
    el algoritmo de forma independiente (no llama a Lasso.ajustar) para instrumentar
    J(w) despues de cada barrida completa.
    """
    rng = np.random.default_rng(3)
    n, p = 120, 6
    X = rng.normal(size=(n, p)) * rng.uniform(1, 8, size=p)
    y = X @ (rng.normal(size=p) * 3) + 4.0 + rng.normal(scale=1.5, size=n)
    lam = lambda_maximo(X, y) * 0.4

    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    normas = (Xc ** 2).sum(axis=0) / n

    w = np.zeros(p)
    historia = [_objetivo_lasso(Xc, yc, w, lam, n)]
    for _ in range(30):
        for j in range(p):
            if normas[j] == 0:
                w[j] = 0.0
                continue
            residuo_parcial = yc - Xc @ w + Xc[:, j] * w[j]
            rho = Xc[:, j] @ residuo_parcial / n
            w[j] = _soft_threshold(rho, lam) / normas[j]
        historia.append(_objetivo_lasso(Xc, yc, w, lam, n))

    diferencias = np.diff(historia)
    assert np.all(diferencias <= 1e-10), (
        f"el objetivo aumento entre barridas en algun punto: {diferencias[diferencias > 1e-10]}"
    )
    print("ok  El objetivo J(w) de Lasso no aumenta entre barridas consecutivas (descenso monotono)")


def main():
    test_ols_lineal_exacto_multivariado()
    test_ols_1d_a_mano()
    test_ols_rmse_train_sin_ruido_es_cero()
    test_ridge_encoge_pero_no_anula()
    test_ridge_forma_cerrada_caso_chico()
    test_ridge_no_penaliza_intercepto()
    test_lambda_maximo_anula_todos_los_coeficientes()
    test_lasso_lam_chico_se_acerca_a_ols()
    test_lasso_selecciona_variables_con_ceros_exactos()
    test_lasso_sparsidad_no_creciente_en_lambda()
    test_lasso_objetivo_no_aumenta_entre_barridas()
    print("TODOS LOS TESTS OK")


if __name__ == "__main__":
    main()
