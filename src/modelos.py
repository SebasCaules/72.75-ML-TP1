"""Los tres modelos de regresion del TP, implementados desde cero sobre numpy.

Los tres comparten un mismo truco para no penalizar el intercepto (D-11): en vez de
agregar una columna de unos a X y dejar que el termino independiente entre a la bolsa de
coeficientes regularizados, se CENTRA X e y antes de resolver. Centrar equivale a estimar
el intercepto por separado, fuera de la penalizacion:

    y = X w + b + error   con   Xc = X - media(X),  yc = y - media(y)

Si w resuelve el problema sobre los datos centrados, el termino independiente que hace
que la recta pase por el centroide (media(X), media(y)) es exactamente

    b = media(y) - media(X) @ w

porque en el centroide el modelo no puede tener error sistematico: la suma de residuos
centrados es cero por construccion (es la condicion de primer orden de minimos cuadrados
respecto de b). Penalizar b no tendria sentido ademas: la regularizacion existe para
evitar que el modelo dependa demasiado de una variable particular, y el intercepto no es
una variable, es simplemente donde esta el promedio de la nube de puntos. Encogerlo hacia
cero correria la prediccion media hacia cero sin ninguna razon estadistica.

Correr los tests con:  python3 -m tests.test_modelos
"""

import numpy as np


# ----------------------------------------------------------------------------------
# Regresion lineal (OLS y Ridge)
# ----------------------------------------------------------------------------------
class RegresionLineal:
    """Minimos cuadrados ordinarios (OLS) si alfa=0, o Ridge (penalizacion L2) si alfa>0.

    La ecuacion normal sale de plantear el error cuadratico

        E(w) = ||y - X w||^2   (o, en Ridge, E(w) = ||y - X w||^2 + alfa ||w||^2)

    y derivar respecto de w e igualar a cero. Para OLS:

        dE/dw = -2 X^T (y - X w) = 0   =>   X^T X w = X^T y   =>   w = (X^T X)^-1 X^T y

    Para Ridge se suma alfa*w a la derivada de la penalizacion (d/dw alfa*w^T w = 2 alfa
    w), lo que da

        (X^T X + alfa I) w = X^T y   =>   w = (X^T X + alfa I)^-1 X^T y

    el termino alfa*I es lo que estabiliza la inversion: X^T X + alfa*I es siempre
    definida positiva (y por lo tanto invertible) para alfa>0, aunque X^T X sola sea
    singular o este mal condicionada.
    """

    def __init__(self, alfa=0.0):
        self.alfa = alfa
        self.coef_ = None
        self.intercepto_ = None

    def ajustar(self, X, y):
        """Ajusta el modelo sobre X (n, p) e y (n,).

        No se agrega una columna de unos a X (a diferencia de la convencion de muchos
        libros) porque eso mezclaria el intercepto con los coeficientes regularizados.
        En su lugar se centra, como se explica en el docstring del modulo.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        media_X = X.mean(axis=0)
        media_y = y.mean()
        Xc = X - media_X
        yc = y - media_y

        if self.alfa == 0:
            # lstsq resuelve el sistema por SVD (descomposicion en valores singulares),
            # no invirtiendo X^T X. Invertir explicitamente (np.linalg.inv) es numerica-
            # mente peor: amplifica el error de redondeo en proporcion al numero de
            # condicion de la matriz, y directamente falla (matriz singular) cuando X^T X
            # no tiene rango completo, que es justo lo que pasa con la expansion
            # polinomica de grado alto: en grado 4 con 1364 features hay columnas casi
            # colineales (potencias altas de variables correlacionadas entre si) y X^T X
            # queda casi singular. lstsq, via SVD, devuelve ademas la solucion de NORMA
            # MINIMA cuando el sistema es indeterminado, en vez de explotar con un error
            # de algebra lineal.
            w = np.linalg.lstsq(Xc, yc, rcond=None)[0]
        else:
            p = Xc.shape[1]
            w = np.linalg.solve(Xc.T @ Xc + self.alfa * np.eye(p), Xc.T @ yc)

        self.coef_ = w
        self.intercepto_ = media_y - media_X @ w
        return self

    def predecir(self, X):
        """Prediccion lineal: X @ coef_ + intercepto_."""
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercepto_


# ----------------------------------------------------------------------------------
# Lasso (penalizacion L1) por descenso por coordenadas
# ----------------------------------------------------------------------------------
def lambda_maximo(X, y):
    """El lambda mas chico que anula TODOS los coeficientes del Lasso.

    La condicion de optimalidad del Lasso en w=0 (subgradiente de |w_j| en 0 es
    cualquier valor en [-1, 1]) exige, para que w=0 sea solucion en la coordenada j:

        |x_j^T (y - media(y))| / n <= lam

    para toda columna j (con Xc, yc centrados, sin intercepto de por medio). El menor
    lam que satisface esto para TODAS las coordenadas a la vez es el maximo de esas
    magnitudes. Por encima de ese lambda, el Lasso queda completamente vacio; por debajo
    empiezan a entrar variables una a una. Es lo que permite armar una grilla de lambda
    relativa (por ejemplo lambda_maximo * [1, 0.5, 0.1, ...]) en vez de tantear numeros
    absolutos sin ninguna escala de referencia.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    yc = y - y.mean()
    # X no se centra explicitamente ademas de yc porque x_j^T yc ya es invariante a
    # trasladar X en una constante: (x_j - c)^T yc = x_j^T yc - c * sum(yc) = x_j^T yc,
    # dado que sum(yc) = 0. Centrar X igual, por prolijidad y para no depender de esa
    # identidad en otro lugar del codigo.
    Xc = X - X.mean(axis=0)
    return np.max(np.abs(Xc.T @ yc)) / n


def _soft_threshold(z, g):
    """sign(z) * max(|z| - g, 0): el operador proximal de la norma L1.

    Es la solucion cerrada de minimizar (1/2)(z - w)^2 + g|w| respecto de w. Si |z| <= g,
    el minimo cae exactamente en w=0 (la penalizacion es mas fuerte que el ajuste), y
    de ahi sale que Lasso puede anular coeficientes exactamente, a diferencia de Ridge.
    """
    return np.sign(z) * np.maximum(np.abs(z) - g, 0.0)


class Lasso:
    """Lasso: regresion lineal con penalizacion L1, resuelta por descenso por coordenadas.

    Objetivo (con Xc, yc centrados, intercepto fuera de la penalizacion por D-11):

        J(w) = (1/(2n)) ||yc - Xc w||^2 + lam * ||w||_1

    El descenso por coordenadas actualiza una coordenada w_j por vez, dejando las demas
    fijas, resolviendo exactamente el subproblema 1-D en esa coordenada (que tiene
    solucion cerrada via soft-thresholding), y repite barridas hasta que el cambio maximo
    en w se estabiliza. A diferencia del gradiente descendente, no hace falta elegir una
    tasa de aprendizaje: cada paso es el minimo exacto de esa coordenada.

    Por que L1 selecciona variables y L2 no: el termino de penalizacion de Ridge es
    diferenciable en todos lados (su gradiente es 2*alfa*w, que se anula solo en w=0
    exactamente en el limite), asi que el optimo de Ridge nunca cae justo en un vertice:
    encoge todos los coeficientes proporcionalmente, pero ninguno llega exactamente a
    cero salvo coincidencia. La penalizacion L1, en cambio, tiene un pico no diferenciable
    en w=0 (la derivada salta de -lam a +lam), lo que crea una zona plana: cualquier rho
    con |rho| <= lam hace que el optimo exacto de esa coordenada sea w_j = 0, no un valor
    chico. Geometricamente, la bola unitaria de la norma L1 en p dimensiones es un
    politopo con vertices sobre los ejes coordenados (donde la mayoria de las coordenadas
    son 0), mientras que la bola L2 es una esfera lisa sin vertices; cuando se busca el
    punto de la bola de penalizacion mas cercano a la solucion sin penalizar, L1 tiende a
    "engancharse" en esos vertices. En un modelo polinomico de grado 3 con 164 features,
    esa es la diferencia entre terminar con, digamos, 12 coeficientes no nulos que se
    pueden interpretar y discutir, y un modelo con 164 numeros chiquitos que nadie puede
    explicar en una defensa oral.

    Por que las features TIENEN que estar estandarizadas antes de aplicar L1: la
    penalizacion lam*|w_j| pesa lo mismo (mismo lam) para cualquier coordenada j, sin
    mirar en que unidades esta esa columna. Si x_j esta en una escala 1000 veces mayor
    que otra columna, su coeficiente optimo (sin penalizar) seria 1000 veces mas chico
    para producir el mismo efecto sobre y, y una penalizacion pareja lo aplasta hacia
    cero mucho antes de que le toque a una columna en escala chica: no es que esa
    variable importe menos, es que las unidades en las que vino el dataset la castigan
    de mas. Estandarizar (media 0, desvio 1) pone a todas las columnas en pie de
    igualdad antes de que lam decida cuales sobreviven.
    """

    def __init__(self, lam, max_iter=1000, tol=1e-7):
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercepto_ = None
        self.n_iter_ = 0
        self.convergio_ = False

    def ajustar(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape

        media_X = X.mean(axis=0)
        media_y = y.mean()
        Xc = X - media_X
        yc = y - media_y

        # normas[j] = ||x_j||^2 / n: el coeficiente que multiplica a w_j^2 en el termino
        # cuadratico del subproblema 1-D, que aparece al despejar w_j de la condicion de
        # optimalidad de esa coordenada.
        normas = (Xc ** 2).sum(axis=0) / n

        w = np.zeros(p)
        # RESIDUO MANTENIDO INCREMENTALMENTE. Es la diferencia entre un algoritmo que
        # termina y uno que no.
        #
        # La forma directa de escribir el paso de coordenada es
        #     residuo_parcial = yc - Xc @ w + Xc[:, j] * w[j]
        # pero ese `Xc @ w` cuesta O(n*p) y esta ADENTRO del bucle sobre j, con lo cual
        # cada barrida completa cuesta O(n*p^2). Con p=1364 (grado 4) y n=856 son unos
        # 1590 millones de operaciones por barrida, y el algoritmo no llega a converger en
        # ningun presupuesto razonable.
        #
        # En vez de recalcularlo, se mantiene r = yc - Xc @ w y se actualiza en O(n) cada
        # vez que cambia una coordenada. Cada barrida pasa a costar O(n*p): con p=1364 es
        # unas 1364 veces mas rapido, y el resultado es EL MISMO — es la misma cuenta,
        # reordenada.
        r = yc - Xc @ w
        convergio = False
        iteracion = 0
        for iteracion in range(1, self.max_iter + 1):
            cambio_max = 0.0
            for j in range(p):
                if normas[j] == 0:
                    # columna constante (o toda cero tras centrar): no aporta informacion,
                    # su coeficiente no puede estar determinado y se fija en 0 para no
                    # dividir por cero.
                    if w[j] != 0.0:
                        cambio_max = max(cambio_max, abs(w[j]))
                        r += Xc[:, j] * w[j]
                        w[j] = 0.0
                    continue
                # rho es la correlacion de la columna j con el residuo dejando afuera la
                # contribucion actual de j. Partiendo de r = yc - Xc @ w:
                #     residuo_parcial = r + Xc[:, j] * w[j]
                #     rho = Xc[:, j] @ residuo_parcial / n
                #         = Xc[:, j] @ r / n  +  (||x_j||^2 / n) * w[j]
                #         = Xc[:, j] @ r / n  +  normas[j] * w[j]
                # que es exactamente lo mismo, pero en O(n) en vez de O(n*p).
                rho = Xc[:, j] @ r / n + normas[j] * w[j]
                w_j_nuevo = _soft_threshold(rho, self.lam) / normas[j]
                delta = w_j_nuevo - w[j]
                if delta != 0.0:
                    # el residuo cambia solo por lo que cambio esta coordenada
                    r -= Xc[:, j] * delta
                    cambio_max = max(cambio_max, abs(delta))
                    w[j] = w_j_nuevo
            if cambio_max < self.tol:
                convergio = True
                break
            if iteracion % 50 == 0:
                # Recalculo periodico: las actualizaciones incrementales acumulan error de
                # redondeo de punto flotante a lo largo de miles de barridas. Refrescar r
                # desde cero cada 50 barridas cuesta una sola operacion O(n*p) y elimina
                # la deriva, sin perder la ventaja de velocidad.
                r = yc - Xc @ w

        self.coef_ = w
        self.intercepto_ = media_y - media_X @ w
        self.n_iter_ = iteracion
        self.convergio_ = convergio
        return self

    def predecir(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercepto_
