import numpy as np


class RegresionLineal:
    def __init__(self, alfa=0.0):
        self.alfa = alfa
        self.coef_ = None
        self.intercepto_ = None

    def ajustar(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        media_X = X.mean(axis=0)
        media_y = y.mean()
        Xc = X - media_X
        yc = y - media_y

        if self.alfa == 0:
            w = np.linalg.lstsq(Xc, yc, rcond=None)[0]
        else:
            p = Xc.shape[1]
            w = np.linalg.solve(Xc.T @ Xc + self.alfa * np.eye(p), Xc.T @ yc)

        self.coef_ = w
        self.intercepto_ = media_y - media_X @ w
        return self

    def predecir(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercepto_


def lambda_maximo(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    yc = y - y.mean()
    Xc = X - X.mean(axis=0)
    return np.max(np.abs(Xc.T @ yc)) / n


def _soft_threshold(z, g):
    return np.sign(z) * np.maximum(np.abs(z) - g, 0.0)


class Lasso:
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

        normas = (Xc ** 2).sum(axis=0) / n

        w = np.zeros(p)
        r = yc - Xc @ w
        convergio = False
        iteracion = 0
        for iteracion in range(1, self.max_iter + 1):
            cambio_max = 0.0
            for j in range(p):
                if normas[j] == 0:
                    if w[j] != 0.0:
                        cambio_max = max(cambio_max, abs(w[j]))
                        r += Xc[:, j] * w[j]
                        w[j] = 0.0
                    continue
                rho = Xc[:, j] @ r / n + normas[j] * w[j]
                w_j_nuevo = _soft_threshold(rho, self.lam) / normas[j]
                delta = w_j_nuevo - w[j]
                if delta != 0.0:
                    r -= Xc[:, j] * delta
                    cambio_max = max(cambio_max, abs(delta))
                    w[j] = w_j_nuevo
            if cambio_max < self.tol:
                convergio = True
                break
            if iteracion % 50 == 0:
                r = yc - Xc @ w

        self.coef_ = w
        self.intercepto_ = media_y - media_X @ w
        self.n_iter_ = iteracion
        self.convergio_ = convergio
        return self

    def predecir(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercepto_
