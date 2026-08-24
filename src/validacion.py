"""Correr con: python -m src.validacion"""

import numpy as np


def rmse(y_real, y_pred):
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_real - y_pred) ** 2)))


def r2(y_real, y_pred):
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_real - y_pred) ** 2))
    ss_tot = float(np.sum((y_real - np.mean(y_real)) ** 2))
    return 1.0 - ss_res / ss_tot


def separar_train_test(n, prop_test=0.2, semilla=42):
    rng = np.random.default_rng(semilla)
    permutacion = rng.permutation(n)

    n_test = round(n * prop_test)
    n_train = n - n_test

    idx_train = permutacion[:n_train]
    idx_test = permutacion[n_train:]
    return idx_train, idx_test


def k_fold(n, k=5, semilla=42):
    rng = np.random.default_rng(semilla)
    permutacion = rng.permutation(n)
    bloques = np.array_split(permutacion, k)

    folds = []
    for i in range(k):
        idx_val = bloques[i]
        idx_train = np.concatenate([bloques[j] for j in range(k) if j != i])
        folds.append((idx_train, idx_val))
    return folds


def resumen_folds(errores_train, errores_val):
    errores_train = np.asarray(errores_train, dtype=float)
    errores_val = np.asarray(errores_val, dtype=float)
    return {
        "rmse_train_medio": float(np.mean(errores_train)),
        "rmse_train_desvio": float(np.std(errores_train, ddof=0)),
        "rmse_val_medio": float(np.mean(errores_val)),
        "rmse_val_desvio": float(np.std(errores_val, ddof=0)),
    }


def main():
    n = 1338

    idx_train, idx_test = separar_train_test(n)
    print(f"separar_train_test(n={n}): train={len(idx_train)}, test={len(idx_test)}")

    folds = k_fold(n, k=5)
    tamanos_val = [len(idx_val) for _, idx_val in folds]
    print(f"k_fold(n={n}, k=5): tamaños de validación por fold = {tamanos_val}")

    resumen = resumen_folds([100.0, 110.0, 105.0], [150.0, 200.0, 170.0])
    print(f"resumen_folds de ejemplo: {resumen}")


if __name__ == "__main__":
    main()
