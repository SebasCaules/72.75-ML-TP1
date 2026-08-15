"""Esquema de validación: partición de datos y métrica de error (punto 2.2 del enunciado).

Este módulo es, a propósito, chico y aislado: no conoce el dataset de seguros ni ningún
modelo en particular. Sólo sabe partir índices `0..n-1` de formas reproducibles y medir
qué tan lejos quedó una predicción de la realidad. Cualquier modelo (OLS, Ridge, Lasso,
con o sin expansión polinómica) se apoya en las mismas funciones de acá, así que el
esquema de evaluación es UNO SOLO para todo el TP y no se reimplementa por modelo.

Correr los tests con:  python -m tests.test_validacion
"""

import numpy as np


# ----------------------------------------------------------------------------------
# Métrica
# ----------------------------------------------------------------------------------
def rmse(y_real, y_pred):
    """Raíz del error cuadrático medio: sqrt(mean((y_real - y_pred)^2)).

    Por qué RMSE y no MSE (que es matemáticamente más simple, un paso menos):
    el MSE queda en las unidades del target AL CUADRADO. Acá el target es `charges`,
    en dólares, así que el MSE queda en dólares². Un número así no tiene lectura
    intuitiva: nadie interpreta "38 millones de dólares al cuadrado de error".

    El RMSE deshace ese cuadrado con la raíz y vuelve a la unidad original (dólares),
    así que "el modelo se equivoca en promedio 4200 dólares" es una frase que un
    humano —el docente en la defensa, o un analista de la aseguradora— entiende sin
    traducir nada. Es exactamente la cantidad que pide el enunciado (RMS/RMSE) y la
    que se usa para comparar modelos en el resto del TP.

    Se calcula sobre arrays de numpy (o algo convertible a array): la resta es
    elemento a elemento, se eleva al cuadrado, se promedia y se toma raíz.
    """
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_real - y_pred) ** 2)))


# ----------------------------------------------------------------------------------
# Split simple train/test
# ----------------------------------------------------------------------------------
def separar_train_test(n, prop_test=0.2, semilla=42):
    """Parte los índices 0..n-1 en train y test, barajando antes de cortar.

    Por qué barajar: el archivo puede tener estructura en su orden (por ejemplo, si
    se cargó por lotes o algún criterio de las columnas categóricas), y cortar los
    primeros/últimos `n` en el orden original arriesga meter ese sesgo en el split.
    Barajar con una permutación aleatoria hace que train y test sean, en expectativa,
    dos muestras de la MISMA distribución subyacente.

    Por qué semilla fija: reproducibilidad. El docente tiene que poder correr este
    mismo código y ver exactamente el mismo split (y por lo tanto los mismos números
    de RMSE) que se reportan en el informe. `np.random.default_rng(semilla)` es un
    generador determinista: la misma semilla siempre produce la misma permutación.

    Por qué NO se estratifica: la estratificación (mantener proporciones de clase en
    cada partición) tiene sentido cuando el target es categórico. Acá `charges` es
    continuo, así que no hay "clases" que preservar. Tampoco hay estructura temporal
    ni de grupos (cada fila es una persona independiente), así que un muestreo
    puramente aleatorio (i.i.d.) es válido y no introduce ningún sesgo adicional.

    El conjunto de test se queda con los ÚLTIMOS `round(n * prop_test)` elementos de
    la permutación (no los primeros): es una convención arbitraria pero fija, que
    alcanza para que el resultado sea determinista y quede documentado acá.

    Devuelve (idx_train, idx_test): dos arrays de enteros con índices POSICIONALES
    (no hay que confundirlos con el índice de un DataFrame si éste no es 0..n-1),
    disjuntos, que juntos cubren exactamente 0..n-1.
    """
    rng = np.random.default_rng(semilla)
    permutacion = rng.permutation(n)

    n_test = round(n * prop_test)
    n_train = n - n_test

    idx_train = permutacion[:n_train]
    idx_test = permutacion[n_train:]
    return idx_train, idx_test


# ----------------------------------------------------------------------------------
# Validación cruzada k-fold
# ----------------------------------------------------------------------------------
def k_fold(n, k=5, semilla=42):
    """Genera los k pares (idx_train, idx_val) de una validación cruzada de k folds.

    Qué es y por qué se usa en vez de un único split de validación: con un solo split
    train/val, el número de RMSE que se obtiene depende de QUÉ filas cayeron del lado
    de validación por azar —con un dataset chico como este (1338 filas), esa varianza
    no es despreciable, y un split "afortunado" o "desafortunado" puede hacer parecer
    mejor o peor a un modelo que en realidad no lo es—.

    La validación cruzada barajea el dataset una sola vez y lo parte en k bloques.
    Cada bloque actúa de conjunto de validación exactamente una vez, mientras los
    otros k-1 bloques entrenan el modelo de ese fold. Al final se promedian los k
    RMSE de validación. Dos beneficios concretos:
      1. Cada dato se usa para validar exactamente una vez y para entrenar k-1 veces:
         no se "desperdicia" ningún dato dejándolo afuera del entrenamiento para
         siempre, como pasaría con un hold-out fijo.
      2. El promedio de k mediciones tiene mucha menos varianza que una sola medición
         (la varianza del promedio de k términos ~independientes cae con 1/k), así
         que el número que se reporta es mucho más confiable para comparar modelos
         o elegir hiperparámetros (por ejemplo, el lambda de Ridge o Lasso).

    Por qué k=5: es el compromiso estándar entre las dos alternativas. Un k grande
    (por ejemplo k=n, leave-one-out) da una estimación casi insesgada del error pero
    computacionalmente cara y de ALTA varianza entre folds (cada conjunto de
    entrenamiento difiere en un solo dato del anterior, así que los k modelos quedan
    muy correlacionados entre sí). Un k chico (por ejemplo k=2) es barato pero cada
    fold entrena con la mitad de los datos, lo que sesga el error hacia arriba
    respecto del modelo final (entrenado con el 100%). k=5 (junto con k=10) es el
    valor que la práctica y la literatura (Hastie et al.) reportan como buen punto
    medio para datasets de tamaño chico/mediano como este.

    Reparto de tamaños: se baraja 0..n-1 una vez y se corta en k bloques lo más
    parejos posible. Si n no es múltiplo de k, los primeros (n % k) bloques quedan
    con un elemento más que el resto (así lo hace np.array_split, que se usa acá).
    Para el fold i, idx_val es el bloque i y idx_train es la concatenación de TODOS
    los demás bloques.

    Devuelve una lista de k tuplas (idx_train, idx_val), cada una con arrays de
    índices posicionales enteros.
    """
    rng = np.random.default_rng(semilla)
    permutacion = rng.permutation(n)
    bloques = np.array_split(permutacion, k)

    folds = []
    for i in range(k):
        idx_val = bloques[i]
        # concatenate del resto de los bloques, en orden, preservando el orden en que
        # quedaron barajados dentro de cada bloque
        idx_train = np.concatenate([bloques[j] for j in range(k) if j != i])
        folds.append((idx_train, idx_val))
    return folds


# ----------------------------------------------------------------------------------
# Resumen de una corrida de k-fold
# ----------------------------------------------------------------------------------
def resumen_folds(errores_train, errores_val):
    """Resume las listas de RMSE (uno por fold) en medias y desvíos.

    Se usa desvío POBLACIONAL (ddof=0, la convención por defecto de numpy) y no el
    muestral (ddof=1): acá los k errores no son una muestra de una población más
    grande de la que interese generalizar la varianza en sí misma —son EXACTAMENTE
    los k resultados que hubo en esta corrida de validación cruzada—, así que
    dividir por k (población) en vez de por k-1 (muestra) describe la dispersión
    observada en esos k números tal cual son, sin la corrección de sesgo que tiene
    sentido cuando se quiere inferir la varianza de una población más amplia a
    partir de una muestra chica. Con k=5 la diferencia entre ambas es de todos modos
    menor, pero se documenta la elección para que quede explícita.

    Devuelve un diccionario con las cuatro cantidades que se reportan en el informe:
    media y desvío del RMSE de train, media y desvío del RMSE de validación. La
    brecha entre ambas medias es la señal de sobreajuste; el desvío de validación es
    la señal de qué tan estable es esa estimación entre folds.
    """
    errores_train = np.asarray(errores_train, dtype=float)
    errores_val = np.asarray(errores_val, dtype=float)
    return {
        "rmse_train_medio": float(np.mean(errores_train)),
        "rmse_train_desvio": float(np.std(errores_train, ddof=0)),
        "rmse_val_medio": float(np.mean(errores_val)),
        "rmse_val_desvio": float(np.std(errores_val, ddof=0)),
    }


def main():
    """Demo rápida sobre n=1338 (el tamaño real del dataset), sin depender de datos.py."""
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
