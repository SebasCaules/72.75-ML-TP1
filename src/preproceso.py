"""Codificación, escalado y expansión polinómica (parte del punto 2 del enunciado).

Este es el módulo donde más fácil se comete una fuga de datos (*data leakage*): si el
escalado o la codificación se "aprenden" mirando TODO el dataset (train + test/validación),
información del conjunto de evaluación se filtra al entrenamiento y el error que se reporta
después queda optimista, sin que se note en el código a simple vista.

Por eso todas las clases de este módulo separan `ajustar` (aprende parámetros SOLO del
fold de entrenamiento) de `transformar` (aplica esos parámetros, ya fijos, a cualquier
fold, incluido el propio de entrenamiento). El esquema de validación cruzada en otro
módulo es el que decide qué es "entrenamiento" en cada iteración; acá sólo se garantiza
que, una vez que ajustar() corrió, transformar() no vuelve a mirar los datos que le pasen.

Correr con:  python -m src.preproceso
"""

from itertools import combinations_with_replacement

import numpy as np
import pandas as pd

from src.datos import CATEGORICAS, DERIVADAS, NUMERICAS, agregar_derivadas, cargar


def quitar_duplicados(df):
    """Elimina filas duplicadas exactas y reindexa desde 0.

    Esto tiene que pasar ANTES de partir en train/test, no después. Si una fila
    duplicada cae con una copia en train y la otra en test, el conjunto de test deja de
    ser independiente del de entrenamiento: el modelo ya "vio" esa fila (o una idéntica)
    durante el ajuste, y el error que se mide sobre ella sale artificialmente bajo. Es
    fuga de datos, aunque no involucre ninguna columna del futuro ni del objetivo.

    Sobre insurance.csv hay un único par de duplicados exactos, en los índices 195 y
    581; `keep="first"` conserva el 195 y descarta el 581, dejando 1337 filas. El
    `reset_index(drop=True)` es necesario para que los índices vuelvan a ser 0..n-1
    contiguos, que es lo que el resto del pipeline (folds por posición) asume.
    """
    return df.drop_duplicates(keep="first").reset_index(drop=True)


class CodificadorCategoricas:
    """Convierte columnas categóricas de texto en columnas numéricas 0/1.

    La estrategia depende de cuántos niveles tiene cada columna:
      - 2 niveles (binaria): UNA sola columna alcanza, porque la segunda sería
        redundante (es 1 - la primera). Se codifica 1 para el nivel alfabéticamente
        MAYOR ('male' sobre 'female', 'yes' sobre 'no'), así el nombre de la columna
        ("sex=male") dice exactamente qué representa el 1.
      - más de 2 niveles (nominal, como `region`): one-hot, pero descartando el primer
        nivel alfabético como categoría de referencia. Esto es la "dummy variable trap":
        si se incluyeran las 4 columnas de `region`, para cualquier fila suman
        exactamente 1, que es idéntico a la columna del intercepto (todos 1). Eso hace
        que X^T X sea singular (una columna es combinación lineal exacta de las otras
        más el intercepto) y la ecuación normal deja de tener solución única. Al
        descartar una columna, esa categoría queda representada implícitamente por
        "todas las demás en 0", y el modelo sigue pudiendo distinguirla vía el
        intercepto.
      - Por qué one-hot y no codificación ordinal (northeast=0, northwest=1,
        southeast=2, southwest=3): no existe ningún orden natural entre regiones. Si se
        usara un único número, el modelo interpretaría la diferencia entre categorías
        como una distancia lineal (southwest, codificada 3, "vale el triple" que
        northeast, codificada 1), una relación que no existe en los datos y que
        contamina los coeficientes con una estructura inventada.
    """

    def __init__(self, categoricas=None, numericas=None):
        self.categoricas = list(categoricas) if categoricas is not None else list(CATEGORICAS)
        # Las derivadas entran como numericas: ya son 0/1, no hay niveles que codificar.
        # Si el DataFrame no las trae, transformar() falla con KeyError, que es lo que
        # queremos: prefiero un error ruidoso a un modelo entrenado en silencio sin la
        # feature que decide el resultado.
        self.numericas = (
            list(numericas) if numericas is not None else list(NUMERICAS) + list(DERIVADAS)
        )
        self.niveles_ = None
        self.nombres_ = None

    def ajustar(self, df):
        """Aprende, por columna categórica, los niveles vistos y ordenados alfabéticamente.

        El orden alfabético es lo que hace determinista el resto: qué nivel se vuelve
        columna de referencia (el primero) y qué nivel "gana" el 1 en las binarias (el
        último) no depende del orden en que aparecen las filas del CSV.
        """
        self.niveles_ = {}
        self.nombres_ = [col for col in self.numericas]
        for col in self.categoricas:
            niveles = sorted(df[col].unique())
            self.niveles_[col] = niveles
            if len(niveles) == 2:
                # binaria: una sola columna, 1 para el nivel alfabéticamente mayor
                self.nombres_.append(f"{col}={niveles[1]}")
            else:
                # nominal: one-hot descartando el primer nivel (categoría de referencia)
                for nivel in niveles[1:]:
                    self.nombres_.append(f"{col}={nivel}")
        return self

    def transformar(self, df):
        """Aplica la codificación aprendida en ajustar() a un DataFrame nuevo.

        Si aparece en `df` un nivel que ajustar() no vio (por ejemplo una región que no
        estaba en el fold de entrenamiento), no hay columna que lo represente: se
        lanza un ValueError en vez de codificarlo silenciosamente como 0 en todo, que
        sería indistinguible de la categoría de referencia y ocultaría el problema.
        """
        if self.niveles_ is None:
            raise ValueError("CodificadorCategoricas: hay que llamar a ajustar() antes de transformar()")

        columnas = []
        for col in self.numericas:
            columnas.append(df[col].to_numpy(dtype=np.float64))

        for col in self.categoricas:
            niveles = self.niveles_[col]
            vistos = set(niveles)
            no_vistos = set(df[col].unique()) - vistos
            if no_vistos:
                raise ValueError(
                    f"CodificadorCategoricas: nivel(es) no visto(s) en ajustar() para la "
                    f"columna '{col}': {sorted(no_vistos)}. Niveles conocidos: {niveles}"
                )
            if len(niveles) == 2:
                columnas.append((df[col].to_numpy() == niveles[1]).astype(np.float64))
            else:
                for nivel in niveles[1:]:
                    columnas.append((df[col].to_numpy() == nivel).astype(np.float64))

        return np.column_stack(columnas).astype(np.float64)

    def ajustar_transformar(self, df):
        """Atajo: ajustar() seguido de transformar() sobre el mismo DataFrame."""
        return self.ajustar(df).transformar(df)


class Estandarizador:
    """Estandariza columnas a media 0 y desvío 1: (X - media) / desvío.

    El escalado importa por dos motivos en este TP: Ridge y Lasso penalizan la norma de
    los coeficientes, y esa penalización sólo es comparable entre features si todas
    están en la misma escala (si no, una feature en miles como `charges` dominaría el
    término de penalización frente a una en unidades como `children`). Y el descenso por
    coordenadas de Lasso converge más rápido y de forma más estable sobre datos
    estandarizados.

    `ajustar` y `transformar` son métodos SEPARADOS, y ese es el punto central del
    módulo: la media y el desvío se calculan ÚNICAMENTE con el fold de entrenamiento
    (ajustar), y esos mismos números, ya fijos, se aplican después al fold de validación
    (transformar). Si en cambio se calcularan media y desvío sobre entrenamiento +
    validación juntos, información de la validación (su propia media, su propio desvío)
    se filtraría al preprocesamiento que ve el entrenamiento, y el error de validación
    que se reporte después va a salir optimista respecto del error real sobre datos
    nuevos. Esta separación es la que hace que el esquema de validación cruzada sea
    honesto.
    """

    def __init__(self):
        self.media_ = None
        self.desvio_ = None

    def ajustar(self, X):
        """Calcula media y desvío por columna sobre X (pensado para ser sólo el fold de train)."""
        self.media_ = X.mean(axis=0)
        desvio = X.std(axis=0)
        # Si una columna es constante, su desvío es 0 y dividir por él da inf/NaN. En ese
        # caso guardamos 1.0: la columna queda en puros ceros después de centrar (X -
        # media = 0), que es exactamente lo correcto, porque una columna constante no
        # aporta ninguna información para distinguir filas.
        desvio = np.where(desvio == 0, 1.0, desvio)
        self.desvio_ = desvio
        return self

    def transformar(self, X):
        """Aplica la media y el desvío ya aprendidos a X (puede ser train o validación)."""
        if self.media_ is None or self.desvio_ is None:
            raise ValueError("Estandarizador: hay que llamar a ajustar() antes de transformar()")
        return (X - self.media_) / self.desvio_

    def ajustar_transformar(self, X):
        """Atajo: ajustar() seguido de transformar() sobre la misma matriz."""
        return self.ajustar(X).transformar(X)


def expandir_polinomica(X, grado):
    """Genera todos los monomios de grado 1 hasta `grado` sobre las columnas de X.

    La regresión "polinómica" sigue siendo una regresión LINEAL: lo que cambia no es el
    modelo (sigue siendo una combinación lineal de coeficientes por feature más un
    intercepto), sino las FEATURES que se le dan de entrada. Si a partir de age y bmi se
    construyen las columnas age, bmi, age^2, age*bmi, bmi^2, el modelo sigue siendo
    "w1*f1 + w2*f2 + ... + b" — lineal en los w — así que se sigue resolviendo con la
    misma ecuación normal (o el mismo descenso por coordenadas para Lasso) que la
    regresión lineal simple. Lo único que cambió es qué hay adentro de cada f_i.

    Los términos cruzados (age*bmi) no son un detalle menor: capturan INTERACCIONES,
    es decir, que el efecto de una variable sobre el objetivo depende del valor de otra.
    En este dataset el efecto del bmi sobre `charges` es distinto según si la persona
    fuma o no (ver el punto 1.4 del análisis exploratorio en src/datos.py); un modelo
    puramente aditivo, sin términos cruzados, no puede representar eso porque en un
    modelo aditivo el efecto de bmi es el mismo constante para todas las filas.

    No se agrega columna de unos (intercepto): eso lo maneja el modelo aparte (ver
    D-11), agregar una columna constante acá rompería al Estandarizador (desvío 0 antes
    de la corrección) y sería redundante con el término independiente que ya calcula el
    modelo al centrar X e y.

    Se usa `itertools.combinations_with_replacement` sobre los índices de columna,
    tomados de a `g` para cada grado g en 1..grado, para que el orden de los monomios
    sea determinista y para no repetir el mismo monomio dos veces: (columna 0, columna
    1) y (columna 1, columna 0) representan el mismo producto x0*x1, así que sólo se
    genera una vez, en el orden en que combinations_with_replacement los produce
    (equivalente a orden lexicográfico ascendente de índices).
    """
    X = np.asarray(X, dtype=np.float64)
    n_columnas = X.shape[1]

    if grado == 1:
        return X.copy()

    bloques = []
    for g in range(1, grado + 1):
        for combinacion in combinations_with_replacement(range(n_columnas), g):
            monomio = np.ones(X.shape[0], dtype=np.float64)
            for indice_columna in combinacion:
                monomio = monomio * X[:, indice_columna]
            bloques.append(monomio)

    return np.column_stack(bloques)


def nombres_polinomicos(nombres, grado):
    """Nombres legibles de los monomios, en el MISMO orden que expandir_polinomica.

    Ese orden compartido es lo que permite, más adelante, mostrar "qué features
    sobreviven al Lasso": si expandir_polinomica genera la columna k y
    nombres_polinomicos genera el nombre k, ambos tienen que corresponder al mismo
    monomio, o el reporte de coeficientes distintos de cero mentiría sobre a qué
    variable pertenece cada uno.

    Un exponente de 1 se escribe sin '^1' ("age", no "age^1"); exponentes mayores se
    escriben "age^2"; factores distintos se separan con "*" ("age*bmi",
    "age^2*bmi").
    """
    nombres = list(nombres)

    if grado == 1:
        return list(nombres)

    resultado = []
    for g in range(1, grado + 1):
        for combinacion in combinations_with_replacement(range(len(nombres)), g):
            # cuenta cuántas veces aparece cada índice de columna en esta combinación,
            # preservando el orden de primera aparición para armar el texto
            exponentes = {}
            orden_aparicion = []
            for indice_columna in combinacion:
                if indice_columna not in exponentes:
                    exponentes[indice_columna] = 0
                    orden_aparicion.append(indice_columna)
                exponentes[indice_columna] += 1

            partes = []
            for indice_columna in orden_aparicion:
                exponente = exponentes[indice_columna]
                if exponente == 1:
                    partes.append(nombres[indice_columna])
                else:
                    partes.append(f"{nombres[indice_columna]}^{exponente}")
            resultado.append("*".join(partes))

    return resultado


def main():
    df = agregar_derivadas(quitar_duplicados(cargar()))
    print(f"Filas tras quitar duplicados: {len(df)}")

    codificador = CodificadorCategoricas()
    X = codificador.ajustar_transformar(df)
    print(f"\nColumnas codificadas ({X.shape[1]}): {codificador.nombres_}")
    print(f"Forma de X: {X.shape}")

    estandarizador = Estandarizador()
    X_esc = estandarizador.ajustar_transformar(X)
    print(f"\nMedia tras estandarizar (debería ser ~0): {np.round(X_esc.mean(axis=0), 8)}")
    print(f"Desvío tras estandarizar (debería ser ~1): {np.round(X_esc.std(axis=0), 8)}")

    for grado in (1, 2, 3, 4):
        X_poly = expandir_polinomica(X, grado)
        nombres = nombres_polinomicos(codificador.nombres_, grado)
        print(f"\nGrado {grado}: {X_poly.shape[1]} columnas, {len(nombres)} nombres")


if __name__ == "__main__":
    main()
