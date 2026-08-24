"""Correr con: python -m src.preproceso"""

from itertools import combinations_with_replacement

import numpy as np
import pandas as pd

from src.datos import CATEGORICAS, DERIVADAS, NUMERICAS, agregar_derivadas, cargar


def quitar_duplicados(df):
    return df.drop_duplicates(keep="first").reset_index(drop=True)


class CodificadorCategoricas:
    def __init__(self, categoricas=None, numericas=None):
        self.categoricas = list(categoricas) if categoricas is not None else list(CATEGORICAS)
        self.numericas = (
            list(numericas) if numericas is not None else list(NUMERICAS) + list(DERIVADAS)
        )
        self.niveles_ = None
        self.nombres_ = None

    def ajustar(self, df):
        self.niveles_ = {}
        self.nombres_ = [col for col in self.numericas]
        for col in self.categoricas:
            niveles = sorted(df[col].unique())
            self.niveles_[col] = niveles
            if len(niveles) == 2:
                self.nombres_.append(f"{col}={niveles[1]}")
            else:
                for nivel in niveles[1:]:
                    self.nombres_.append(f"{col}={nivel}")
        return self

    def transformar(self, df):
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
        return self.ajustar(df).transformar(df)


class Estandarizador:
    def __init__(self):
        self.media_ = None
        self.desvio_ = None

    def ajustar(self, X):
        self.media_ = X.mean(axis=0)
        desvio = X.std(axis=0)
        desvio = np.where(desvio == 0, 1.0, desvio)
        self.desvio_ = desvio
        return self

    def transformar(self, X):
        if self.media_ is None or self.desvio_ is None:
            raise ValueError("Estandarizador: hay que llamar a ajustar() antes de transformar()")
        return (X - self.media_) / self.desvio_

    def ajustar_transformar(self, X):
        return self.ajustar(X).transformar(X)


def expandir_polinomica(X, grado):
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
    nombres = list(nombres)

    if grado == 1:
        return list(nombres)

    resultado = []
    for g in range(1, grado + 1):
        for combinacion in combinations_with_replacement(range(len(nombres)), g):
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
