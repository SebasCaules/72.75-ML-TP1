# Decisiones metodológicas

Cada punto del enunciado pide **justificar** las decisiones, no sólo tomarlas. Este documento
las numera (`D-01` … `D-20`) para poder citarlas desde el código y desde el informe sin repetir
el argumento en cada lugar.

El criterio para que algo entre acá es que **haya tenido alternativa razonable**. Lo que no
admitía discusión no se documenta.

---

## 1. Decisiones sobre el método

| # | Decisión | Por qué |
|---|---|---|
| **D-01** | El duplicado exacto se elimina **antes** del split (queda n=1337) | Si una copia cae en train y la otra en test, el test deja de ser independiente. Es fuga de datos |
| **D-02** | Los outliers **se conservan** | El 97,8 % son fumadores: es una subpoblación real, no error de carga. Eliminarlos sesgaría el modelo contra el grupo más caro |
| **D-03** | Split **80/20**, `semilla=42`, barajado aleatorio simple | El dataset no es temporal ni agrupado, así que i.i.d. es válido. Sin estratificar: el target es continuo, no hay clases que preservar |
| **D-04** | **k = 5** folds, sobre train **únicamente** | Con 1070 filas de train da ~214 por fold, suficiente para estimar el error con estabilidad. El punto 2.2 del enunciado exige explícitamente que la CV no toque test |
| **D-05** | El `Estandarizador` se ajusta **dentro de cada fold**, sólo con las filas de entrenamiento de ese fold | Ajustarlo antes de partir filtra información de validación hacia el entrenamiento. Es la fuga de datos más común y la más fácil de cometer sin darse cuenta |
| **D-06** | Orden del pipeline: codificar → **estandarizar** → expandir polinómica → **estandarizar de nuevo** | Sin el primer escalado, `age³ = 262 144` contra `children³ = 125` destruye el condicionamiento. Sin el segundo, la penalización L1 no es comparable entre features de escalas distintas |
| **D-07** | Se estandarizan **todas** las columnas, incluidas las dummies 0/1 | Para OLS es una reparametrización sin efecto en las predicciones; para Lasso pone a todos los coeficientes en pie de igualdad. La consistencia vale más que la pureza |
| **D-08** | Grados del polinomio evaluados: **1, 2, 3, 4** | 1 y 2 son el rango útil esperado; 3 y 4 se incluyen para **mostrar el sobreajuste**, que es lo que el punto 5 quiere ver |
| **D-09** | El test se evalúa **una sola vez**, al final, con el modelo ya elegido por validación | Doctrina del test set. Reiterarlo lo convierte en un segundo conjunto de validación y el número deja de estimar el error de generalización |
| **D-20** | **Una configuración que no converge no puede ser elegida.** Se excluye de la selección y se declara cuál era | Su RMSE no es el del modelo Lasso: es dónde quedó la optimización al cortarla. No es reproducible ni interpretable, y elegirla invalidaría la respuesta del punto 5 |

## 2. Decisiones de implementación

| # | Decisión | Por qué |
|---|---|---|
| **D-10** | Categóricas: binarias → **una** columna 0/1; `region` → one-hot **menos una** columna | Dos columnas que suman 1, más el intercepto, hacen singular a $X^TX$ (*dummy variable trap*) |
| **D-11** | El intercepto **no se penaliza**: se obtiene centrando $X$ e $y$, y luego $b = \bar{y} - \bar{x}^T w$ | Penalizarlo encogería la predicción media hacia cero, que no es lo que la regularización quiere hacer |
| **D-12** | OLS se resuelve con `np.linalg.lstsq` (SVD); ridge con `np.linalg.solve` | En grado 4 la matriz de diseño queda numéricamente singular (ver §4) y `solve` devuelve basura. `lstsq` da la solución de norma mínima. Con $\alpha>0$, $X^TX+\alpha I$ es definida positiva y `solve` es válido y más rápido |
| **D-13** | `expandir_polinomica` **no** agrega columna de unos | El intercepto lo maneja el modelo (D-11). Una columna constante además rompe el estandarizador (desvío 0) |
| **D-14** | Objetivo de Lasso: $\frac{1}{2n}\lVert y - Xw - b\rVert^2 + \lambda\lVert w\rVert_1$, por descenso por coordenadas | Es la convención estándar; el factor $1/n$ hace que $\lambda$ no dependa del tamaño de la muestra |
| **D-15** | Grilla de $\lambda$ **relativa a $\lambda_{max}$**, calculada una sola vez sobre el train completo | $\lambda_{max} = \max_j\lvert x_j^T(y-\bar y)\rvert/n$ es el menor $\lambda$ que anula todos los coeficientes. Una grilla relativa es interpretable y comparable entre grados; una absoluta elegida a ojo, no |
| **D-16** | Tests con asserts planos, sin `pytest` | El repo se clona y corre con `numpy`, `pandas`, `matplotlib` y nada más. Cero dependencias de desarrollo |
| **D-17** | Código, nombres de funciones y docstrings **en español** | El TP se defiende oralmente en español; el código es material de defensa |
| **D-18** | El Lasso mantiene el **residuo incrementalmente** ($r = y_c - X_c w$, actualizado en $O(n)$ por coordenada) en vez de recalcular $X_c w$ dentro del bucle | La forma directa cuesta $O(np^2)$ por barrida: con $p=494$ y $n=856$ son 209 M de operaciones por barrida y el algoritmo no converge en ningún presupuesto razonable. La versión incremental es **la misma cuenta reordenada** — coeficientes idénticos a $10^{-15}$ contra la implementación directa, y óptimo global confirmado contra `scipy.optimize`. Se recalcula $r$ desde cero cada 50 barridas para evitar deriva de punto flotante |
| **D-19** | `tol = 1e-4` en vez de `1e-7` | El criterio es $\max_j\lvert\Delta w_j\rvert < tol$ **en unidades absolutas**, y los coeficientes están en dólares (el target tiene media 13 447 y desvío 12 289). Pedir $10^{-7}$ es converger a once órdenes de magnitud por debajo de la señal: una centésima de centavo en un coeficiente, precisión que ningún número reportado usa |

---

## 3. Contrato de módulos

```python
# src/validacion.py
rmse(y_real, y_pred) -> float
separar_train_test(n, prop_test=0.2, semilla=42) -> (idx_train, idx_test)
k_fold(n, k=5, semilla=42) -> [(idx_train, idx_val)] * k     # índices posicionales
resumen_folds(errores_train, errores_val) -> dict

# src/preproceso.py
quitar_duplicados(df) -> pd.DataFrame
class CodificadorCategoricas:  ajustar(df) · transformar(df) -> np.ndarray · nombres_
class Estandarizador:          ajustar(X)  · transformar(X)  -> np.ndarray · media_ · desvio_
expandir_polinomica(X, grado) -> np.ndarray       # monomios 1..grado, SIN columna de unos
nombres_polinomicos(nombres, grado) -> [str]

# src/modelos.py
class RegresionLineal(alfa=0.0):  ajustar(X, y) · predecir(X) · coef_ · intercepto_
class Lasso(lam, max_iter, tol):  ajustar(X, y) · predecir(X) · coef_ · intercepto_ · n_iter_
lambda_maximo(X, y) -> float
```

Las clases devuelven `self` en `ajustar` para poder encadenar. La separación
`ajustar` / `transformar` no es estilística: es lo que permite ajustar con un fold y aplicar a
otro, que es la base de D-05.

---

## 4. Diagnóstico de rango de la matriz polinómica

Las variables binarias rompen la expansión polinómica. El módulo lo reporta al correr:

| grado | columnas | rango efectivo | redundantes | cond. completo |
|---|---:|---:|---:|---:|
| 1 | 8 | 8 | 0 | 2,2 |
| 2 | 44 | 36 | 8 (18,2 %) | 1,8 · 10¹⁶ |
| 3 | 164 | 100 | 64 (39,0 %) | 8,7 · 10¹⁶ |
| 4 | 494 | 216 | **278 (56,3 %)** | 3,1 · 10¹⁸ |

Dos causas distintas, y conviene no confundirlas:

1. **Una dummy elevada a una potencia sigue teniendo dos valores**, así que es una función
   **afín exacta** de la dummy original: `smoker=yes²` = 1,456866 · `smoker=yes` + 1,0, con
   residuo 1,5 · 10⁻¹⁴. `smoker²`, `smoker³` y `smoker⁴` no aportan nada.
2. **El producto de dos dummies del mismo one-hot** cae en el espacio generado por las dummies
   y la constante: 3 dummies más una constante ya generan todas las funciones sobre las 4
   regiones, y el producto es una de ellas.

> **Cuidado con un atajo tentador y falso:** ese producto **no es idénticamente cero**. Lo es
> en la codificación cruda 0/1, pero D-06 estandariza **antes** de expandir, y las dummies
> estandarizadas no valen 0 y 1. El producto toma tres valores distintos. Sigue siendo
> redundante, por la razón 2, no por ser nulo.

De grado 2 en adelante el número de condición completo supera la precisión de `float64`
($\approx 10^{16}$): la matriz es **numéricamente singular**, y por eso D-12 exige `lstsq`
(SVD). Restringido al subespacio de rango completo el condicionamiento es benigno (4,0 / 18,4 /
129,4), así que las **predicciones** son estables. Lo que no es estable es el reparto de
coeficientes entre columnas colineales: **no se pueden interpretar de a uno**.

---

## 5. Verificación

Los números del informe se recalcularon con una reimplementación independiente del pipeline,
que no pasa por `src/experimentos.py`:

| Chequeo | Resultado |
|---|---|
| RMSE de validación cruzada, grados 1–4 | Coincide a 0,1 |
| RMSE de test (producción, referencia lineal, baseline) | Coincide a 0,01 |
| Usos de `X_test` / `y_test` antes del punto 5 | 4 usos, 0 sospechosos (sólo construcción y `len()`) |
| Firma de sobreajuste: brecha train–val | 88 (grado 1) → 2508 (grado 4) |
| Lasso contra `scipy.optimize` sobre el mismo objetivo | Coincide a 8 decimales en 3 valores de $\lambda$ |
| Alineación de `nombres_polinomicos` con las columnas | Los 494 monomios de grado 4, verificados por factorización en primos |
| Suites `test_validacion`, `test_preproceso`, `test_modelos` | 3/3 en verde |

El chequeo de alineación de nombres merece una nota: se asigna un **primo distinto a cada
columna**, con lo cual el valor de cada monomio es un producto de primos que factoriza
unívocamente al monomio. Si un nombre no coincidiera con su columna, el informe estaría
reportando el coeficiente de una feature con el nombre de otra — un error silencioso que
ningún test de conteo detecta.

---

## 6. Resultado

| | Ganador de la validación cruzada | **Producción (regla de 1 ES)** |
|---|---|---|
| Modelo | Lasso grado 4, λ=286,4 | **Lasso grado 2, λ=286,4** |
| Features vivas | 20 de 494 | **10 de 44** |
| RMSE validación | 4920,0 | 4955,3 |
| **RMSE test** | 4647,3 | **4739,3** |

Referencias: lineal grado 1 → 6057,7; baseline (media de train) → 11 963,4.

Costo de la simplicidad: **+92 dólares de RMSE (2 %)** por un espacio de features 11× menor.

El desarrollo completo, con las tablas y las figuras, está en
[`informe/informe.pdf`](informe/informe.pdf).
