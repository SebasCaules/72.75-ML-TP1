# Decisiones metodológicas

Cada punto del enunciado pide **justificar** las decisiones, no sólo tomarlas. Este documento
las numera (`D-01` … `D-30`) para poder citarlas desde el código y desde el informe sin repetir
el argumento en cada lugar.

El criterio para que algo entre acá es que **haya tenido alternativa razonable**. Lo que no
admitía discusión no se documenta.

---

## 1. Decisiones sobre el método

| # | Decisión | Por qué |
|---|---|---|
| **D-01** | El duplicado exacto se elimina **antes** del split (queda n=1337) | Si una copia cae en train y la otra en test, el test deja de ser independiente. Es fuga de datos |
| **D-02** | Los outliers **se conservan** | El 97,8 % son fumadores: es una subpoblación real, no error de carga. Eliminarlos sesgaría el modelo contra el grupo más caro |
| **D-03** | Split **80/20**, `semilla=42`, barajado aleatorio simple | El dataset no es temporal ni agrupado, así que i.i.d. es válido. Sin estratificar — **pero no por el motivo que decía esta fila hasta el EDA de la Clase 3; ver D-24** |
| **D-04** | **k = 5** folds, sobre train **únicamente** | Con 1070 filas de train da ~214 por fold, suficiente para estimar el error con estabilidad. El punto 2.2 del enunciado exige explícitamente que la CV no toque test |
| **D-05** | El `Estandarizador` se ajusta **dentro de cada fold**, sólo con las filas de entrenamiento de ese fold | Ajustarlo antes de partir filtra información de validación hacia el entrenamiento. Es la fuga de datos más común y la más fácil de cometer sin darse cuenta |
| **D-06** | Orden del pipeline: codificar → **estandarizar** → expandir polinómica → **estandarizar de nuevo** | Sin el primer escalado, `age³ = 262 144` contra `children³ = 125` destruye el condicionamiento. Sin el segundo, la penalización L1 no es comparable entre features de escalas distintas |
| **D-07** | Se estandarizan **todas** las columnas, incluidas las dummies 0/1 | Para OLS es una reparametrización sin efecto en las predicciones; para Lasso pone a todos los coeficientes en pie de igualdad. La consistencia vale más que la pureza |
| **D-08** | Grados del polinomio evaluados: **1, 2, 3, 4** | 1 y 2 son el rango útil esperado; 3 y 4 se incluyen para **mostrar el sobreajuste**, que es lo que el punto 5 quiere ver |
| **D-09** | El test se evalúa **una sola vez**, al final, con el modelo ya elegido por validación | Doctrina del test set. Reiterarlo lo convierte en un segundo conjunto de validación y el número deja de estimar el error de generalización |
| **D-20** | **Una configuración que no converge no puede ser elegida.** Se excluye de la selección y se declara cuál era | Su RMSE no es el del modelo Lasso: es dónde quedó la optimización al cortarla. No es reproducible ni interpretable, y elegirla invalidaría la respuesta del punto 5 |
| **D-22** | **La elección de $k=5$ se sostiene con evidencia, no con la cita a Hastie.** Se repite la selección completa del punto 5 con $k=5$, $10$ y $20$, y un barrido controlado hasta LOO. Ver §4.ter | $k=10$ acá cuesta unos 15 minutos, no las ~13 horas de LOO: el argumento de costo no alcanza para descartarlo, así que había alternativa razonable y por el criterio de entrada de este documento hay que justificarla. El barrido muestra que el modelo elegido **no cambia** con $k$, y de paso cuantifica lo que §4.bis había dejado abierto |
| **D-23** | Se agrega la feature derivada **`fumador_obeso` = fuma ∧ bmi > 30** | Es la opción (a) que prescribe la Clase 3 al detectar más de una población en un histograma (slides 36–38), y la población está en la figura 8. El término cruzado `bmi*smoker` del polinomio no la reemplaza: modela una **pendiente**, y lo que hay es un **escalón**. Ver §4.quater |
| **D-24** | **Se sigue sin estratificar, pero por evidencia, no por la definición.** La razón vieja ("el target es continuo, no hay clases") queda refutada por la figura 8 | Hay tres poblaciones de facto, y los folds actuales van de 17 a 31 fumadores obesos. Pero estratificar por ellas **no cambia** la varianza entre folds: 593 contra 591 promediando 8 particiones, cuando entre dos particiones del mismo método va de 410 a 715. Igualar los conteos no iguala las magnitudes |
| **D-25** | **`children` se mantiene numérica**, pese a que el slide 35 prescribe one-hot para las numéricas discretizadas | Medido sobre 8 particiones: dentro del ruido en grado 1 (+5 OLS / +36 Lasso, con ±14 y ±8 de desvío) y peor en grado 2 (+453 OLS / +91 Lasso). Es casi monótona en `charges` hasta 3 hijos, y los niveles 4 y 5 tienen 18 y 16 filas: one-hot les daría un parámetro propio a celdas de ese tamaño |
| **D-26** | **El test se evalúa por segunda vez**, y se reportan **los dos** números (modelo viejo y modelo nuevo) | D-09 protege contra elegir mirando el test. La selección de D-23 se hizo con CV sobre train: ninguna decisión ajustable usó test —el barrido que fija el umbral (bmi>30) corre sólo sobre train, en `evidencia_features.py`; las figuras 7-8 y las tablas descriptivas del EDA usan el dataset completo, como todo el punto 1—, y publicar los dos números elimina la posibilidad de haber elegido el más favorable, que es exactamente lo que la doctrina quiere impedir. La segunda evaluación reutiliza a propósito la misma partición (semilla 42): así la diferencia 4739,33 → 4465,32 mide el efecto de D-23 sobre las mismas 267 filas de test, no el azar de una partición nueva. El protocolo ideal —el que describe el docstring de `evaluar_test.py`— pide partición nueva al cambiar de modelo; acá se registra la desviación y por qué se la acepta |
| **D-27** | Se agrega la característica derivada **`edad_al_cuadrado` = age²** | El costo médico crece de forma convexa con la edad; una recta subestima a los mayores. Evidencia: −29,76 dólares de RMSE de validación promediados sobre 8 particiones. Ver §4.quinquies |
| **D-28** | Se agrega la característica derivada **`bmi_si_fuma` = bmi · 1[smoker=yes]** | Entre fumadores el costo crece más rápido con el bmi —eso es una **pendiente**—, y `fumador_obeso` (D-23) modela un **escalón**, no una pendiente: son dos estructuras distintas y coexisten. Evidencia: −45,47, y aditiva con D-27 (−75,16 dólares con las dos juntas, contra −29,76 y −45,47 por separado). Ver §4.quinquies |
| **D-29** | **El test se evalúa por tercera vez**, y se reportan **los tres** números | Es la continuación honesta de D-26: la selección de D-27/D-28 se hizo íntegramente con CV sobre train, y publicar 4739,33 / 4465,32 / 4288,52 —los tres sobre las mismas 267 filas, semilla 42— hace comparables las tres etapas en vez de exhibir sólo la más favorable. Reutilizar la partición se aparta del protocolo ideal que describe el docstring de `evaluar_test.py` (pide partición nueva al cambiar de modelo), y acá el costo es real y no sólo nominal: **cada evaluación extra erosiona un poco la independencia del test**, y tres evaluaciones sobre la misma partición es más de lo que la doctrina recomienda. Se acepta la desviación porque la comparabilidad entre las tres etapas —medir el efecto de cada decisión sobre las mismas filas— vale más que la partición nueva, no porque el costo sea nulo |
| **D-30** | Se documenta el diagnóstico de residuos y el error irreducible (`src/diagnostico_residuos.py`) | Es la respuesta fundamentada a «¿cómo se baja más el error?»: el 41,9 % del error cuadrático de train se concentra en 28 personas (2,6 % de las filas) que ninguna combinación de las columnas del dataset distingue del resto. Ver §4.sexies |

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
# src/sensibilidad_k.py            # D-22; ~3 h (parte A) + minutos (parte B), no toca test. Escribe resultados/sensibilidad_k.{json,csv}
preparar_train() -> (X_train, y_train)
grilla_completa(X_train, y_train, k, lam_max) -> [dict]      # las 19 configuraciones con k folds
seleccionar(candidatos, k) -> dict                           # ganador + regla de 1 ES + parsimonia
barrido_controlado(X_train, y_train) -> [dict]               # configuracion fija, k variable
rmse_agrupado(X_train, y_train, k) -> float                  # una raiz sobre los residuos out-of-fold

# src/diagnostico_residuos.py       # D-30; segundos, no toca test. Escribe resultados/diagnostico_residuos.csv
residuos_out_of_fold(train, grado, lam) -> (residuos, y)     # cada fila predicha por un modelo que no la vio

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

| grado | columnas | rango efectivo | redundantes | cond. completo | cond. restringido |
|---|---:|---:|---:|---:|---:|
| 1 | 11 | 11 | 0 (0,0 %) | 17,5 | 17,5 |
| 2 | 77 | 62 | 15 (19,5 %) | 5,8 · 10¹⁶ | 401,6 |
| 3 | 363 | 202 | 161 (44,4 %) | 1,5 · 10¹⁸ | 31.398,9 |
| 4 | 1364 | 436 | **928 (68,0 %)** | 3,8 · 10¹⁷ | 1.496.341,4 |

La última columna es el condicionamiento **restringido al subespacio de rango completo**, y es
la que explica por qué las predicciones son estables aunque la matriz sea numéricamente
singular: descartados los valores singulares nulos, el problema está muy por debajo del límite
de `float64` hasta grado 3 y recién en grado 4 crece a un nivel que empieza a pesar (1,5 millón,
48 veces el de grado 3).

Tres causas distintas, y conviene no confundirlas:

1. **Una dummy elevada a una potencia sigue teniendo dos valores**, así que es una función
   **afín exacta** de la dummy original: `smoker=yes²` = 1,456866 · `smoker=yes` + 1,0, con
   residuo 1,5 · 10⁻¹⁴. `smoker²`, `smoker³` y `smoker⁴` no aportan nada.
2. **El producto de dos dummies del mismo one-hot** cae en el espacio generado por las dummies
   y la constante: 3 dummies más una constante ya generan todas las funciones sobre las 4
   regiones, y el producto es una de ellas.
3. **Desde D-27/D-28, la expansión duplica exactamente a las dos características derivadas.**
   `age · age` es la misma columna que `edad_al_cuadrado`, y `bmi · smoker=yes` es la misma
   columna que `bmi_si_fuma`: la expansión polinómica las vuelve a generar a partir de grado 2 y
   quedan dos copias idénticas de cada una. Es una causa nueva que el pipeline de 9 features no
   tenía, y explica por qué la redundancia sube a 68,0 % en grado 4 (contra 59,7 % antes) y por
   qué esta corrida deja 4 configuraciones sin converger en vez de 2: más columnas exactamente
   colineales hacen más lento al descenso por coordenadas. En **grado 1** —el de producción— no
   hay duplicación: ahí las dos columnas nuevas son la única vía de esas dos estructuras.

> **Cuidado con un atajo tentador y falso:** el producto de dos dummies del mismo one-hot **no es
> idénticamente cero**. Lo es en la codificación cruda 0/1, pero D-06 estandariza **antes** de
> expandir, y las dummies estandarizadas no valen 0 y 1. El producto toma tres valores distintos.
> Sigue siendo redundante, por la razón 2, no por ser nulo.

De grado 2 en adelante el número de condición completo supera la precisión de `float64`
($\approx 10^{16}$): la matriz es **numéricamente singular** desde ese punto, y por eso D-12
exige `lstsq` (SVD). Con D-27/D-28 el crecimiento del condicionamiento completo **ya no es
monótono con el grado**: el máximo (1,5 · 10¹⁸) está en grado 3, no en grado 4 (3,8 · 10¹⁷) —
la causa 3 mete redundancia extra ya en grado 2, y el punto en que el condicionamiento completo
es peor deja de coincidir con el de más columnas. Restringido al subespacio de rango completo el
condicionamiento es varios órdenes de magnitud más chico en los cuatro grados (17,5 / 401,6 /
31.398,9 / 1.496.341,4), así que las **predicciones** siguen siendo estables. Lo que no es
estable es el reparto de coeficientes entre columnas colineales: **no se pueden interpretar de a
uno** — y desde D-27/D-28 eso alcanza también a `age`/`edad_al_cuadrado` y a
`smoker=yes`/`bmi_si_fuma` en cuanto grupo, aunque en grado 1 (producción) no haya colinealidad
exacta entre ellas.

---

## 4.bis Una objeción honesta a la regla de 1 error estándar

El error estándar se calcula como $\sigma/\sqrt{k}$, con $\sigma$ el desvío del RMSE **entre
folds**. Es la práctica estándar (Hastie, Tibshirani & Friedman; es lo que hace `glmnet`), pero
tiene un supuesto que conviene declarar antes de que lo pregunten: **$\sqrt{k}$ trata a los $k$
folds como muestras independientes, y no lo son**. Los conjuntos de entrenamiento de dos folds
cualesquiera comparten $3/4$ de sus filas, así que sus errores están correlacionados
positivamente y el desvío entre folds **subestima** la variabilidad real.

Consecuencia práctica: el ES real depende de con qué $k$ se lo mida, y $k=5$ no es el que da el
valor más alto. Eso no debilita la conclusión, la refuerza — si el umbral efectivo es más ancho
que los 164,1 que usa la selección, las configuraciones estadísticamente indistinguibles del
mejor son **más** de 7 (de 15), no menos, y el argumento para elegir el modelo simple queda más
firme. Con D-27/D-28 esta objeción ya no decide nada de por sí —el ganador crudo de la CV **es**
el modelo de producción, así que no hay un umbral que amplíe la banda para bajar a un modelo más
simple (ver §6)—, pero sigue valiendo como advertencia sobre el ES que reporta cualquier CV con
$k=5$.

Cuantificar ese sesgo con precisión requeriría validación cruzada repetida o un estimador
corregido. El barrido controlado sobre el pipeline vigente (`resultados/sensibilidad_k.csv`, D-22)
lo cuantifica por otra vía y confirma la dirección: el ES sube a 293,9 en $k=10$ —1,79 veces los
164,1 que reporta la selección— y de ahí **cae** de forma monótona (250,9 en $k=20$); no hay una
meseta que lo sostenga por encima de $k=5$.

---

## 4.ter Sensibilidad al número de folds (D-22)

`src/sensibilidad_k.py` corre dos experimentos que conviene no mezclar. Ninguno toca test: el
split se hace con la misma semilla y `idx_test` se descarta sin usarlo, así que todo pasa dentro
de las 1070 filas de train. Por eso es lícito correrlo **después** de la evaluación de test sin
violar D-09 — no re-evalúa test, mide cuán estable es un procedimiento de selección ya ejecutado.

> **Esta sección (partes A y B) documenta la corrida hecha antes de D-23**, con los números
> registrados en las tablas de este apartado; esa corrida está preservada en
> `resultados/sensibilidad_k_previo_d23.json`. **El barrido completo se rehízo con el pipeline de
> D-23 (9 features) el 19/08** (2,8 h de cómputo), con la misma conclusión de estabilidad que
> abajo: **el modelo de producción sale lineal de grado 1 en los tres $k$** (ganador Lasso g2
> λ=98,8 en los tres; ES 173,7 / 291,4 / 248,1; dentro de 1 ES: 9 de 17, 10 de 18, 10 de 18).
>
> **Con D-27/D-28 (11 features) la parte B se volvió a correr** y es la que hoy vive en
> `resultados/sensibilidad_k.csv` —resumida en §4.bis, ES 164,1 en $k=5$ con pico de 293,9 en
> $k=10$—, mientras que esa corrida de 9 features de D-23 quedó preservada aparte en
> `resultados/sensibilidad_k_previo_d27.json`. **La parte A no se volvió a correr con las 11
> features**: rehacer la selección completa (19 configuraciones, grado 4 con 1364 columnas) cuesta
> ~5 h, y no corrió en esta ola. La corrida vigente de la parte A sigue siendo, por lo tanto, la de
> 9 features de D-23, y es la que hoy vive en `resultados/sensibilidad_k.json` (misma corrida que
> `sensibilidad_k_previo_d27.json`, preservada bajo los dos nombres) — hay que declarar esto cada
> vez que se cite un número de la parte A: **no** es del pipeline vigente de 11 features.

### A — ¿Cambia el modelo elegido?

Se repite la selección completa del punto 5 (las 19 configuraciones, regla de 1 ES, criterio de
parsimonia) para cada $k$:

| | $k=5$ | $k=10$ | $k=20$ |
|---|---|---|---|
| Filas de entrenamiento por fold | 856 | 963 | 1016 |
| Ganador crudo de la CV | lasso g4, λ=286,4 | lasso g4, **λ=95,5** | lasso g4, **λ=95,5** |
| RMSE de validación del ganador | 4920,0 | 4833,2 | 4744,8 |
| σ entre folds | 224,3 | 724,7 | 1031,7 |
| Error estándar | **100,3** | **229,2** | **230,7** |
| Configuraciones dentro de 1 ES | 8 | 9 | 9 |
| **Producción (regla de 1 ES)** | **lasso g2, λ=286,4** | **lasso g2, λ=286,4** | **lasso g2, λ=286,4** |
| Costo (orientativo, según máquina) | ≈7 min | ≈14,5 min | ≈26 min |

**El modelo de producción es idéntico en los tres.** Esa es la afirmación que importa: la
respuesta del punto 5 no es un artefacto de haber puesto 5.

Dos observaciones honestas que van con la tabla:

1. **El ganador crudo sí se mueve**, y $k=5$ es el que queda solo: con 856 filas por fold el
   grado 4 necesita más regularización (λ=286,4) que con 963 o 1016 (λ=95,5). Es el sesgo de la
   curva de aprendizaje, medible y no teórico. No cambia nada de lo que se entrega —el ganador
   crudo no es el modelo de producción— pero conviene decirlo antes de que lo pregunten.
2. **La banda de 1 ES se ensancha** (8 → 9 configuraciones indistinguibles), que es exactamente
   lo que §4.bis anticipaba.

### B — ¿Qué le pasa a los números que se reportan?

En A el ganador cambia de configuración entre $k=5$ y $k=10$, así que comparar su RMSE mezcla dos
efectos. Acá se **fija** la configuración de producción y se varía sólo $k$:

| $k$ | puntos por fold | media de los $k$ RMSE | **RMSE agrupado** | σ entre folds | ES |
|---:|---:|---:|---:|---:|---:|
| 5 | 214 | 4955,3 | 4960,4 | 226,2 | **101,1** |
| 10 | 107 | 4896,0 | 4946,7 | 705,9 | **223,2** |
| 20 | 54 | 4836,5 | 4935,9 | 978,4 | **218,8** |
| 50 | 21 | 4681,9 | 4935,6 | 1550,1 | **219,2** |
| 100 | 11 | 4529,6 | 4935,2 | 1977,0 | 197,7 |
| 200 | 5 | 4161,8 | 4934,5 | 2620,1 | 185,3 |
| 500 | 2 | 3565,9 | 4934,1 | 3399,6 | 152,0 |
| 1070 (LOO) | 1 | **3073,5** | 4934,3 | 3860,2 | 118,0 |

> **Por qué la grilla es densa.** Con sólo (5, 10, 20, 50, 1070) el tramo final es un salto de
> 20× y la figura lo dibuja como una recta, o sea **afirma** una forma intermedia que nadie midió.
> Los puntos 100, 200 y 500 cuestan segundos (grado 2 son 44 columnas) y convierten esa recta en
> dato. No fue cosmético: **desmintieron una lectura** que los tres puntos originales sugerían
> (ver hallazgo 2).

Hay **dos hallazgos distintos**, y el primero corrige una lectura tentadora:

**1. El nivel del error casi no depende de $k$ — la caída es un artefacto de cómo se promedia.**
A primera vista la columna "media de los $k$ RMSE" baja 1882 dólares (un 38 %) de $k=5$ a LOO, y
es tentador leerlo como que $k=5$ es fuertemente pesimista. No lo es. El pipeline reporta la
**media de los $k$ RMSE de fold**, y la raíz es cóncava, así que por desigualdad de Jensen

$$\operatorname{media}_i \sqrt{\mathrm{ECM}_i} \;<\; \sqrt{\operatorname{media}_i \mathrm{ECM}_i},$$

con una brecha que crece cuando los folds se achican. La columna **RMSE agrupado** —una sola raíz
sobre los 1070 residuos *out-of-fold* juntos— no tiene ese sesgo, y va de 4960,4 a 4934,3: **26
dólares en total, un 0,5 %**, agotados ya en $k=20$. O sea: el sesgo pesimista real de $k=5$ es
despreciable, y lo que se movía era la métrica, no el modelo.

Es el mismo fenómeno que hace inservible a LOO con este pipeline, llevado al extremo: **con un
solo dato por fold, $\mathrm{RMSE}_i = |y_i - \hat y_i|$ y el promedio de los $k$ RMSE es
literalmente el MAE** (3073,5), un 38 % por debajo del RMSE y no comparable contra el RMSE de test
que reporta el informe. LOO no es "más validación": rompe la métrica en silencio.

**2. El ES tiene un máximo en $k=10$–$50$; no es una meseta, y el atípico sigue siendo $k=5$.**
Con los tres puntos originales (10, 20, 50) el ES parecía **estabilizarse** en ≈ 220. La grilla
densa muestra que no: sube hasta 223,2 en $k=10$, se sostiene en ≈ 220 hasta $k=50$, y después
**baja monótonamente** (197,7 → 185,3 → 152,0 → 118,0). Es una joroba, y los tres puntos medidos
al principio caían justo arriba de ella.

El mecanismo se puede verificar contra la propia tabla. Si $\sigma$ creciera exactamente como
$\sqrt{k}$, el ES sería constante. Comparando la razón observada de $\sigma$ contra $\sqrt{k}$
tramo a tramo:

| tramo | razón de σ | $\sqrt{k_2/k_1}$ | qué implica |
|---|---:|---:|---|
| 5 → 10 | **3,12** | 1,41 | σ crece mucho más: **la σ de $k=5$ es anormalmente chica** |
| 10 → 20 | 1,39 | 1,41 | sigue a $\sqrt{k}$ → ES plano |
| 20 → 50 | 1,58 | 1,58 | sigue a $\sqrt{k}$ → ES plano |
| 50 → 100 | 1,28 | 1,41 | empieza a quedarse atrás |
| 200 → 500 | 1,30 | 1,58 | σ **se satura** → ES cae |
| 500 → 1070 | 1,14 | 1,46 | σ se satura → ES cae |

Las dos puntas de la joroba tienen causas distintas, y ninguna es la del medio. A la izquierda,
$k=5$ estima una dispersión con **cinco números** y `ddof=0`: un estimador ruidoso y sesgado hacia
abajo, y con folds de 214 puntos la cola pesada de `charges` (los outliers que D-02 conserva)
casi no se muestrea. A la derecha, $\sigma$ deja de crecer como $\sqrt{k}$ porque **satura contra
la dispersión de los residuos individuales**: con 11, 5, 2 o 1 punto por fold ya no mide
variabilidad entre remuestreos sino entre observaciones.

> **Inferencia:** las razones de la tabla son dato; que la causa a la izquierda sea el estimador
> con `ddof=0` y a la derecha la saturación de $\sigma$ es la lectura más razonable de ese patrón,
> no una descomposición formal de la varianza.

Consecuencia práctica: **el rango en el que el ES significa algo es $k=10$–$50$**, donde vale
≈ 220. Los 101,1 de $k=5$ son el **45 %** de ese valor. El ES de LOO (118,0) no entra en la
comparación por la misma razón que el de $k=500$: es otra cantidad con el mismo nombre.

### Qué se hace con esto

**No se cambia el pipeline.** El modelo elegido es el mismo bajo los tres $k$, el nivel del error
no depende de $k$ una vez que se corrige la forma de promediar, y el test ya está evaluado
(D-09; sobre las dos corridas y la partición reutilizada, ver D-26): migrar a $k=10$ no
compraría nada y sí rompería esa garantía.

Lo que sí cambiaba, con esta corrida, era lo que el informe **declaraba**: el ES verdadero era al
menos ≈ 220, no 100,3, y por lo tanto la banda de 1 error estándar era al menos el doble de ancha
que la reportada entonces. Con el pipeline actual el argumento es el mismo pero los números son
otros —ver §4.bis, que lo recalcula sobre `resultados/sensibilidad_k.csv`—: eso **refuerza** la
elección del modelo simple, hay más configuraciones estadísticamente indistinguibles del mejor,
no menos.

## 4.quater El EDA de la Clase 3 y lo que cambió (D-23, D-24, D-25)

Las figuras 7 y 8 —los histogramas de las siete variables, y el de `charges` separado por
población— se hicieron después de haber cerrado el pipeline, siguiendo el método de la Clase 3.
Salieron tres cosas: una feature que baja el error más que cualquier otra decisión de este
trabajo, una justificación del método que estaba mal escrita, y una prescripción del deck que
acá no funciona. Las tres se miden en `src/evidencia_features.py`, que corre sobre train y
escribe `resultados/evidencia_features.csv`.

Todas las comparaciones de esta sección se promedian sobre **ocho particiones de folds
distintas**, no sobre una. Con una sola, una diferencia de treinta dólares es indistinguible de
que un fold te haya tocado mejor; con ocho, el desvío entre particiones queda a la vista y se
puede decir cuál diferencia es real.

### D-23 — La característica `fumador_obeso`

El histograma de `charges` muestra tres poblaciones. La Clase 3 prescribe, para ese caso, (a)
darle esa información al sistema como variable binaria o (b) partir el dataset y entrenar dos
modelos. Se eligió (a): (b) habría dejado 274 filas para el modelo de fumadores.

**Lo que separa a las poblaciones es un escalón, no una pendiente.** Partiendo el bmi en tramos
finos, entre los fumadores (dataset completo, `python3 -m src.datos`):

| tramo de bmi | [25,28) | [28,29) | [29,30) | **[30,31)** | [31,32) | [32,35) |
|---|---:|---:|---:|---:|---:|---:|
| **fumadores** | 22 445 | 22 044 | 23 555 | **38 799** | 37 969 | 40 758 |
| no fumadores | 8 524 | 7 589 | 8 257 | 8 058 | 8 543 | 8 716 |

Quince mil dólares en una unidad de bmi, con pendiente suave a los dos lados del corte, y sin
nada parecido entre los no fumadores. Esto es lo que distingue la feature nueva del término
`bmi*smoker=yes` que la expansión de grado 2 ya generaba: **ese término modela un cambio de
pendiente**, y un producto con una variable continua no puede representar un salto.

El efecto, medido (RMSE de validación, ± desvío entre las ocho particiones):

| grado | modelo | sin la feature | con la feature | diferencia |
|---:|---|---:|---:|---:|
| 1 | OLS | 6 094 ± 20 | **4 455 ± 14** | **−1 639** |
| 1 | Lasso | 6 114 ± 19 | 4 493 ± 11 | −1 621 |
| 2 | OLS | 4 953 ± 29 | 4 470 ± 15 | −482 |
| 2 | Lasso | 4 913 ± 21 | 4 457 ± 9 | −456 |
| 3 | OLS | 5 174 ± 62 | 4 898 ± 43 | −276 |
| 3 | Lasso | 4 905 ± 23 | 4 449 ± 9 | −456 |

La fila que decide el trabajo es la primera: **un modelo lineal sin regularizar, con una
característica más, queda por debajo de cualquier configuración del pipeline anterior** —el mejor
de aquéllos era Lasso grado 2 con 4 913. Y las diferencias son de mil seiscientos dólares contra
un ruido de veinte: no hay ambigüedad que resolver.

**El umbral no se ajustó a los datos.** El 30 es el umbral clínico de obesidad de la OMS, una
constante médica externa a este dataset, y es el que ya usaba `interaccion_fumador_bmi` desde el
punto 1.4. El barrido de umbrales lo confirma —el mínimo cae exactamente en 30, con 29 y 31
claramente peores—, pero la confirmación llegó después de la elección, no antes: si el corte se
hubiera elegido por barrido, sería un hiperparámetro ajustado sobre validación y habría que
declararlo como tal.

**El efecto es de la interacción, no de sus partes.** Agregar sólo `bmi > 30` deja el error
prácticamente donde estaba; `smoker` ya estaba en la matriz. Es la conjunción la que aporta.

**No es fuga de datos.** Las dos columnas de las que sale están disponibles al momento de
predecir, el umbral es externo, y `charges` no interviene en el cálculo.

### D-24 — La razón de D-03 estaba mal, la decisión no

D-03 decía: *«sin estratificar: el target es continuo, no hay clases que preservar»*. La figura 8
lo refuta. `charges` no es una distribución continua unimodal sino tres poblaciones casi
disjuntas, y los cinco folds las reparten desigual: entre **17 y 31** fumadores obesos por fold
(del 7,9 % al 14,5 %), con correlación **+0,62** entre esa proporción y el `charges` medio del
fold. Hay clases de facto, y se reparten mal.

Eso hacía esperar que estratificar redujera la varianza entre folds. **No la reduce.**

| desvío del RMSE entre folds | promedio de 8 particiones | rango entre particiones |
|---|---:|---|
| folds aleatorios (D-03, actual) | 593 | 410 – 715 |
| folds estratificados por población | 591 | 330 – 876 |

Ocho dólares de diferencia, cuando entre dos particiones del **mismo** método la diferencia va de
410 a 715. Es ruido.

El motivo es que lo que mueve el RMSE de un fold no es *cuántos* fumadores obesos le tocaron sino
*cuáles*: dentro de ese grupo los costos van de 32 548 a 63 770 dólares (dataset completo), así
que igualar los conteos deja intacta la varianza que importa.

Esto no es un detalle cosmético para la defensa. El error estándar que usa la regla de 1 ES sale
de ese desvío, así que si estratificar lo hubiera bajado, el umbral de la regla se habría angostado
y podría haber cambiado el modelo elegido. No pasa, y ahora está medido.

### D-25 — `children` se queda numérica

El slide 35 muestra una variable que parecía continua y que el histograma revela discretizada, y
prescribe tratarla como discreta: one-hot, o una binaria. `children` es exactamente ese caso —seis
escalones enteros— así que la prescripción aplica y hay que responderla.

| grado | modelo | `children` numérica | `children` one-hot | diferencia |
|---:|---|---:|---:|---:|
| 1 | OLS | 4 380 ± 14 | 4 385 ± 11 | +5 |
| 1 | Lasso | 4 414 ± 8 | 4 450 ± 8 | +36 |
| 2 | OLS | 4 521 ± 17 | 4 974 ± 177 | +453 |
| 2 | Lasso | 4 457 ± 9 | 4 535 ± 46 | +78 |

Neutro en grado 1 y peor en grado 2. Se ve por qué en los datos: `charges` medio es casi monótono
en `children` hasta 3 hijos (12 004 → 12 580 → 15 666 → 16 211), así que la codificación numérica
es una aproximación barata y de baja varianza; y los niveles 4 y 5 tienen 18 y 16 filas, a las que
one-hot les daría un parámetro propio. La receta del deck es correcta en general y no lo es acá,
que es la clase de cosa que sólo se sabe midiendo.

---

## 4.quinquies Dos características más: curvatura y pendiente (D-27, D-28)

`fumador_obeso` (D-23) modela un **escalón**: una diferencia de nivel entre dos poblaciones. Pero
cerrar esa comparación con la Clase 3 dejó dos preguntas abiertas que no son de escalón sino de
**forma de la curva**, y que un histograma de poblaciones no contesta: ¿el costo crece en línea
recta con la edad, o se curva? ¿el costo de los fumadores crece a la misma tasa por punto de bmi
que el de los no fumadores, o distinto? Las dos se responden agregando una característica
derivada, y las dos bajan el error de validación. Se miden, igual que en §4.quater, promediando
sobre **8 particiones de folds distintas**, con `src/evidencia_features.py`
(`resultados/evidencia_features.csv`, bloque `D-27-D-28`):

| característica agregada | RMSE de validación | diferencia |
|---|---:|---:|
| base D-23 (9 features) | 4 455,23 | — |
| + `edad_al_cuadrado` (D-27) | 4 425,47 | **−29,76** |
| + `bmi_si_fuma` (D-28) | 4 409,76 | **−45,47** |
| + las dos (pipeline vigente) | 4 380,07 | **−75,16** |

**Las dos son aditivas**: −29,76 más −45,47 dan −75,23, contra el −75,16 medido con las dos
juntas — la diferencia (7 centavos) es ruido, no interacción. Es la confirmación de que modelan
**estructuras distintas**: si compitieran por explicar la misma variación, el efecto conjunto
sería menor que la suma de los efectos individuales, y no lo es.

### D-27 — `edad_al_cuadrado`

El costo médico no crece en línea recta con la edad: la atención de un paciente mayor no cuesta
un incremento constante por año, crece de forma **convexa**. Una recta ajustada por mínimos
cuadrados subestima sistemáticamente a los extremos de edad. `age²` le da al modelo lineal la
curvatura que una recta no tiene, sin necesitar la expansión polinómica completa —que trae la
duplicación de la causa 3 de la sección 4— para conseguirla.

### D-28 — `bmi_si_fuma`

Entre los fumadores, el costo no sólo tiene el escalón de `fumador_obeso`: además **crece más
rápido con el bmi** que entre los no fumadores. Eso es una **pendiente distinta por grupo**, y es
justo lo que un escalón no puede representar —un escalón desplaza el nivel, no cambia cuánto
sube el costo por unidad de bmi—. `bmi_si_fuma = bmi · 1[smoker=yes]` es el término de
interacción que le da esa pendiente al modelo lineal sin expandir a grado 2.

Nótese que la expansión polinómica de grado 2 ya generaba el monomio `bmi · smoker=yes` —es
literalmente la misma cuenta—, así que D-28 no es una estructura nueva que el pipeline no podía
alcanzar: es la manera de dársela al modelo **de grado 1**, que es donde vive la producción, en
vez de esperar a que el barrido de grados la encuentre a costa de 66 columnas más.

### La contrapartida: más redundancia, más configuraciones sin converger

Las dos características nuevas no son gratis fuera de grado 1. Como ya `age` y `bmi`/`smoker=yes`
están en la matriz, la expansión polinómica **vuelve a generar** `age²` y `bmi·smoker=yes` a
partir de grado 2 — y esas columnas son, hasta el redondeo de punto flotante, idénticas a
`edad_al_cuadrado` y a `bmi_si_fuma`. Es la causa 3 de la sección 4: sube la redundancia de grado
4 de 59,7 % (pipeline de 9 features) a **68,0 %** (1364 columnas, 928 redundantes), y es la razón
concreta por la que esta corrida de la selección deja **4 configuraciones sin converger** en vez
de las 2 de antes —lasso g3 con λ=310,83/103,61/31,08 y g4 con λ=31,08—: más columnas
exactamente colineales hacen más lento al descenso por coordenadas (D-18), y D-20 exige
excluirlas de la selección en vez de reportar dónde quedó cortada la optimización.

En **grado 1**, que es donde vive la producción, esta contrapartida no existe: ahí `age²` y
`bmi·smoker=yes` no se generan por ningún otro camino, así que `edad_al_cuadrado` y
`bmi_si_fuma` son la única vía de esas dos estructuras y no hay duplicación que pagar.

## 4.sexies El error que queda: diagnóstico de residuos (D-30)

Bajado el error con D-23, D-27 y D-28, la pregunta que sigue es si se puede bajar más. `D-30`
contesta con un diagnóstico *out-of-fold* sobre train: para cada fila, la predicción sale de un
modelo que **no la vio** durante el ajuste —el mismo esquema de los 5 folds de D-04, pero
guardando el residuo de cada fila en vez de sólo el RMSE agregado—, en
`src/diagnostico_residuos.py` (`resultados/diagnostico_residuos.csv`).

### El error no está repartido

| fracción de filas con mayor residuo | filas | % del error cuadrático total |
|---|---:|---:|
| 1 % | 10 | **23,53 %** |
| 5 % | 53 | **75,38 %** |
| 10 % | 107 | 91,54 % |
| 20 % | 214 | 93,47 % |

Diez personas —el 1 % del train— explican casi una cuarta parte del error cuadrático total. El
5 % explica tres cuartas partes. El modelo no está mal en todos lados por igual: está muy bien en
la enorme mayoría de las filas y muy mal en un núcleo chico.

### Por población

| población | n | RMSE | % del error cuadrático total |
|---|---:|---:|---:|
| no fumadores | 850 | 4 615 | **86,3 %** |
| fumadores, bmi ≤ 30 | 105 | 3 407 | 5,8 % |
| fumadores, bmi > 30 | 115 | 3 804 | 7,9 % |

Es un resultado que a primera vista sorprende: el error se concentra en los **no fumadores**, la
población que `fumador_obeso` y `bmi_si_fuma` no distinguen entre sí, no en los fumadores obesos
que motivaron D-23. Adentro de "no fumadores" hay, otra vez, una subpoblación de facto.

### El núcleo: 28 personas que ninguna columna distingue

Filtrando a los no fumadores con `charges > 25 000` —28 filas, **2,6 %** del train— ese grupo
aporta el **41,88 %** del error cuadrático total, y el modelo los **subestima en promedio 17 378
dólares**. Es, con amplio margen, el bloque más grande de error de todo el dataset.

Y no se los puede identificar con las columnas disponibles:

| variable | estas 28 personas | resto de los no fumadores |
|---|---:|---:|
| age (media) | **50,79** | 39,11 |
| bmi (media) | 31,23 | 30,69 |
| children (media) | 1,39 | 1,10 |
| sex | 54 % F / 46 % M | 51 % F / 49 % M |
| region | pareja a la del resto (northeast 29 %, northwest 29 %, southeast 32 %, southwest 11 %) | northeast 24 %, northwest 26 %, southeast 25 %, southwest 25 % |

La única diferencia que se nota es la edad (50,8 contra 39,1); bmi, `children`, sexo y región son
prácticamente el mismo perfil que el resto de los no fumadores. Ninguna combinación de las
columnas originales —ni siquiera con las cuatro características derivadas del pipeline vigente—
separa a este grupo del resto antes de ver `charges`.

### Conclusión: error irreducible con este dataset

Ese ~42 % del error cuadrático **no es falta de capacidad del modelo**: es que falta la variable
que explica el gasto —una patología, una cirugía, una condición preexistente— y esa variable no
está en el CSV. Ninguna familia de modelos, por más flexible que sea, puede recuperar una señal
que no está en las columnas de entrada; agregar grados de libertad ahí sólo memoriza ruido (es
exactamente el sobreajuste que documenta la sección 4 en grados 3 y 4).

Esto se puede verificar contra el propio historial de este documento: el diagnóstico se rehizo
**después** de D-27 y D-28, y el núcleo no se movió —era 41,1 % del error con el pipeline de 9
features, es 41,9 % ahora—. Mejorar el modelo (D-23, D-27, D-28) bajó el RMSE de test 450,81
dólares, pero no tocó este núcleo: es la evidencia de que el ~42 % es un piso, no una debilidad de
esta familia de modelos en particular.

---

## 5. Verificación

Los números del informe se recalcularon con una reimplementación independiente del pipeline,
que no pasa por `src/experimentos.py`:

| Chequeo | Resultado |
|---|---|
| RMSE de validación cruzada, grados 1–4 | Coincide a 0,1 |
| RMSE de test (producción, referencia lineal, baseline) | Coincide a 0,01 |
| Usos de `X_test` / `y_test` antes del punto 5 | 4 usos, 0 sospechosos (sólo construcción y `len()`) |
| Firma de sobreajuste: brecha train–val | 61,8 (grado 1) → 84.537,9 (grado 4) — el grado 4 es 19,9 veces el error del grado 1, y el desvío entre folds pasa de ±367,0 a ±47.000,1 (128 veces) |
| Lasso contra `scipy.optimize` sobre el mismo objetivo | Coincide a 8 decimales en 3 valores de $\lambda$ |
| Alineación de `nombres_polinomicos` con las columnas | Los monomios de grado 4, verificados por factorización en primos |
| Suites `test_validacion`, `test_preproceso`, `test_modelos`, `test_paleta` | 4/4 en verde |
| Selección del punto 5 recalculada con $k=5$, 10 y 20 (D-22) | Mismo modelo de producción (lineal grado 1) en los tres, pero **esta corrida es del pipeline de 9 features (D-23), no se re-corrió con las 11 de D-27/D-28** — preservada en `resultados/sensibilidad_k.json` y `resultados/sensibilidad_k_previo_d27.json` (misma corrida, dos nombres); la corrida pre-D-23 (§4.ter, en `sensibilidad_k_previo_d23.json`) daba la misma conclusión de estabilidad, con su producción de entonces (Lasso g2 λ=286,4 en los tres) |
| `src/sensibilidad_k.py` (parte B) reproduce la producción vigente en $k=5$ | Verificado contra `resultados/sensibilidad_k.csv` (rehecho con el pipeline de 11 features): RMSE de validación 4413,45, igual al de `modelo_elegido.json`; el JSON vigente coincide (error estándar 164,11, ganador lineal grado 1 sin regularizar) |
| Diagnóstico de residuos (D-30) reproduce el núcleo de error | Verificado contra `resultados/diagnostico_residuos.csv`: 28 no fumadores con `charges` > 25 000 (2,6 % de train) aportan 41,88 % del SSE; coincide con el 41,1 % medido antes de D-27/D-28, confirmando que el núcleo no se mueve al mejorar el modelo |

El chequeo de alineación de nombres merece una nota: se asigna un **primo distinto a cada
columna**, con lo cual el valor de cada monomio es un producto de primos que factoriza
unívocamente al monomio. Si un nombre no coincidiera con su columna, el informe estaría
reportando el coeficiente de una feature con el nombre de otra — un error silencioso que
ningún test de conteo detecta.

---

## 6. Resultado

Con D-27/D-28 desaparece la tensión que este documento venía arrastrando desde D-23: **el ganador
crudo de la validación cruzada y el modelo de producción son el mismo modelo.** El lineal de
grado 1 con las 11 features es, a la vez, el de menor RMSE de validación entre las 15
configuraciones elegibles **y** el más simple de todo el espacio de búsqueda (D-08). Ya no hay
que bajar de un ganador más complejo pagando un costo de simplicidad: no hay segundo lugar del
que bajar.

| | Ganador de la validación cruzada = Producción |
|---|---|
| Modelo | **lineal grado 1, sin regularizar, 11 features** |
| Features vivas | **11 de 11** |
| RMSE validación | **4413,45 ± 366,97** |
| RMSE train (CV) | 4351,63 ± 93,21 |
| **RMSE test** | **4288,52** |

La regla de 1 error estándar (D-22) se sigue aplicando —el error estándar es 164,11, el umbral
1 ES es 4577,56 y hay 7 de 15 configuraciones elegibles dentro de esa banda— y **confirma** la
elección: la más simple de esas 7 vuelve a ser el mismo modelo lineal de grado 1. Pero ya no es
la regla la que decide, porque el punto de partida (el mínimo crudo) y el punto de llegada (el
más simple dentro de la banda) coinciden desde el primer paso.

Referencias sobre test: predecir siempre la media da 11 963,43 —el modelo vigente es **2,79
veces** mejor que ese baseline. La historia completa del test, sobre las mismas 267 filas
(semilla 42; D-09, D-26, D-29):

| etapa | pipeline | RMSE de test |
|---|---|---:|
| antes de D-23 | 8 features | 4 739,33 (`resultados/evaluacion_test_previo_d23.json`) |
| D-23 (`fumador_obeso`) | 9 features | 4 465,32 (`resultados/evaluacion_test_previo_d27.json`) |
| D-27/D-28 (`edad_al_cuadrado`, `bmi_si_fuma`) | 11 features, **vigente** | **4 288,52** |

Mejora total: **450,81 dólares, un 9,5 %** sobre el punto de partida; la última etapa (D-27/D-28)
sola aportó **176,80**. El test reentrenado da un RMSE de train de 4360,21, y la diferencia entre
test y validación es **−124,93** (D-29): el test salió más bajo que la validación, dentro de lo
esperable para una sola partición de 267 filas y no una señal de fuga —D-09 y la verificación de
§5 descartan esa posibilidad.

Queda, eso sí, un costo que D-27/D-28 sí introducen: `age`/`edad_al_cuadrado` y
`smoker=yes`/`bmi_si_fuma` están fuertemente correlacionadas entre sí (§4), así que **los
coeficientes individuales de esos dos grupos ya no son interpretables de a uno** —el grupo sí lo
es—. Es visible en los propios coeficientes del modelo de producción: `age` cae de 3754 a 45,10
y `bmi` de 333 a 111,20 porque `edad_al_cuadrado` (3749,13) y `bmi_si_fuma` (5236,99) absorbieron
buena parte de su efecto; `smoker=yes` cae de 5437 a 1161,22 por lo mismo, con `fumador_obeso`
(4891,83) llevándose el resto del escalón. Es un costo real de interpretabilidad que el modelo de
9 features no tenía, y que D-30 (§4.sexies) señala que no alcanza para explicar el ~42 % de error
que sigue siendo irreducible con este dataset.

El desarrollo completo, con las tablas y las figuras, está en
[`informe/informe.pdf`](informe/informe.pdf).
