# TP1 — Regresión e Introducción a la evaluación de modelos

**72.75 Aprendizaje Automático (Machine Learning) — ITBA — 2026 Q2**
Defensa: 26/08/2026 · Enunciado: [`enunciado.pdf`](enunciado.pdf)

Predicción del costo anual de gastos médicos (`charges`) mediante regresión lineal y
polinómica, evaluadas con *k-fold cross-validation*.

**Resultado:** RMSE de **4289 dólares** en test con un modelo lineal de once características
y sin regularizar. Tres de esas características no venían en el CSV: la principal salió del
análisis exploratorio (ver *Hallazgo principal*).

---

## Dataset elegido: Insurance Charges

De los tres candidatos del punto 0 del enunciado (Bike Sharing, Insurance Charges, Wine
Quality) se eligió **Insurance Charges** (1338 filas × 7 columnas, Kaggle).

El criterio no fue el tamaño ni la popularidad, sino **cuál de los tres permite responder
efectivamente los puntos 1 y 5 del enunciado**:

| Criterio | Insurance | Bike Sharing | Wine Quality |
|---|---|---|---|
| Categóricas para el punto 1.1 | 3 reales (`sex`, `smoker`, `region`) | Sólo enteros que *simulan* ser ordinales | **Ninguna** — todo continuo |
| Outliers con criterio defendible (1.3) | 10.4 % de `charges`, y son señal | Escasos | Muchos, en 8 columnas |
| ¿Se puede mostrar sobreajuste? (punto 5) | **Sí**, y de forma extrema: el grado 4 se va a 20× el error del grado 1 | No en `day.csv` (465 features / 731 filas) | No: el error apenas se mueve |
| Riesgo metodológico | Ninguno | **Es serie temporal**: el split i.i.d. queda expuesto | Target ordinal 3–8 en el tinto, disfrazado de continuo |

Insurance es el único donde los cuatro subpuntos de limpieza tienen contenido real **y** la
comparación de modelos produce la curva en U del punto 5 con una explicación causal detrás
(ver *Hallazgo principal*).

> La comparación entre los tres datasets se hizo con un análisis exploratorio previo, **fuera
> de este repo**. Los únicos números que este repo calcula y respalda son los de Insurance, que
> están en `resultados/` y en `informe/salida-seleccion.txt`.

## Instalación y uso

Requiere Python 3.10+ y sólo `numpy`, `pandas` y `matplotlib`. **No usa scikit-learn**: el
enunciado pide *implementar* el esquema de validación cruzada (punto 2.2), así que el split,
el k-fold, la codificación, el escalado, la resolución de mínimos cuadrados, la expansión
polinómica y el Lasso están escritos desde cero sobre numpy. Tampoco usa pytest.

```bash
pip install numpy pandas matplotlib

python -m src.datos          # punto 1: análisis exploratorio
python -m src.experimentos   # puntos 2 a 5  (~27 min desde D-27/D-28: grado 4 pasó a 1364 features)
python -m src.evidencia_features  # la evidencia de D-23, D-24, D-25, D-27 y D-28
python -m src.diagnostico_residuos  # D-30 — dónde vive el error que queda
python -m src.graficos       # figuras de la presentación
python -m src.tablas         # regenera las tablas del informe desde los CSV

python -m tests.test_validacion    # los tres módulos base tienen su suite
python -m tests.test_preproceso
python -m tests.test_modelos
python -m tests.test_paleta       # la paleta de las figuras, bajo simulación de daltonismo
```

`src.experimentos` tarda unos 27 minutos: el Lasso de grado 4 son **1364** features casi
colineales —eran 714 antes de que D-27/D-28 agregaran `edad_al_cuadrado` y `bmi_si_fuma`, y
494 antes de que D-23 agregara `fumador_obeso`— y el descenso por coordenadas necesita
decenas de miles de barridas. **No hace falta correrlo** para ver los resultados: están
versionados en `resultados/` y la salida completa en `informe/salida-seleccion.txt`.

## Estructura

```
src/datos.py         punto 1 — la evidencia que justifica cada decisión de limpieza,
                     y la construcción de las tres features derivadas: `fumador_obeso`
                     (D-23), `edad_al_cuadrado` y `bmi_si_fuma` (D-27, D-28)
src/validacion.py    rmse, split train/test, k-fold, resumen entre folds
src/preproceso.py    codificación one-hot, estandarizado, expansión polinómica
src/modelos.py       OLS por mínimos cuadrados vía SVD (lstsq), ridge, Lasso por coordenadas
src/experimentos.py  puntos 2 a 5 — CV, grados 1-4, barrido de lambda, test final
src/evidencia_features.py  la evidencia numérica de D-23, D-24, D-25, D-27 y D-28 (no toca test)
src/tablas.py        genera informe/tablas-cv.tex desde resultados/*.csv
src/graficos.py      las figuras de la presentación y del informe (incluye los
                     histogramas del EDA: figuras 7 y 8)
src/sensibilidad_k.py  D-22 — barrido del número de folds k (no toca test; parte A ~3 h,
                     vigente con el pipeline anterior de 9 features; parte B barata,
                     rehecha con el pipeline actual)
src/diagnostico_residuos.py  D-30 — dónde vive el error que queda (no toca test)
tests/               suites con casos de respuesta conocida, sin pytest
                     (incluye test_paleta.py: los colores de las figuras se validan,
                     no se eligen a ojo)
data/raw/            insurance.csv, copiado tal cual se descargó
resultados/          cv_lineal.csv, cv_lasso.csv, modelo_elegido.json, evaluacion_test.json,
                     evaluacion_test_previo_d23.json, evaluacion_test_previo_d27.json,
                     diagnostico_residuos.csv, sensibilidad_k.{json,csv},
                     sensibilidad_k_previo_d23.json, sensibilidad_k_previo_d27.json
                     (corridas anteriores preservadas)
figuras/             PNG para la presentación
informe/informe.tex  informe completo en LaTeX -> informe.pdf (20 páginas)
informe/presentacion.tex  las diapositivas de la defensa -> presentacion.pdf (19 slides)
informe/guion.md     guion hablado de la defensa, con reloj por slide y preparación de preguntas
informe/             salida-seleccion.txt + resultados-test.tex (macros de test; plantilla hasta evaluar)
DECISIONES.md        las decisiones metodológicas numeradas, con su justificación
```

---

## Hallazgo principal

`bmi` correlaciona apenas **0.198** con `charges`, y sin embargo es determinante. La razón es
que su efecto **depende de si la persona fuma**:

| | bmi ≤ 30 | bmi > 30 | efecto de la obesidad |
|---|---:|---:|---|
| **No fumador** | 7 967 | 8 853 | +11 % |
| **Fumador** | 21 369 | 41 693 | **+95 %** |

La obesidad casi no cuesta si no fumás, y **duplica el costo si fumás**. Eso es una interacción
`smoker × bmi`, que un modelo puramente aditivo no puede representar.

**Y es un escalón, no una pendiente.** Esta es la corrección que trajo el EDA de la Clase 3
(figuras 7 y 8). Partiendo el bmi en tramos finos alrededor de 30, entre los fumadores:

| tramo de bmi | [28, 29) | [29, 30) | **[30, 31)** | [31, 32) |
|---|---:|---:|---:|---:|
| charges medio | 22 044 | 23 555 | **38 799** | 37 969 |

Quince mil dólares en una unidad de bmi, con pendiente suave a los dos lados. Entre los no
fumadores el mismo corte casi no mueve nada (8 257 → 8 058).

La expansión polinómica de grado 2 genera `bmi*smoker=yes` y el Lasso lo encuentra solo, pero
ese término modela un cambio de **pendiente**: un producto con una variable continua no puede
representar un salto. Por eso el pipeline agrega la característica binaria
`fumador_obeso = (smoker=yes) ∧ (bmi > 30)` — la opción (a) que prescribe la Clase 3 al detectar
más de una población en un histograma (slides 36–38). Es la decisión **D-23**, y es la que más
mueve el error de todo el trabajo.

El umbral 30 es el de la OMS: una constante médica externa al dataset, no el corte que minimiza
el error. El barrido de umbrales lo confirma igual (mínimo exacto en 30), y está en
`resultados/evidencia_features.csv`.

El mismo método —mirar qué estructura le falta al modelo aditivo y agregar la columna que la
representa, en vez de confiar en que el polinomio la encuentre sola— dio dos características
más: `edad_al_cuadrado` (la curvatura del costo con la edad, D-27) y `bmi_si_fuma` (la
pendiente distinta entre fumadores, D-28). Las dos bajan el error por separado y de forma
aditiva entre sí (ver *Resultados*), y con ellas el test bajó de 4465,32 a **4288,52** dólares.

## Resultados

Validación cruzada de 5 folds sobre las 1070 filas de entrenamiento (el test, 267 filas, se
reserva para la evaluación final — tres corridas sobre la misma partición: ver D-26 y D-29):

| Modelo | Grado | λ | RMSE train | RMSE validación | Features |
|---|---:|---:|---:|---:|---:|
| **lineal** | **1** | **—** | 4351.6 ± 93.2 | **4413.4 ± 367.0** | **11** |
| lineal | 2 | — | 4227.2 ± 104.7 | 4516.7 ± 380.0 | 77 |
| lineal | 3 | — | 3931.1 ± 117.1 | 6757.5 ± 857.0 | 363 |
| lineal | 4 | — | 3379.1 ± 105.3 | **87916.9 ± 47000.1** | 1364 |
| lasso | 2 | 103.6 | 4330.4 ± 89.2 | 4422.9 ± 360.3 | 22.2 de 77¹ |

¹ Coeficientes vivos promediados entre folds; es el mejor Lasso del barrido, pero no gana la
CV (ver más abajo).

El grado 4 sin regularizar es el retrato del sobreajuste: **el mejor error de train de toda la
tabla** (3379.1) y un error de validación de 87 917, casi veinte veces (19.9×) el del grado 1,
con un desvío entre folds de ±47 000 — ciento veintiocho veces el desvío del grado 1 (±367.0).
No es sólo peor: es inestable. La brecha train-validación pasa de 61.8 en grado 1 a 84 537.9 en
grado 4. Además de la colinealidad de siempre entre potencias de variables correlacionadas,
desde D-27/D-28 la expansión polinómica duplica EXACTAMENTE `edad_al_cuadrado` y `bmi_si_fuma`
a partir de grado 2 (`age*age` y `bmi*smoker=yes` son las mismas columnas), lo que sube la
redundancia a 68.0 % en grado 4 (era 59.7 %) y es la razón de que ahora sean 4 las
configuraciones que no convergen (antes 2). En grado 1 —el de producción— no hay duplicación.

**El modelo de producción es el ganador de la CV: lineal de grado 1, sin regularizar.** Ya no
hay tensión entre "el que gana" y "el que se elige": es a la vez el de menor error de
validación de las 15 configuraciones elegibles (de 19 corridas, 4 descartadas por no
converger — D-20) y el modelo más simple de todo el espacio de búsqueda. La regla de 1 error
estándar (ES = 164.1, umbral 4577.6) sigue aplicándose y confirma la elección —7 de las 15
elegibles caen dentro de un ES del mejor, y la más simple de esas 7 es la misma configuración—,
pero ya no es lo que decide: no hay "costo de simplicidad" que pagar porque el ganador crudo y
el modelo simple coinciden.

| | features | RMSE test |
|---|---:|---:|
| **Modelo de producción = ganador de la CV** (lineal grado 1) | **11** | **4288.52** |
| *Modelo de producción de D-23* (lineal grado 1, 9 features) | *9* | *4465.32* |
| *Modelo de producción anterior a D-23* (Lasso grado 2, λ=286.4) | *10 de 44* | *4739.33* |
| Baseline trivial (predecir siempre la media) | — | 11963.43 |

Tres etapas, una sola partición de test (semilla 42, D-26 y D-29): **4739.33 → 4465.32 →
4288.52** dólares. El EDA (D-23, `fumador_obeso`) bajó el error 274.01 dólares; las dos
features de curvatura y pendiente (D-27/D-28) bajaron otros 176.80; la mejora total es 450.81
dólares, un 9.5 % sobre el punto de partida. El modelo final es 2.79 veces mejor que el
baseline de predecir siempre la media. El RMSE de test (4288.52) queda 124.93 dólares por
DEBAJO del RMSE de validación de la CV (4413.45) — nada indica que el modelo generalice peor
de lo que promete la validación. (El train reentrenado sobre las 1070 filas completas da
4360.21, coherente con los dos números anteriores.) Las corridas anteriores están
versionadas a propósito en `resultados/evaluacion_test_previo_d23.json` y
`evaluacion_test_previo_d27.json`: D-26 y D-29 obligan a reportar los tres números, no sólo el
favorable.

Los once coeficientes del modelo de producción (sobre features estandarizadas, así que las
magnitudes son comparables entre sí):

```
bmi_si_fuma        5236.99      region=southwest    -498.36
fumador_obeso      4891.83      region=southeast    -424.86
edad_al_cuadrado   3749.13      sex=male            -218.32
smoker=yes         1161.22      region=northwest    -142.53
children            815.02      bmi                  111.20
                                age                    45.10
```

**Las dos features nuevas y `fumador_obeso` son los tres coeficientes más grandes del
modelo**, por encima de `smoker=yes`. Pero hay un costo de interpretabilidad real que el
modelo de nueve features no tenía: `age` y `edad_al_cuadrado` están muy correlacionadas entre
sí (por construcción), igual que `bmi`/`smoker=yes`/`bmi_si_fuma`, así que dentro de cada
grupo el reparto exacto del coeficiente es en buena medida arbitrario —es lo mismo que ya pasa
con la expansión polinómica de grado ≥ 2 (ver diagnóstico de rango más arriba)— y **los
coeficientes individuales de esos grupos ya no se pueden leer de a uno, sólo el grupo**: por
eso `age` cae de 3754 a 45 y `bmi` de 333 a 111 (el efecto se lo llevaron las columnas nuevas),
y por lo mismo `smoker=yes` cae de 5437 a 1161. El modelo entero sigue cabiendo en once
números: no hay penalización que apague nada porque no hay nada de más que apagar.

## Estado

- [x] Punto 0 — elección del dataset
- [x] Punto 1 — limpieza de datos (categóricas, faltantes, outliers, features y escalado)
- [x] Punto 2 — regresión lineal con k-fold CV implementado desde numpy
- [x] Punto 3 — regresión polinómica (grados 1–4) y regularización L1
- [x] Puntos 4 y 5 — evaluación, test final y las tres respuestas
- [x] **Presentación** — `informe/presentacion.pdf` (19 slides) + guion hablado con reloj
      en `informe/guion.md`. El enunciado exige mandarla junto con el código **24 h antes**
      de la defensa del **26/08/2026**.
- [x] **Evaluación final de test** — RMSE **4288.52** (lineal grado 1, 11 features).
      `resultados-test.tex` ya tiene los números y el informe y la presentación los toman
      solos al compilar.
- [x] **EDA de la Clase 3** — histogramas de las 7 variables (figuras 7 y 8), la
      característica `fumador_obeso` que salió de ahí (D-23), y la revisión de D-03 y del
      slide 35 que trajo (D-24, D-25).
- [x] **D-27 / D-28 — dos features derivadas más** — `edad_al_cuadrado` (la curvatura del
      costo con la edad) y `bmi_si_fuma` (la pendiente distinta entre fumadores), evidencia
      en `resultados/evidencia_features.csv` (−29.76 y −45.47 por separado, −75.16 juntas:
      aditivas). Con las dos, el pipeline pasa de 9 a 11 features y el modelo de producción
      pasa a ser también el ganador de la CV, sin costo de simplicidad. Test: **4288.52**.
- [x] **D-30 — diagnóstico de residuos** (`src/diagnostico_residuos.py`) — el 41.9 % del
      error cuadrático de train se concentra en 28 no fumadores (2.6 % de las filas) con
      `charges` > 25 000, subestimados en 17 378 dólares en promedio, y que ninguna
      combinación de las columnas del dataset distingue del resto de los no fumadores (la
      única diferencia es la edad). Es error **irreducible con este dataset**: falta la
      variable que explica el gasto y no está en el CSV. El núcleo no se movió con
      D-27/D-28 (era 41.1 % con 9 features).
- [x] **D-22 con el modelo nuevo, parcial** — la parte B (barrido controlado, figura 6) está
      rehecha con el pipeline actual (11 features) en `resultados/sensibilidad_k.csv`: el
      nivel del error casi no depende de k (4428.7 a k=5, 4405.9 en LOO — 0.51 % de
      diferencia), pero el error estándar sí, y pega su pico en k=10 (293.9, 1.79 veces el
      de la selección con k=5). La parte A (selección completa con k=5, 10 y 20) **sigue
      siendo la corrida del 19/08 con el pipeline anterior, de 9 features**
      (`resultados/sensibilidad_k.json`) — recorrerla con 11 features cuesta horas y no se
      hizo. La sección correspondiente de `DECISIONES.md` documenta también la corrida
      anterior a D-23, preservada en `sensibilidad_k_previo_d23.json` (y la de 9 features,
      previa a D-27/D-28, en `sensibilidad_k_previo_d27.json`).
