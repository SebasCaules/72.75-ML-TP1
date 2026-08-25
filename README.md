# TP1 — Regresión e Introducción a la evaluación de modelos

**72.75 Aprendizaje Automático (Machine Learning) — ITBA — 2026 Q2**
**Grupo 7** — Andrés Cortese (64612) · Sebastián Caules (64331)
Defensa: 02/09/2026, 17:35, Aula 701F (2ª fecha) · Enunciado: [`enunciado.pdf`](enunciado.pdf)

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

python -m src.datos          # punto 1: análisis exploratorio (sólo sobre train)
python -m src.experimentos   # puntos 2 a 5  (~27 min: el grado 4 son 1364 features)
python -m src.evidencia_features  # la evidencia numérica de las decisiones de features
python -m src.diagnostico_residuos  # dónde vive el error que queda
python -m src.graficos       # figuras de la presentación
python -m src.tablas         # regenera las tablas del informe desde los CSV

python -m tests.test_validacion    # los tres módulos base tienen su suite
python -m tests.test_preproceso
python -m tests.test_modelos
python -m tests.test_paleta       # la paleta de las figuras, bajo simulación de daltonismo
```

`src.experimentos` tarda unos 27 minutos: el Lasso de grado 4 son **1364** features casi
colineales y el descenso por coordenadas necesita decenas de miles de barridas. **No hace
falta correrlo** para ver los resultados: están versionados en `resultados/` y la salida
completa en `informe/salida-seleccion.txt`.

## Estructura

```
src/datos.py         punto 1 — la evidencia que justifica cada decisión de limpieza,
                     y la construcción de las tres features derivadas (`fumador_obeso`,
                     `edad_al_cuadrado`, `bmi_si_fuma`). `cargar_train()` es la puerta
                     de entrada del EDA: devuelve sólo las 1070 filas de train
src/validacion.py    rmse, split train/test, k-fold, resumen entre folds
src/preproceso.py    codificación one-hot, estandarizado, expansión polinómica
src/modelos.py       OLS por mínimos cuadrados vía SVD (lstsq), ridge, Lasso por coordenadas
src/experimentos.py  puntos 2 a 5 — CV, grados 1-4, barrido de lambda, test final
src/evidencia_features.py  la evidencia numérica de las features derivadas, de la
                     estratificación y de `children` (no toca test)
src/tablas.py        genera informe/tablas-cv.tex desde resultados/*.csv
src/graficos.py      las figuras de la presentación y del informe (incluye los
                     histogramas del EDA: figuras 7 y 8)
src/sensibilidad_k.py  barrido del número de folds k (no toca test). El informe usa la
                     parte B, el barrido controlado, que es el barato
src/diagnostico_residuos.py  dónde vive el error que queda (no toca test)
tests/               suites con casos de respuesta conocida, sin pytest
                     (incluye test_paleta.py: los colores de las figuras se validan,
                     no se eligen a ojo)
data/raw/            insurance.csv, copiado tal cual se descargó
resultados/          cv_lineal.csv, cv_lasso.csv, modelo_elegido.json, evaluacion_test.json,
                     evidencia_features.csv, diagnostico_residuos.csv, sensibilidad_k.csv
figuras/             PNG para la presentación
informe/informe.tex  informe completo en LaTeX -> informe.pdf (23 páginas)
informe/presentacion.tex  las diapositivas de la defensa -> presentacion.pdf (20 slides)
informe/guion.md     guion hablado de la defensa, con reloj por slide y preparación de preguntas
informe/             salida-seleccion.txt + resultados-test.tex (macros de test; plantilla hasta evaluar)
entregables/         lo que se manda a la cátedra: la presentación y el zip del código,
                     con el nombre del grupo. El zip lleva src/, tests/, data/ y su
                     propio README. Es contenido DERIVADO de este repo: se regenera,
                     no se edita a mano
```

---

## Hallazgo principal

`bmi` correlaciona apenas **0.194** con `charges` (medido sobre train), y sin embargo es
determinante. La razón es
que su efecto **depende de si la persona fuma**:

| | bmi ≤ 30 | bmi > 30 | efecto de la obesidad |
|---|---:|---:|---|
| **No fumador** | 8 009 | 8 890 | +11 % |
| **Fumador** | 21 472 | 41 811 | **+95 %** |

La obesidad casi no cuesta si no fumás, y **duplica el costo si fumás**. Eso es una interacción
`smoker × bmi`, que un modelo puramente aditivo no puede representar.

**Y es un escalón, no una pendiente.** Partiendo el bmi en tramos finos alrededor de 30,
entre los fumadores:

| tramo de bmi | [28, 29) | [29, 30) | **[30, 31)** | [31, 32) |
|---|---:|---:|---:|---:|
| charges medio | 22 358 | 24 079 | **38 878** | 39 114 |

Casi quince mil dólares en una unidad de bmi, con pendiente suave a los dos lados. Entre los no
fumadores el mismo corte casi no mueve nada (8 563 → 8 089).

La expansión polinómica de grado 2 genera `bmi*smoker=yes` y el Lasso lo encuentra solo, pero
ese término modela un cambio de **pendiente**: un producto con una variable continua no puede
representar un salto. Por eso el pipeline agrega la característica binaria
`fumador_obeso = (smoker=yes) ∧ (bmi > 30)`: cuando el histograma revela más de una población,
darle esa información al modelo como una columna binaria. Es la decisión que más mueve el error
de todo el trabajo.

El umbral 30 es el de la OMS: una constante médica externa al dataset, no el corte que minimiza
el error. El barrido de umbrales lo confirma igual (mínimo exacto en 30), y está en
`resultados/evidencia_features.csv`.

El mismo método —mirar qué estructura le falta al modelo aditivo y agregar la columna que la
representa, en vez de confiar en que el polinomio la encuentre sola— dio dos características
más: `edad_al_cuadrado` (la curvatura del costo con la edad) y `bmi_si_fuma` (la pendiente
distinta entre fumadores). Las dos bajan el error por separado y de forma aditiva entre sí: el
RMSE de validación pasa de 4455 con sólo `fumador_obeso` a **4380** con las tres.

## Resultados

Validación cruzada de 5 folds sobre las 1070 filas de entrenamiento. El test (267 filas) se
reserva para la evaluación final y no participa de ninguna decisión, ni siquiera del análisis
exploratorio:

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
grado 4. Una regla práctica lo anticipa sin mirar ningún error: hacen falta del orden de diez
ejemplos por parámetro, y cada fold entrena con 856 filas — alcanzan para los 11 parámetros del
grado 1, quedan justas para los 77 del grado 2 y no alcanzan para los 363 del grado 3 ni los
1364 del grado 4. Además de la colinealidad entre potencias de variables correlacionadas, la
expansión polinómica duplica EXACTAMENTE `edad_al_cuadrado` y `bmi_si_fuma` a partir de grado 2
(`age*age` y `bmi*smoker=yes` son las mismas columnas), lo que lleva la redundancia al 68.0 % en
grado 4 y explica que 4 configuraciones no converjan. En grado 1 —el de producción— no hay
duplicación.

**El modelo de producción es el ganador de la CV: lineal de grado 1, sin regularizar.** Ya no
hay tensión entre "el que gana" y "el que se elige": es a la vez el de menor error de
validación de las 15 configuraciones elegibles (de 19 corridas, 4 descartadas por no
converger) y el modelo más simple de todo el espacio de búsqueda. La regla de 1 error
estándar (ES = 164.1, umbral 4577.6) sigue aplicándose y confirma la elección —7 de las 15
elegibles caen dentro de un ES del mejor, y la más simple de esas 7 es la misma configuración—,
pero ya no es lo que decide: no hay "costo de simplicidad" que pagar porque el ganador crudo y
el modelo simple coinciden.

| | features | RMSE test | R² test |
|---|---:|---:|---:|
| **Modelo de producción = ganador de la CV** (lineal grado 1) | **11** | **4288.52** | **0.871** |
| Baseline trivial (predecir siempre la media) | — | 11963.43 | 0.000 |

El modelo es **2.79 veces mejor** que el baseline y explica el **87.1 %** de la varianza de
`charges` en datos que nunca vio. El RMSE de test (4288.52) queda 124.93 dólares por DEBAJO del
RMSE de validación de la CV (4413.45) — nada indica que el modelo generalice peor de lo que
promete la validación. (El train reentrenado sobre las 1070 filas completas da 4360.21,
coherente con los dos números anteriores.)

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

**Las tres features derivadas son los tres coeficientes más grandes del modelo**, por encima
de `smoker=yes`. Pero hay un costo de interpretabilidad real: `age` y `edad_al_cuadrado` están
muy correlacionadas entre sí por construcción, igual que `bmi`/`smoker=yes`/`bmi_si_fuma`, así
que dentro de cada grupo el reparto exacto del coeficiente es en buena medida arbitrario —es lo
mismo que ya pasa con la expansión polinómica de grado ≥ 2 (ver diagnóstico de rango más
arriba)— y **los coeficientes individuales de esos grupos no se leen de a uno, sólo el grupo**.
Por eso `age` (45.10), `bmi` (111.20) y `smoker=yes` (1161.22) quedan chicos: el efecto se lo
llevaron las columnas derivadas. El modelo entero cabe en once números: no hay penalización que
apague nada porque no hay nada de más que apagar.

## Estado

- [x] Punto 0 — elección del dataset
- [x] Punto 1 — limpieza de datos (categóricas, faltantes, outliers, features y escalado)
- [x] Punto 2 — regresión lineal con k-fold CV implementado desde numpy
- [x] Punto 3 — regresión polinómica (grados 1–4) y regularización L1
- [x] Puntos 4 y 5 — evaluación, test final y las tres respuestas
- [x] **Presentación** — `informe/presentacion.pdf` (20 slides) + guion hablado con reloj
      en `informe/guion.md`. El enunciado exige mandarla junto con el código **24 h antes**
      de la defensa del **02/09/2026**.
- [x] **Evaluación final de test** — RMSE **4288.52** (lineal grado 1, 11 features).
      `resultados-test.tex` ya tiene los números y el informe y la presentación los toman
      solos al compilar.
- [x] **Análisis exploratorio sobre train** — histogramas de las 7 variables, outliers,
      correlaciones y las tres poblaciones de `charges`, todo medido sobre las 1070 filas de
      entrenamiento: el EDA decide qué features entran, así que no puede mirar el test.
- [x] **Tres features derivadas** — `fumador_obeso` (el escalón en bmi = 30),
      `edad_al_cuadrado` (la curvatura del costo con la edad) y `bmi_si_fuma` (la pendiente
      distinta entre fumadores). Evidencia en `resultados/evidencia_features.csv`: la primera
      baja el RMSE de validación de 6094 a 4455; las otras dos, de 4455 a 4380, y su efecto es
      aditivo entre sí.
- [x] **Diagnóstico de residuos** (`src/diagnostico_residuos.py`) — el 41.9 % del error
      cuadrático de train se concentra en 28 no fumadores (2.6 % de las filas) con `charges`
      > 25 000, subestimados en 17 378 dólares en promedio, y que ninguna combinación de las
      columnas del dataset distingue del resto de los no fumadores (la única diferencia es la
      edad). Es error **irreducible con este dataset**: falta la variable que explica el gasto
      y no está en el CSV.
- [x] **Sensibilidad al número de folds** — el barrido controlado (figura 6) está en
      `resultados/sensibilidad_k.csv`: el nivel del error casi no depende de k (4428.7 a k=5,
      4405.9 en LOO — 0.51 % de diferencia), pero el error estándar sí, y pega su pico en k=10
      (293.9, 1.79 veces el de la selección con k=5).
