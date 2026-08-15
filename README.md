# TP1 — Regresión e Introducción a la evaluación de modelos

**72.75 Aprendizaje Automático (Machine Learning) — ITBA — 2026 Q2**
Defensa: 26/08/2026 · Enunciado: [`enunciado.pdf`](enunciado.pdf)

Predicción del costo anual de gastos médicos (`charges`) mediante regresión lineal y
polinómica, evaluadas con *k-fold cross-validation*.

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
| ¿El polinomio mejora al lineal? (punto 5) | **Sí**, y después empeora — la curva en U completa | No en `day.csv` (465 features / 731 filas) | No: el error apenas se mueve |
| Riesgo metodológico | Ninguno | **Es serie temporal**: el split i.i.d. queda expuesto | Target ordinal 3–8 disfrazado de continuo |

Insurance es el único donde los cuatro subpuntos de limpieza tienen contenido real **y** la
comparación de modelos produce la curva en U del punto 5 con una explicación causal detrás
(ver *Hallazgo principal*).

> La comparación entre los tres datasets se hizo con un análisis exploratorio previo, **fuera
> de este repo**. Los únicos números que este repo calcula y respalda son los de Insurance, que
> están en `resultados/` y en `informe/salida-seleccion.txt`.

## Instalación y uso

Requiere Python 3.10+ y sólo `numpy`, `pandas` y `matplotlib`. **No usa scikit-learn**: el
enunciado pide *implementar* el esquema de validación cruzada (punto 2.2), así que el split,
el k-fold, la codificación, el escalado, la ecuación normal, la expansión polinómica y el
Lasso están escritos desde cero sobre numpy. Tampoco usa pytest.

```bash
pip install numpy pandas matplotlib

python -m src.datos          # punto 1: análisis exploratorio
python -m src.experimentos   # puntos 2 a 5  (~6.5 min)
python -m src.graficos       # figuras de la presentación

python -m tests.test_validacion    # los tres módulos base tienen su suite
python -m tests.test_preproceso
python -m tests.test_modelos
```

`src.experimentos` tarda unos 6.5 minutos: el Lasso de grado 4 son 494 features casi
colineales y el descenso por coordenadas necesita decenas de miles de barridas. **No hace
falta correrlo** para ver los resultados: están versionados en `resultados/` y la salida
completa en `informe/salida-seleccion.txt`.

## Estructura

```
src/datos.py         punto 1 — la evidencia que justifica cada decisión de limpieza
src/validacion.py    rmse, split train/test, k-fold, resumen entre folds
src/preproceso.py    codificación one-hot, estandarizado, expansión polinómica
src/modelos.py       OLS (ecuación normal vía SVD), ridge, Lasso por coordenadas
src/experimentos.py  puntos 2 a 5 — CV, grados 1-4, barrido de lambda, test final
src/graficos.py      las cinco figuras de la presentación
tests/               suites con casos de respuesta conocida, sin pytest
data/raw/            insurance.csv, copiado tal cual se descargó
resultados/          cv_lineal.csv, cv_lasso.csv, final.json
figuras/             PNG para la presentación
informe/informe.tex  informe completo en LaTeX -> informe.pdf (12 páginas)
informe/             conclusiones.md (guion de la defensa) + salida-seleccion.txt
DECISIONES.md        las 20 decisiones metodológicas numeradas, con su justificación
```

---

## Hallazgo principal

`bmi` correlaciona apenas **0.198** con `charges`, y sin embargo es determinante. La razón es
que su efecto **depende de si la persona fuma**:

| | bmi ≤ 30 | bmi > 30 | efecto de la obesidad |
|---|---:|---:|---|
| **No fumador** | 7 967 | 8 853 | +11 % |
| **Fumador** | 21 369 | 41 693 | **+95 %** |

La obesidad casi no cuesta si no fumás, y **duplica el costo si fumás**. Eso es un término de
interacción `smoker × bmi`, que un modelo puramente aditivo no puede representar — y es
exactamente lo que la expansión polinómica de grado 2 recupera de forma automática. Este es el
mecanismo concreto detrás de la mejora del punto 5, no un resultado numérico sin explicación.

## Resultados

Validación cruzada de 5 folds sobre las 1070 filas de entrenamiento (el test, 267 filas, se
evalúa una sola vez al final):

| Modelo | Grado | λ | RMSE train | RMSE validación | Features vivas |
|---|---:|---:|---:|---:|---:|
| lineal | 1 | — | 6034.7 ± 89.1 | 6122.8 ± 355.3 | 8 |
| lineal | 2 | — | 4741.7 ± 64.8 | 4986.2 ± 244.0 | 44 |
| lineal | 3 | — | 4501.2 ± 71.6 | 5228.9 ± 256.5 | 164 |
| lineal | 4 | — | 4058.3 ± 91.6 | **6566.4 ± 1031.5** | 494 |
| **lasso** | **2** | **286.4** | 4879.8 ± 61.2 | 4955.3 ± 226.2 | **10** |
| lasso | 4 | 286.4 | 4761.8 ± 65.1 | **4920.0 ± 224.3** | 26 |

El grado 4 sin regularizar es el retrato del sobreajuste: **el mejor error de train de toda la
tabla** (4058.3) y el peor de validación (6566.4), con un desvío entre folds de ±1031 contra
±244 del grado 2. No es sólo peor: es inestable.

**Modelo elegido para producción: Lasso de grado 2 con λ=286.4** — no el de menor error. Ocho
de las dieciocho configuraciones caen dentro de un error estándar (100.3) del mejor, o sea que
son estadísticamente indistinguibles y cuál gana lo decide el azar de la partición. Entre
indistinguibles, el más simple (regla de 1 ES).

| | RMSE test |
|---|---:|
| **Modelo de producción** (Lasso grado 2, 10 features de 44) | **pendiente** |
| Ganador de la CV (Lasso grado 4, 20 features de 494) | pendiente |
| Lineal simple (grado 1) | pendiente |
| Baseline trivial (predecir siempre la media) | pendiente |

La simplicidad cuesta unos pocos dólares de RMSE y compra un espacio de features 11 veces
más chico. **El número exacto sale de tu corrida de `evaluar_test.py`**: este repo no lo
afirma antes de que el test se haya evaluado.

Las diez features que sobreviven a la penalización L1, ordenadas por magnitud:

```
smoker=yes        9301.54      children             519.39
age               3463.37      age^2                186.36
bmi*smoker=yes    3317.35      region=se*region=sw  124.98
bmi               1595.81      bmi*region=se        -83.98
                               smoker=yes^2          20.05
                               bmi^2                 -6.88
```

**El Lasso encontró la interacción solo.** `bmi*smoker=yes` queda tercera en magnitud, por
encima de `bmi` sola — el mismo hallazgo que el análisis exploratorio había detectado a mano,
recuperado esta vez sin que nadie se lo indicara.

## Estado

- [x] Punto 0 — elección del dataset
- [x] Punto 1 — limpieza de datos (categóricas, faltantes, outliers, features y escalado)
- [x] Punto 2 — regresión lineal con k-fold CV implementado desde numpy
- [x] Punto 3 — regresión polinómica (grados 1–4) y regularización L1
- [x] Puntos 4 y 5 — evaluación, test final y las tres respuestas
- [ ] **Presentación** — el guion está en `informe/informe.pdf` §7; falta armar las
      diapositivas. El enunciado exige mandarlas junto con el código **24 h antes** de la
      defensa del **26/08/2026**.
