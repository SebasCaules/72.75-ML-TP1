# Entrega — TP1: Regresión e introducción a la evaluación de modelos

**Grupo 7** — Andrés Cortese (64612) · Sebastián Caules (64331)
**72.75 Aprendizaje Automático (Machine Learning) — ITBA — 2026 Q2**

Defensa: **02/09/2026, 17:35, Aula 701F** (2ª fecha, presencial)
Envío: hasta el **01/09/2026** (24 h antes, según el enunciado)

Dataset elegido: **Insurance Charges** (punto 0.2 del enunciado).
Resultado: **RMSE de 4288,52 dólares en test** (R² = 0,871) con un modelo lineal de grado 1
y once características, sin regularizar.

---

## Qué se manda

El enunciado pide dos cosas, y son las dos que están acá: *"mandar la presentación y el código
24 horas antes de la clase en la que se presentará el TP"* (TP1, p. 1).

| Archivo | Qué es |
|---|---|
| `TP1-grupo7-presentacion.pdf` | Las diapositivas de la defensa. 20 slides de contenido + portada y cierre (54 páginas de PDF porque Beamer expande las animaciones en páginas sucesivas). Duración prevista: 10 min. |
| `TP1-grupo7-codigo.zip` | El código Python, los tests y el dataset. Se descomprime en una única carpeta `TP1-grupo7-codigo/`. |

---

## Qué hay en el zip

Sólo lo necesario para correr el pipeline y **recrear los resultados desde cero**, más los
tests:

```
src/            12 módulos: limpieza, validación cruzada, preproceso, modelos, experimentos,
                evaluación de test, figuras y tablas
tests/          4 suites con casos de respuesta conocida, sin pytest
data/raw/       insurance.csv, copiado tal cual se descargó (1338 filas × 7 columnas)
README.md       cómo correrlo y qué hace cada archivo
```

No van al zip ni las figuras ya generadas, ni los CSV/JSON de resultados, ni las fuentes
LaTeX, ni el informe: todo eso lo **produce el código**. `python -m src.experimentos` crea
`resultados/` y lo llena; `python -m src.graficos` crea `figuras/`.

### Cómo verificarlo

```bash
unzip TP1-grupo7-codigo.zip && cd TP1-grupo7-codigo
pip install numpy pandas matplotlib

python -m tests.test_validacion    # las cuatro suites corren en segundos
python -m tests.test_preproceso
python -m tests.test_modelos
python -m tests.test_paleta

python -m src.datos                # punto 1: EDA, sólo sobre train
python -m src.experimentos         # puntos 2 a 5 — recrea resultados/, tarda ~27 min
```

Sólo requiere **numpy, pandas y matplotlib** (Python 3.10+). **No usa scikit-learn**: el punto
2.2 pide *implementar* el esquema de validación cruzada, así que el split, el k-fold, la
codificación one-hot, el escalado, los mínimos cuadrados, la expansión polinómica y el Lasso
por descenso por coordenadas están escritos desde cero sobre numpy. La semilla está fijada en
42 en todo el pipeline, así que los números se reproducen exactos.

---

## Cobertura del enunciado

Todos los puntos del enunciado se responden en la presentación.

| Punto del enunciado | Dónde está en la presentación |
|---|---|
| 1.1 Introducción teórica — separación train/validación/test | *Los tres conjuntos, y por qué hacen falta los tres* |
| 0. Elección del dataset | *Qué hay que predecir* |
| 1.1 Variables categóricas (one-hot) | *Dentro de cada fold, en este orden* |
| 1.2 Valores faltantes | *Dentro de cada fold…* (el dataset no tiene faltantes; queda documentado) |
| 1.3 Outliers | *Los outliers: ¿error de carga o subpoblación real?* + *El criterio importa: IQR contra z-score* |
| 1.4 Características y escalado | *Dentro de cada fold, en este orden* + *Una columna nueva vale más que todo el polinomio* |
| 2.1 Separación train/test | *Cómo se parten los datos, y por qué así* |
| 2.2 k-fold cross-validation (implementado a mano) | *Dentro de cada fold, en este orden* |
| 2.3 Entrenamiento lineal + RMSE train/validación | *Sin el grado 4: acá se elige* |
| 3.1 / 3.2 Transformación polinómica y entrenamiento | *Qué pasa al subir el grado del polinomio* |
| 3.3 Regularización L1 (opcional) | *Lasso: regularizar recupera la estabilidad* |
| 4. RMSE por grado y por λ | *Sin el grado 4: acá se elige* + *Los dos indicadores de sobreajuste* |
| 5.1 ¿Qué modelo obtuvo menor error? | *1. ¿Qué modelo obtuvo menor error?* |
| 5.2 ¿Cuál implementaría en producción? | *2. ¿Cuál implementaría en una aplicación real?* |
| 5.3 ¿Qué RMSE esperar en datos nuevos? | *3. ¿Qué RMSE esperar en datos nuevos?* + *El resultado sobre test* |

---

## El número que se defiende

| | features | RMSE test | R² test |
|---|---:|---:|---:|
| **Modelo de producción** (lineal, grado 1, sin regularizar) | **11** | **4288,52** | **0,871** |
| Baseline trivial (predecir siempre la media) | — | 11963,43 | 0,000 |

El test son 267 filas reservadas desde el principio, que no participaron de ninguna decisión
—ni siquiera del análisis exploratorio— y se evaluaron **una sola vez**. El RMSE de test queda
124,93 dólares *por debajo* del RMSE de validación cruzada (4413,45): nada indica que el modelo
generalice peor de lo que promete la validación.

El hallazgo que más mueve el error no es un modelo más complejo sino una columna nueva:
`fumador_obeso = (smoker = yes) ∧ (bmi > 30)`, que captura un salto de casi quince mil dólares
en una unidad de bmi entre los fumadores. El grado 4 sin regularizar, en cambio, tiene el mejor
error de train de toda la tabla (3379) y un error de validación de 87 917 — veinte veces el del
grado 1. Es el retrato del sobreajuste.

---

*Esta carpeta se genera a partir de la raíz de este repositorio; no se edita a mano.
La presentación es copia de `informe/presentacion.pdf`, y el zip se arma con `src/`,
`tests/`, `data/` y un README propio.*
