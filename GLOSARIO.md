# Glosario y decisiones

Qué significa cada término que aparece en el repo, y por qué se decidió cada cosa.
Sin relleno: una línea por ítem.

---

## 1. Términos

### Evaluación

| Término | Qué es | Por qué aparece acá |
|---|---|---|
| **RMSE** | Raíz del error cuadrático medio: $\sqrt{\frac{1}{n}\sum(y-\hat y)^2}$ | Está en dólares, igual que el target. "me equivoco en unos 5000 dólares" se entiende; el MSE está en dólares² y no significa nada |
| **MSE** | Lo mismo sin la raíz | No se reporta: unidades ininterpretables |
| **Baseline** | Predecir siempre la media de train | El piso. Si un modelo no lo supera, no sirve. Su RMSE sale de la evaluación de test |
| **train** | Las 1070 filas con las que se ajustan los coeficientes | — |
| **validación** | Las filas que en cada fold quedan afuera del ajuste | Sirven para **elegir** entre modelos, no para ajustar |
| **test** | Las 267 filas reservadas | Estiman el error en datos nuevos. Se tocan **una vez** |
| **k-fold cross-validation** | Partir train en $k$ bloques y repetir ajustar/validar $k$ veces, dejando uno afuera cada vez | Cada dato valida 1 vez y entrena $k-1$. El promedio de los $k$ errores tiene mucha menos varianza que un solo split |
| **fold** | Cada uno de esos $k$ bloques | Acá $k=5$, ~214 filas cada uno |
| **error estándar (ES)** | Desvío de una media: $\sigma/\sqrt{k}$ | Con $k=5$: $366{,}97/\sqrt5 = 164{,}11$. Es la vara para decidir si dos modelos difieren de verdad |
| **regla de 1 ES** | Entre modelos dentro de 1 ES del mejor, elegir el más simple | 7 de 15 configuraciones caen adentro. Acá **no decide nada**: el más simple del espacio (lineal grado 1) es además el de menor error, así que la regla sólo confirma. **Ojo con una objeción legítima:** dividir por $\sqrt k$ supone folds independientes, y no lo son —sus conjuntos de entrenamiento se superponen en 3/4 de las filas—, así que el ES real es **mayor** que 164,1 (el barrido de $k$ lo mide: llega a 293,9 en $k=10$). Eso no invalida la conclusión: la subestima |
| **sesgo del mínimo** | El mínimo de varias estimaciones ruidosas está sesgado hacia abajo, aunque cada una sea insesgada | Por eso el RMSE de validación del ganador no se puede prometer: se lo eligió *por ser* el mínimo |

### Datos

| Término | Qué es | Por qué aparece acá |
|---|---|---|
| **outlier** | Valor atípico según un criterio explícito | 139 en `charges` (10,4 %) |
| **IQR** | Rango intercuartílico. Atípico = fuera de $[Q_1-1{,}5\,\text{IQR},\; Q_3+1{,}5\,\text{IQR}]$ | Se apoya en cuartiles: robusto a la asimetría |
| **z-score** | $(x-\bar x)/\sigma$; atípico si $|z|>3$ | **No** se usa como criterio: los propios extremos inflan $\sigma$ y el criterio se sabotea |
| **asimetría (*skew*)** | Cuánto se estira una cola respecto de la otra | `charges` tiene 1,516 — por eso falla el z-score |
| **one-hot** | Una columna 0/1 por categoría | `region` → 4 dummies |
| **dummy variable trap** | Si están las 4 dummies, suman 1 siempre = la columna del intercepto → $X^TX$ singular | Por eso se descarta una: la **categoría de referencia** |
| **estandarizar** | $(x-\bar x)/\sigma$ por columna | Sin esto la penalización L1 castiga más a las variables de escala grande |
| **fuga de datos (*leakage*)** | Que información de validación o test entre al entrenamiento | El error más fácil de cometer sin notarlo |
| **duplicado exacto** | Fila idéntica en las 7 columnas | Hay 1 (índices 195 y 581). Si una copia cae en train y otra en test, el test deja de ser independiente |

### Modelos

| Término | Qué es | Por qué aparece acá |
|---|---|---|
| **OLS** | Mínimos cuadrados ordinarios | El modelo base del punto 2 |
| **ecuación normal** | $X^TXw = X^Ty$, de derivar el error e igualar a 0 | La solución cerrada de OLS |
| **expansión polinómica** | Agregar potencias y productos de las columnas | Grado 2 sobre 11 columnas → 77 |
| **monomio** | Cada término generado: `bmi`, `bmi^2`, `age*bmi` | — |
| **término de interacción** | Producto de dos variables distintas | `bmi*smoker` es **el hallazgo** del TP |
| **lineal en los parámetros** | El modelo sigue siendo combinación lineal de columnas | Se transforman las features, no el modelo: se resuelve igual que grado 1 |
| **Ridge (L2)** | Penaliza $\sum w_j^2$ | Encoge todos los coeficientes, no anula ninguno |
| **Lasso (L1)** | Penaliza $\sum|w_j|$ | **Anula** coeficientes en cero exacto: hace selección de variables |
| **$\lambda$** | Cuánto pesa la penalización | Más $\lambda$ = más regularización = menos features vivas |
| **$\lambda_{max}$** | El $\lambda$ más chico que anula *todos* los coeficientes: $\max_j |x_j^T(y-\bar y)|/n$ | Da una escala de referencia: la grilla es relativa a él, no números elegidos a ojo |
| **descenso por coordenadas** | Optimizar una coordenada por vez, con solución cerrada en cada una | Cómo se resuelve Lasso. No necesita tasa de aprendizaje |
| **soft-thresholding** | $\text{sign}(z)\max(|z|-\gamma,0)$ | El paso de cada coordenada. El $\max(\cdot,0)$ es lo que produce ceros exactos |
| **intercepto** | El término independiente $b$ | **No se penaliza**: se obtiene centrando $X$ e $y$ |
| **convergencia / tolerancia** | Cortar cuando el cambio máximo en $w$ baja de `tol` | Un modelo que no converge no es el modelo Lasso: es dónde quedó la optimización al cortarla |

### Álgebra numérica

| Término | Qué es | Por qué aparece acá |
|---|---|---|
| **colinealidad** | Una columna es combinación lineal de otras | En grado 4, **928 de 1364 columnas** lo son |
| **rango efectivo** | Cuántas columnas son realmente independientes | Grado 4: 436 de 1364 |
| **número de condición** | $\sigma_{max}/\sigma_{min}$. Mide cuánto amplifica el error de redondeo | Grado 4: $3{,}1\times10^{18}$, por encima de la precisión de `float64` ($\approx10^{16}$) |
| **SVD** | Descomposición en valores singulares | Cómo `lstsq` resuelve OLS sin invertir $X^TX$ |
| **solución de norma mínima** | Cuando el sistema es indeterminado, la de menor $\|w\|$ | Lo que devuelve `lstsq` en grado alto |

### Diagnóstico

| Término | Qué es | Por qué aparece acá |
|---|---|---|
| **sobreajuste** | El modelo memoriza el train y falla afuera | Grado 4: el **mejor** RMSE de train (3379) y el peor de validación (87 917) |
| **subajuste** | El modelo es demasiado simple | Grado 1: los dos errores altos y parecidos |
| **brecha train–validación** | La diferencia entre ambos | Crece de **62** (grado 1) a **84 538** (grado 4). Es la firma directa del sobreajuste |
| **desvío entre folds** | Cuánto varía el error según la partición | Pasa de ±367 a ±47 000: el grado 4 no es sólo peor, es **inestable** |

---

## 2. Decisiones

Cada una tenía alternativa razonable. Lo que no la tenía, no está.

### Datos

| # | Decisión | Por qué |
|---|---|---|
| D-01 | Eliminar el duplicado **antes** del split | Si no, una copia queda en train y otra en test: fuga |
| D-02 | **Conservar** los outliers | El 97,8 % son fumadores. Son subpoblación real, no error de carga |
| D-03 | Split 80/20, semilla 42, barajado, **sin estratificar** | No es serie temporal: i.i.d. vale. **Ojo:** la razón vieja ("el target es continuo, no hay clases") quedó refutada por el EDA — hay tres poblaciones. Se sigue sin estratificar porque estratificar por ellas **no baja** la varianza entre folds (593 contra 591). Ver D-24 |
| D-10 | Binarias → 1 columna; `region` → one-hot menos una | Dummy variable trap |
| — | Incluir **las 6 variables**, sin descartar ninguna a mano | Descartar mirando la correlación con el target es decidir *con los datos*. La selección la hace L1, dentro de cada fold |

### Evaluación

| # | Decisión | Por qué |
|---|---|---|
| D-04 | $k=5$ folds, sólo sobre train | ~214 filas por fold. El enunciado exige que la CV no toque test |
| D-05 | Estandarizador ajustado **dentro de cada fold** | Ajustarlo antes de partir filtra la media de validación al entrenamiento |
| D-06 | Orden: codificar → estandarizar → expandir → estandarizar | 1.º evita que `age³`=262 144 conviva con `children³`=125; 2.º hace comparable la penalización L1 |
| D-07 | Estandarizar **también** las dummies | Para OLS no cambia nada; para Lasso las pone en pie de igualdad |
| D-08 | Evaluar grados 1, 2, 3 y 4 | 3 y 4 están para **mostrar** el sobreajuste, no porque se esperen buenos |
| D-09 | Test una sola vez, al final | Doctrina del test set |
| **D-21** | **La evaluación de test es un paso manual y separado** | `experimentos.py` ni siquiera construye `X_test`. Garantía estructural, verificable con `grep`, no una promesa |
| **D-22** | **$k=5$ se sostiene con un barrido, no con una cita** | Con $k=5$, 10 y 20 sale **el mismo** modelo de producción. De paso mide el ES en el rango donde significa algo, $k$=10–50 (pico de 293,9, no 164,1), y muestra que promediar RMSEs de fold chicos sesga la métrica |

### Implementación

| # | Decisión | Por qué |
|---|---|---|
| D-11 | Intercepto fuera de la penalización, vía centrar | Penalizarlo encogería la predicción media hacia cero sin razón |
| D-12 | OLS con `lstsq` (SVD), no `inv()` | En grado ≥2 la matriz es numéricamente singular: invertir da basura |
| D-13 | La expansión **no** agrega columna de unos | El intercepto lo maneja el modelo, y una columna constante rompe el estandarizador (desvío 0) |
| D-14 | Objetivo Lasso con factor $1/(2n)$ | Convención estándar: hace que $\lambda$ no dependa del tamaño de muestra |
| D-15 | Grilla de $\lambda$ relativa a $\lambda_{max}$ | Interpretable y comparable entre grados |
| D-16 | Tests con asserts planos, sin `pytest` | El repo corre con numpy, pandas y matplotlib, nada más |
| D-17 | Todo en español | Se defiende oralmente en español |
| D-18 | Lasso con residuo **incremental** | La forma directa es $O(np^2)$ por barrida: con $p=1364$ no converge nunca. Incremental es $O(np)$ — la misma cuenta reordenada |
| D-19 | `tol = 1e-4`, no `1e-7` | El criterio es absoluto y los coeficientes están en dólares. Pedir $10^{-7}$ es converger a una centésima de centavo |
| D-20 | Una configuración que no converge **no puede ser elegida** | Su RMSE no es el del Lasso: es dónde se cortó la optimización |

### Presentación de resultados

| Decisión | Por qué |
|---|---|
| Paleta azul/naranja validada, no elegida a ojo | La anterior daba ΔE 7,3 bajo deuteranopía: las curvas de grado 2 y 4 eran indistinguibles para un daltónico |
| Los grados van con **rampa de un tono**, claro→oscuro | El grado es una escala **ordenada**, no categorías nominales. Se lee sin leyenda |
| Histogramas **apilados**, no superpuestos | Superponer con transparencia crea un tercer color que no está en la leyenda |
| Gráfico de 1-ES con **puntos, no barras** | Los valores van de 4900 a 5400: una barra que no arranca en cero exageraría justo lo que el gráfico quiere mostrar como despreciable |
| Los números de test entran por `\input` desde `resultados-test.tex` | Ningún valor de test se transcribe a mano: o está el que produjo la evaluación, o dice `[PENDIENTE]` |

---

## 3. El flujo, de punta a punta

```
1338 filas
  └─ quitar duplicado exacto ................................ 1337
       └─ split 80/20, semilla 42
            ├─ TRAIN 1070 ─┬─ codificar (ajustado sólo acá) → 11 columnas
            │              └─ k-fold, 5 bloques de ~214
            │                   └─ por fold: estandarizar → expandir → estandarizar → ajustar
            │                        └─ 19 configuraciones (4 lineales + 15 Lasso)
            │                             └─ 4 descartadas por no converger
            │                                  └─ 15 elegibles
            │                                       ├─ [5.1] menor RMSE val → lineal g1
            │                                       └─ [5.2] regla 1 ES → lineal g1  ← producción
            │                                            (el ganador ya era el más simple)
            └─ TEST 267 ...... intacto hasta que corras `python -m src.evaluar_test`
```

---

## 4. Las tres respuestas, en una línea cada una

1. **¿Cuál dio menor error?** El **lineal de grado 1** (11 características, sin regularizar): RMSE validación 4413,45.
2. **¿Cuál a producción?** El **mismo**, el lineal de grado 1: no hay que elegir entre error y simplicidad porque el más simple del espacio es además el de menor error. La regla de 1 ES (7 de 15 indistinguibles) lo confirma.
3. **¿Qué RMSE prometés?** El de **test**, no el de validación —que está sesgado a la baja por haber elegido el mínimo—. Es **4288,52** (`resultados/evaluacion_test.json`).
