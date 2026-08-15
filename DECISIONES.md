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
| **D-22** | **La elección de $k=5$ se sostiene con evidencia, no con la cita a Hastie.** Se repite la selección completa del punto 5 con $k=5$, $10$ y $20$, y un barrido controlado hasta LOO. Ver §4.ter | $k=10$ acá cuesta unos 15 minutos, no las ~13 horas de LOO: el argumento de costo no alcanza para descartarlo, así que había alternativa razonable y por el criterio de entrada de este documento hay que justificarla. El barrido muestra que el modelo elegido **no cambia** con $k$, y de paso cuantifica lo que §4.bis había dejado abierto |

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
# src/sensibilidad_k.py            # D-22; ~45 min, no toca test. Escribe resultados/sensibilidad_k.{json,csv}
preparar_train() -> (X_train, y_train)
grilla_completa(X_train, y_train, k, lam_max) -> [dict]      # las 19 configuraciones con k folds
seleccionar(candidatos, k) -> dict                           # ganador + regla de 1 ES + parsimonia
barrido_controlado(X_train, y_train) -> [dict]               # configuracion fija, k variable
rmse_agrupado(X_train, y_train, k) -> float                  # una raiz sobre los residuos out-of-fold

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

## 4.bis Una objeción honesta a la regla de 1 error estándar

El error estándar se calcula como $\sigma/\sqrt{k}$, con $\sigma$ el desvío del RMSE **entre
folds**. Es la práctica estándar (Hastie, Tibshirani & Friedman; es lo que hace `glmnet`), pero
tiene un supuesto que conviene declarar antes de que lo pregunten: **$\sqrt{k}$ trata a los $k$
folds como muestras independientes, y no lo son**. Los conjuntos de entrenamiento de dos folds
cualesquiera comparten $3/4$ de sus filas, así que sus errores están correlacionados
positivamente y el desvío entre folds **subestima** la variabilidad real.

Consecuencia práctica: el ES verdadero es **mayor** que los 100,3 que reportamos. Eso no
debilita la conclusión, la refuerza — si el umbral real es más ancho, las configuraciones
estadísticamente indistinguibles del mejor son **más** de 8, no menos, y el argumento para
elegir el modelo simple queda más firme.

Cuantificar ese sesgo con precisión requeriría validación cruzada repetida o un estimador
corregido. **§4.ter lo cuantifica por otra vía** —barriendo $k$— y confirma la dirección: el ES
en el rango donde significa algo ($k=10$–$50$) es ≈ 220, más del doble de los 100,3 que
reporta el informe.

---

## 4.ter Sensibilidad al número de folds (D-22)

`src/sensibilidad_k.py` corre dos experimentos que conviene no mezclar. Ninguno toca test: el
split se hace con la misma semilla y `idx_test` se descarta sin usarlo, así que todo pasa dentro
de las 1070 filas de train. Por eso es lícito correrlo **después** de la evaluación de test sin
violar D-09 — no re-evalúa test, mide cuán estable es un procedimiento de selección ya ejecutado.

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
no depende de $k$ una vez que se corrige la forma de promediar, y el test ya se evaluó una única
vez (D-09): migrar a $k=10$ no compraría nada y sí rompería esa garantía.

Lo que sí cambia es lo que el informe **declara**: el ES verdadero es al menos ≈ 220, no 100,3, y
por lo tanto la banda de 1 error estándar es al menos el doble de ancha que la reportada. Como
argumentaba §4.bis, eso **refuerza** la elección del modelo simple —hay más configuraciones
estadísticamente indistinguibles del mejor, no menos—.

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
| Selección del punto 5 recalculada con $k=5$, 10 y 20 (D-22, §4.ter) | Mismo modelo de producción en los tres |
| `src/sensibilidad_k.py` reproduce el $k=5$ ya comiteado | Diferencia **exactamente 0** contra `modelo_elegido.json` en RMSE, σ, ES, umbral y λ |

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
| **RMSE test** | pendiente | **pendiente** |

Referencias sobre test (lineal grado 1 y baseline de la media): pendientes, salen de la
misma corrida.

Costo de la simplicidad: un espacio de features 11× menor, a cambio de unos pocos dólares
de RMSE. El número exacto lo produce la evaluación de test.

El desarrollo completo, con las tablas y las figuras, está en
[`informe/informe.pdf`](informe/informe.pdf).
