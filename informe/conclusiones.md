# Conclusiones — TP1 Regresión

## 1. Introducción teórica: separación train / validación / test

El objetivo de un modelo de regresión no es ajustar bien los datos que ya vio, sino predecir
bien datos que **todavía no vio**. Medir el error sobre el propio conjunto de entrenamiento no
sirve para eso: el modelo ajusta sus parámetros exactamente para minimizar ese error, así que un
RMSE de train bajo puede significar dos cosas radicalmente distintas — que el modelo capturó el
patrón real, o que memorizó el ruido particular de esas 1070 filas. Sin un conjunto separado no
hay forma de distinguir una cosa de la otra.

Por eso el trabajo separa tres roles, cada uno con un dato distinto y ninguno intercambiable:

- **Entrenamiento (train):** ajusta los parámetros del modelo (los coeficientes de la regresión).
- **Validación:** no participa del ajuste; se usa para **elegir entre modelos** — qué grado
  polinómico, qué valor de $\lambda$ en Lasso. Es la vara con la que se compite entre
  configuraciones.
- **Test:** se toca **una única vez**, al final, después de que grado y $\lambda$ ya están
  fijados. Estima el error que el modelo va a tener en producción, sobre datos genuinamente
  nuevos.

La razón por la que el test tiene que quedar intacto hasta el final es sutil pero central: en el
momento en que su error se usa para decidir algo — elegir un hiperparámetro, comparar dos
arquitecturas, hasta simplemente "mirarlo y probar de nuevo" — deja de ser una medida honesta del
error de generalización y pasa a ser, de hecho, un segundo conjunto de validación. El número que
reporta ya no dice cuánto va a fallar el modelo en el mundo real, sino cuánto se sobreajustó
también al test. En este trabajo el conjunto de test (267 filas) se evaluó una sola vez, después
de fijar el modelo de producción por validación cruzada.

¿Por qué no alcanza con un único split de validación, y hace falta validación cruzada (*k-fold
cross-validation*) de $k=5$ *folds*? Con un solo split, el resultado depende de qué filas cayeron
del lado de validación por azar — con 1070 datos de train, una partición particular puede
favorecer o perjudicar a un modelo simplemente por cómo cayeron los outliers o los casos raros.
La validación cruzada evita depender de una sola tirada: se parte el train en 5 bloques de
aproximadamente 214 filas cada uno, y se repite el ciclo ajustar-validar 5 veces, dejando cada
vez un bloque distinto afuera del ajuste. Así **cada dato valida exactamente una vez y entrena en
las otras 4 vueltas**, y el promedio de los 5 errores resultantes tiene mucha menos varianza que
el error de un único split — la señal que importa (¿esta configuración generaliza mejor que
aquella?) deja de estar enmascarada por el ruido de qué le tocó a cada partición. Con un $n$
chico como este (1070 filas de train) esa reducción de varianza es decisiva: es la diferencia
entre poder distinguir dos configuraciones o no.

## 2. Punto 1 — limpieza de datos

### 1.1 Variables categóricas

El dataset tiene tres variables categóricas: `sex` (binaria), `smoker` (binaria) y `region`
(cuatro categorías: northeast, northwest, southeast, southwest). Se codificaron con **one-hot**
y no con codificación ordinal (asignar 0, 1, 2, 3 a las regiones) porque una codificación ordinal
le impone al modelo un orden y una distancia numérica que no existen en los datos: le estaría
diciendo, por ejemplo, que "southwest" está más lejos de "northeast" que de "southeast" en algún
sentido matemático, cuando la variable es puramente nominal. One-hot no asume ninguna estructura
de orden.

Para `region` se generaron 4 dummies pero se **descartó una columna** (queda como categoría de
referencia, absorbida en la ordenada al origen): es la forma estándar de evitar la
*dummy variable trap* — si las 4 dummies estuvieran todas presentes, sumarían siempre 1, lo cual
las hace linealmente dependientes de la columna constante del modelo y vuelve singular la matriz
de diseño (no se puede invertir $X^TX$ para resolver por mínimos cuadrados).

Para las binarias (`sex`, `smoker`) alcanza con **una sola columna** (`sex=male`, `smoker=yes`)
en lugar de dos: la segunda categoría es exactamente el complemento de la primera, así que
agregarla sería la misma trampa de colinealidad que con `region`, solo que con dos categorías en
vez de cuatro.

El codificador resultante produce 8 columnas: `age`, `bmi`, `children`, `sex=male`,
`smoker=yes`, `region=northwest`, `region=southeast`, `region=southwest`.

### 1.2 Datos faltantes

El dataset no tiene ningún valor nulo (0 en las 1338 filas × 7 columnas originales). Sí apareció
**una fila duplicada exacta** (los índices 195 y 581 son idénticos en las 7 columnas), que se
eliminó, quedando 1337 filas. La eliminación se hace **antes del split** train/test: un
duplicado exacto que quedara repartido entre ambos conjuntos —una copia en train y la otra en
test— filtraría información del test hacia el entrenamiento sin que el modelo tuviera que
generalizar nada; el error de test estaría inflado artificialmente a la baja para esa fila.

### 1.3 Outliers

Se comparó la detección por **rango intercuartílico (IQR)** contra **z-score** y se optó por
IQR como criterio de decisión. La tabla de discrepancia muestra por qué:

| Variable | IQR detecta | Z-score detecta |
|---|---|---|
| `charges` | 139 (10,4 %) | 7 |
| `children` | 0 | 18 |

En `charges`, z-score detecta muchísimo menos porque la propia definición de z-score depende de
la media y el desvío estándar, y `charges` tiene una asimetría de 1,516: los valores extremos —
que son justamente los que se querría detectar— arrastran la media hacia arriba y **inflan el
desvío estándar**, con lo cual "tres desvíos de la media" deja de ser un umbral exigente: los
propios outliers se disfrazan a sí mismos. IQR, al basarse en cuartiles, es robusto a esa
distorsión.

En `children` pasa lo inverso: z-score marca 18 valores como outliers y IQR ninguno, porque
`children` es un entero acotado entre 0 y 5 — "estar a tres desvíos de la media" no tiene el
mismo significado en una variable discreta de rango chico que en una continua sin cota natural.

De los 139 outliers de `charges` detectados por IQR, el **97,8 % son fumadores**, contra apenas
11,5 % de fumadores en el resto de la población. Esa desproporción es la evidencia de que **no
son errores de carga de datos, son una subpoblación real**: fumar dispara el costo médico de
forma no lineal, y esos 139 casos son la cola derecha esperable de esa subpoblación, no ruido.
Por eso se **conservan**: eliminarlos le estaría escondiendo al modelo justo el fenómeno más
importante del dataset (la interacción fumador × costo), que es además la variable más
correlacionada con `charges`.

### 1.4 Features y escalado

El enunciado pide dos decisiones justificadas: **qué características se incluyen** y **si se
escala**. Van por separado.

**Qué features se incluyen: las seis, sin descartar ninguna.** El enunciado admite
explícitamente que "pueden ser todas", y acá esa es además la respuesta correcta:

| Variable | ¿Se incluye? | Por qué |
|---|---|---|
| `age` | Sí | Segunda correlación más alta con `charges` (0.299) y relación monótona clara |
| `bmi` | Sí | Correlación baja (0.198) **pero no descartable**: su efecto depende de `smoker`, y Pearson mide relación lineal, no interacciones (ver §1.3 y el hallazgo principal) |
| `children` | Sí | Correlación 0.068, la más débil. Se conserva igual: es barata (una columna), el Lasso puede apagarla si no sirve, y de hecho **sobrevive** a la penalización con coeficiente 519.39 |
| `sex` | Sí | Binaria, una columna. El Lasso la apaga en el modelo final, que es la forma correcta de descartarla: por evidencia, no por prejuicio |
| `smoker` | Sí | La variable más informativa del dataset (correlación 0.787) |
| `region` | Sí | Tres columnas tras el one-hot. Aporta poco, pero es la única variable geográfica y su costo es mínimo |

El criterio general: **no se descarta ninguna variable a mano**. Con 1337 filas y sólo 8
columnas tras codificar, no hay problema de dimensionalidad que lo justifique, y sacar una
variable mirando su correlación con el objetivo es una decisión tomada con los datos —
exactamente el tipo de elección que después contamina la evaluación. La selección de variables
se delega a la penalización L1, que la hace **dentro de cada fold** y por lo tanto sin fuga: de
las 44 features de grado 2, el Lasso deja vivas 10.

Ninguna variable se descarta por redundancia tampoco: no hay pares con correlación alta entre
sí, y ninguna es una función de otra (el caso `casual + registered = cnt` que descalificaba a
Bike Sharing no tiene equivalente acá).

**El escalado.** Las variables numéricas (`age`, `bmi`, `children` y, después de expandir, sus
potencias y
productos) se **estandarizan** (media 0, desvío 1) antes de ajustar cualquier modelo con
regularización. Es necesario porque Lasso penaliza la magnitud de los coeficientes: si las
variables están en escalas distintas (`age` en años, `bmi` en kg/m², `children` en unidades
enteras chicas), la penalización castiga más a los coeficientes de las variables que
naturalmente necesitan valores grandes para explicar la misma cantidad de variación, sin que eso
tenga que ver con su importancia real.

El punto crítico es **cuándo** se calculan los parámetros del escalado (media y desvío):
**siempre sobre el fold de entrenamiento únicamente**, nunca incluyendo el fold de validación. Es
la misma lógica que con el test: si la media y el desvío se calcularan sobre todos los datos
(train + validación juntos), la validación dejaría de ser una medición honesta, porque el
escalado ya habría "visto" esos datos antes de que el modelo intentara predecirlos. El pipeline
implementado respeta esto en dos pasos dentro de cada fold: estandarizar con los parámetros del
fold de entrenamiento → expandir la base polinómica → estandarizar otra vez (las potencias y
productos tienen escalas propias) con parámetros, de nuevo, ajustados solo con ese fold de
entrenamiento.

## 3. Puntos 2 y 3 — qué se implementó

El pipeline completo, de punta a punta:

1. `quitar_duplicados`: 1338 → 1337 filas.
2. `separar_train_test` con semilla 42: 1070 filas de train, 267 de test (80/20).
3. `CodificadorCategoricas`, ajustado **solo con train**: produce las 8 columnas descriptas en
   1.1.
4. `k_fold`: implementación propia de la partición en 5 folds sobre el train (no existe
   `sklearn`, así que la partición, el ciclo de ajuste-validación y el promedio de métricas están
   escritos a mano sobre `numpy`).
5. Dentro de cada fold: `Estandarizador` (ajustado solo con el fold de entrenamiento) →
   `expandir_polinomica` (genera potencias y productos cruzados hasta el grado elegido) →
   `Estandarizador` otra vez → ajuste del modelo.
6. Modelos: `RegresionLineal` (mínimos cuadrados ordinarios vía `lstsq`, o ridge vía resolución
   de sistema lineal cuando corresponde) y `Lasso` (descenso por coordenadas con actualización
   incremental del residuo).

### 2.1 Cómo se separaron train y test, y por qué así

El enunciado pide explicar **cómo** además de por qué. El procedimiento concreto:

1. Se genera una **permutación aleatoria** de los índices `0..1336` con
   `np.random.default_rng(42).permutation(n)`.
2. Se corta: las **últimas** `round(1337 × 0,2) = 267` posiciones van a test, las **1070**
   restantes a train.

Tres decisiones dentro de eso, y cada una tiene su motivo:

- **Se baraja antes de cortar.** Tomar las últimas 267 filas del archivo tal como viene sería
  válido sólo si el orden de las filas fuera aleatorio, y eso no se puede dar por sentado: los
  CSV suelen venir ordenados por algún criterio (fecha de carga, región, un `id` correlacionado
  con algo). Barajar rompe cualquier estructura latente en el orden.

- **La semilla es fija (42).** Sin semilla fija, cada corrida daría una partición distinta y
  ningún número del informe sería reproducible — quien corrige no podría verificar nada. Fijarla
  también evita una trampa sutil: si la semilla variara, sería posible correr varias veces y
  quedarse con la partición que da los mejores números, que es una forma encubierta de ajustar
  contra el test.

- **No se estratifica.** La estratificación mantiene la proporción de *clases* entre train y
  test, y acá el objetivo (`charges`) es **continuo**: no hay clases que preservar. Tampoco hay
  estructura temporal ni de grupos (cada fila es una persona distinta, no hay mediciones
  repetidas del mismo individuo), así que el supuesto de independencia que justifica el
  barajado aleatorio se cumple. Éste es precisamente el punto donde Bike Sharing habría
  fallado: al ser una serie temporal, filas contiguas están correlacionadas y un split
  aleatorio deja vecinos de cada fila de validación dentro de train, lo que hace que el error
  de validación salga optimista.

Un punto conceptual importante: la **regresión polinómica sigue siendo lineal en los
parámetros**. Elevar `age` al cuadrado o multiplicar `bmi` por `smoker=yes` no cambia la
naturaleza del modelo — sigue siendo una combinación lineal de columnas de la matriz de diseño,
solo que esas columnas ahora son transformaciones no lineales de las variables originales. Lo que
se ajusta con mínimos cuadrados (o con Lasso) es exactamente el mismo tipo de problema que con
grado 1; lo único que cambia es qué features se le presentan al modelo.

## 4. Punto 4 — resultados

### Validación cruzada, regresión lineal

| Grado | RMSE train | RMSE val | Features |
|---|---|---|---|
| 1 | 6034,7 ± 89,1 | 6122,8 ± 355,3 | 8 |
| 2 | 4741,7 ± 64,8 | 4986,2 ± 244,0 | 44 |
| 3 | 4501,2 ± 71,6 | 5228,9 ± 256,5 | 164 |
| 4 | 4058,3 ± 91,6 | 6566,4 ± 1031,5 | 494 |

### Lasso ($\lambda_{max}$ = 9545,67 en los tres grados evaluados)

| Configuración | $\lambda$ | RMSE train | RMSE val | Coefs vivos (prom.) |
|---|---|---|---|---|
| grado 2 | 2864 | 7279,8 | 7337,4 | 4,0 |
| grado 2 | 954,6 | 5272,8 | 5315,0 | 4,8 |
| grado 2 | 286,4 | 4879,8 | 4955,3 | 10,0 |
| grado 2 | 95,46 | 4773,2 | 4928,5 | 23,6 |
| grado 2 | 28,64 | 4746,2 | 4959,5 | 33,0 |
| grado 3 | 286,4 | 4822,3 | 4943,5 | 18,4 |
| grado 4 | 286,4 | 4761,8 | **4920,0** | 26,4 |
| grado 4 | 28,64 | — | — | descartado: 1 de 5 folds no convergió |

### Análisis

En regresión lineal sin regularizar el sobreajuste es evidente a partir de grado 3: el error de
**train** sigue bajando monótonamente con el grado (6034,7 → 4741,7 → 4501,2 → 4058,3 dólares),
porque más features siempre le dan al modelo más grados de libertad para ajustar los datos que
ya vio. Pero el error de **validación** deja de acompañarlo: mejora de grado 1 a grado 2 (6122,8
→ 4986,2), pero después empeora en grado 3 (5228,9) y empeora fuerte en grado 4 (6566,4) — el
modelo de grado 4 termina siendo, en datos nuevos, peor que el de grado 2 e incluso comparable al
de grado 1.

El indicador más claro de sobreajuste no es solo que el RMSE de validación suba: es que la
**brecha** entre train y validación crece sistemáticamente con el grado — de 88 dólares en grado
1 (6122,8 − 6034,7) a 2508 dólares en grado 4 (6566,4 − 4058,3). Un modelo que generaliza bien
tiene un error de validación parecido al de train; una brecha que se ensancha es la firma directa
de que el modelo está aprendiendo particularidades del fold de entrenamiento que no se sostienen
en datos que no vio.

El segundo indicador, más sutil pero igual de contundente, es la **estabilidad entre folds**: el
desvío estándar del RMSE de validación pasa de ±89,1 en grado 1 a ±1031,5 en grado 4 — más de
diez veces mayor. Esto dice algo distinto de "el modelo es peor en promedio": dice que el modelo
de grado 4 es **inestable**, que su desempeño cambia drásticamente según qué filas caen en cada
partición. Un modelo así no solo predice peor, predice de forma poco confiable — dos corridas con
particiones distintas podrían darle resultados marcadamente diferentes.

Lasso corrige buena parte de este problema regularizando: con regularización fuerte (grado 4,
$\lambda$ = 286,4) se llega al menor RMSE de validación de todas las configuraciones evaluadas
(4920,0), muy por debajo de los 6566,4 de la regresión lineal sin regularizar del mismo grado —
la penalización $L_1$ apaga la mayoría de los 494 coeficientes (deja en promedio 26,4 vivos) y
con eso recupera la estabilidad que el modelo sin regularizar había perdido.

## 5. Punto 5 — las tres respuestas

### 5.1 ¿Cuál modelo ganó?

El de menor RMSE de validación cruzada es **Lasso de grado 4, $\lambda$ = 286,37**, con
RMSE val = 4920,0 ± 224,3.

### 5.2 ¿Cuál modelo iría a producción?

No el ganador de 5.1. El argumento central es que la diferencia entre las mejores
configuraciones no es estadísticamente significativa: aplicando la **regla de 1 error estándar**
(ES = 224,3 / $\sqrt{5}$ = 100,3, umbral = 4920,0 + 100,3 = 5020,3), **8 de las 18
configuraciones evaluadas caen dentro de ese umbral**. Eso significa que esas 8 configuraciones
son, para los datos que se tienen, **indistinguibles entre sí**: cuál termina apareciendo primera
en la tabla depende del azar de cómo cayó la partición en 5-fold, no de una diferencia real de
capacidad predictiva.

Frente a un empate estadístico, la elección razonable es la configuración **más simple** del
grupo — menos features, menos coeficientes vivos, menos riesgo de que esa aparente ventaja sea
artefacto del ruido de validación. Esa configuración es **Lasso de grado 2, $\lambda$ = 286,37**
(RMSE val = 4955,3).

El costo de esa simplicidad, medido en el conjunto de test: pasar de 494 a 44 features (de las
que Lasso deja vivas 20 contra 10 respectivamente) cuesta **+92,0 dólares de RMSE** (4739,33
frente a 4647,34). Es un costo pequeño frente a la ganancia en robustez e interpretabilidad de un
modelo con una décima parte de los parámetros.

### 5.3 ¿Qué RMSE se promete en producción?

**4739,33 dólares** — el RMSE de **test** del modelo de producción (Lasso grado 2,
$\lambda$ = 286,37), **no** su RMSE de validación (4955,3).

Es importante entender por qué el número de validación está sesgado a la baja y no se puede usar
como promesa: esa configuración se eligió, entre otras cosas, por tener uno de los RMSE de
validación más bajos entre 19 estimaciones distintas evaluadas (los 4 grados de regresión lineal
más las 15 combinaciones de grado × $\lambda$ de Lasso, descontada la que no convergió). Cada una
de esas 19 estimaciones tiene ruido de muestreo — es un promedio sobre solo 5 folds. Cuando se
elige el **mínimo** de un conjunto de estimaciones ruidosas, ese mínimo queda sesgado hacia
abajo aunque cada estimación individual sea insesgada: es más probable que el "ganador" lo sea en
parte por haber tenido una racha favorable de ruido, no solo por ser mejor de verdad. El test, al
no haber participado en ninguna decisión, no tiene ese sesgo.

Dos advertencias honestas sobre este número:

- El conjunto de test tiene **267 filas**: el 4739,33 es en sí mismo una estimación, no un valor
  exacto, y tiene su propia incertidumbre de muestreo que este trabajo no cuantificó con un
  intervalo de confianza.
- El RMSE promedia sobre una población heterogénea. Dado que la variable más correlacionada con
  `charges` es `smoker` (0,787) y que la interacción fumador × BMI dispara el costo hasta un 95 %
  por encima del caso no fumador con el mismo BMI, el error real **no se reparte parejo**: se
  concentra en los fumadores, que son justamente los casos más caros y los que más le importa
  acertar a una aseguradora.

## 6. Limitaciones

- **Colinealidad en grados altos:** en grado 4, el 56,3 % de las 494 columnas son redundantes
  (216 de rango efectivo sobre 494 generadas). Hay dos causas identificadas: (1) una dummy
  binaria elevada a una potencia sigue teniendo solo dos valores posibles, así que es una función
  afín exacta de la dummy original —por ejemplo `smoker=yes`² = 1,456866 · `smoker=yes` + 1,0,
  con un residuo de 1,5×10⁻¹⁴, es decir, matemáticamente exacto salvo error de punto flotante—; y
  (2) el producto de dos dummies del mismo one-hot (p. ej. dos regiones) cae dentro del espacio
  generado por las dummies individuales más la constante, porque 3 dummies más una constante ya
  generan todas las funciones posibles sobre las 4 categorías de región. Importa aclarar que ese
  producto **no es idénticamente cero**: lo sería en la codificación cruda 0/1, pero como el
  pipeline estandariza antes de expandir la base polinómica, cada dummy toma tres valores
  distintos y el producto también, aunque siga siendo linealmente redundante. Dentro de un grupo
  colineal como este, el reparto de coeficientes entre las columnas correlacionadas es
  arbitrario — no se pueden interpretar de a uno cuando el grado es alto.
- **Una sola partición train/test, con una sola semilla (42):** el RMSE de test reportado
  (4739,33) tiene varianza asociada a qué filas específicas cayeron del lado de test, que este
  trabajo no midió — repetir el split con otra semilla daría un número distinto, presumiblemente
  cercano pero no idéntico.
- **El dataset es simulado**, no el registro de una aseguradora real: proviene de estadísticas
  demográficas (el origen es el dataset del libro de Lantz), así que las relaciones que el modelo
  aprendió reflejan cómo se construyó la simulación, no necesariamente la dinámica de costos
  médicos reales de una población.
- **El RMSE penaliza el error al cuadrado**, con lo cual la métrica está dominada por los casos
  caros (fumadores con BMI alto): un modelo puede tener un RMSE relativamente bueno y aun así
  errar de forma proporcionalmente grande en los casos baratos, porque esos errores pesan poco en
  términos absolutos frente a los errores en la cola cara de la distribución.

## 7. Guion de la presentación (10 minutos)

| Minuto | Qué se dice | Figura |
|---|---|---|
| 0–1 | Presentación del dataset (insurance.csv, 1338 filas, 7 columnas) y de la pregunta que responde el TP: predecir `charges` y elegir un modelo defendible, no solo el de menor error. | — |
| 1–3 | Introducción teórica: por qué train no alcanza, qué rol cumple cada conjunto, por qué el test se toca una sola vez, y qué gana la validación cruzada de 5 folds frente a un único split (1070 de train, folds de ~214). Es la base conceptual de todo lo que sigue. | — |
| 3–4 | Punto 1: por qué se conservan los 139 outliers de `charges` (97,8 % fumadores) en vez de eliminarlos, y por qué IQR y no z-score. | 04-outliers-charges.png |
| 4–5 | Puntos 2 y 3: el pipeline (codificar solo con train, estandarizar dentro de cada fold, expandir polinómica) y que la regresión polinómica sigue siendo lineal en los parámetros. | — |
| 5–7 | Punto 4: las curvas de train vs. validación por grado, dónde aparece el sobreajuste (la brecha que crece de 88 a 2508 dólares, el desvío entre folds que se dispara de ±89 a ±1031) y cómo Lasso lo corrige con el camino de regularización. | 01-curvas-train-val.png, 02-camino-lasso.png |
| 7–9 | Punto 5, el núcleo de la defensa: el ganador de CV (grado 4), por qué 8 de 18 configuraciones son indistinguibles bajo la regla de 1 error estándar, la elección del modelo de producción (grado 2, +92 dólares de costo por la simplicidad) y el RMSE de test que se promete (4739,33), con la advertencia del sesgo del mínimo y de la heterogeneidad del error entre fumadores y no fumadores. | 03-interaccion-smoker-bmi.png, 05-predicho-vs-real.png |
| 9–10 | Limitaciones (colinealidad en grado alto, una sola semilla, dataset simulado) y cierre. | — |
