# Guion de la defensa — TP1 · 26/08/2026

Guion hablado de la presentación (`presentacion.pdf`), slide por slide, con reloj y marcas
de avance. Sigue las notas de orador (`\note{}`) de `presentacion.tex`, con ajustes de
oralidad y los números vigentes, acá recortado a lo que efectivamente se dice en 10 minutos y
ordenado para ensayar.

## Reglas de la cátedra que este guion respeta (TP1, p. 1)

- **10 minutos** de presentación + **8 minutos** de preguntas.
- La **introducción teórica 1.1** —separación train-validación-test: qué es y por qué es
  necesaria— tiene que estar en la presentación, "brevemente". Acá ocupa el bloque
  1:00–2:20 (slides 3 y 4), y la validación cruzada se completa en la slide 7.
- **Presentación y código se mandan 24 horas antes** de la clase: deadline **25/08**.
- Defensa: **26/08/2026**.

## Convenciones

- `[→]` = avanzar un overlay (una pulsación). La cantidad de `[→]` por slide coincide con
  los overlays del PDF; el último estado de cada slide queda en pantalla mientras se termina
  de hablar.
- Ritmo de referencia: **~2,3 palabras por segundo**. Cada bloque indica su presupuesto.
- Los números de test entran por `\input` desde `resultados-test.tex`, que produce
  `python3 -m src.evaluar_test`. Ya está corrido: el deck muestra los valores definitivos.

## Cobertura de la consigna

| Punto del enunciado | Dónde se dice | Minuto |
| --- | --- | --- |
| Intro teórica 1.1 (obligatoria) | Slides 3 y 4 (+ CV en slide 7) | 1:00–2:20 |
| 1. Limpieza: faltantes, outliers, criterio | Slides 5–6 (faltantes: una frase en slide 5) | 2:20–3:25 |
| 1. EDA: distribución de las variables | Slides 15–17 (el hallazgo que cambió el modelo) | 8:20–9:20 |
| 1. Limpieza: categóricas y escalado | Slide 7 (one-hot y doble estandarizado) | 3:25–4:15 |
| 2. Split explicado + CV sólo sobre train + RMSE | Slides 3, 4, 7, 8 | 1:00–5:05 |
| 3. Polinómica + regularización L1 | Slides 7, 10 | 3:25–4:15 y 5:45–6:15 |
| 4. RMSE por grado y lambda | Slides 8–10 | 4:15–6:15 |
| 5. Las tres preguntas (núcleo) | Slides 11–14 | 6:15–8:20 |
| Limitaciones y cierre | Slides 18–20 | 9:20–10:00 |

---

## El guion

### 1 · Portada — 0:00 → 0:15 (~35 palabras)

> Buenas. Vamos a presentar el TP1: regresión e introducción a la evaluación de modelos,
> sobre el dataset *Insurance Charges*: 1338 registros de personas, siete variables, y una
> variable a predecir: el costo médico anual.

### 2 · Qué hay que predecir — 0:15 → 1:00 (~105 palabras)

> El problema: predecir `charges`, el costo médico anual de una persona.
> **[→]** Con tres variables numéricas: edad, índice de masa corporal y cantidad de hijos.
> **[→]** Dos binarias: sexo y fumador.
> **[→]** Y una nominal de cuatro categorías: la región.
> **[→]** Ahora, la pregunta real del TP no es cuál modelo ajusta mejor. Es cuál modelo
> elegiríamos para usarlo de verdad, y qué error nos animaríamos a prometer sobre datos
> nuevos. Toda la presentación es la respuesta a esas dos preguntas.

### 3-4 · Los tres conjuntos y cómo se parten los datos — 1:00 → 2:20 · **intro teórica obligatoria** (~190 palabras)

> Antes de modelar, la decisión que ordena todo el trabajo: cómo se parten los datos.
> Entrenar y evaluar sobre los mismos datos premia memorizar; para saber si un modelo
> generaliza hay que medirlo en datos que no vio. De ahí los tres roles: **train** ajusta los
> parámetros; **validación** compara modelos y elige hiperparámetros; y **test** se reserva
> para estimar, una sola vez y al final, el error del modelo ya elegido. Si el test participa
> de alguna decisión, deja de ser una medición independiente.
> Adentro de train usamos validación cruzada de cinco folds: cada modelo se entrena cinco
> veces y cada fila valida exactamente una vez. Eso da un error promedio más estable que un
> split único, y además un desvío entre folds, que después va a importar.
> Y hay una segunda razón para el tercer conjunto, más sutil: el modelo aprende del train,
> pero **nosotros** aprendemos del set con el que elegimos. Si probás diecinueve
> configuraciones y te quedás con la mejor, en parte elegiste ruido favorable. Por eso ese
> error también está sesgado y hace falta un tercero.
> **[→]** Tres decisiones concretas. Partimos de 1337 filas y no 1338: había un duplicado
> exacto y se elimina antes de partir, porque si una copia cae en train y la otra en test, el
> test deja de ser independiente.
> **[→]** El split es 80/20 con semilla fija, sin estratificar —y el motivo no es que el
> target sea continuo: hay tres poblaciones de facto. Es que estratificar por ellas, medido, no
> baja la varianza entre folds. Las filas son personas independientes, no una serie temporal.
> **[→]** Y lo que más nos importa remarcar: el split va **antes** del análisis exploratorio,
> no justo antes de entrenar. El EDA también decide —de él salieron las tres columnas
> derivadas—, así que todo el punto 1 está medido sobre las 1070 filas de train.

### 5 · Outliers — 2:20 → 2:55 (~85 palabras)

> Punto uno, limpieza. Valores faltantes no hay: las 1338 por 7 celdas están completas; lo
> único que apareció fue ese duplicado. Lo que sí hay son outliers: 115 en `charges` según el
> criterio IQR, toda la cola derecha cara. ¿Error de carga o subpoblación real? Un error de
> carga estaría repartido al azar entre fumadores y no fumadores.
> **[→]** Y no lo está: el 97,4 % de los outliers son fumadores, contra el 11,3 % del resto.
> Es una subpoblación real, así que se conservan: recortarlos sería borrar justo lo que el
> modelo tiene que aprender.

### 6 · IQR contra z-score — 2:55 → 3:25 (~70 palabras)

> ¿Y por qué IQR y no z-score? Porque acá no discrepan en el margen: se contradicen.
> **[→]** En `charges` el z-score detecta veinte veces menos: los propios extremos inflan el
> desvío que usa de referencia; el criterio se sabotea solo.
> **[→]** En `children` pasa lo inverso: es una entera de cero a cinco, y "tres desvíos" ahí
> no significa nada.
> **[→]** Usamos IQR: se apoya en cuartiles y es robusto a la asimetría.

### 7 · El pipeline — 3:25 → 4:15 (~115 palabras)

Las cinco cajas entran de a una: cada `[→]` de la primera oración trae la caja que se está
nombrando (con su flecha).

> Puntos dos y tres: el pipeline, dentro de cada fold y en este orden: primero codificar
> one-hot, **[→]** estandarizar, **[→]** expandir en polinomios, **[→]** estandarizar de
> nuevo, **[→]** y recién ahí ajustar el modelo.
> **[→]** Los dos estandarizadores se ajustan sólo con el sub-train del fold. Ajustarlos
> antes de partir sería fuga de datos: la media de validación se filtraría al entrenamiento.
> **[→]** El primer escalado evita mezclar escalas absurdas: edad al cubo llega a 262 mil;
> hijos al cubo, a 125.
> **[→]** El segundo hace que la penalización L1 castigue a todas las features con la misma
> vara.
> **[→]** Y una aclaración conceptual: la regresión polinómica sigue siendo lineal en los
> parámetros: se transforman las columnas, no el modelo. Por eso la ecuación normal sigue
> sirviendo.

### 8 · La curva en U — 4:15 → 5:05 (~115 palabras)

> Punto cuatro, resultados. Esta figura se construye en cuatro pasos, y cada paso desmiente
> al anterior. Primero, el error de train solo: baja siempre; la conclusión ingenua sería
> "más grado es mejor".
> **[→]** Entra la validación, y acá está el resultado central: no mejora en ningún grado.
> Empata en grado 2 y de ahí empeora, hasta que el grado 4 se sale del gráfico: 87 916,9, casi
> veinte veces el del grado 1.
> **[→]** Tercero, la banda de un desvío entre folds: más-menos 367 en grado 1, más-menos
> 47 000 en grado 4.
> **[→]** Y la lectura final: el grado 4 no es sólo peor en promedio. Es inestable: su error
> depende de cómo caiga la partición.

### 9 · Los dos indicadores — 5:05 → 5:45 (~90 palabras)

> El sobreajuste se ve en dos indicadores independientes. La brecha train-validación: de 61,8
> dólares en grado 1 a casi ochenta y cinco mil (84 537,9) en grado 4. Una brecha que se
> ensancha así es la firma directa del sobreajuste.
> **[→]** Y el desvío entre folds: de más-menos 367 a más-menos 47 000, ciento veintiocho
> veces. Eso no dice "peor en promedio": dice poco confiable.
> **[→]** El remate: el error de train de grado 4 es el mejor de toda la tabla. Si
> eligiéramos por train, elegiríamos exactamente el peor modelo. Por eso el train no sirve
> para elegir.

### 10 · Lasso — 5:45 → 6:15 (~70 palabras)

> La regularización recupera lo que el grado había roto. Al aflojar lambda entran más
> features: el error primero baja —el modelo gana capacidad— y después vuelve a subir
> —empieza a sobreajustar—; el mínimo queda en el medio. Y en grado 4, la penalización L1
> deja vivos 17 de 1364 coeficientes: el 98,8 % muere en cero exacto, y con eso el grado 4
> pasa de 87 916,9 a 4433,7, un factor de casi veinte. Lo que no logra es superar al lineal
> simple sin regularizar.

### 11 · Primera respuesta — 6:15 → 6:30 (~40 palabras)

> Punto cinco, las tres preguntas del enunciado. Primera: ¿qué modelo obtuvo menor error de
> validación? La regresión lineal de grado 1, once features, sin regularizar: RMSE 4413,45
> más-menos 366,97. Y es, a la vez, el modelo más simple de todo el espacio de búsqueda.

### 12 · Segunda respuesta — 6:30 → 7:15 (~105 palabras)

> Segunda: ¿cuál implementaríamos en una aplicación real? Acá está el giro respecto de lo que
> este TP mostraba antes: ya no hay tensión entre el ganador de la validación cruzada y el
> modelo de producción. Son el mismo.
> **[→]** El desvío entre folds es 366,97, así que el error estándar de cada estimación es
> 164,11.
> **[→]** Con esa vara, 7 de las 15 configuraciones elegibles —de 19 corridas, 4 quedaron
> descartadas por no converger— son estadísticamente indistinguibles del ganador. Igual
> aplicamos la regla de un error estándar: entre esas 7, la más simple. Y la más simple es la
> misma que ya había ganado por error. La regla ya no decide: confirma.

### 13 · El resultado sobre test — 7:15 → 7:45 (~70 palabras)

> Y este es el resultado sobre las 267 filas de test, que no participaron de ninguna
> decisión: ni del split, ni del análisis exploratorio, ni de la elección de grado o lambda.
> **[→]** Las dos métricas juntas: 4288,52 dólares de RMSE y
> un R² de 0,871. Contra el baseline de predecir siempre la media, que da 11 963: el modelo es
> 2,79 veces mejor y explica el 87 % de la varianza en datos que nunca vio.

### 14 · Tercera respuesta — 7:45 → 8:20 (~85 palabras)

> Tercera: ¿qué RMSE prometeríamos?
> El número que prometemos es el de **test**: 4288,52 dólares, medido una sola vez sobre las
> 267 filas reservadas —unos 4300—.
> **[→]** Y esta vez la causa es más simple que antes: no se eligió por ser el mínimo de una
> tabla ajena. Producción es, directamente, el modelo que ganó la validación cruzada por error
> y por simplicidad a la vez. El sesgo de elegir el mínimo entre quince estimaciones ruidosas
> sigue existiendo —por eso igual lo declaramos—, pero ya no hay una segunda cifra de la que
> desconfiar: ganador y producción miden lo mismo.

### 15 · El histograma que cambió el modelo — 8:20 → 8:50 (~80 palabras)

> El hallazgo no salió de un modelo: salió de un histograma. El de `charges` muestra un pico
> grande y dos jorobas a la derecha, y la Clase 3 dice qué hacer con eso: ver el histograma
> separado por población.
> **[→]** Separado, son tres. Y el tercer grupo —fumadores con BMI arriba de 30— arranca en
> 32 548 dólares, por encima del 99,1 % de todos los demás. Es un escalón, no una pendiente:
> entre fumadores, pasar del tramo 29-30 al 30-31 suma quince mil dólares de golpe.

### 16 · Una columna vale más que todo el polinomio — 8:50 → 9:20 (~75 palabras)

> Eso importa porque el término cruzado del polinomio modela un cambio de *pendiente*: un
> producto con una variable continua no puede representar un salto. Así que creamos la
> binaria: fuma **y** BMI mayor a 30. El lineal sin ella da 6094.
> **[→]** Con ella, 4455.
> **[→]** Y el mejor Lasso *sin* la columna daba 4913, con 44 features. Una sola columna bien
> elegida le gana a toda la expansión polinómica más la regularización.

### 17 · El modelo entero, en once números — dentro de los 30 s anteriores

> Y en el modelo final el coeficiente más grande ya no es `fumador_obeso`: es `bmi_si_fuma`
> —la pendiente extra del BMI entre fumadores—, con 5236,99; `fumador_obeso` queda segundo,
> con 4891,83. `age` y `bmi` solos caen a 45,10 y 111,20 porque esas dos columnas nuevas les
> absorbieron el efecto. El modelo entero cabe en once números.

### 18 · Limitaciones — 9:20 → 9:45 (~60 palabras)

> Las limitaciones declaradas. La colinealidad en grados altos: en grado 4, el 68 % de las
> 1364 columnas es redundante, y los coeficientes no se interpretan de a uno.
> **[→]** Una sola semilla: la varianza del split no está medida.
> **[→]** El dataset es simulado.
> **[→]** Y el RMSE penaliza al cuadrado: lo dominan los casos caros.

### 19 · En una línea — 9:45 → 9:55 (~45 palabras)

> En una línea: a producción llevaríamos una regresión lineal con once características.
> «TEST» *pendiente:* Y el error que prometamos va a salir del test: una sola evaluación,
> sobre filas que siguen sin tocarse. / *definitiva:* Y esperaríamos un error típico de
> `\rmsetestproduccion` dólares.
> Es, a la vez, el de menor error de validación y el más simple de todo el espacio de
> búsqueda: no hubo que resignar precisión por simplicidad.
> **[→]** Todo implementado desde cero sobre numpy.

### 20 · Gracias / ¿Preguntas? — 9:55 → 10:00 (~10 palabras)

> Gracias. Quedamos abiertos a preguntas.

Esta slide queda puesta durante los 8 minutos de preguntas; las respuestas preparadas están
en la sección siguiente.

---

## Los 8 minutos de preguntas

Respuestas cortas preparadas, con el dato al frente. Las cuatro primeras eran las slides de
respaldo del deck; el detalle completo está en `informe.pdf`.

**¿Por qué k = 5 y no 10, o LOO?** — Lo medimos. El barrido controlado —barrer k con el modelo ya
elegido, hasta LOO— está rehecha con el pipeline de once features: el nivel del error agrupado
casi no depende de k —de 4428,7 (k=5) a 4405,9 (k=1070, LOO), un 0,51 %—; lo que se mueve es la
dispersión: el desvío entre folds salta de 367,0 (k=5) a 929,5 (k=10), 2,53 veces, y el error
estándar pasa de 164,1 a un pico de 293,9 en k=10 —1,79 veces el de la selección—. Repetir la
**selección completa** para varios k —las 19 configuraciones— cuesta horas por cada k, porque
el Lasso de grado 4 sobre 1364 columnas domina el costo, así que no lo reportamos: el informe
se apoya sólo en el barrido controlado. Con 1070 filas, k = 5 da ~214 por fold.

**¿La colinealidad de grado 4 no invalida el modelo?** — Invalida la lectura de coeficientes,
no las predicciones. El rango efectivo en grado 4 es 436 de 1364 columnas: una dummy al
cuadrado es función afín exacta de sí misma, y hay una causa extra —la
expansión duplica exactamente `edad_al_cuadrado` y `bmi_si_fuma` a partir de grado 2—.
Restringido al subespacio de rango completo, el condicionamiento es benigno (17,5 en grado 1,
401,6 en grado 2) y las predicciones son estables. Por eso los coeficientes de un grupo
colineal no se interpretan de a uno.

**¿Por qué `lstsq` y no invertir X^T X?** — El número de condición ya no crece monótono con
el grado: supera la precisión de float64 (~10^16) desde grado 2 —5,8 × 10^16—, salta a un
máximo de 1,5 × 10^18 en grado 3 y baja a 3,8 × 10^17 en grado 4. Desde grado 2 la matriz es
numéricamente singular e invertirla devolvería basura. `lstsq` usa SVD y da la solución de
norma mínima en cualquiera de los tres grados.

**¿Cómo saben que sus números están bien?** — Verificación cruzada: el RMSE de CV se
recalculó por un camino independiente del módulo (coincide a 0,1); el Lasso se contrastó
contra `scipy.optimize` (8 decimales); se auditaron los usos de `X_test` en el código; y la
alineación de los nombres de features se verificó asignando un primo a cada columna, de
modo que cada monomio factoriza unívocamente. Las cuatro suites de tests en verde.

**¿Por qué la regla de 1 error estándar y no el mínimo?** — Porque el mínimo de 19
estimaciones ruidosas está sesgado hacia abajo aunque cada estimación sea insesgada. Ahora que
ese mínimo también es el modelo más simple, la regla ya no tiene que decidir entre dos modelos
distintos: confirma. Y sigue mereciendo declararse, porque 7 configuraciones caen dentro de un
error estándar del ganador —a esa resolución el ranking exacto es ruido— y entre
indistinguibles el más simple es más barato, más explicable y más mantenible, incluso cuando ya
ganó por error.

**¿Y si el test da bastante peor que validación?** — Sería la evidencia directa del sesgo
del mínimo que acabamos de describir, no una sorpresa. El número que se promete es el de
test; además tiene su propia incertidumbre (267 filas) y promedia una población heterogénea:
el diagnóstico de residuos muestra que el error se concentra en un núcleo chico de no
fumadores caros —28 personas, 2,6 % del train— que ninguna de las columnas del dataset
distingue del resto.

**¿No es raro que test (4288,52) diera más bajo que validación (4413,45)?** — No: la
diferencia —124,93— es chica frente al ruido entre folds (±366,97). El argumento del sesgo de
selección no dice que el test deba dar peor: dice que la cifra de validación no es insesgada
porque salió de comparar quince estimaciones entre sí. El test puede caer para cualquier lado;
acá cayó abajo, sobre la misma partición que ya evaluamos dos veces antes de esta.

**¿Dónde pudo haber fuga de datos y cómo la evitaron?** — Tres lugares: el duplicado (se
elimina antes del split), los estandarizadores (se ajustan sólo con el sub-train de cada
fold) y el test (se toca una sola vez, al final; los usos de `X_test` en el código están
auditados).

**¿Por qué no estratificaron el split?** — *Ojo con esta: la respuesta obvia es la
equivocada.* No decir "el target es continuo, no hay clases": el histograma de `charges`
muestra que **sí** hay tres poblaciones de facto, y los cinco folds las reparten desigual
—entre 17 y 31 fumadores obesos, con correlación +0,62 entre esa proporción y el `charges`
medio del fold—. La respuesta correcta es: **lo medimos y no ayuda**. El desvío del RMSE
entre folds es 593 con folds aleatorios y 591 estratificando, promediando ocho particiones,
cuando entre dos particiones del mismo método va de 410 a 715. Lo que mueve el error de un
fold no es *cuántos* fumadores obesos le tocaron sino *cuáles* —dentro de ese grupo los
costos van de 32 548 a 63 770—, así que igualar los conteos deja intacta la varianza que
importa.

**¿La característica `fumador_obeso` no es fuga de datos?** — No. Sale de dos columnas que
están disponibles al momento de predecir, `smoker` y `bmi`, y `charges` no interviene en el
cálculo. Lo único que podría serlo es el umbral, y el umbral es el de la OMS: una constante
médica externa al dataset, la misma que el análisis del punto 1.4 ya usaba.

**¿No ajustaron el umbral 30 contra validación?** — No, y la distinción importa. El corte se
eligió *antes*, por ser el umbral clínico. El barrido de umbrales existe y da el mínimo
exactamente en 30 —29 da 4776 y 31 da 4888— pero es una confirmación posterior. Si el corte
se hubiera elegido *por* ese barrido sería un hiperparámetro ajustado sobre validación y el
RMSE de validación quedaría sesgado a la baja; lo diríamos así.

**¿Evaluaron el test más de una vez?** — Sí, sobre la misma partición, a medida que el
conjunto de características se fue cerrando; está declarado en las limitaciones del informe.
Lo que no pasó en ninguna de esas corridas es que el test participara de una decisión: la
elección se hizo íntegramente con validación cruzada sobre train, y ningún módulo del pipeline
de selección construye `X_test` —se verifica con un `grep`—. El costo igual es real: el
protocolo ideal pide partición nueva cada vez que cambia el modelo, y reutilizarla erosiona un
poco su independencia. Lo decimos porque ocultarlo sería peor.

**¿Y cómo mejorarían el modelo a partir de acá?** — No con otro modelo: con otra variable.
El diagnóstico de residuos muestra que el 41,9 % del error cuadrático de train se
concentra en 28 personas —no fumadoras, 2,6 % del train, con `charges` mayor a 25 000— a las
que el modelo subestima en 17 378 dólares en promedio. Y no se las puede separar del resto:
contra los demás no fumadores tienen BMI, cantidad de hijos, sexo y región parecidos; la única
diferencia es la edad (50,8 contra 39,1 años), y ninguna combinación de las siete columnas
originales las distingue. Es error irreducible con este dataset —falta la variable que explica
el gasto, una patología o una cirugía que no está en el CSV—. El diagnóstico corre sobre el
pipeline de once características, o sea con las tres derivadas ya incorporadas: no las tocan,
porque las tres están construidas alrededor de `smoker=yes` y este núcleo es de no fumadores.

**El slide 35 dice que una variable discretizada va one-hot. `children` lo es. ¿Por qué la
dejaron numérica?** — Porque lo medimos y no conviene: con one-hot el RMSE queda
igual en grado 1 —+5 con un ruido de ±14— y empeora en grado 2 —+453—. `charges` es casi
monótono en `children` hasta 3 hijos, así que la codificación numérica es una aproximación
barata; y los niveles 4 y 5 tienen 18 y 16 filas, a las que one-hot les daría un parámetro
propio.

---

## Reloj de ensayo

| Slide | Título corto | Arranca | Dura |
| --- | --- | --- | --- |
| 1 | Portada | 0:00 | 0:15 |
| 2 | Qué predecir | 0:15 | 0:45 |
| 3 | Los tres conjuntos (intro teórica) | 1:00 | 0:45 |
| 4 | Cómo se parte (intro teórica) | 1:45 | 0:35 |
| 5 | Outliers | 2:20 | 0:35 |
| 6 | IQR vs z-score | 2:55 | 0:30 |
| 7 | Pipeline | 3:25 | 0:50 |
| 8 | Curva en U | 4:15 | 0:50 |
| 9 | Dos indicadores | 5:05 | 0:40 |
| 10 | Lasso | 5:45 | 0:30 |
| 11 | Respuesta 1 | 6:15 | 0:15 |
| 12 | Respuesta 2 | 6:30 | 0:45 |
| 13 | El resultado sobre test | 7:15 | 0:30 |
| 14 | Respuesta 3 | 7:45 | 0:35 |
| 15 | El histograma que cambió el modelo | 8:20 | 0:30 |
| 16 | Una columna vale más que el polinomio | 8:50 | 0:20 |
| 17 | El modelo en once números | 9:10 | 0:10 |
| 18 | Limitaciones | 9:20 | 0:25 |
| 19 | En una línea | 9:45 | 0:10 |
| 20 | Gracias / ¿Preguntas? | 9:55 | 0:05 |

Puntos de control al ensayar: si al terminar la slide 7 pasaron más de 4:20, recortar en
las slides 8–10 (los números están en pantalla); si al terminar la 14 pasaron más de 8:25,
el hallazgo (15–17) admite comprimirse a una sola pasada sin overlays. Las slides 3 y 4 son
la intro teórica obligatoria: no se recortan.
