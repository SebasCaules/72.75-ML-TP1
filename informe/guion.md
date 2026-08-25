# Guion de la defensa — TP1 · Grupo 7 · 02/09/2026, 17:35, Aula 701F

Guion hablado de la presentación (`presentacion.pdf`), slide por slide, con reloj y marcas
de avance. Sigue las notas de orador (`\note{}`) de `presentacion.tex`, con ajustes de
oralidad y los números vigentes, acá recortado a lo que efectivamente se dice en 10 minutos y
ordenado para ensayar.

## El criterio de reparto: la teoría se nombra, los resultados se defienden

Lo que ya se dictó en la teórica —qué es train/validación/test, qué es una validación
cruzada, qué hace one-hot, qué es fuga de datos, qué penaliza L1— **se nombra en una pasada
y no se desarrolla**. El tribunal lo sabe: repetirlo no suma nota y gasta el único recurso
escaso, que son diez minutos. Lo que sí hay que defender es lo que este trabajo hizo con eso:
los números, por qué elegimos el modelo que elegimos, y qué error prometemos.

La regla práctica al ensayar: **si una frase se entendería igual en cualquier TP de la
materia, es teoría y va rápido; si sólo tiene sentido con nuestros datos adelante, es
resultado y se dice completa.**

| Bloque | Slides | Antes | Ahora | |
| --- | --- | --- | --- | --- |
| Marco (portada, problema, cierre) | 1, 2, 20–22 | 1:40 | 1:20 | −0:20 |
| **Teoría y método** | 3, 4, 6, 7 | **2:40** | **1:50** | **−0:50** |
| Limpieza (dato propio) | 5 | 0:35 | 0:30 | −0:05 |
| **Resultados y hallazgo** | 8–19 | **5:05** | **6:20** | **+1:15** |

Los 75 segundos que salen de la teoría, el marco y la limpieza entran enteros en resultados.
La mitad la absorben las dos slides nuevas —el zoom sin grado 4, donde se elige, y el
histograma separado por población, donde está el hallazgo—; la otra mitad va a decir en voz
alta lo que antes estaba escrito en las slides y ahora no está en ningún lado.

## Reglas de la cátedra que este guion respeta (TP1, p. 1)

- **10 minutos** de presentación + **8 minutos** de preguntas.
- La **introducción teórica 1.1** —separación train-validación-test: qué es y por qué es
  necesaria— tiene que estar en la presentación, y el enunciado dice **"brevemente"**. Acá
  ocupa 0:50–1:45 (slides 3 y 4). Comprimida, no eliminada: es un punto evaluable, así que
  los tres roles y el porqué del tercer conjunto se dicen sí o sí. Lo que se fue es el
  desarrollo.
- **Presentación y código se mandan 24 horas antes** de la clase: deadline **25/08**.
- Defensa: **02/09/2026**, 17:35, Aula 701F (2ª fecha, presencial).

## Convenciones

- `[→]` = avanzar un overlay (una pulsación). La cantidad de `[→]` por slide coincide con
  los overlays del PDF; el último estado de cada slide queda en pantalla mientras se termina
  de hablar.
- Ritmo de referencia: **~2,7 palabras por segundo** (~165 por minuto). Es el ritmo real de
  este guion, medido: el presupuesto de cada slide sale de dividir sus palabras por ese
  número. Ojo que es un ritmo de presentación *rápida*; si al ensayar no entra, lo que se
  recorta está indicado abajo, en los puntos de control.
- **Las slides casi no tienen texto.** El deck es figuras y números; la prosa se sacó a
  propósito. O sea que este guion no acompaña a la pantalla: *es* el contenido. Si una frase
  de acá no se dice, esa información no aparece en ningún lado.
- Los números de test entran por `\input` desde `resultados-test.tex`, que produce
  `python3 -m src.evaluar_test`. Ya está corrido: el deck muestra los valores definitivos.

## Cobertura de la consigna

| Punto del enunciado | Dónde se dice | Minuto |
| --- | --- | --- |
| Intro teórica 1.1 (obligatoria) | Slides 3 y 4 (+ CV en slide 7) | 0:50–1:45 |
| 1. Limpieza: faltantes, outliers, criterio | Slides 5–6 (faltantes: una frase en slide 5) | 1:45–2:35 |
| 1. EDA: distribución de las variables | Slides 16–19 (el hallazgo que cambió el modelo) | 7:45–9:25 |
| 1. Limpieza: categóricas y escalado | Slide 7 (one-hot y doble estandarizado) | 2:35–3:10 |
| 2. Split explicado + CV sólo sobre train + RMSE | Slides 3, 4, 7, 8 | 0:50–3:50 |
| 3. Polinómica + regularización L1 | Slides 7, 11 | 2:35–3:10 y 5:00–5:25 |
| 4. RMSE por grado y lambda | Slides 8–11 | 3:10–5:25 |
| 5. Las tres preguntas (núcleo) | Slides 12–15 | 5:25–7:45 |
| Limitaciones y cierre | Slides 20–22 | 9:25–10:00 |

---

## El guion

### 1 · Portada — 0:00 → 0:10 (~27 palabras)

> Buenas. TP1: regresión e introducción a la evaluación de modelos, sobre el dataset
> *Insurance Charges*. 1338 personas, siete variables, y una a predecir: el costo médico anual.

### 2 · Qué hay que predecir — 0:10 → 0:35 (~67 palabras)

> Predecir `charges`, el costo médico anual de una persona.
> **[→]** Con tres numéricas: edad, índice de masa corporal y cantidad de hijos.
> **[→]** Dos binarias: sexo y fumador.
> **[→]** Y una nominal de cuatro categorías: la región.
> La pregunta real del TP no es cuál ajusta mejor: es cuál elegiríamos para usar de verdad, y
> qué error nos animaríamos a prometer. Toda la presentación responde esas dos.

### 3 · Los tres conjuntos — 0:35 → 1:05 · **intro teórica obligatoria** (~86 palabras)

Punto evaluable. Se dice completo pero *rápido*: son definiciones que el tribunal ya conoce.
No desarrollar ninguna, no dar ejemplos.

> La intro teórica que pide el enunciado, breve. **Train** ajusta los parámetros.
> **[→] Validación** no ajusta nada: elige entre configuraciones. Acá, una validación cruzada
> de cinco folds adentro de train.
> **[→] Test** no participa de ninguna decisión: se toca una vez, al final.
> **[→]** ¿Por qué no alcanza con dos? Porque el modelo aprende del train, pero **nosotros**
> aprendemos del set con el que elegimos: probamos diecinueve configuraciones y nos quedamos
> con la mejor. Ese error también queda sesgado, y por eso hace falta un tercero.

### 4 · Cómo se parten los datos — 1:05 → 1:35 (~74 palabras)

Acá ya no hay teoría: son tres decisiones nuestras. La tercera es la que vale.

*Si preguntan por qué no estratificaron, la respuesta larga —y la trampa de contestar "el
target es continuo"— está preparada en la sección de preguntas.*

> Nuestra partición: 80/20, 1070 y 267 filas.
> **[→]** 1337 y no 1338: había un duplicado exacto, y se elimina **antes** de partir.
> **[→]** Semilla fija, sin estratificar: lo medimos y no baja la varianza entre folds.
> **[→]** Y lo que más nos importa: el split va **antes** del análisis exploratorio. El EDA
> también decide —de él salieron las tres columnas derivadas—, así que todo el punto 1 está
> medido sobre las 1070 de train.

### 5 · Outliers — 1:35 → 2:05 (~84 palabras)

> Limpieza. Faltantes no hay: las 1338 por 7 celdas están completas; lo único fue el
> duplicado. Outliers sí: 115 en `charges` por criterio IQR, toda la cola cara. ¿Error de
> carga o subpoblación real? Un error de carga estaría repartido al azar entre fumadores y no
> fumadores. Y no lo está: el **97,4 %** de los outliers son fumadores, contra el 11,3 % del
> resto. Es una subpoblación real, así que se conservan: recortarlos sería borrar justo lo
> que el modelo tiene que aprender.

### 6 · IQR contra z-score — 2:05 → 2:25 (~55 palabras)

> ¿Por qué IQR y no z-score? Acá no discrepan en el margen: se contradicen.
> **[→]** En `charges` el z-score detecta 5 contra 115: los propios extremos inflan el desvío
> que usa de referencia.
> **[→]** En `children` pasa lo inverso, 16 contra 0: es entera de cero a cinco.
> **[→]** Usamos IQR, robusto a la asimetría.

### 7 · El pipeline — 2:25 → 2:55 (~85 palabras)

Las cinco cajas entran de a una. **Esta slide es la que más se recortó**: one-hot,
estandarizar y fuga de datos son de la teórica. Nombrarlas al pasar; el único punto propio es
*por qué dos* estandarizaciones.

> El pipeline, dentro de cada fold: codificar one-hot, **[→]** estandarizar, **[→]** expandir
> en polinomios, **[→]** estandarizar de nuevo, **[→]** y ajustar.
> **[→]** Los dos estandarizadores se ajustan sólo con el sub-train del fold: sin fuga.
> **[→]** El primero evita mezclar escalas absurdas: edad al cubo llega a 262 mil, hijos al
> cubo a 125.
> **[→]** El segundo hace que L1 castigue a todas las features con la misma vara. Y la
> polinómica sigue siendo lineal en los parámetros: se transforman las columnas, no el modelo.

### 8 · La curva, hasta grado 4 — 2:55 → 3:40 (~121 palabras)

> Resultados. Tres pasos, y cada uno desmiente al anterior. Primero el error de train solo:
> baja siempre, y la conclusión ingenua sería "más grado es mejor".
> **[→]** Entra validación, y acá está el resultado central: **no mejora en ningún grado**.
> Empata en 2 y de ahí empeora, hasta que el grado 4 llega a 87 917, casi veinte veces el del
> grado 1.
> **[→]** Y la banda de un desvío entre folds: más-menos 367 en grado 1, más-menos 47 000 en
> grado 4. El grado 4 no es sólo peor: es inestable, su error depende de cómo caiga la
> partición.
> Aclaro la escala porque se nota: el eje es logarítmico, es la única forma de que los cuatro
> grados entren juntos.

### 9 · Sin el grado 4: acá se elige — 3:40 → 4:30 (~139 palabras)

Slide nueva. **Es la que justifica la elección**, así que no apurarla. La honestidad del
segundo párrafo es lo que la hace fuerte, no lo que la debilita.

> Mismo gráfico sin el grado 4 y en escala lineal. Lo saco porque ya quedó descartado: su
> única función era mostrar hasta dónde llega el desastre, y mientras está adentro obliga a
> una escala que no deja comparar a los otros tres.
> Sacado, el rango baja a 3900–7600 y recién ahí se ve la diferencia real. Grado 1: 4413.
> Grado 2: 4517, **103 dólares peor**. Grado 3: 6758, un 53 % peor.
> **[→]** El mínimo de validación es el grado 1. Pero digamos lo incómodo: esos 103 dólares
> son **menos** que el desvío entre folds del grado 1, que es más-menos 367. O sea que el
> grado 1 no le gana al grado 2 por error: **empata, y gana por más simple**. Eso es la regla
> de un error estándar, y es el argumento de las slides que vienen.

### 10 · Los dos indicadores — 4:30 → 4:55 (~72 palabras)

> El sobreajuste se ve en dos indicadores independientes. La brecha train-validación: de 61,8
> dólares en grado 1 a 84 538 en grado 4.
> **[→]** Y el desvío entre folds: de 367 a 47 000, ciento veintiocho veces. Eso no dice
> "peor en promedio", dice poco confiable.
> El remate: el error de **train** de grado 4 es el mejor de toda la tabla, 3379. Si
> eligiéramos por train, elegiríamos exactamente el peor modelo.

### 11 · Lasso — 4:55 → 5:20 (~67 palabras)

> La regularización recupera lo que el grado había roto. Al aflojar lambda entran más
> features: el error baja y después vuelve a subir; el mínimo queda en el medio. En grado 4,
> L1 deja vivos **17 de 1364** coeficientes —el 98,8 % muere en cero exacto— y con eso el
> grado 4 pasa de 87 917 a 4434. Lo que no logra es superar al lineal simple.

### 12 · Primera respuesta — 5:20 → 5:35 (~38 palabras)

> Las tres preguntas del enunciado. Primera: ¿qué modelo obtuvo menor error de validación? La
> lineal de grado 1, once features, sin regularizar: **4413,45 más-menos 366,97**. Y es, a la
> vez, el modelo más simple de todo el espacio.

### 13 · Segunda respuesta — 5:35 → 6:20 (~117 palabras)

> Segunda: ¿cuál implementaríamos de verdad? Acá está el giro de este trabajo: ya no hay
> tensión entre el ganador de la validación cruzada y el modelo de producción. Son el mismo.
> **[→]** El desvío entre folds es 366,97, así que el error estándar de cada estimación es
> 164,11. Con esa vara, **7 de las 15 configuraciones elegibles** son estadísticamente
> indistinguibles del ganador —corrimos 19; 4 quedaron descartadas por no converger—. A esa
> resolución, cuál sale primera lo decide en buena medida el azar de la partición.
> Igual aplicamos la regla de un error estándar: entre esas 7, la más simple. Y la más simple
> es la que ya había ganado por error. La regla no decide: **confirma**.

### 14 · El resultado sobre test — 6:20 → 7:05 (~115 palabras)

> Y este es el resultado, sobre las 267 filas que el modelo nunca vio y que no participaron
> de ninguna decisión: ni del split, ni del EDA, ni de la elección de grado o lambda.
> **[→]** Las dos métricas juntas, y a propósito. El RMSE es el que pide el enunciado y está
> en dólares: **4288,52**. Pero un RMSE solo no se puede interpretar —¿4288 es mucho o poco?—,
> así que al lado va el baseline de predecir siempre la media, 11 963: el modelo es **2,79
> veces mejor**. Y el R² es esa misma comparación normalizada: **0,871**, o sea que explica el
> 87 % de la variabilidad de `charges` en datos que nunca vio.

### 15 · Tercera respuesta — 7:05 → 7:35 (~77 palabras)

> Tercera: ¿qué RMSE prometeríamos en datos nuevos? El de **test**: 4288,52 dólares, medido
> una sola vez sobre las 267 filas reservadas.
> No los 4413 de validación. Y el porqué acá es directo: producción **es** el mínimo de
> validación entre las quince configuraciones elegibles, y el mínimo de un conjunto de
> estimaciones ruidosas queda sesgado a la baja aunque cada una sea insesgada. Por eso el test
> se reserva hasta el final y se toca una sola vez.

### 16 · El histograma que cambió el modelo — 7:35 → 8:00 (~66 palabras)

> El hallazgo no salió de un modelo: salió de un histograma. Este es el de `charges`, entero.
> Un pico grande y dos jorobas a la derecha, donde una distribución con cola no tendría
> ninguna. Con la distribución entera no se puede decir mucho más: la pregunta es si esas
> jorobas son algo o son ruido del binning. La Clase 3 dice qué hacer: separarlo por
> población.

### 17 · Separado por población: son tres — 8:00 → 8:30 (~83 palabras)

Slide nueva. Mismo eje x que la anterior: lo único que cambia es el color, y eso es lo que
hay que hacer mirar.

> Separado, las dos jorobas tienen nombre. No fumadores. Fumadores con BMI hasta 30. Y
> fumadores con BMI arriba de 30, que arrancan en **32 548 dólares**, por encima del 99,1 %
> de todos los demás. Cada joroba es un grupo.
> Y la separación es un **escalón**, no una pendiente: entre fumadores, pasar del tramo de
> BMI 29-30 al de 30-31 suma quince mil dólares de golpe, con pendiente suave a los dos
> lados. Entre los no fumadores, el mismo corte no mueve nada.

### 18 · Una columna vale más que todo el polinomio — 8:30 → 9:00 (~86 palabras)

> Eso importa porque el término cruzado del polinomio, BMI por fumador, modela un cambio de
> *pendiente*: un producto con una variable continua no puede representar un salto. Así que
> creamos la binaria: fuma **y** BMI mayor a 30. El lineal sin ella da 6094.
> **[→]** Con ella, **4455**. Mil seiscientos dólares menos, contra un ruido de veinte entre
> particiones. Y el mejor Lasso *sin* la columna daba 4913, con 44 features: una sola columna
> bien elegida le gana a toda la expansión polinómica más la regularización.

### 19 · El modelo entero, en once números — 9:00 → 9:15 (~33 palabras)

> El modelo entero cabe en once números, y los tres más grandes son las tres columnas
> derivadas: `bmi_si_fuma`, `fumador_obeso`, `edad_al_cuadrado`.
> **[→]** Ninguna venía en el CSV. Las tres salieron de mirar los datos.

### 20 · Limitaciones — 9:15 → 9:35 (~56 palabras)

> Las limitaciones declaradas. Colinealidad en grados altos: en grado 4 el 68 % de las 1364
> columnas es redundante, y los coeficientes no se interpretan de a uno.
> **[→]** Una sola semilla: la varianza del split no está medida.
> **[→]** El dataset es simulado.
> **[→]** Y el RMSE penaliza al cuadrado: lo dominan los casos caros.

### 21 · El modelo que llevo a producción — 9:35 → 9:55 (~54 palabras)

> El cierre. A producción llevamos una regresión lineal con once características.
> **[→]** Y esperamos un error típico de **4288,52** dólares.
> **[→]** Es, a la vez, el de menor error de validación y el más simple de todo el espacio: no
> hubo que resignar precisión por simplicidad. Todo implementado desde cero sobre numpy, sin
> scikit-learn.

### 22 · Gracias / ¿Preguntas? — 9:55 → 10:00 (~5 palabras)

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

| Slide | Título corto | Arranca | Dura | Bloque |
| --- | --- | --- | --- | --- |
| 1 | Portada | 0:00 | 0:10 | marco |
| 2 | Qué predecir | 0:10 | 0:25 | marco |
| 3 | Los tres conjuntos | 0:35 | 0:30 | teoría |
| 4 | Cómo se parte | 1:05 | 0:30 | teoría |
| 5 | Outliers | 1:35 | 0:30 | limpieza |
| 6 | IQR vs z-score | 2:05 | 0:20 | teoría |
| 7 | Pipeline | 2:25 | 0:30 | teoría |
| 8 | La curva, hasta grado 4 | 2:55 | 0:45 | **resultados** |
| 9 | Sin el grado 4: acá se elige | 3:40 | 0:50 | **resultados** |
| 10 | Dos indicadores | 4:30 | 0:25 | **resultados** |
| 11 | Lasso | 4:55 | 0:25 | **resultados** |
| 12 | Respuesta 1 | 5:20 | 0:15 | **resultados** |
| 13 | Respuesta 2 | 5:35 | 0:45 | **resultados** |
| 14 | El resultado sobre test | 6:20 | 0:45 | **resultados** |
| 15 | Respuesta 3 | 7:05 | 0:30 | **resultados** |
| 16 | El histograma | 7:35 | 0:25 | **hallazgo** |
| 17 | Separado por población | 8:00 | 0:30 | **hallazgo** |
| 18 | Una columna vale más | 8:30 | 0:30 | **hallazgo** |
| 19 | El modelo en once números | 9:00 | 0:15 | **hallazgo** |
| 20 | Limitaciones | 9:15 | 0:20 | cierre |
| 21 | El modelo que llevo a producción | 9:35 | 0:20 | cierre |
| 22 | Gracias / ¿Preguntas? | 9:55 | 0:05 | cierre |

**Puntos de control al ensayar.**

- Al terminar la **slide 7** tienen que haber pasado **2:55**. Es el control más importante
  del guion: todo lo que está antes es marco y teoría, y si ahí ya se fueron cuatro minutos,
  el trabajo propio se cuenta apurado. Si se pasa, se recorta en 3, 4, 6 y 7 —son las cuatro
  slides que repiten la teórica— y nunca en 8–19.
- Al terminar la **slide 15** tienen que haber pasado **7:35**: las tres respuestas del punto
  5 ya están dadas, o sea que lo evaluable está entero. A partir de ahí queda el hallazgo,
  que es lo mejor del trabajo pero no es un punto suelto de la consigna.
- Si hay que recortar en vivo, el orden es: primero la aclaración de la escala logarítmica en
  la 8; después el segundo estandarizador en la 7; después la 19, que se puede decir en una
  frase. **No se recorta la 9** —es donde se justifica la elección— ni la 17.
- Las slides **3 y 4 no se eliminan**: son la intro teórica obligatoria del punto 1.1. Pero
  se dicen rápido y sin desarrollar, que es lo que el enunciado pide con "brevemente".
