# Guion de la defensa — TP1 · 26/08/2026

Guion hablado de la presentación (`presentacion.pdf`), slide por slide, con reloj y marcas
de avance. Es el mismo texto que llevan los `\note{}` de `presentacion.tex`, acá recortado
a lo que efectivamente se dice en 10 minutos y ordenado para ensayar.

## Reglas de la cátedra que este guion respeta (TP1, p. 1)

- **10 minutos** de presentación + **8 minutos** de preguntas.
- La **introducción teórica 1.1** —separación train-validación-test: qué es y por qué es
  necesaria— tiene que estar en la presentación, "brevemente". Acá ocupa el bloque
  1:00–2:20 (slide 3), y la validación cruzada se completa en la slide 6.
- **Presentación y código se mandan 24 horas antes** de la clase: deadline **25/08**.
- Defensa: **26/08/2026**.

## Convenciones

- `[→]` = avanzar un overlay (una pulsación). La cantidad de `[→]` por slide coincide con
  los overlays del PDF; el último estado de cada slide queda en pantalla mientras se termina
  de hablar.
- Ritmo de referencia: **~2,3 palabras por segundo**. Cada bloque indica su presupuesto.
- `«TEST»` marca los tres lugares donde entra el número de test. Hasta correr
  `python3 -m src.evaluar_test` el deck muestra el placeholder y el guion usa la variante
  *pendiente*; después de correrlo, usar la variante *definitiva* con el número que quede en
  `resultados-test.tex`.

## Cobertura de la consigna

| Punto del enunciado | Dónde se dice | Minuto |
| --- | --- | --- |
| Intro teórica 1.1 (obligatoria) | Slide 3 (+ CV en slide 6) | 1:00–2:20 |
| 1. Limpieza: faltantes, outliers, criterio | Slides 4–5 (faltantes: una frase en slide 4) | 2:20–3:25 |
| 1. Limpieza: categóricas y escalado | Slide 6 (one-hot y doble estandarizado) | 3:25–4:15 |
| 2. Split explicado + CV sólo sobre train + RMSE | Slides 3, 6, 7 | 1:00–5:05 |
| 3. Polinómica + regularización L1 | Slides 6, 9 | 3:25–4:15 y 5:45–6:15 |
| 4. RMSE por grado y lambda | Slides 7–9 | 4:15–6:15 |
| 5. Las tres preguntas (núcleo) | Slides 10–15 | 6:15–9:20 |
| Limitaciones y cierre | Slides 16–18 | 9:20–10:00 |

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

### 3 · Cómo se parten los datos — 1:00 → 2:20 · **intro teórica obligatoria** (~190 palabras)

> Antes de modelar, la decisión que ordena todo el trabajo: cómo se parten los datos.
> Entrenar y evaluar sobre los mismos datos premia memorizar; para saber si un modelo
> generaliza hay que medirlo en datos que no vio. De ahí los tres roles: **train** ajusta los
> parámetros; **validación** compara modelos y elige hiperparámetros; y **test** se reserva
> para estimar, una sola vez y al final, el error del modelo ya elegido. Si el test participa
> de alguna decisión, deja de ser una medición independiente.
> Adentro de train usamos validación cruzada de cinco folds: cada modelo se entrena cinco
> veces y cada fila valida exactamente una vez. Eso da un error promedio más estable que un
> split único, y además un desvío entre folds, que después va a importar.
> **[→]** Tres decisiones concretas. Partimos de 1337 filas y no 1338: había un duplicado
> exacto y se elimina antes de partir, porque si una copia cae en train y la otra en test, el
> test deja de ser independiente.
> **[→]** El split es 80/20 con semilla fija, sin estratificar: el target es continuo y las
> filas son personas independientes, no una serie temporal.
> **[→]** Quedan 267 filas de test: alcanzan para estimar el error, con una incertidumbre
> propia que declaramos en las limitaciones.

### 4 · Outliers — 2:20 → 2:55 (~85 palabras)

> Punto uno, limpieza. Valores faltantes no hay: las 1338 por 7 celdas están completas; lo
> único que apareció fue ese duplicado. Lo que sí hay son outliers: 139 en `charges` según el
> criterio IQR, toda la cola derecha cara. ¿Error de carga o subpoblación real? Un error de
> carga estaría repartido al azar entre fumadores y no fumadores.
> **[→]** Y no lo está: el 97,8 % de los outliers son fumadores, contra el 11,5 % del resto.
> Es una subpoblación real, así que se conservan: recortarlos sería borrar justo lo que el
> modelo tiene que aprender.

### 5 · IQR contra z-score — 2:55 → 3:25 (~70 palabras)

> ¿Y por qué IQR y no z-score? Porque acá no discrepan en el margen: se contradicen.
> **[→]** En `charges` el z-score detecta veinte veces menos: los propios extremos inflan el
> desvío que usa de referencia; el criterio se sabotea solo.
> **[→]** En `children` pasa lo inverso: es una entera de cero a cinco, y "tres desvíos" ahí
> no significa nada.
> **[→]** Usamos IQR: se apoya en cuartiles y es robusto a la asimetría.

### 6 · El pipeline — 3:25 → 4:15 (~115 palabras)

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

### 7 · La curva en U — 4:15 → 5:05 (~115 palabras)

> Punto cuatro, resultados. Esta figura se construye en cuatro pasos, y cada paso desmiente
> al anterior. Primero, el error de train solo: baja siempre; la conclusión ingenua sería
> "más grado es mejor".
> **[→]** Entra la validación: mejora hasta grado 2 y después empeora. El grado 4 es peor que
> el lineal simple.
> **[→]** Tercero, la banda de un desvío entre folds: más-menos 244 en grado 2, más-menos
> 1031 en grado 4.
> **[→]** Y la lectura final: el grado 4 no es sólo peor en promedio. Es inestable: su error
> depende de cómo caiga la partición.

### 8 · Los dos indicadores — 5:05 → 5:45 (~90 palabras)

> El sobreajuste se ve en dos indicadores independientes. La brecha train-validación: de 88
> en grado 1 a 2508 en grado 4. Una brecha que se ensancha así es la firma directa del
> sobreajuste.
> **[→]** Y el desvío entre folds: de más-menos 89 a más-menos 1031, más de diez veces. Eso
> no dice "peor en promedio": dice poco confiable.
> **[→]** El remate: el error de train de grado 4 es el mejor de toda la tabla. Si
> eligiéramos por train, elegiríamos exactamente el peor modelo. Por eso el train no sirve
> para elegir.

### 9 · Lasso — 5:45 → 6:15 (~70 palabras)

> La regularización recupera lo que el grado había roto. Al aflojar lambda entran más
> features: el error primero baja —el modelo gana capacidad— y después vuelve a subir
> —empieza a sobreajustar—; el mínimo queda en el medio. Y en grado 4, la penalización L1
> deja vivos 26 de 494 coeficientes: el 95 % muere en cero exacto, y con eso vuelve la
> estabilidad que la curva en U había perdido.

### 10 · Primera respuesta — 6:15 → 6:30 (~40 palabras)

> Punto cinco, las tres preguntas del enunciado. Primera: ¿qué modelo obtuvo menor error?
> Lasso de grado 4, lambda 286, RMSE de validación 4920 más-menos 224. Y sin embargo, no es
> el que llevaríamos a producción.

### 11 · Segunda respuesta — 6:30 → 7:15 (~105 palabras)

> Segunda: ¿cuál implementaríamos en una aplicación real? Acá está el argumento central del
> TP. Las mejores configuraciones están separadas por decenas de dólares…
> **[→]** …pero el desvío entre folds es 224, así que el error estándar de cada estimación
> es cien.
> **[→]** Con esa vara, 8 de las 18 configuraciones son estadísticamente indistinguibles del
> ganador: cuál sale primera lo decide el azar de la partición, no el modelo. Elegir por el
> ranking crudo sería leer ruido. Aplicamos la regla de un error estándar: entre
> indistinguibles, el más simple.

### 12 · Entre indistinguibles, el más simple — 7:15 → 7:45 (~75 palabras)

> El ganador de la validación cruzada usa 494 features. El modelo de producción, grado 2,
> usa 44, con 10 coeficientes vivos.
> **[→]** «TEST» *pendiente:* La simplicidad compra un espacio de features once veces más
> chico; el costo en dólares lo va a decir el test. / *definitiva:* La simplicidad cuesta
> +`\costosimplicidad` dólares de RMSE —un 2 %— y compra un espacio once veces más chico.
> **[→]** Las dos referencias encuadran: el lineal simple y predecir la media quedan en otra
> liga. La discusión fina es entre los dos de arriba.

### 13 · Tercera respuesta — 7:45 → 8:20 (~85 palabras)

> Tercera: ¿qué RMSE prometeríamos?
> «TEST» *pendiente:* El número que se promete es el de test, que todavía no medimos: se mide
> una sola vez, sobre las 267 filas reservadas, al final. / *definitiva:* El número que
> prometemos es el de test: `\rmsetestproduccion` dólares.
> **[→]** Lo que sí sabemos: no va a ser el 4955 de validación. Esa configuración se eligió
> por ser el mínimo de 19 estimaciones ruidosas, y el mínimo de estimaciones ruidosas está
> sesgado hacia abajo aunque cada una sea insesgada. Por eso el test se reserva hasta el
> final.

### 14 · El hallazgo — 8:20 → 8:55 (~85 palabras)

> ¿Y por qué el polinomio mejora? Mirada sola, `bmi` parece poco informativa: correlación
> 0,198.
> **[→]** Pero separando por fumador aparecen dos poblaciones superpuestas.
> **[→]** En los no fumadores, el BMI casi no mueve el costo: 83 dólares por punto. En los
> fumadores lo dispara: 1473. Un factor de 17,7 entre las pendientes: eso es un término de
> interacción `bmi` por `smoker`, que un modelo aditivo no puede representar. Por eso gana el
> grado 2 —por la interacción, no por las potencias.

### 15 · El Lasso lo encontró solo — 8:55 → 9:20 (~60 palabras)

> Y el Lasso lo encontró solo: en el modelo de producción, `bmi` por `smoker` quedó con
> coeficiente 3317…
> **[→]** …por encima de `bmi` sola. La penalización recuperó, sin que nadie se lo indicara,
> el mismo fenómeno que el análisis exploratorio había encontrado a mano.
> **[→]** De 44 features sobreviven 10; las demás, apagadas en cero exacto.

### 16 · Limitaciones — 9:20 → 9:45 (~60 palabras)

> Cuatro limitaciones declaradas. La colinealidad en grados altos: en grado 4, el 56 % de
> las columnas es redundante, y los coeficientes no se interpretan de a uno.
> **[→]** Una sola semilla: la varianza del split no está medida.
> **[→]** El dataset es simulado.
> **[→]** Y el RMSE penaliza al cuadrado: lo dominan los casos caros.

### 17 · En una línea — 9:45 → 9:55 (~45 palabras)

> En una línea: a producción llevaríamos un Lasso de grado 2 con 10 features vivas.
> «TEST» *pendiente:* Y el error que prometamos va a salir del test: una sola evaluación,
> sobre filas que siguen sin tocarse. / *definitiva:* Y esperaríamos un error típico de
> `\rmsetestproduccion` dólares.
> No es el de menor error de validación: es el más simple entre los indistinguibles.
> **[→]** Todo implementado desde cero sobre numpy.

### 18 · Gracias / ¿Preguntas? — 9:55 → 10:00 (~10 palabras)

> Gracias. Quedamos abiertos a preguntas.

Esta slide queda puesta durante los 8 minutos de preguntas; las respuestas preparadas están
en la sección siguiente.

---

## Los 8 minutos de preguntas

Respuestas cortas preparadas, con el dato al frente. Las cuatro primeras eran las slides de
respaldo del deck anterior; el detalle completo está en `informe.pdf` y en `DECISIONES.md`.

**¿Por qué k = 5 y no 10, o LOO?** — Lo medimos (D-22): repetimos la selección completa con
k = 5, 10 y 20 y un barrido hasta LOO. El modelo elegido no cambia con k. Con 1070 filas,
k = 5 da ~214 por fold; subir k encarece y no cambia la decisión.

**¿La colinealidad de grado 4 no invalida el modelo?** — Invalida la lectura de coeficientes,
no las predicciones. El rango efectivo en grado 4 es 216 de 494 columnas: una dummy al
cuadrado es función afín exacta de sí misma. Restringido al subespacio de rango completo, el
condicionamiento es benigno y las predicciones son estables. Por eso los coeficientes de un
grupo colineal no se interpretan de a uno.

**¿Por qué `lstsq` y no invertir X^T X?** — El número de condición llega a 3,1 × 10^18 en
grado 4, por encima de la precisión de float64 (~10^16): la matriz es numéricamente singular
e invertirla devolvería basura. `lstsq` usa SVD y da la solución de norma mínima.

**¿Cómo saben que sus números están bien?** — Verificación cruzada: el RMSE de CV se
recalculó por un camino independiente del módulo (coincide a 0,1); el Lasso se contrastó
contra `scipy.optimize` (8 decimales); se auditaron los usos de `X_test` en el código; y la
alineación de los 494 nombres de features se verificó asignando un primo a cada columna, de
modo que cada monomio factoriza unívocamente. Tres suites de tests en verde.

**¿Por qué la regla de 1 error estándar y no el mínimo?** — Porque el mínimo de 19
estimaciones ruidosas está sesgado hacia abajo aunque cada estimación sea insesgada, y
porque 8 configuraciones caen dentro de un error estándar del ganador: a esa resolución el
ranking es ruido, y entre indistinguibles el más simple es más barato, más explicable y más
mantenible.

**¿Y si el test da bastante peor que validación?** — Sería la evidencia directa del sesgo
del mínimo que acabamos de describir, no una sorpresa. El número que se promete es el de
test; además tiene su propia incertidumbre (267 filas) y promedia una población heterogénea:
el error se concentra en los fumadores.

**¿Dónde pudo haber fuga de datos y cómo la evitaron?** — Tres lugares: el duplicado (se
elimina antes del split), los estandarizadores (se ajustan sólo con el sub-train de cada
fold) y el test (se toca una sola vez, al final; los usos de `X_test` en el código están
auditados).

**¿Por qué no estratificaron el split?** — El target es continuo: no hay clases que
preservar. Y las filas son personas independientes, no una serie temporal: el barajado
simple con semilla fija alcanza.

---

## Reloj de ensayo

| Slide | Título corto | Arranca | Dura |
| --- | --- | --- | --- |
| 1 | Portada | 0:00 | 0:15 |
| 2 | Qué predecir | 0:15 | 0:45 |
| 3 | Partición (intro teórica) | 1:00 | 1:20 |
| 4 | Outliers | 2:20 | 0:35 |
| 5 | IQR vs z-score | 2:55 | 0:30 |
| 6 | Pipeline | 3:25 | 0:50 |
| 7 | Curva en U | 4:15 | 0:50 |
| 8 | Dos indicadores | 5:05 | 0:40 |
| 9 | Lasso | 5:45 | 0:30 |
| 10 | Respuesta 1 | 6:15 | 0:15 |
| 11 | Respuesta 2 | 6:30 | 0:45 |
| 12 | Indistinguibles | 7:15 | 0:30 |
| 13 | Respuesta 3 | 7:45 | 0:35 |
| 14 | Interacción | 8:20 | 0:35 |
| 15 | Lasso lo encontró | 8:55 | 0:25 |
| 16 | Limitaciones | 9:20 | 0:25 |
| 17 | En una línea | 9:45 | 0:10 |
| 18 | Gracias / ¿Preguntas? | 9:55 | 0:05 |

Puntos de control al ensayar: si al terminar la slide 6 pasaron más de 4:20, recortar en
las slides 7–9 (los números están en pantalla); si al terminar la 13 pasaron más de 8:25,
el hallazgo (14–15) admite comprimirse a una sola pasada sin overlays.
