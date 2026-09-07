# Informe Técnico — Predicción de Popularidad de Canciones

**Spotify Tracks Dataset**  
**MLY1101 — Machine Learning | Evaluación Parcial N°1**  
**Caso C: Inteligencia musical y predicción de popularidad de canciones**

---

## Descripción del problema de negocio

La industria musical y las plataformas de streaming necesitan anticipar qué canciones tienen mayor probabilidad de tener buen desempeño, para decidir dónde invertir en marketing, en qué playlists priorizar un lanzamiento o qué temas promocionar.

Hoy esa decisión suele tomarse de forma intuitiva o basada en la reputación del artista, sin un análisis sistemático de las características de la canción.

Este proyecto explora si las características técnicas del audio —como energía, bailabilidad, tempo, entre otras— se relacionan con la popularidad de una canción dentro de Spotify, utilizando el **Spotify Tracks Dataset**.

---

## Objetivos del proyecto

### Objetivo general

Analizar el dataset de canciones de Spotify para evaluar la calidad de los datos y la relación entre las variables de audio y la popularidad, dejando la base preparada para una futura fase de modelado.

### Objetivos específicos

- Diagnosticar la calidad del dataset: valores nulos, duplicados y outliers.
- Analizar la distribución de las variables numéricas y su relación con `popularity`.
- Evaluar sesgos éticos y problemas de representatividad presentes en los datos.

---

## Definición de KPIs que resolverán el problema de negocio

> **Nota:** en esta evaluación no se realiza modelamiento. Por ello, los KPIs definidos corresponden a indicadores de calidad y viabilidad para la etapa actual del proyecto.

| KPI | Meta | Resultado obtenido |
|---|---|---|
| Registros utilizables después de limpieza | > 96% | **99,9991%** — 113.999 de 114.000 registros conservados |
| Variables de audio asociadas a popularidad | Identificar factores con asociación relevante | **0 variables** — todas las variables auditivas presentan correlaciones muy débiles |
| Consistencia de `popularity = 0` | Evaluar la confiabilidad de la variable objetivo | **25% vs. 7%** — la ocurrencia es aproximadamente 3,6 veces mayor en `track_id` repetidos |
| Errores críticos después de limpieza | 0 errores críticos pendientes | **0 errores** — sin nulos, duplicados exactos ni registros con duración cero |

---

## Descripción de las fuentes de datos utilizadas

- **Fuente:** Spotify Tracks Dataset, publicado en Kaggle por el usuario `maharshipandya`.
- **Formato:** CSV — dato estructurado.
- **Dimensiones:** 114.000 filas × 20 columnas.
- **Herramientas colaborativas:**
  - **Google Colab:** utilizado para desarrollar y ejecutar el notebook sin requerir instalación local.
  - **GitHub:** utilizado como repositorio de versionado y trabajo en equipo.
- **Reproducibilidad:** el notebook descarga el dataset directamente desde el repositorio del equipo mediante `wget`, por lo que puede ejecutarse de principio a fin sin pasos manuales de carga.
- **Variables principales:**
  - Identificación: `track_id`, `artists`, `album_name`, `track_name`, `track_genre`.
  - Variable objetivo: `popularity`, escala de 0 a 100.
  - Variables de audio: `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `time_signature`, `duration_ms`, `explicit`.

---

## Reproducción del proyecto

1. Abrir `notebooks/notebook_ejecutable.ipynb` en Google Colab o Jupyter Notebook.
2. Ejecutar las celdas en el orden definido.
3. El dataset se descarga automáticamente desde el repositorio mediante `wget`.
4. No se requieren pasos manuales adicionales para cargar los datos.
5. El notebook contiene el código y las visualizaciones utilizadas para reproducir las etapas descritas en este informe.

---

## Preparación y análisis exploratorio de los datos (EDA)

El detalle completo, con código y gráficos ejecutados, se encuentra en:

`notebooks/notebook_ejecutable.ipynb`

### Calidad de datos

- **Valores faltantes y registro corrupto:** se encontró un único registro —fila 65900— con `artists`, `album_name` y `track_name` nulos y, además, `duration_ms = 0`. Corresponde a una misma fila corrupta, no a dos problemas independientes. Se elimina mediante `.dropna()`, pasando de **114.000 a 113.999 filas**.

- **Duplicados de `track_id`:** se registran 24.259 repeticiones. Corresponden a canciones que pueden encontrarse etiquetadas en varios géneros al mismo tiempo. Por ejemplo, `"Baby Blue - Remastered 2010"` de Badfinger aparece en 9 géneros distintos. Estos registros no se eliminan de forma general porque contienen información real de pertenencia a múltiples géneros.

- **`popularity = 0`:** se identifican 16.019 registros, equivalentes al **14,1%** del dataset. Se investigó si estos valores corresponden a una señal real o a una anomalía relacionada con la estructura del dataset.

### Distribución y correlación

- `popularity` presenta una **media de 33,24**, una **mediana de 35,00** y un sesgo (`skew`) de **0,05**, por lo que su distribución general es prácticamente simétrica.
- Ninguna variable de audio muestra una correlación relevante con `popularity`.
  - Máximo observado: aproximadamente **0,05** en `loudness`.
  - Mínimo observado: aproximadamente **-0,10** en `instrumentalness`.

**Interpretación:** en este análisis exploratorio no se encontraron relaciones lineales fuertes entre una característica de audio individual y la popularidad. Esto sugiere que, para una futura etapa de modelamiento, podrían ser relevantes factores adicionales no presentes en el dataset, como fama del artista, presencia en playlists o estrategias de marketing.

### Outliers de duración — Regla IQR

Se detectaron:

- **273 atípicos inferiores**, asociados principalmente a efectos de sonido, piezas breves y fragmentos musicales.
- **5.344 atípicos superiores**, asociados principalmente a canciones o mezclas extensas, especialmente en géneros electrónicos.

La revisión de los casos extremos indica que corresponden principalmente a contenidos musicalmente plausibles, por lo que **no se eliminan automáticamente del dataset**.

### Sesgo por género

Los géneros con mayor popularidad promedio son:

- `pop-film`: 59,3
- `k-pop`: 57,0

Entre los géneros con menor popularidad promedio se encuentran:

- `iranian`: 2,2
- `romance`: 3,2
- `latin`: 8,3

Estas diferencias deben interpretarse con cautela, ya que pueden reflejar diferencias en representación, audiencia o penetración de Spotify en distintos mercados, y no calidad musical objetiva.

### Variabilidad de popularidad dentro de un mismo artista

Se agruparon las canciones por artista, considerando únicamente artistas con al menos 10 canciones. **2.252 artistas** cumplen este criterio.

Entre los artistas con mayor dispersión en `popularity` aparecen:

- Taylor Swift — `std = 42,8`
- The Weeknd — `std = 42,3`
- Bruno Mars — `std = 42,0`
- Doja Cat — `std = 41,7`
- Shawn Mendes — `std = 41,3`

En contraste, Bad Bunny presenta una mayor consistencia dentro del dataset, con:

- `mean = 87,1`
- `std = 6,0`

Al revisar registros particulares, se encontraron casos como:

- `"Lover"` de Taylor Swift con valores de `popularity` 85 y 0.
- `"You Belong With Me (Taylor's Version)"` con valores 9 y 0.
- 26 canciones de Marvin Gaye registradas con `popularity = 0`.

Estos casos llevaron a investigar la relación entre `popularity = 0` y la repetición de `track_id`.

### Análisis específico de `popularity = 0`

Se comparó la frecuencia de `popularity = 0` entre registros con `track_id` repetido y registros con `track_id` único.

- En registros con `track_id` repetido: aproximadamente **25%** presentan `popularity = 0`.
- En registros con `track_id` único: aproximadamente **7%** presentan `popularity = 0`.

La ocurrencia de `popularity = 0` es, por lo tanto, aproximadamente **3,6 veces mayor** entre los registros repetidos.

Esto demuestra una asociación relevante entre la repetición de `track_id` y los valores cero de popularidad. Sin embargo, **no existe evidencia suficiente para concluir que todos los valores `popularity = 0` sean errores**, por lo que no se eliminan automáticamente y deberán tratarse con cautela en futuras etapas de modelamiento.

---

## Evaluación de sesgos, ética y privacidad

- La variable `popularity` mide comportamiento de consumo dentro de Spotify y **no representa calidad musical objetiva**. Es una métrica propia de la plataforma y está condicionada por su mercado y forma de recopilación de datos.

- Se observan diferencias importantes en popularidad promedio entre géneros. Estas diferencias pueden estar asociadas a representación, penetración de Spotify y comportamiento de sus usuarios, por lo que no deben interpretarse como diferencias de calidad musical.

- `popularity = 0` presenta una asociación importante con registros repetidos. Debido a ello, esta variable requiere especial cautela antes de utilizarse como objetivo de un futuro modelo.

- Un eventual modelo desarrollado con este dataset debería utilizarse únicamente para estimar comportamiento de consumo dentro de Spotify y **no para inferir valor cultural, artístico o social de un género o artista**.

- El dataset no contiene datos personales de usuarios (`PII`), sino metadatos de canciones y artistas. Se recomienda evitar cruces con otras fuentes que puedan generar perfiles de terceros fuera del alcance original del proyecto.

---

## Metodología utilizada — CRISP-DM

| Fase | Estado en esta entrega |
|---|---|
| 1. Comprensión del negocio | Completa |
| 2. Comprensión de los datos | Completa |
| 3. Preparación de los datos | Completa |
| 4. Modelado | Pendiente — Evaluación 2 |
| 5. Evaluación del modelo | Pendiente — Evaluación 2/3 |
| 6. Despliegue | Pendiente — Evaluación 3 |

### Alcance de esta entrega

Esta evaluación cubre las etapas iniciales de CRISP-DM: comprensión del negocio, comprensión de los datos, análisis exploratorio y preparación de los datos.

El entrenamiento, comparación y evaluación de modelos queda fuera del alcance de esta entrega y se abordará en evaluaciones posteriores.

---

## Conclusiones

El análisis permitió identificar y tratar los principales problemas de calidad del dataset, conservar 113.999 de los 114.000 registros originales y documentar anomalías relevantes antes de una futura etapa de modelamiento.

Los principales hallazgos fueron:

- No se encontraron asociaciones lineales fuertes entre variables individuales de audio y `popularity`.
- Se detectó una concentración relevante de `popularity = 0`, especialmente entre registros con `track_id` repetido.
- Los outliers de duración corresponden principalmente a contenidos musicalmente plausibles, por lo que no se eliminaron de forma automática.
- Existen diferencias importantes de popularidad promedio entre géneros que deben interpretarse considerando posibles sesgos de representación y de plataforma.

Con estas consideraciones, el dataset queda documentado y preparado para continuar posteriormente con la etapa de modelamiento.
