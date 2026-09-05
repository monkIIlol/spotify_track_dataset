# Informe Técnico — Predicción de Popularidad de Canciones (Spotify Tracks Dataset)

MLY1101 — Machine Learning | Evaluación Parcial N°1 | **Caso C**: Inteligencia musical y predicción de popularidad de canciones

## Descripción del problema de negocio

La industria musical y las plataformas de streaming necesitan anticipar qué canciones tienen mayor probabilidad de tener buen desempeño, para decidir dónde invertir en marketing, en qué playlists priorizar un lanzamiento o qué temas promocionar. Hoy esa decisión suele tomarse de forma intuitiva o basada en la reputación del artista, sin un análisis sistemático de las características de la canción. Este proyecto explora si las características técnicas del audio (energía, bailabilidad, tempo, etc.) explican la popularidad de una canción dentro de Spotify, usando el Spotify Tracks Dataset.

## Objetivos del proyecto

**General:** analizar el dataset de canciones de Spotify para evaluar la calidad de los datos y la relación entre las variables de audio y la popularidad, dejando la base preparada para una futura fase de modelado.

**Específicos:**
- Diagnosticar la calidad del dataset (nulos, duplicados, outliers).
- Analizar la distribución de las variables numéricas y su relación con `popularity`.
- Evaluar sesgos éticos y de representatividad presentes en los datos.

## Definición de KPIs que resolverán el problema de negocio

| KPI | Meta | Resultado obtenido |
|---|---|---|
| % de registros con valores faltantes tras la limpieza | < 1% | 0.0009% (1 de 114.000 registros) |
| Outliers de duración identificados y caracterizados | Detectar, separar por dirección y justificar tratamiento | 273 atípicos inferiores + 5.344 atípicos superiores |
| Correlación máxima entre variables de audio y `popularity` | Referencia para evaluar viabilidad de un modelo basado solo en audio | 0.05 (`loudness`) → correlación muy débil |
| Confiabilidad de `popularity = 0` | Determinar si es señal real o ruido de captura | `popularity = 0` aparece significativamente más en registros con `track_id` repetido (25.3% vs. 7.7%), lo que sugiere una anomalía asociada a duplicaciones/versiones del catálogo, pero no permite concluir que todos los valores cero sean errores. |
| Representatividad por género | Verificar balance del dataset | 114 géneros, 1.000 canciones por género (balanceado por diseño) |

## Descripción de las fuentes de datos utilizadas

- **Fuente:** Spotify Tracks Dataset, publicado en Kaggle por el usuario *maharshipandya* ([enlace](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)).
- **Formato:** CSV, dato estructurado.
- **Dimensiones:** 114.000 filas × 20 columnas.
- **Herramientas colaborativas:** Google Colab para desarrollo y ejecución sin instalación local, y GitHub como repositorio de versionado y trabajo en equipo. El notebook descarga el dataset directamente desde el repositorio del equipo (`!wget`), por lo que se ejecuta de principio a fin sin pasos manuales.
- **Variables:** identificación (`track_id`, `artists`, `album_name`, `track_name`, `track_genre`), variable objetivo (`popularity`, escala 0-100) y variables de audio (`danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `time_signature`, `duration_ms`, `explicit`).

## Preparación y análisis exploratorio de los datos (EDA)

El detalle completo, con código y gráficos ejecutados, está en `notebooks/notebook_ejecutable.ipynb`.

### Calidad de datos

- **Valores faltantes y registro corrupto:** un único registro (fila 65900) con `artists`, `album_name`, `track_name` nulos y además `duration_ms = 0`. Es la misma fila corrupta, no dos problemas distintos. Se elimina con `.dropna()` (114.000 → 113.999 filas).
- **Duplicados de `track_id`:** se repite 24.259 veces; corresponde a la misma canción etiquetada en varios géneros a la vez (ej. *"Baby Blue - Remastered 2010"* de Badfinger aparece en 9 géneros distintos). No se eliminan a nivel de dataset completo porque representan información real de multi-género.
- **`popularity = 0`:** 16.019 canciones (14.1%). Se investigó si es una señal real o un artefacto — ver KPI correspondiente y sección de Ética.

### Distribución y correlación

- `popularity`: media 33.24, mediana 35.00, sesgo (skew) de 0.05 → distribución prácticamente simétrica.
- Ninguna variable de audio muestra correlación relevante con `popularity` (máximo 0.05 en `loudness`, mínimo -0.10 en `instrumentalness`). **Conclusión de negocio:** la popularidad no se explica principalmente por el audio, sino probablemente por factores externos (fama del artista, presencia en playlists, marketing).

### Outliers de duración (regla IQR)

Se detectaron 273 atípicos inferiores (efectos de sonido y piezas breves, ej. género `iranian` y fragmentos de música clásica) y 5.344 atípicos superiores (mezclas continuas de DJ en géneros electrónicos como `house` y `techno`). Ninguno es un error de captura: son diferencias legítimas según el tipo de contenido, por lo que no se filtran del dataset.

### Sesgo por género

Los géneros con mayor popularidad promedio son `pop-film` (59.3) y `k-pop` (57.0); los de menor son `iranian` (2.2), `romance` (3.2) y `latin` (8.3). Esta brecha probablemente refleja la penetración de mercado de Spotify más que la calidad musical real (ver sección de Ética).

### Variabilidad de popularidad dentro de un mismo artista

Se agruparon las canciones por artista (mínimo 10 canciones, 2.252 artistas cumplen el filtro), calculando promedio y desviación estándar de `popularity`. Hallazgo relevante: los artistas con mayor dispersión no son artistas irregulares, son superestrellas — Taylor Swift (`std=42.8`), The Weeknd (`std=42.3`), Bruno Mars (`std=42.0`), Doja Cat (`std=41.7`) y Shawn Mendes (`std=41.3`), todos con canciones que van de 0 hasta más de 85 de popularidad. En contraste, Bad Bunny muestra alta consistencia (`mean=87.1`, `std=6.0`).

Revisando el detalle, se encontró que la propia canción de Taylor Swift *"Lover"* aparece dos veces en el dataset con valores de `popularity` de 85 y 0, y *"You Belong With Me (Taylor's Version)"* aparece con 9 y 0. Esto, junto con el caso de Marvin Gaye (26 canciones en el dataset, las 26 con `popularity = 0`, algo implausible para un artista de ese nivel), llevó a investigar directamente si `popularity = 0` está asociado a registros duplicados — confirmado en el KPI correspondiente.

## Evaluación de sesgos, ética y privacidad

- La variable `popularity` mide consumo dentro de Spotify, no calidad musical objetiva — es una métrica de plataforma, sujeta a su penetración de mercado.
- Se detectó que géneros como `iranian`, `romance` y `latin` tienen popularidad promedio muy inferior a géneros como `pop-film` o `k-pop`. Esto probablemente refleja menor penetración histórica de Spotify en esos mercados, no menor calidad musical real — un caso de **sesgo histórico/de selección** heredado de cómo la plataforma recolectó y expuso los datos.
- Adicionalmente, se comprobó que `popularity = 0` no es una medición confiable de bajo consumo real, sino que está asociado a duplicados y reediciones del catálogo. Tratar estos valores como ceros reales sin una variable indicadora habría sesgado un futuro modelo, penalizando artificialmente a artistas y géneros con más reediciones en su catálogo.
- **Recomendación:** un modelo entrenado sobre este dataset no debe usarse para inferir "valor cultural" de un género, solo para estimar comportamiento de consumo dentro de Spotify, dejando ambas limitaciones (sesgo de plataforma y ruido en `popularity=0`) explícitas en cualquier reporte o producto derivado.
- El dataset no contiene datos personales de usuarios (PII), solo metadata pública de canciones y artistas. Se recomienda evitar cruces con otras fuentes que pudieran generar perfiles de terceros fuera del alcance original del dataset.

## Metodología utilizada (CRISP-DM)


| Fase | Estado en esta entrega |
|---|---|
| 1. Comprensión del negocio | Completa |
| 2. Comprensión de los datos | Completa |
| 3. Preparación de los datos | Completa |
| 4. Modelado | Pendiente — Evaluación 2 |
| 5. Evaluación del modelo | Pendiente — Evaluación 2/3 |
| 6. Despliegue | Pendiente — Evaluación 3 |
