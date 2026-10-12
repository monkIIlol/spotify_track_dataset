# Spotify Tracks — MLY1101

Proyecto semestral de **Machine Learning (MLY1101)** desarrollado sobre el **Spotify Tracks Dataset**.

**Integrantes**:  - Diego Torres
                  - Vicente Tapia
                  - Cristian Urcullú

La Evaluación Parcial N°2 continúa el trabajo realizado en la EVA1 e incorpora modelamiento y evaluación mediante:

- **Regresión Lineal** y **Árbol de Decisión** para estimar la `popularity` observada.
- **K-Means** para identificar perfiles sonoros dentro del catálogo.
- **PCA** como apoyo para representar e interpretar los perfiles obtenidos.

## Resultados principales

- Baseline: `MAE = 17,12`.
- Regresión Lineal con audio + estructura + género: `MAE = 11,83`, `R² = 0,35`.
- Árbol de Decisión: `MAE = 11,16`, `R² = 0,37`.
- K-Means: `K = 4`, con `Silhouette ≈ 0,17`.
- Los dos primeros componentes de PCA explican aproximadamente el `48%` de la variabilidad.

El principal hallazgo supervisado es que **la información de género aporta una mejora mucho mayor que el cambio entre los dos algoritmos evaluados**. En paralelo, las características de audio aportan poca capacidad para estimar `popularity`, pero sí permiten describir perfiles sonoros del catálogo.

## Estructura

```text
spotify_track_dataset/
├── README.md
├── requirements.txt
├── data/
│   └── Spotify_Tracks_Dataset.csv
├── notebooks/
│   └── Eva2_Spotify.ipynb
├── images/
│   └── gráficos generados por el notebook
├── models/
│   └── modelos y métricas generados
└── report/
    ├── Informe_Tecnico_Spotify_EP1.md
    └── Informe_Tecnico_Spotify_EVA2.md
```

## Ejecución en PC

Desde la raíz del repositorio:

```bash
python -m pip install -r requirements.txt
```

Luego abrir `notebooks/Eva2_Spotify.ipynb` en VS Code o Jupyter Notebook y ejecutar todas las celdas.

El notebook detecta automáticamente la raíz del proyecto y utiliza:

- `data/` para el dataset;
- `images/` para los gráficos;
- `models/` para los modelos y métricas.

Si el dataset no existe localmente, puede descargarlo automáticamente desde este repositorio.

La ejecución local fue comprobada en **Windows con Python 3.12.6**.

## Ejecución en Google Colab

El mismo archivo `notebooks/Eva2_Spotify.ipynb` puede abrirse y ejecutarse en Google Colab. Si el dataset no está presente en el entorno, el notebook lo descarga automáticamente.

La ejecución completa también fue comprobada en un entorno limpio de Colab.

## Informes

- [Informe técnico EVA1](report/Informe_Tecnico_Spotify_EP1.md)
- [Informe técnico EVA2](report/Informe_Tecnico_Spotify_EVA2.md)

## Fuente de datos

Se utiliza el **Spotify Tracks Dataset**, publicado originalmente en Kaggle por `maharshipandya`. El archivo utilizado por el proyecto se mantiene en `data/Spotify_Tracks_Dataset.csv`.
