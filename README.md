# Actividad formativa 4: Implementación y análisis de modulación AM

**Estudiante:** Israel Erasmo Maldonado Zamora  
**Institución:** Universidad Ciudadana de Nuevo León

## Descripción

Este proyecto implementa una simulación de modulación en amplitud (AM) convencional con portadora. Se genera un mensaje senoidal de 500 Hz y una portadora de 10 kHz, con índice de modulación de 0.70. Se analizan la forma de onda, el espectro, el ruido aditivo, la distorsión por recorte y la atenuación del canal.

## Requisitos

- Python 3.9 o superior
- NumPy
- SciPy
- Matplotlib

Instalación:

```bash
pip install -r requirements.txt
```

## Ejecución

Desde esta carpeta:

```bash
python actividad_am.py --output-dir outputs
```

El programa genera las siguientes salidas:

- `01_senal_y_modulacion.png`: mensaje, portadora y señal AM en el tiempo.
- `02_espectro_am.png`: espectro de frecuencia y bandas laterales.
- `03_ruido.png`: comparación de señal ideal y señal con ruido.
- `04_distorsion.png`: efecto del recorte no lineal.
- `05_atenuacion.png`: efecto de la atenuación del canal.
- `06_mensaje_recuperado.png`: comparación del mensaje original y recuperado.
- `metricas_demodulacion.csv`: métricas de error por escenario.
- `resultados.json`: parámetros y resultados numéricos.

## Modelo matemático

La modulación implementada es:

`s(t) = Ac [1 + μ m(t)] cos(2π fc t)`

con `Ac = 1`, `μ = 0.70`, `fm = 500 Hz` y `fc = 10 kHz`. La señal de información se recupera mediante una estimación ideal de la envolvente basada en la transformada de Hilbert. Para el caso atenuado se utiliza la ganancia conocida del canal como compensación ideal.

## Escenarios analizados

1. Señal ideal.
2. Ruido blanco gaussiano aditivo con SNR objetivo de 15 dB.
3. Distorsión no lineal mediante recorte simétrico a ±1.25.
4. Atenuación lineal con ganancia 0.50.

La semilla aleatoria está fijada en el código para facilitar la reproducción de los resultados.

## Publicación en GitHub

El proyecto está preparado para subirlo a un repositorio. Desde la carpeta del proyecto, después de crear el repositorio en GitHub:

```bash
git init
git add .
git commit -m "Implementacion y analisis de modulacion AM"
git branch -M main
git remote add origin https://github.com/USUARIO/actividad-am-python.git
git push -u origin main
```

Sustituye `USUARIO` por tu nombre de usuario real de GitHub.

## Archivos

- `actividad_am.py`: código fuente de la simulación.
- `requirements.txt`: dependencias de Python.
- `outputs/`: gráficas y métricas generadas.
- `Informe_Actividad_Formativa_4_AM.docx`: informe de la actividad.
- `Presentacion_Actividad_Formativa_4_AM.pptx`: presentación de apoyo.
