# Clasificación Multiclase de Animales a partir de Imágenes

Este repositorio documenta y provee una línea base para resolver una tarea de **clasificación multiclase** de animales con aprendizaje profundo.

## Definición

La clasificación multiclase asigna **una única etiqueta** por imagen, donde cada etiqueta representa una especie o categoría de animal (por ejemplo: `perro`, `gato`, `elefante`, `tigre`).

## Características principales

- **Entrada:** imágenes digitales de animales.
- **Salida:** una clase única por imagen.
- **Número de clases:** desde pocas categorías hasta cientos o miles.

## Arquitectura base implementada

La implementación en `src/modelo_multiclase_animales.py` incluye:

1. **Preprocesamiento**
   - Redimensionado de imágenes.
   - Normalización de píxeles a `[0, 1]`.
   - Aumento de datos (volteo horizontal, rotación y contraste).
2. **Extracción de características (CNN)**
   - `Conv2D(32) + MaxPooling2D`
   - `Conv2D(64) + MaxPooling2D`
3. **Capas finales**
   - `Flatten`
   - `Dense(128, relu)`
   - `Dense(num_clases, softmax)`
4. **Entrenamiento**
   - Optimizador: `adam`
   - Pérdida: `categorical_crossentropy`
   - Métricas: `accuracy`, `precision`, `recall`, `top-k accuracy` (k adaptable hasta 3 según el número de clases)

## Retos típicos de este problema

- Alta similitud entre clases.
- Gran variabilidad dentro de la misma clase.
- Oclusiones y fondos complejos.
- Desequilibrio de clases.

## Métricas de evaluación recomendadas

- Accuracy.
- Precision, Recall y F1-score (por clase y promedios macro/micro).
- Matriz de confusión.
- Top-k accuracy.

## Uso rápido

```bash
python src/modelo_multiclase_animales.py \
  --train-dir /ruta/train \
  --val-dir /ruta/val \
  --test-dir /ruta/test \
  --epochs 10 \
  --batch-size 32 \
  --img-size 224
```

> Las carpetas deben organizarse por clase (formato compatible con `image_dataset_from_directory` de TensorFlow/Keras).
