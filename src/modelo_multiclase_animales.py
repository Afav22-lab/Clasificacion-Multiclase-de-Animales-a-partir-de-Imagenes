"""Línea base para clasificación multiclase de animales con TensorFlow/Keras."""

from __future__ import annotations

import argparse

import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE


def construir_modelo(num_clases: int, img_size: int) -> tf.keras.Model:
    modelo = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(img_size, img_size, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(num_clases, activation="softmax"),
        ]
    )

    modelo.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
        ],
    )
    return modelo


def _pipeline_dataset(dataset: tf.data.Dataset, augment: bool) -> tf.data.Dataset:
    normalizacion = tf.keras.layers.Rescaling(1.0 / 255)
    aumento = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ]
    )

    def transformacion(x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x = tf.cast(x, tf.float32)
        if augment:
            x = aumento(x, training=True)
        x = normalizacion(x)
        return x, y

    return dataset.map(transformacion, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


def cargar_datasets(
    train_dir: str,
    val_dir: str,
    img_size: int,
    batch_size: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )
    class_names = train_ds.class_names
    return _pipeline_dataset(train_ds, augment=True), _pipeline_dataset(val_ds, augment=False), class_names


def matriz_confusion(modelo: tf.keras.Model, test_ds: tf.data.Dataset) -> tf.Tensor:
    y_true_batches = []
    y_pred_batches = []
    for x_batch, y_batch in test_ds:
        predicciones = modelo(x_batch, training=False)
        y_true_batches.append(tf.argmax(y_batch, axis=1))
        y_pred_batches.append(tf.argmax(predicciones, axis=1))

    y_true = tf.concat(y_true_batches, axis=0)
    y_pred = tf.concat(y_pred_batches, axis=0)
    return tf.math.confusion_matrix(y_true, y_pred)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento base: clasificación multiclase de animales.")
    parser.add_argument("--train-dir", required=True, help="Directorio de entrenamiento (subcarpetas por clase).")
    parser.add_argument("--val-dir", required=True, help="Directorio de validación (subcarpetas por clase).")
    parser.add_argument("--test-dir", help="Directorio de prueba (subcarpetas por clase).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()

    train_ds, val_ds, class_names = cargar_datasets(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )

    modelo = construir_modelo(num_clases=len(class_names), img_size=args.img_size)
    modelo.fit(train_ds, validation_data=val_ds, epochs=args.epochs)
    metricas = modelo.evaluate(val_ds, verbose=0)
    print("Métricas de validación:", dict(zip(modelo.metrics_names, metricas)))

    if args.test_dir:
        test_ds = tf.keras.utils.image_dataset_from_directory(
            args.test_dir,
            image_size=(args.img_size, args.img_size),
            batch_size=args.batch_size,
            label_mode="categorical",
            shuffle=False,
        )
        test_ds = _pipeline_dataset(test_ds, augment=False)
        print("Matriz de confusión:")
        print(matriz_confusion(modelo, test_ds).numpy())


if __name__ == "__main__":
    main()
