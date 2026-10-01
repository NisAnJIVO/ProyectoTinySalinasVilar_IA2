import os
import warnings
import logging

# Forzar modo CPU y suprimir advertencias de GPU/oneDNN
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf
from tensorflow.keras import layers, models

def create_tiny_cnn(input_shape=(64, 64, 1), num_classes=4):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Primer bloque convolucional
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Segundo bloque convolucional
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Regularizacion ligera para prevenir sobreajuste
        layers.Dropout(0.2),

        # Aplanamiento y clasificador
        layers.Flatten(),
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def print_model_info():
    model = create_tiny_cnn()
    model.summary()

    total_params = model.count_params()
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)

    print("=" * 60)
    print("DETALLES DE LA ARQUITECTURA TINYML CNN")
    print("=" * 60)
    print(f"Total de parametros      : {total_params:,}")
    print(f"Parametros entrenables   : {trainable_params:,}")
    print(f"Tamano estimado en FP32  : {total_params * 4 / 1024:.2f} KB")
    print(f"Tamano estimado en INT8  : {total_params * 1 / 1024:.2f} KB")
    print("=" * 60)

if __name__ == "__main__":
    print_model_info()
