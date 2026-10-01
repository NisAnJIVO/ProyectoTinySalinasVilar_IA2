import os
import warnings
import logging

# Forzar modo CPU y suprimir advertencias de GPU/oneDNN
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from preprocess import get_train_val_test_split
from model import create_tiny_cnn

# Fijar semillas para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_model():
    print("=" * 60)
    print("FASE 7: ENTRENAMIENTO DEL MODELO")
    print("=" * 60)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_train_val_test_split()

    model = create_tiny_cnn(input_shape=(64, 64, 1), num_classes=4)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
        verbose=1
    )

    epochs = 40
    batch_size = 16

    print(f"Iniciando entrenamiento por hasta {epochs} epocas (batch size: {batch_size})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )

    # Guardar modelo en formato Keras
    model_save_path = os.path.join(MODELS_DIR, "model_fp32.keras")
    model.save(model_save_path)
    print(f"Modelo guardado exitosamente en: {model_save_path}")

    # Generar grafica de Accuracy
    plt.figure(figsize=(8, 5))
    epochs_range = range(1, len(history.history["accuracy"]) + 1)
    plt.plot(epochs_range, history.history["accuracy"], label="Training Accuracy", color="#1f77b4", linewidth=2)
    plt.plot(epochs_range, history.history["val_accuracy"], label="Validation Accuracy", color="#ff7f0e", linewidth=2, linestyle="--")
    plt.title("Evolucion de Accuracy (Entrenamiento vs Validacion)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Epocas", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    acc_plot_path = os.path.join(RESULTS_DIR, "training_validation_accuracy.png")
    plt.savefig(acc_plot_path, dpi=300)
    plt.close()
    print(f"Grafica guardada: {acc_plot_path}")

    # Generar grafica de Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history.history["loss"], label="Training Loss", color="#1f77b4", linewidth=2)
    plt.plot(epochs_range, history.history["val_loss"], label="Validation Loss", color="#d62728", linewidth=2, linestyle="--")
    plt.title("Evolucion de Loss (Entrenamiento vs Validacion)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Epocas", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    loss_plot_path = os.path.join(RESULTS_DIR, "training_validation_loss.png")
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    print(f"Grafica guardada: {loss_plot_path}")

    return model, history

if __name__ == "__main__":
    train_model()
