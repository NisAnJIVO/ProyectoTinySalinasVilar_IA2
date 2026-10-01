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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

from preprocess import get_train_val_test_split, CLASSES

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def evaluate_model():
    print("=" * 60)
    print("FASE 8 Y 9: EVALUACION DEL MODELO FP32")
    print("=" * 60)

    _, _, (X_test, y_test) = get_train_val_test_split()

    model_path = os.path.join(MODELS_DIR, "model_fp32.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontro el modelo: {model_path}")

    model = tf.keras.models.load_model(model_path)

    # Predicciones sobre el conjunto de prueba
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Metricas
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")

    print(f"\nAccuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")

    print("\nReporte de Clasificacion:")
    print("-" * 60)
    report = classification_report(y_test, y_pred, target_names=CLASSES, digits=4)
    print(report)

    # Matriz de Confusion
    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusion (valores absolutos):")
    print(cm)

    # Grafica de Matriz de Confusion
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        title="Matriz de Confusion (Conjunto de Prueba - FP32)",
        ylabel="Clase Real",
        xlabel="Clase Predicha"
    )

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold", fontsize=12)

    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"\nGrafica guardada: {cm_path}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cm": cm
    }

if __name__ == "__main__":
    evaluate_model()
