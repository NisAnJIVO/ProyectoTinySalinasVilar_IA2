import os
import warnings
import logging

# Forzar modo CPU y suprimir advertencias de GPU/oneDNN
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score

from preprocess import get_train_val_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def evaluate_tflite_model(model_path, X_test, y_test):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    preds = []
    for i in range(len(X_test)):
        sample = np.expand_dims(X_test[i], axis=0)
        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        preds.append(np.argmax(output))

    acc = float(accuracy_score(y_test, preds))
    return acc, preds

def run_quantization_and_comparison():
    print("=" * 60)
    print("FASE 10, 11 Y 12: CONVERSION A TFLITE Y CUANTIZACION INT8")
    print("=" * 60)

    (X_train, _), _, (X_test, y_test) = get_train_val_test_split()

    keras_model_path = os.path.join(MODELS_DIR, "model_fp32.keras")
    if not os.path.exists(keras_model_path):
        raise FileNotFoundError(f"No se encontro el modelo Keras: {keras_model_path}")

    model = tf.keras.models.load_model(keras_model_path)

    # 1. Conversion a TFLite FP32
    print("\n--- 1. Convirtiendo a TensorFlow Lite (FP32) ---")
    conv_fp32 = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32_content = conv_fp32.convert()

    fp32_path = os.path.join(MODELS_DIR, "model_fp32.tflite")
    with open(fp32_path, "wb") as f:
        f.write(tflite_fp32_content)

    fp32_bytes = os.path.getsize(fp32_path)
    fp32_kb = fp32_bytes / 1024.0
    print(f"Modelo FP32 guardado: {fp32_path}")
    print(f"Tamano FP32 : {fp32_bytes:,} bytes ({fp32_kb:.2f} KB)")

    # 2. Cuantizacion INT8 con dataset representativo
    print("\n--- 2. Aplicando cuantizacion INT8 con representative dataset ---")
    def representative_dataset_gen():
        for i in range(min(60, len(X_train))):
            yield [np.expand_dims(X_train[i], axis=0).astype(np.float32)]

    conv_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
    conv_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    conv_int8.representative_dataset = representative_dataset_gen
    tflite_int8_content = conv_int8.convert()

    int8_path = os.path.join(MODELS_DIR, "model_int8.tflite")
    with open(int8_path, "wb") as f:
        f.write(tflite_int8_content)

    int8_bytes = os.path.getsize(int8_path)
    int8_kb = int8_bytes / 1024.0
    print(f"Modelo INT8 guardado: {int8_path}")
    print(f"Tamano INT8 : {int8_bytes:,} bytes ({int8_kb:.2f} KB)")

    # Reduccion de tamano
    abs_reduction_bytes = fp32_bytes - int8_bytes
    abs_reduction_kb = abs_reduction_bytes / 1024.0
    pct_reduction = (abs_reduction_bytes / fp32_bytes) * 100.0

    print("\n--- Metricas de Reduccion de Tamano ---")
    print(f"Reduccion absoluta   : {abs_reduction_bytes:,} bytes ({abs_reduction_kb:.2f} KB)")
    print(f"Reduccion porcentual : {pct_reduction:.2f}%")

    # 3. Evaluacion comparativa en el conjunto de prueba
    print("\n--- 3. Evaluacion sobre el conjunto de prueba ---")
    acc_fp32, _ = evaluate_tflite_model(fp32_path, X_test, y_test)
    acc_int8, _ = evaluate_tflite_model(int8_path, X_test, y_test)
    acc_diff = acc_int8 - acc_fp32

    print(f"Accuracy Modelo FP32 : {acc_fp32:.4f} ({acc_fp32 * 100:.2f}%)")
    print(f"Accuracy Modelo INT8 : {acc_int8:.4f} ({acc_int8 * 100:.2f}%)")
    print(f"Diferencia Accuracy  : {acc_diff:+.4f} ({acc_diff * 100:+.2f}%)")

    # 4. Grafica comparativa de tamano
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["FP32", "INT8"], [fp32_kb, int8_kb], color=["#2b5c8f", "#27ae60"], width=0.45, edgecolor="black")
    ax.set_title("Comparacion del Tamano del Modelo (FP32 vs INT8)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Tamano (KB)", fontsize=11)
    ax.set_ylim(0, max(fp32_kb, int8_kb) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f} KB",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.annotate(f"Reduccion:\n-{pct_reduction:.1f}%",
                xy=(1, int8_kb),
                xytext=(0.5, fp32_kb * 0.6),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5),
                fontsize=11, fontweight="bold", color="#c0392b", ha="center")

    plt.tight_layout()
    size_plot_path = os.path.join(RESULTS_DIR, "model_size_comparison.png")
    plt.savefig(size_plot_path, dpi=300)
    plt.close()
    print(f"\nGrafica guardada: {size_plot_path}")

    # 5. Grafica comparativa de Accuracy
    fig, ax = plt.subplots(figsize=(6, 5))
    bars_acc = ax.bar(["FP32", "INT8"], [acc_fp32 * 100, acc_int8 * 100], color=["#2b5c8f", "#27ae60"], width=0.45, edgecolor="black")
    ax.set_title("Comparacion de Precision en Test (FP32 vs INT8)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars_acc:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    acc_plot_path = os.path.join(RESULTS_DIR, "model_accuracy_comparison.png")
    plt.savefig(acc_plot_path, dpi=300)
    plt.close()
    print(f"Grafica guardada: {acc_plot_path}")

    # 6. Guardar resumen consolidado en CSV
    csv_path = os.path.join(RESULTS_DIR, "results_summary.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "format", "accuracy", "size_bytes", "size_kb"])
        writer.writerow(["TinyCNN_FP32", "FP32", f"{acc_fp32:.4f}", fp32_bytes, f"{fp32_kb:.2f}"])
        writer.writerow(["TinyCNN_INT8", "INT8", f"{acc_int8:.4f}", int8_bytes, f"{int8_kb:.2f}"])

    print(f"Resumen guardado en: {csv_path}")

    return {
        "fp32_bytes": fp32_bytes,
        "fp32_kb": fp32_kb,
        "int8_bytes": int8_bytes,
        "int8_kb": int8_kb,
        "pct_reduction": pct_reduction,
        "acc_fp32": acc_fp32,
        "acc_int8": acc_int8
    }

if __name__ == "__main__":
    run_quantization_and_comparison()
