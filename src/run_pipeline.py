import os
import sys

def main():
    print("=" * 70)
    print("PROYECTO ACADEMICO TINYML:")
    print("Clasificacion de sonidos cotidianos mediante un modelo TinyML ultracompacto")
    print("=" * 70)

    # 1. Verificacion del dataset
    from verify_dataset import verify_dataset
    verify_dataset()

    # 2. Exploracion de datos y generacion de figuras iniciales
    from explore_data import plot_class_distribution, plot_waveforms, plot_spectrograms
    print("\n[Paso 2] Generando graficas de exploracion (distribucion, ondas, espectrogramas)...")
    plot_class_distribution()
    plot_waveforms()
    plot_spectrograms()

    # 3. Entrenamiento del modelo FP32
    from train import train_model
    print("\n[Paso 3] Entrenando Tiny CNN...")
    train_model()

    # 4. Evaluacion del modelo en el conjunto de prueba
    from evaluate import evaluate_model
    print("\n[Paso 4] Evaluando modelo y generando matriz de confusion...")
    evaluate_model()

    # 5. Conversion a TFLite y Cuantizacion INT8
    from quantization import run_quantization_and_comparison
    print("\n[Paso 5] Cuantizando a INT8 y comparando con FP32...")
    run_quantization_and_comparison()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETADO CON EXITO")
    print("Todos los modelos y graficas han sido generados en 'models/' y 'results/'")
    print("=" * 70)

if __name__ == "__main__":
    main()
