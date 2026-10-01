import json
import nbformat as nbf

def generate_notebook():
    nb = nbf.v4.new_notebook()

    cells = [
        nbf.v4.new_markdown_cell(
            "# Clasificacion de Sonidos Cotidianos mediante un Modelo TinyML Ultracompacto\n\n"
            "**Proyecto Academico de TinyML**\n\n"
            "Este notebook presenta el flujo completo de ingenieria para clasificar 4 sonidos cotidianos "
            "del dataset ESC-50 (`clapping`, `clock_alarm`, `door_wood_knock`, `keyboard_typing`), "
            "utilizando una Red Neuronal Convolucional (CNN) ultracompacta disenada para microcontroladores "
            "y dispositivos de borde, aplicando cuantizacion INT8 con TensorFlow Lite."
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Verificacion del Dataset y Comprobacion de Archivos\n"
            "Comprobamos la presencia de los 160 archivos WAV (40 archivos por clase), "
            "verificando frecuencia de muestreo (44.1 kHz), canales mono y duracion (5.0 s)."
        ),
        nbf.v4.new_code_cell(
            "import os\n"
            "import sys\n"
            "import warnings\n"
            "import logging\n"
            "import importlib\n\n"
            "# Configurar ejecucion en CPU y silenciar avisos de GPU/oneDNN\n"
            "os.environ['CUDA_VISIBLE_DEVICES'] = '-1'\n"
            "os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'\n"
            "os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'\n"
            "warnings.filterwarnings('ignore')\n"
            "logging.getLogger('tensorflow').setLevel(logging.ERROR)\n\n"
            "# Asegurar que el directorio de trabajo apunte a la raiz del proyecto\n"
            "if os.path.basename(os.getcwd()) == 'notebooks':\n"
            "    os.chdir('..')\n\n"
            "# Incluir carpeta src en el path de Python\n"
            "src_dir = os.path.abspath('src')\n"
            "if src_dir not in sys.path:\n"
            "    sys.path.append(src_dir)\n\n"
            "# Importar y recargar verify_dataset para evitar modulos cacheados\n"
            "import verify_dataset\n"
            "importlib.reload(verify_dataset)\n"
            "verify_dataset.verify_dataset()"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Exploracion de Datos y Visualizacion de Ondas Sonoras\n"
            "Generamos la grafica de distribucion uniforme de clases y comparamos la firma "
            "temporal de las formas de onda."
        ),
        nbf.v4.new_code_cell(
            "from IPython.display import Image, display\n"
            "import explore_data\n"
            "importlib.reload(explore_data)\n\n"
            "explore_data.plot_class_distribution()\n"
            "explore_data.plot_waveforms()\n\n"
            "res_dir = 'results' if os.path.exists('results') else os.path.join('..', 'results')\n"
            "display(Image(filename=os.path.join(res_dir, 'class_distribution.png')))\n"
            "display(Image(filename=os.path.join(res_dir, 'waveform_sample.png')))\n"
            "display(Image(filename=os.path.join(res_dir, 'waveform_comparison.png')))"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Conversion de Audio a Espectrogramas (STFT)\n"
            "Convertimos las senales a representaciones bidimensionales de tiempo vs frecuencia en escala logaritmica (dB)."
        ),
        nbf.v4.new_code_cell(
            "explore_data.plot_spectrograms()\n"
            "res_dir = 'results' if os.path.exists('results') else os.path.join('..', 'results')\n"
            "display(Image(filename=os.path.join(res_dir, 'spectrograms.png')))"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Pipeline de Preprocesamiento y Division Estratificada\n"
            "Re-muestreo a 16 kHz, extraccion de espectrograma 64x64, normalizacion Min-Max [0, 1] "
            "y division de datos en 70% entrenamiento (112), 15% validacion (24) y 15% prueba (24)."
        ),
        nbf.v4.new_code_cell(
            "import preprocess\n"
            "importlib.reload(preprocess)\n\n"
            "(X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocess.get_train_val_test_split()\n"
            "print(f'Dimensiones X_train : {X_train.shape}, y_train: {y_train.shape}')\n"
            "print(f'Dimensiones X_val   : {X_val.shape}, y_val: {y_val.shape}')\n"
            "print(f'Dimensiones X_test  : {X_test.shape}, y_test: {y_test.shape}')"
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Arquitectura de la TinyML CNN\n"
            "Definicion de la arquitectura compacta adecuada para microcontroladores (2 capas Conv2D + MaxPooling + Dense)."
        ),
        nbf.v4.new_code_cell(
            "import model\n"
            "importlib.reload(model)\n"
            "model.print_model_info()"
        ),
        nbf.v4.new_markdown_cell(
            "## 6. Curvas de Aprendizaje (Accuracy y Loss)\n"
            "Analisis de la convergencia y regularizacion durante el entrenamiento."
        ),
        nbf.v4.new_code_cell(
            "res_dir = 'results' if os.path.exists('results') else os.path.join('..', 'results')\n"
            "display(Image(filename=os.path.join(res_dir, 'training_validation_accuracy.png')))\n"
            "display(Image(filename=os.path.join(res_dir, 'training_validation_loss.png')))"
        ),
        nbf.v4.new_markdown_cell(
            "## 7. Evaluacion en el Conjunto de Prueba y Matriz de Confusion\n"
            "Evaluacion rigurosa de metricas ponderadas y deteccion de patrones acusticos confusos."
        ),
        nbf.v4.new_code_cell(
            "import evaluate\n"
            "importlib.reload(evaluate)\n"
            "metrics = evaluate.evaluate_model()\n"
            "res_dir = 'results' if os.path.exists('results') else os.path.join('..', 'results')\n"
            "display(Image(filename=os.path.join(res_dir, 'confusion_matrix.png')))"
        ),
        nbf.v4.new_markdown_cell(
            "## 8. Conversion a TensorFlow Lite y Cuantizacion INT8\n"
            "Optimizacion post-entrenamiento de FP32 (32 bits) a INT8 (8 bits enteros) con dataset representativo."
        ),
        nbf.v4.new_code_cell(
            "import quantization\n"
            "importlib.reload(quantization)\n"
            "summary = quantization.run_quantization_and_comparison()\n"
            "res_dir = 'results' if os.path.exists('results') else os.path.join('..', 'results')\n"
            "display(Image(filename=os.path.join(res_dir, 'model_size_comparison.png')))\n"
            "display(Image(filename=os.path.join(res_dir, 'model_accuracy_comparison.png')))"
        ),
        nbf.v4.new_markdown_cell(
            "## 9. Resultados Consolidados\n"
            "Lectura del archivo consolidado CSV con las metricas reales."
        ),
        nbf.v4.new_code_cell(
            "import pandas as pd\n"
            "res_dir = 'results' if os.path.exists('results') else os.path.join('..', 'results')\n"
            "csv_path = os.path.join(res_dir, 'results_summary.csv')\n"
            "df = pd.read_csv(csv_path)\n"
            "display(df)"
        )
    ]

    nb.cells.extend(cells)
    target_path = "notebooks/tinyml_audio_classification.ipynb"
    with open(target_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook guardado exitosamente en: {target_path}")

if __name__ == "__main__":
    generate_notebook()
