import os
import soundfile as sf
import scipy.signal as signal
import scipy.ndimage as ndimage
import numpy as np
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "ESC-50-TinyML")
CLASSES = ["clapping", "clock_alarm", "door_wood_knock", "keyboard_typing"]
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}

TARGET_SR = 16000
SPEC_SIZE = (64, 64)
RANDOM_STATE = 42

def audio_to_spectrogram(audio_path, target_sr=TARGET_SR, target_shape=SPEC_SIZE):
    audio, sr = sf.read(audio_path)

    # Convertir a mono si existiera mas de un canal
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Re-muestreo a la frecuencia objetivo
    if sr != target_sr:
        num_target_samples = int(len(audio) * target_sr / sr)
        audio = signal.resample(audio, num_target_samples)

    # Asegurar longitud uniforme (5 segundos = 80000 muestras a 16 kHz)
    expected_len = target_sr * 5
    if len(audio) < expected_len:
        audio = np.pad(audio, (0, expected_len - len(audio)), mode="constant")
    elif len(audio) > expected_len:
        audio = audio[:expected_len]

    # Calculo del espectrograma
    _, _, Sxx = signal.spectrogram(audio, fs=target_sr, nperseg=512, noverlap=256)
    log_spec = 10 * np.log10(Sxx + 1e-10)

    # Redimensionamiento a resolucion compacta TinyML
    zoom_factors = (target_shape[0] / log_spec.shape[0], target_shape[1] / log_spec.shape[1])
    spec_resized = ndimage.zoom(log_spec, zoom_factors)

    # Normalizacion Min-Max a rango [0, 1]
    min_val = spec_resized.min()
    max_val = spec_resized.max()
    spec_norm = (spec_resized - min_val) / (max_val - min_val + 1e-8)

    return spec_norm.astype(np.float32)

def load_dataset():
    data = []
    labels = []
    filepaths = []

    for cls_name in CLASSES:
        cls_dir = os.path.join(DATASET_PATH, cls_name)
        files = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith(".wav")])

        for fname in files:
            full_path = os.path.join(cls_dir, fname)
            spec = audio_to_spectrogram(full_path)
            data.append(spec)
            labels.append(CLASS_TO_IDX[cls_name])
            filepaths.append(full_path)

    # Dimensiones: (N, 64, 64, 1)
    X = np.expand_dims(np.array(data, dtype=np.float32), axis=-1)
    y = np.array(labels, dtype=np.int32)
    return X, y, filepaths

def get_train_val_test_split():
    X, y, _ = load_dataset()

    # Division estratificada: 70% entrenamiento, 30% restante
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )

    # Del 30% restante, dividimos a partes iguales: 15% validacion, 15% prueba
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

if __name__ == "__main__":
    print("Cargando y preprocesando dataset...")
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = get_train_val_test_split()
    print(f"Conjunto de Entrenamiento : X={X_tr.shape}, y={y_tr.shape}")
    print(f"Conjunto de Validacion    : X={X_va.shape}, y={y_va.shape}")
    print(f"Conjunto de Prueba        : X={X_te.shape}, y={y_te.shape}")
    print(f"Muestras por clase en train: {np.bincount(y_tr)}")
    print(f"Muestras por clase en val  : {np.bincount(y_va)}")
    print(f"Muestras por clase en test : {np.bincount(y_te)}")
