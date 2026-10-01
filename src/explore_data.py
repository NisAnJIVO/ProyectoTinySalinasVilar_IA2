import os
import soundfile as sf
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "ESC-50-TinyML")
RESULTS_PATH = os.path.join(BASE_DIR, "results")
CLASSES = ["clapping", "clock_alarm", "door_wood_knock", "keyboard_typing"]

os.makedirs(RESULTS_PATH, exist_ok=True)

def plot_class_distribution():
    counts = {}
    for cls in CLASSES:
        cls_dir = os.path.join(DATASET_PATH, cls)
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(".wav")]
        counts[cls] = len(files)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.keys(), counts.values(), color="#2c3e50", width=0.5, edgecolor="black")

    ax.set_title("Distribucion de Clases en el Dataset Reducido", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Categoria de Sonido", fontsize=11, labelpad=10)
    ax.set_ylabel("Cantidad de Muestras", fontsize=11, labelpad=10)
    ax.set_ylim(0, 50)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    output_path = os.path.join(RESULTS_PATH, "class_distribution.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Grafica guardada: {output_path}")

def plot_waveforms():
    # Audio individual
    sample_cls = "clapping"
    sample_file = os.listdir(os.path.join(DATASET_PATH, sample_cls))[0]
    sample_path = os.path.join(DATASET_PATH, sample_cls, sample_file)

    audio, sr = sf.read(sample_path)
    time_axis = np.linspace(0, len(audio) / sr, num=len(audio))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(time_axis, audio, color="#1f77b4", linewidth=0.7)
    ax.set_title(f"Forma de Onda Individual - {sample_cls} ({sample_file})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Tiempo (s)", fontsize=11)
    ax.set_ylabel("Amplitud", fontsize=11)
    ax.set_xlim(0, max(time_axis))
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    output_individual = os.path.join(RESULTS_PATH, "waveform_sample.png")
    plt.savefig(output_individual, dpi=300)
    plt.close()
    print(f"Grafica guardada: {output_individual}")

    # Comparacion de las cuatro clases
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, cls in enumerate(CLASSES):
        cls_dir = os.path.join(DATASET_PATH, cls)
        first_file = sorted(os.listdir(cls_dir))[0]
        audio_i, sr_i = sf.read(os.path.join(cls_dir, first_file))
        time_i = np.linspace(0, len(audio_i) / sr_i, num=len(audio_i))

        axes[i].plot(time_i, audio_i, color=colors[i], linewidth=0.6)
        axes[i].set_title(f"Clase: {cls}", fontsize=11, fontweight="bold", loc="left")
        axes[i].set_ylabel("Amplitud", fontsize=10)
        axes[i].grid(True, linestyle="--", alpha=0.4)
        axes[i].set_ylim(-1.05, 1.05)

    axes[-1].set_xlabel("Tiempo (s)", fontsize=11)
    fig.suptitle("Comparacion de Formas de Onda entre Clases", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    output_comparison = os.path.join(RESULTS_PATH, "waveform_comparison.png")
    plt.savefig(output_comparison, dpi=300)
    plt.close()
    print(f"Grafica guardada: {output_comparison}")

def plot_spectrograms():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for i, cls in enumerate(CLASSES):
        cls_dir = os.path.join(DATASET_PATH, cls)
        first_file = sorted(os.listdir(cls_dir))[0]
        audio, sr = sf.read(os.path.join(cls_dir, first_file))

        # Re-muestreo a 16 kHz para estandarizacion didactica
        target_sr = 16000
        n_samples = int(len(audio) * target_sr / sr)
        audio_16k = signal.resample(audio, n_samples)

        f, t, Sxx = signal.spectrogram(audio_16k, fs=target_sr, nperseg=512, noverlap=256)
        log_Sxx = 10 * np.log10(Sxx + 1e-10)

        im = axes[i].pcolormesh(t, f, log_Sxx, shading="gouraud", cmap="inferno")
        axes[i].set_title(f"Espectrograma: {cls}", fontsize=11, fontweight="bold")
        axes[i].set_ylabel("Frecuencia (Hz)", fontsize=10)
        axes[i].set_xlabel("Tiempo (s)", fontsize=10)
        axes[i].set_ylim(0, 8000)
        cbar = fig.colorbar(im, ax=axes[i], format="%+2.0f dB")
        cbar.set_label("Intensidad (dB)", fontsize=9)

    fig.suptitle("Representacion en Espectrograma de las Cuatro Clases", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    output_spectrogram = os.path.join(RESULTS_PATH, "spectrograms.png")
    plt.savefig(output_spectrogram, dpi=300)
    plt.close()
    print(f"Grafica guardada: {output_spectrogram}")

if __name__ == "__main__":
    print("Generando exploracion y graficas...")
    plot_class_distribution()
    plot_waveforms()
    plot_spectrograms()
    print("Exploracion completada exitosamente.")
