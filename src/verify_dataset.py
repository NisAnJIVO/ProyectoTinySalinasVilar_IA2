import os
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "ESC-50-TinyML")
EXPECTED_CLASSES = ["clapping", "clock_alarm", "door_wood_knock", "keyboard_typing"]
EXPECTED_PER_CLASS = 40
EXPECTED_TOTAL = 160

def verify_dataset():
    print("=" * 60)
    print("FASE 1: VERIFICACION DEL DATASET")
    print("=" * 60)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"No se encontro el directorio: {DATASET_PATH}")

    found_classes = sorted([
        d for d in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, d))
    ])

    print(f"Directorio base: {DATASET_PATH}")
    print(f"Clases detectadas: {found_classes}")

    missing_classes = set(EXPECTED_CLASSES) - set(found_classes)
    if missing_classes:
        raise ValueError(f"Faltan clases esperadas: {missing_classes}")

    total_files = 0
    all_valid = True
    class_stats = {}

    for cls in EXPECTED_CLASSES:
        cls_dir = os.path.join(DATASET_PATH, cls)
        wav_files = [f for f in os.listdir(cls_dir) if f.lower().endswith(".wav")]
        count = len(wav_files)
        total_files += count
        class_stats[cls] = count

        durations = []
        samplerates = []
        channels_list = []

        for wav_file in wav_files:
            wav_path = os.path.join(cls_dir, wav_file)
            info = sf.info(wav_path)
            durations.append(info.duration)
            samplerates.append(info.samplerate)
            channels_list.append(info.channels)

        avg_dur = sum(durations) / len(durations) if durations else 0
        unique_sr = set(samplerates)
        unique_ch = set(channels_list)

        print(f"\nClase: {cls}")
        print(f"  Archivos encontrados : {count} (esperado: {EXPECTED_PER_CLASS})")
        print(f"  Frecuencia muestreo  : {unique_sr} Hz")
        print(f"  Canales              : {unique_ch}")
        print(f"  Duracion promedio    : {avg_dur:.2f} s")

        if count != EXPECTED_PER_CLASS:
            print(f"  ERROR: Se esperaban {EXPECTED_PER_CLASS} archivos en {cls}, pero se encontraron {count}.")
            all_valid = False

    print("\n" + "-" * 60)
    print(f"Total de archivos encontrados : {total_files}")
    print(f"Total de archivos esperados   : {EXPECTED_TOTAL}")

    if total_files != EXPECTED_TOTAL or not all_valid:
        raise ValueError(f"La verificacion fallo. Total={total_files} vs Esperado={EXPECTED_TOTAL}")

    print("Verificacion completada con exito: los 160 archivos estan presentes y correctos.")
    print("=" * 60)
    return class_stats

if __name__ == "__main__":
    verify_dataset()
