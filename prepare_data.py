import os

import librosa
import soundfile as sf


INPUT_PATH = "dataset"
OUTPUT_PATH = "processed_dataset"

TARGET_SR = 16000
DURATION = 1
TARGET_LENGTH = TARGET_SR * DURATION


def process_file(file_path, output_path):
    try:
        audio, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)

        # Light normalization keeps recordings on a comparable scale
        # without destroying real loudness differences entirely.
        peak = max(abs(audio.max()), abs(audio.min()), 1e-6)
        audio = audio / peak

        if len(audio) > TARGET_LENGTH:
            audio = audio[:TARGET_LENGTH]
        else:
            audio = librosa.util.fix_length(audio, size=TARGET_LENGTH)

        sf.write(output_path, audio, TARGET_SR)
        return True
    except Exception as exc:
        print(f"Error processing {file_path}: {exc}")
        return False


def collect_audio_files(root_folder):
    audio_files = []
    for current_root, _, files in os.walk(root_folder):
        for file_name in files:
            if file_name.lower().endswith(".wav"):
                audio_files.append(os.path.join(current_root, file_name))
    return sorted(audio_files)


def main():
    print("Starting dataset processing...")

    for label in ["scream", "non_scream"]:
        input_folder = os.path.join(INPUT_PATH, label)
        output_folder = os.path.join(OUTPUT_PATH, label)

        print(f"\nProcessing {label}...")

        if not os.path.exists(input_folder):
            print("Folder not found:", input_folder)
            continue

        os.makedirs(output_folder, exist_ok=True)
        files = collect_audio_files(input_folder)

        print(f"Total WAV files found (recursive): {len(files)}")

        success_count = 0
        for index, input_file in enumerate(files, start=1):
            relative_name = os.path.relpath(input_file, input_folder)
            safe_name = relative_name.replace(os.sep, "__")
            output_file = os.path.join(output_folder, safe_name)

            if process_file(input_file, output_file):
                success_count += 1

            if index % 100 == 0:
                print(f"Processed {index} files...")

        print(f"Finished {label}: {success_count} files written")

    print("\nRecommended dataset layout before preprocessing:")
    print("dataset/")
    print("  scream/")
    print("  non_scream/")
    print("    cough/")
    print("    clap/")
    print("    speech/")
    print("    background_noise/")
    print("    urbansound_clean/")


if __name__ == "__main__":
    main()
