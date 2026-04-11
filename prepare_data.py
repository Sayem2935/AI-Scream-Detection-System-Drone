import os
import librosa
import soundfile as sf

INPUT_PATH = "dataset"
OUTPUT_PATH = "processed_dataset"

TARGET_SR = 16000
DURATION = 1

print(" Starting processing...")

def process_file(file_path, output_path):
    try:
        audio, sr = librosa.load(file_path, sr=TARGET_SR)

        target_length = TARGET_SR * DURATION

        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = librosa.util.fix_length(audio, size=target_length)

        sf.write(output_path, audio, TARGET_SR)

    except Exception as e:
        print(" Error:", file_path, e)

for label in ["scream", "non_scream"]:
    input_folder = os.path.join(INPUT_PATH, label)
    output_folder = os.path.join(OUTPUT_PATH, label)

    print(f"\n Processing {label}...")

    if not os.path.exists(input_folder):
        print(" Folder not found:", input_folder)
        continue

    os.makedirs(output_folder, exist_ok=True)

    files = os.listdir(input_folder)

    print(f"👉 Total files found: {len(files)}")

    count = 0

    for file in files:
        if not file.endswith(".wav"):
            continue

        input_file = os.path.join(input_folder, file)
        output_file = os.path.join(output_folder, file)

        process_file(input_file, output_file)
        count += 1

        if count % 100 == 0:
            print(f"Processed {count} files...")

    print(f" Finished {label}: {count} files")

