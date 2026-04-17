import os
import shutil

import pandas as pd


DATASET_PATH = "UrbanSound8K/audio"
METADATA_PATH = "UrbanSound8K/metadata/UrbanSound8K.csv"
OUTPUT_ROOT = "dataset/non_scream"

# Remove classes that often overlap with scream-like energy or
# contain human vocalizations/sharp impulses that cause false positives.
EXCLUDED_CLASSES = {
    "children_playing",
    "drilling",
    "gun_shot",
    "jackhammer",
    "siren",
    "car_horn",
}

# Keep only safer environmental negatives from UrbanSound8K.
SAFE_CLASSES = [
    "air_conditioner",
    "dog_bark",
    "engine_idling",
    "street_music",
]


def main():
    df = pd.read_csv(METADATA_PATH)
    selected_df = df[df["class"].isin(SAFE_CLASSES)].copy()

    print("UrbanSound8K extraction plan")
    print("Excluded confusing classes:", ", ".join(sorted(EXCLUDED_CLASSES)))
    print("Selected safe classes:", ", ".join(SAFE_CLASSES))
    print("Total selected files:", len(selected_df))

    copied_count = 0

    for class_name in SAFE_CLASSES:
        class_df = selected_df[selected_df["class"] == class_name]
        output_dir = os.path.join(OUTPUT_ROOT, "urbansound_clean", class_name)
        os.makedirs(output_dir, exist_ok=True)

        class_count = 0
        for _, row in class_df.iterrows():
            fold = f"fold{row['fold']}"
            file_name = row["slice_file_name"]

            src = os.path.join(DATASET_PATH, fold, file_name)
            dst = os.path.join(output_dir, file_name)

            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied_count += 1
                class_count += 1

        print(f"Copied {class_count} files for {class_name}")

    print("Done! Total copied:", copied_count)
    print("\nRecommended additional folders to add manually under dataset/non_scream/:")
    print("- cough/")
    print("- clap/")
    print("- speech/")
    print("- background_noise/")


if __name__ == "__main__":
    main()
