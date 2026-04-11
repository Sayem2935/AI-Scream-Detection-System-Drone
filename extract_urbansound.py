import os
import shutil
import pandas as pd

DATASET_PATH = "UrbanSound8K/audio"
METADATA_PATH = "UrbanSound8K/metadata/UrbanSound8K.csv"

OUTPUT_PATH = "dataset/non_scream"


ALLOWED_CLASSES = [
    "dog_bark",
    "children_playing",
    "engine_idling",
    "air_conditioner",
    "drilling",
    "jackhammer",
    "street_music"
]


df = pd.read_csv(METADATA_PATH)


df = df[df["class"].isin(ALLOWED_CLASSES)]

print("Total selected files:", len(df))

os.makedirs(OUTPUT_PATH, exist_ok=True)


count = 0

for _, row in df.iterrows():
    fold = f"fold{row['fold']}"
    file_name = row["slice_file_name"]

    src = os.path.join(DATASET_PATH, fold, file_name)
    dst = os.path.join(OUTPUT_PATH, file_name)

    if os.path.exists(src):
        shutil.copy(src, dst)
        count += 1

        if count % 200 == 0:
            print(f"Copied {count} files...")

print(" DONE! Total copied:", count)