"""Step C: split frames by whole day into train / val / test (never random).

Reads data/processed/dedup.csv, removes repeats for the chosen rule and writes:
  configs/splits/train.txt, val.txt, test.txt   (image paths)
  configs/data_split.yaml                       (for YOLO)
Image paths are relative to the repo root, so always run YOLO from the repo root.
"""
from pathlib import Path
import argparse

import pandas as pd
import yaml

DAY_TO_SPLIT = {
    "2025-02-17": "train",   # old layout
    "2025-02-18": "train",   # pre-shift + changeover + new layout
    "2025-02-19": "train",
    "2025-02-21": "val",     # afternoon only
    "2025-02-20": "test",    # morning + afternoon; do not look at it until the end
}
SPLITS = ["train", "val", "test"]
IMG_DIR = "data/raw/train/images"
LBL_DIR = Path("data/raw/train/labels")
OUT = Path("configs")
PERSON = 5

ap = argparse.ArgumentParser()
ap.add_argument("--dup", default="dup_px0.25",
                help="repeat column in dedup.csv, e.g. dup_px0.25, dup_px0.5, dup_t0")
args = ap.parse_args()

df = pd.read_csv("data/processed/dedup.csv", parse_dates=["timestamp"])
missing = set(df["day"]) - set(DAY_TO_SPLIT)
assert not missing, f"Days without a split: {missing}"

# 1) remove repeats, add split / time of day / person count
dup_col = args.dup
assert dup_col in df.columns, f"{dup_col} is not a column in dedup.csv"
removed = int(df[dup_col].sum())
df = df[~df[dup_col]].copy()
df["split"] = df["day"].map(DAY_TO_SPLIT)
df["time_of_day"] = df["timestamp"].dt.hour.map(lambda h: "am" if h < 12 else "pm")


def person_boxes(name):
    p = LBL_DIR / (Path(name).stem + ".txt")
    if not p.exists():
        return 0
    return sum(1 for line in p.read_text().splitlines() if line.split()[:1] == [str(PERSON)])


df["persons"] = df["file"].map(person_boxes)

# 2) write the image lists
(OUT / "splits").mkdir(parents=True, exist_ok=True)
for split in SPLITS:
    files = df.loc[df["split"] == split].sort_values("frame")["file"]
    (OUT / "splits" / f"{split}.txt").write_text(
        "\n".join(f"{IMG_DIR}/{f}" for f in files) + "\n")

# 3) write the YOLO settings file (class names copied from the Roboflow export)
raw = yaml.safe_load(open("data/raw/data.yaml"))
data = {"train": "splits/train.txt", "val": "splits/val.txt", "test": "splits/test.txt",
        "nc": raw["nc"], "names": raw["names"]}
with open(OUT / "data_split.yaml", "w") as f:
    f.write(f"# Made by src/make_splits.py with dedup rule {args.dup}\n")
    yaml.safe_dump(data, f, sort_keys=False)

# 4) checks
assert df["file"].is_unique, "a frame appears twice"
assert df.groupby("day")["split"].nunique().max() == 1, "a day is in two splits"

# 5) summary
print(f"Dedup rule {args.dup}: {removed} repeats removed, {len(df)} frames kept\n")
summary = df.groupby("split").agg(frames=("file", "size"), days=("day", "nunique"),
                                  sessions=("session", "nunique"),
                                  person_boxes=("persons", "sum"))
summary["persons_per_frame"] = (summary["person_boxes"] / summary["frames"]).round(1)
print(summary.reindex(SPLITS).to_string())
print("\nFrames per split, morning / afternoon:")
print(pd.crosstab(df["split"], df["time_of_day"]).reindex(SPLITS, fill_value=0).to_string())
print("\nFrames per split and period:")
print(pd.crosstab(df["split"], df["period"]).reindex(SPLITS, fill_value=0).to_string())
print("\nWritten: configs/splits/{train,val,test}.txt and configs/data_split.yaml")
