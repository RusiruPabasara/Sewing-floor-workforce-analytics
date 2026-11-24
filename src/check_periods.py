"""Ths will help to fix the period boundaries in configs/periods.json.

First of all, prints how many 'person' boxes each frame has around the shift start
Next, saves crops of the front-centre line around the end of the changeover
"""
from pathlib import Path
import cv2

IMG_DIR = Path("data/raw/train/images")
LBL_DIR = Path("data/raw/train/labels")
OUT = Path("data/processed/period_check")
PERSON = 5          # class id of 'person' in data/raw/data.yaml


def find(frame, folder, ext):
    return next(folder.glob(f"frame_{frame}_jpg*{ext}"))


def persons(frame):
    lines = find(frame, LBL_DIR, ".txt").read_text().splitlines()
    return sum(1 for line in lines if line.split()[:1] == [str(PERSON)])


# 1) when do workers arrive?
print("Person boxes per frame (pre_shift -> changeover):")
for f in range(750, 791):
    n = persons(f)
    print(f"  frame {f}: {n:2d} " + "#" * n)

# 2) when do the front-centre tables stop moving?
OUT.mkdir(parents=True, exist_ok=True)
for f in range(836, 881, 2):
    img = cv2.imread(str(find(f, IMG_DIR, ".jpg")))[540:1080, 480:1440]  # bottom-centre
    cv2.putText(img, str(f), (10, 520), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.imwrite(str(OUT / f"front_{f}.jpg"), img)
print(f"\nFront-centre crops saved in {OUT}")
