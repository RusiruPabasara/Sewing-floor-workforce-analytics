from pathlib import Path
from datetime import datetime
import csv
import re
import cv2
import numpy as np
import easyocr

Y1, Y2 = 50, 112
DATE_X = (36, 298)            # "02-19-2025"
TIME_LEFT, TIME_RIGHT = 408, 596   # measured digit span of "HH:MM:SS"
CELL = (TIME_RIGHT - TIME_LEFT) / 8
DIGIT_CELLS = [0, 1, 3, 4, 6, 7]   # skip the colons (cells 2 and 5)

IMG_DIR = Path("data/raw/train/images")
OUT_DIR = Path("data/processed/ocr_crops")
OUT_DIR.mkdir(parents=True, exist_ok=True)

reader = easyocr.Reader(["en"], gpu=False)


def to_gray(img, x1, x2):
    crop = img[Y1:Y2, x1:x2]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def prep_date(img):
    crop = to_gray(img, *DATE_X)
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(crop, 20, 20, 20, 20, cv2.BORDER_REPLICATE)


def read_line(crop, allow):
    return " ".join(reader.recognize(crop, allowlist=allow, detail=0)).strip()


def stitch_digits(img):
    """Cut the 8 character cells, drop the 2 colons, join the 6 digits."""
    gray = to_gray(img, 0, img.shape[1])
    pieces = []
    for c in DIGIT_CELLS:
        x1 = int(round(TIME_LEFT + c * CELL)) - 1
        x2 = int(round(TIME_LEFT + (c + 1) * CELL)) + 1
        cell = gray[:, x1:x2]
        cell = cv2.resize(cell, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
        _, cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pieces.append(cell)
        pieces.append(np.full((cell.shape[0], 20), 0, dtype=np.uint8))  # gap
    strip = np.hstack(pieces)
    return cv2.copyMakeBorder(strip, 20, 20, 20, 20, cv2.BORDER_REPLICATE), pieces[::2]


def read_time(img):
    strip, cells = stitch_digits(img)
    text = re.sub(r"\D", "", read_line(strip, "0123456789"))
    if len(text) != 6:            # fallback: read each cell on its own
        text = ""
        for cell in cells:
            padded = cv2.copyMakeBorder(cell, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
            d = re.sub(r"\D", "", read_line(padded, "0123456789"))
            text += d[:1] if d else "?"
    return text, strip


def parse(date_text, t):
    d = re.sub(r"\D", "", date_text)
    if len(d) != 8 or len(t) != 6 or "?" in t:
        return None, False
    hh, mm, ss = int(t[:2]), int(t[2:4]), int(t[4:6])
    if not (5 <= hh <= 19 and mm < 60 and ss < 60):
        return None, False
    try:
        dt = datetime.strptime(d, "%m%d%Y").replace(hour=hh, minute=mm, second=ss)
    except ValueError:
        return None, False
    return dt, True


images = sorted(IMG_DIR.glob("*.jpg"))
sample = images[::200][:20]
rows = []
for i, path in enumerate(sample, start=1):
    img = cv2.imread(str(path))
    date_crop = prep_date(img)
    date_text = read_line(date_crop, "0123456789-")
    time_digits, strip = read_time(img)
    cv2.imwrite(str(OUT_DIR / f"date_{i:02d}.png"), date_crop)
    cv2.imwrite(str(OUT_DIR / f"time_{i:02d}.png"), strip)

    dt, ok = parse(date_text, time_digits)
    frame_no = re.search(r"frame_(\d+)", path.name).group(1)
    rows.append([i, frame_no, date_text, time_digits, dt.isoformat() if dt else "", ok])
    print(f"{i:02d} | frame {frame_no} | {date_text!r} | {time_digits!r} | {dt} | ok={ok}")

with open("data/processed/ocr_test_v4.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n", "frame_no", "date_text", "time_digits", "parsed", "ok"])
    w.writerows(rows)

print("Fully parsed (date + HH:MM:SS):", sum(1 for r in rows if r[5]), "of", len(rows))