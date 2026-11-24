"""Step B: find near-duplicate frames.

Two measures are computed for every frame (the clock is covered first):
  1) phash: a 64-bit fingerprint of the whole picture. Kept for comparison only:
     the visual check showed it is too coarse for this wide view (it misses workers moving).
  2) pixel change: % of pixels that changed (small grey picture, slightly blurred).
     This one sees a single worker moving, so it is used to decide.
"""
from pathlib import Path
import json
import random

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from timestamp_cells import Y1, Y2, X_OFF, X_END

IMG_DIR = Path("data/raw/train/images")
OUT = Path("data/processed")
CHECK = OUT / "dedup_check"
SESSION_GAP_MIN = 10                 # gap bigger than this = new session
PHASH_T = [0, 2, 4]                  # phash: repeat if distance <= t
PIXEL_T = [0.25, 0.5, 1.0]           # pixel: repeat if % changed pixels <= p
DIFF_LEVEL = 25                      # grey change (0-255) that counts as "changed"
SMALL = (480, 270)                   # picture size used for the pixel check
CHANGE_BINS = [0, 0.25, 0.5, 1, 2]   # example pairs are saved for each bin
PAIRS_PER_BIN = 5

periods = json.load(open("configs/periods.json"))


def period_of(frame):
    for name, (first, last) in periods.items():
        if first <= frame <= last:
            return name
    return "unknown"


def load(name):
    """Return (phash, small blurred grey picture), with the clock covered."""
    img = cv2.imread(str(IMG_DIR / name))
    img[Y1:Y2, X_OFF:X_END] = 128
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, SMALL, interpolation=cv2.INTER_AREA)
    return imagehash.phash(Image.fromarray(gray)), cv2.GaussianBlur(small, (5, 5), 0)


def changed_mask(a, b):
    return cv2.absdiff(a, b) > DIFF_LEVEL


def changed_pct(a, b):
    return 100.0 * float(changed_mask(a, b).mean())


# 1) frames in time order, with day, session and period
df = pd.read_csv(OUT / "timestamps.csv").sort_values("frame").reset_index(drop=True)
df["timestamp"] = pd.to_datetime(df["timestamp"],format='ISO8601')
df["day"] = df["timestamp"].dt.strftime("%Y-%m-%d")
gap_min = df["timestamp"].diff().dt.total_seconds() / 60
df["session"] = (gap_min.isna() | (gap_min > SESSION_GAP_MIN)).cumsum()
df["period"] = df["frame"].map(period_of)
new_session = df["session"].ne(df["session"].shift()).tolist()

# 2) one pass over all frames (takes a few minutes)
cols = {"hash": [], "dist_prev": [], "change_prev": []}
for t in PHASH_T:
    cols[f"dup_t{t}"] = []
for p in PIXEL_T:
    cols[f"dup_px{p}"] = []
kept_hash, kept_img = {}, {}         # last kept frame, one per threshold
prev_hash = prev_img = None

for i, name in enumerate(df["file"]):
    h, img = load(name)
    start = new_session[i]
    cols["hash"].append(str(h))
    cols["dist_prev"].append(None if start else h - prev_hash)
    cols["change_prev"].append(None if start else round(changed_pct(img, prev_img), 3))
    for t in PHASH_T:
        dup = (not start) and (h - kept_hash[t] <= t)
        cols[f"dup_t{t}"].append(dup)
        if not dup:
            kept_hash[t] = h
    for p in PIXEL_T:
        dup = (not start) and (changed_pct(img, kept_img[p]) <= p)
        cols[f"dup_px{p}"].append(dup)
        if not dup:
            kept_img[p] = img
    prev_hash, prev_img = h, img
    if (i + 1) % 500 == 0:
        print(f"{i + 1}/{len(df)} frames done")

df["hash"] = cols.pop("hash")
df["dist_prev"] = pd.array(cols.pop("dist_prev"), dtype="Int64")
df["change_prev"] = np.array(cols.pop("change_prev"), dtype=float)
for k, v in cols.items():
    df[k] = v
dup_cols = list(cols)

keep = ["file", "frame", "timestamp", "day", "session", "period", "hash",
        "dist_prev", "change_prev"] + dup_cols
df[keep].to_csv(OUT / "dedup.csv", index=False)

# 3) summary
print(f"\nSessions found: {df['session'].nunique()}")
print("\nRepeats per rule (total):")
print(df[dup_cols].sum().to_string())
print("\nRepeats per period:")
print(df.groupby("period")[dup_cols].sum().to_string())
print("\nPixel change vs the frame before (% of pixels), per period:")
print(df.groupby("period")["change_prev"]
        .describe(percentiles=[0.05, 0.25, 0.5]).round(2).to_string())
print("\nPixel change for frames phash called IDENTICAL (distance 0):")
same_mask = df["dist_prev"].fillna(-1) == 0
print(df.loc[same_mask].groupby("period")["change_prev"].describe().round(2).to_string())

# 4) example pairs: before | after | after with changed pixels in red
CHECK.mkdir(exist_ok=True)
for old in CHECK.glob("*.jpg"):
    old.unlink()


def save_pair(i, tag):
    a_name, b_name = df.at[i - 1, "file"], df.at[i, "file"]
    _, sa = load(a_name)
    _, sb = load(b_name)
    a = cv2.resize(cv2.imread(str(IMG_DIR / a_name)), (960, 540))
    b = cv2.resize(cv2.imread(str(IMG_DIR / b_name)), (960, 540))
    mask = cv2.resize(changed_mask(sa, sb).astype(np.uint8), (960, 540),
                      interpolation=cv2.INTER_NEAREST).astype(bool)
    red = b.copy()
    red[mask] = (0, 0, 255)
    pair = np.hstack([a, b, red])
    f1, f2 = df.at[i - 1, "frame"], df.at[i, "frame"]
    text = (f"frame {f1} | frame {f2} | changes in red   ({df.at[i, 'period']}, "
            f"phash {df.at[i, 'dist_prev']}, pixels {df.at[i, 'change_prev']:.2f}%)")
    cv2.putText(pair, text, (20, 525), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(str(CHECK / f"{tag}_f{f1}_f{f2}.jpg"), pair)


rng = random.Random(0)
# a) frames that phash called identical
same = df.index[same_mask].tolist()
for i in rng.sample(same, min(PAIRS_PER_BIN, len(same))):
    save_pair(i, "phash0")
# b) working-time frames grouped by pixel change (pre_shift left out: nobody works there)
work = df["period"] != "pre_shift"
change = df["change_prev"]
for lo, hi in zip(CHANGE_BINS[:-1], CHANGE_BINS[1:]):
    rows = df.index[work & (change >= lo) & (change < hi)].tolist()
    for i in rng.sample(rows, min(PAIRS_PER_BIN, len(rows))):
        save_pair(i, f"px{lo}-{hi}")
print(f"\nExample pairs saved in {CHECK}")
