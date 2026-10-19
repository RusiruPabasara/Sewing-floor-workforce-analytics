from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import bisect
import csv
import json
import random
import re
import cv2
import numpy as np
from timestamp_cells import get_cells, Y1, Y2, X_OFF, X_END

IMG_DIR = Path("data/raw/train/images")
OUT = Path("data/processed")
centres = np.load(OUT / "centres.npy")
labels = json.load(open("configs/digit_labels.json"))
templates = [(centres[int(k)].astype(np.float32), v)
             for k, v in labels.items() if v in list("0123456789")]
assert templates, "Fill configs/digit_labels.json first"


def frame_no(p):
    return int(re.search(r"frame_(\d+)", p.name).group(1))


def classify(wide):
    q = wide.astype(np.float32) / 255
    best, best_digit = -1.0, "?"
    for t, d in templates:
        s = float(np.nan_to_num(cv2.matchTemplate(q, t, cv2.TM_CCOEFF_NORMED).max()))
        if s > best:
            best, best_digit = s, d
    return best_digit, best


def to_dt(d):
    if "?" in d:
        return None
    try:
        return datetime.strptime(d[:8], "%m%d%Y").replace(
            hour=int(d[8:10]), minute=int(d[10:12]), second=int(d[12:14]))
    except ValueError:
        return None


def longest_nondecreasing(values):
    """Indices of the longest non-decreasing subsequence."""
    tails, tails_idx, prev = [], [], [-1] * len(values)
    for i, v in enumerate(values):
        pos = bisect.bisect_right(tails, v)
        if pos == len(tails):
            tails.append(v); tails_idx.append(i)
        else:
            tails[pos] = v; tails_idx[pos] = i
        prev[i] = tails_idx[pos - 1] if pos > 0 else -1
    out, k = [], tails_idx[-1]
    while k != -1:
        out.append(k); k = prev[k]
    return out


# 1) read every frame
files = sorted(IMG_DIR.glob("*.jpg"), key=frame_no)
rows = []
for n, p in enumerate(files, 1):
    _, wide = get_cells(cv2.imread(str(p)))
    res = [classify(w) for w in wide]
    digits = "".join(d for d, _ in res)
    rows.append({"file": p.name, "frame": frame_no(p), "digits": digits,
                 "min_score": round(min(s for _, s in res), 3), "dt": to_dt(digits)})
    if n % 500 == 0:
        print(f"{n}/{len(files)} frames read")

# 2) frame order must follow time order: drop readings that break it
valid = [i for i, r in enumerate(rows) if r["dt"]]
keep = {valid[j] for j in longest_nondecreasing([rows[i]["dt"] for i in valid])}
good = sorted(keep)
for i, r in enumerate(rows):
    r["status"] = "ok" if i in keep else "bad"
    r["final"] = r["dt"] if i in keep else None

# 3) repair bad rows by interpolating between neighbouring good frames
for i, r in enumerate(rows):
    if r["status"] == "bad":
        k = bisect.bisect(good, i)
        if 0 < k < len(good):
            a, b = rows[good[k - 1]], rows[good[k]]
            frac = (r["frame"] - a["frame"]) / (b["frame"] - a["frame"])
            r["final"] = a["dt"] + (b["dt"] - a["dt"]) * frac
            r["status"] = "repaired"

with open(OUT / "timestamps.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "frame", "digits", "min_score", "status", "timestamp"])
    for r in rows:
        w.writerow([r["file"], r["frame"], r["digits"], r["min_score"], r["status"],
                    r["final"].isoformat() if r["final"] else ""])

# 4) summary (task 3: days and hours)
print("\nStatus:", dict(Counter(r["status"] for r in rows)))
days = defaultdict(list)
for r in rows:
    if r["final"]:
        days[r["final"].date()].append(r["final"].hour)
print("\nDistinct days:", len(days))
for d in sorted(days):
    hrs = days[d]
    print(f"  {d}: {len(hrs)} frames, hours {min(hrs):02d} to {max(hrs):02d}, "
          f"{len(set(hrs))} distinct hours")

# 5) 50 random crops for your eye check
chk = OUT / "check"
chk.mkdir(exist_ok=True)
random.seed(42)
for r in random.sample(rows, 50):
    img = cv2.imread(str(IMG_DIR / r["file"]))[Y1:Y2, X_OFF:X_END]
    d = r["digits"]
    pred = f"{d[:2]}-{d[2:4]}-{d[4:8]}_{d[8:10]}.{d[10:12]}.{d[12:14]}"
    cv2.imwrite(str(chk / f"f{r['frame']}__{pred}__{r['status']}.png"), img)
print("\nEye-check crops saved in data/processed/check")