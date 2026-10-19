from pathlib import Path
import sys
import cv2
import numpy as np
from timestamp_cells import get_cells

K = int(sys.argv[1])                    # cluster number to inspect
centres = np.load("data/processed/centres.npy")
c = centres[K].astype(np.float32)

files = sorted(Path("data/raw/train/images").glob("*.jpg"))[::8]
hits = []
for p in files:
    tight, _ = get_cells(cv2.imread(str(p)))
    for cell in tight:
        q = cell.astype(np.float32) / 255
        if int(np.argmin([((q - centres[j]) ** 2).sum() for j in range(len(centres))])) == K:
            hits.append(cell)
    if len(hits) >= 12:
        break

strip = np.hstack([np.hstack([h, np.zeros((h.shape[0], 8), np.uint8)]) for h in hits[:12]])
strip = cv2.resize(strip, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
cv2.imwrite(f"data/processed/cluster_{K}_members.png", strip)
print(f"Saved data/processed/cluster_{K}_members.png with {len(hits[:12])} examples")
