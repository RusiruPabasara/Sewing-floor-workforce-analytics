from pathlib import Path
import json
import cv2
import numpy as np
from sklearn.cluster import KMeans
from timestamp_cells import get_cells, W

IMG_DIR = Path("data/raw/train/images")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)
K = 24                                   # more clusters than digits, on purpose

files = sorted(IMG_DIR.glob("*.jpg"))[::8]
cells = []
for p in files:
    tight, _ = get_cells(cv2.imread(str(p)))
    cells += [c.flatten() for c in tight]
h = tight[0].shape[0]

X = np.array(cells, dtype=np.float32) / 255
km = KMeans(n_clusters=K, n_init=5, random_state=42).fit(X)
counts = np.bincount(km.labels_, minlength=K)
centres = km.cluster_centers_.reshape(K, h, W)
np.save(OUT / "centres.npy", centres)

# montage so you can see every cluster
tile_h, tile_w, cols = h * 2 + 30, 100, 6
rows = (K + cols - 1) // cols
canvas = np.zeros((rows * tile_h, cols * tile_w), np.uint8)
for k in range(K):
    r, c = divmod(k, cols)
    tile = cv2.resize((centres[k] * 255).astype(np.uint8), None, fx=2, fy=2,
                      interpolation=cv2.INTER_NEAREST)
    y, x = r * tile_h, c * tile_w
    cv2.putText(canvas, f"#{k} n={counts[k]}", (x + 2, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1)
    canvas[y + 26:y + 26 + tile.shape[0], x + 10:x + 10 + tile.shape[1]] = tile
cv2.imwrite(str(OUT / "cluster_montage.png"), canvas)

labels_path = Path("configs/digit_labels.json")
if not labels_path.exists():
    labels_path.write_text(json.dumps({str(k): "" for k in range(K)}, indent=2))
print("Done. Open data/processed/cluster_montage.png and fill configs/digit_labels.json")