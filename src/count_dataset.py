from pathlib import Path
import yaml

root = Path("data/raw")
cfg = yaml.safe_load(open(root / "data.yaml"))
names = cfg["names"]
if isinstance(names, dict):
    names = [names[k] for k in sorted(names)]
print("Classes:", names)

if "person" not in names:
    raise SystemExit("No 'person' class found. Check data.yaml.")
pid = names.index("person")
print("person class id:", pid)

img_dir = root / "train" / "images"
lbl_dir = root / "train" / "labels"

images = list(img_dir.glob("*.*"))
labels = list(lbl_dir.glob("*.txt"))

with_person = boxes = 0
for lf in labels:
    n = sum(1 for line in lf.read_text().splitlines()
            if line.strip() and int(line.split()[0]) == pid)
    boxes += n
    with_person += n > 0

print(f"Images:                {len(images)}")
print(f"Label files:           {len(labels)}")
print(f"Frames with person:    {with_person}")
print(f"Total person boxes:    {boxes}")
print(f"Avg boxes per frame:   {boxes / max(with_person, 1):.1f}")