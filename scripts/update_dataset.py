import argparse
import csv
import hashlib
import random
from pathlib import Path
from typing import List, Tuple, Dict

from PIL import Image
import numpy as np
from utils.face_align import detect_face_5pt, align_face_256


def file_md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def scan_new_images(root: Path) -> List[Tuple[str, Path]]:
    """
    Scan a directory tree and return list of (identity, image_path).
    Assumes directory name is identity if images are organized as root/ID/*.jpg
    """
    out = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            ident = p.parent.name
            out.append((ident, p))
    return out

def ensure_aligned_one(ident: str, raw_path: Path, dst_root: Path) -> Path | None:
    """
    Detect + align to 256x256; save under dst_root/ident/<stem>.png
    Returns saved path, or None if no face detected.
    """
    img = Image.open(raw_path).convert("RGB")
    img_np = np.asarray(img)
    pts = detect_face_5pt(img_np)
    if pts is None:
        return None
    aligned = align_face_256(img_np, pts)
    dst_dir = (dst_root / ident)
    dst_dir.mkdir(parents=True, exist_ok=True)
    out_path = dst_dir / (raw_path.stem + ".png")
    Image.fromarray(aligned).save(out_path.as_posix())
    return out_path

def read_images_index(index_csv: Path) -> List[Dict]:
    if not index_csv.exists():
        return []
    rows = []
    with open(index_csv, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    return rows

def write_images_index(index_csv: Path, rows: List[Dict]):
    if not rows:
        return
    # deduplicate by path hash
    seen = set()
    uniq = []
    for r in rows:
        k = (r["path"], r.get("md5", ""))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    with open(index_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["identity", "path", "md5", "width", "height"])
        wr.writeheader()
        wr.writerows(uniq)

def sample_pairs(
    images: List[Dict],
    same_ratio: float,
    max_pairs: int,
    min_per_id: int
) -> List[Dict]:
    """
    Build src-tgt pairs with a controllable same/diff ratio.
    """
    from collections import defaultdict
    by_id: Dict[str, List[Dict]] = defaultdict(list)
    for r in images:
        by_id[r["identity"]].append(r)

    # filter identities with enough images
    by_id = {k: v for k, v in by_id.items() if len(v) >= max(min_per_id, 2)}
    id_list = list(by_id.keys())
    rng = random.Random(42)

    pairs = []
    n_same = int(max_pairs * same_ratio)
    n_diff = max_pairs - n_same

    # same-identity pairs
    for _ in range(n_same):
        if not id_list:
            break
        ident = rng.choice(id_list)
        xs = by_id[ident]
        if len(xs) < 2:
            continue
        a, b = rng.sample(xs, 2)
        pairs.append({"src": a["path"], "tgt": b["path"], "same": 1})

    # different-identity pairs
    for _ in range(n_diff):
        if len(id_list) < 2:
            break
        ia, ib = rng.sample(id_list, 2)
        a = rng.choice(by_id[ia])
        b = rng.choice(by_id[ib])
        pairs.append({"src": a["path"], "tgt": b["path"], "same": 0})

    rng.shuffle(pairs)
    return pairs

def train_val_split(pairs: List[Dict], val_ratio: float) -> Tuple[List[Dict], List[Dict]]:
    n = len(pairs)
    nv = int(n * val_ratio)
    return pairs[nv:], pairs[:nv]

def main():
    ap = argparse.ArgumentParser(description="Incrementally update aligned dataset and pairs CSV.")
    ap.add_argument("--new-raw-dir", required=True, help="Directory of new raw images, organized as new_raw/ID/*.jpg")
    ap.add_argument("--dst-align-dir", default="data/lfw_aligned", help="Aligned images root")
    ap.add_argument("--index-csv", default="data/metadata/images.csv", help="Images index CSV to create/update")
    ap.add_argument("--pairs-train-csv", default="data/metadata/pairs_train.csv")
    ap.add_argument("--pairs-val-csv", default="data/metadata/pairs_val.csv")
    ap.add_argument("--min-per-id", type=int, default=2, help="Minimum images per identity to be considered")
    ap.add_argument("--max-per-id", type=int, default=1000, help="Cap images per identity in index")
    ap.add_argument("--max-pairs", type=int, default=20000, help="Target #pairs for (train+val)")
    ap.add_argument("--same-ratio", type=float, default=0.2, help="Fraction of same-identity pairs")
    ap.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    args = ap.parse_args()

    new_root = Path(args.new_raw_dir)
    dst_root = Path(args.dst_align_dir)
    index_csv = Path(args.index_csv)

    # 1) scan new raw images
    todo = scan_new_images(new_root)
    print(f"Scanned {len(todo)} files under {new_root}")

    # 2) align and write to dst_root
    added = []
    for ident, p in todo:
        outp = ensure_aligned_one(ident, p, dst_root)
        if outp is None:
            continue
        # collect metadata
        with Image.open(outp) as im:
            w, h = im.size
        added.append({
            "identity": ident,
            "path": outp.as_posix(),
            "md5": file_md5(outp),
            "width": str(w),
            "height": str(h)
        })
    print(f"Aligned & saved: {len(added)} images")

    # 3) merge into images index
    old = read_images_index(index_csv)
    merged = old + added
    # enforce per-identity cap
    from collections import defaultdict
    by_id = defaultdict(list)
    for r in merged:
        by_id[r["identity"]].append(r)
    capped = []
    for ident, xs in by_id.items():
        xs = xs[: args.max_per_id]
        capped.extend(xs)
    write_images_index(index_csv, capped)
    print(f"Updated index: {index_csv} (total {len(capped)} images)")

    # 4) regenerate pairs (train/val)
    images = read_images_index(index_csv)
    pairs_all = sample_pairs(images, same_ratio=args.same_ratio,
                             max_pairs=args.max_pairs, min_per_id=args.min_per_id)
    train, val = train_val_split(pairs_all, val_ratio=args.val_ratio)

    # 5) save pairs CSVs
    for path, data in [(Path(args.pairs_train_csv), train), (Path(args.pairs_val_csv), val)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=["src", "tgt", "same"])
            wr.writeheader()
            wr.writerows(data)
        print(f"Wrote {len(data)} pairs -> {path}")

if __name__ == "__main__":
    main()
