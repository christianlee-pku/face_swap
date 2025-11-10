import os
import csv
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import pandas as pd

from utils.exif_fix import load_image_fix_exif
from utils.checksums import sha256_file, dhash_image
from utils.face_align import detect_face_5pt, align_face_256
from utils.mask import soft_ellipse_mask
from utils.splits import build_identity_splits, save_splits_json
from utils.pairs import make_pairs, save_pairs_csv

RAW_DIR = Path("data/lfw_raw")         # where original LFWPeople is placed
ALIGNED_DIR = Path("data/aligned_256")
MASK_DIR = Path("data/masks_256")
META_DIR = Path("data/metadata")
TMP_DIR = Path("data/tmp")

def ensure_dirs():
    for d in [ALIGNED_DIR, MASK_DIR, META_DIR, TMP_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def scan_and_align():
    """
    Iterate raw LFW images by identity, align to 256, save aligned image & mask.
    Also record metadata (path, sha256, dhash, blur, prob, box size).
    """
    rows = []
    people = sorted([p for p in RAW_DIR.iterdir() if p.is_dir()])
    for person_dir in tqdm(people, desc="Identities"):
        person = person_dir.name
        out_person_dir = ALIGNED_DIR / person
        out_person_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(person_dir.iterdir()):
            if not img_path.suffix.lower() in [".jpg",".jpeg",".png"]: 
                continue

            # 1) basic metadata
            sha256 = sha256_file(img_path)
            try:
                pil_img = load_image_fix_exif(img_path)
            except Exception as e:
                print(f"[WARN] Failed to load {img_path}: {e}")
                continue

            # 2) detect face (box + 5pt landmarks)
            box, lm5, prob = detect_face_5pt(pil_img, prob_thresh=0.90)
            if box is None:
                status = "no_face"
                rows.append({
                    "person": person, "raw_path": str(img_path),
                    "sha256": sha256, "status": status
                })
                continue

            # 3) align to 256
            aligned, M = align_face_256(pil_img, lm5)
            if aligned is None:
                status = "align_fail"
                rows.append({
                    "person": person, "raw_path": str(img_path),
                    "sha256": sha256, "status": status
                })
                continue

            # 4) compute quality scores
            # blur score
            arr_gray = np.array(aligned.convert("L"))
            blur = cv2.Laplacian(arr_gray, cv2.CV_64F).var()
            # box size
            x1,y1,x2,y2 = box
            face_w, face_h = (x2-x1), (y2-y1)

            # 5) save aligned image (PNG keeps quality)
            fname = img_path.stem + ".png"
            out_img = out_person_dir / fname
            aligned.save(out_img)

            # 6) make and save mask
            mask = soft_ellipse_mask(h=256, w=256, scale_x=0.9, scale_y=1.0, blur=25)
            out_mask_dir = MASK_DIR / person
            out_mask_dir.mkdir(parents=True, exist_ok=True)
            out_mask_path = out_mask_dir / fname
            Image.fromarray(mask).save(out_mask_path)

            # 7) dhash for near-dup flag (optional)
            dhash_val = dhash_image(aligned)

            # 8) record metadata
            rows.append({
                "person": person,
                "raw_path": str(img_path),
                "aligned_path": str(out_img),
                "mask_path": str(out_mask_path),
                "sha256": sha256,
                "dhash": int(dhash_val),
                "prob": float(prob),
                "box_w": float(face_w),
                "box_h": float(face_h),
                "blur": float(blur),
                "status": "ok"
            })

    # write CSV
    meta_csv = META_DIR / "image_index.csv"
    with open(meta_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[INFO] Wrote metadata to {meta_csv}")
    return meta_csv

def build_splits_and_pairs():
    splits = build_identity_splits(str(ALIGNED_DIR), test_size=0.10, val_size=0.10, seed=42)
    save_splits_json(splits, META_DIR / "splits.json")
    print("[INFO] Saved splits.json")

    # build pairs per split
    pairs_train = make_pairs(str(ALIGNED_DIR), splits, "train", same_ratio=0.5, max_pairs=200000)
    pairs_val   = make_pairs(str(ALIGNED_DIR), splits, "val",   same_ratio=0.5, max_pairs=20000)
    pairs_test  = make_pairs(str(ALIGNED_DIR), splits, "test",  same_ratio=0.5, max_pairs=20000)

    save_pairs_csv(pairs_train, META_DIR / "pairs_train.csv")
    save_pairs_csv(pairs_val,   META_DIR / "pairs_val.csv")
    save_pairs_csv(pairs_test,  META_DIR / "pairs_test.csv")
    print("[INFO] Saved pairs CSVs")

def main():
    ensure_dirs()

    meta_csv = scan_and_align()
    # basic summary
    df = pd.read_csv(meta_csv)
    print(df["status"].value_counts())
    ok_df = df[df["status"]=="ok"]
    if len(ok_df) == 0:
        print("[WARN] No valid aligned images found.")
        return
    print(f"[INFO] OK images: {len(ok_df)} across identities: {ok_df['person'].nunique()}")

    build_splits_and_pairs()
    print("[DONE] Phase 1 preprocessing complete.")

if __name__ == "__main__":
    # OpenCV import local (avoid circular import)
    import cv2
    main()
