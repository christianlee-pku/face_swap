import hashlib
import json
import random
import shutil
import logging
from pathlib import Path
from typing import Any, Dict, List

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None
    np = None

try:
    from facenet_pytorch import MTCNN  # type: ignore
    from PIL import Image
except Exception:  # pragma: no cover
    MTCNN = None
    Image = None


def compute_checksum(path: Path) -> str:
    """Compute SHA256 checksum for a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_and_align(image_path: Path, output_path: Path) -> Dict[str, Any]:
    """Detection/alignment via RetinaFace when available; fallback to MTCNN; returns aligned file path and metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not image_path.exists():
        return {"aligned_path": str(image_path), "bbox": None, "landmarks": None}

    # RetinaFace path (disabled: dependency removed). Falls through to MTCNN.

    # MTCNN fallback
    if MTCNN is None or Image is None:
        return {"aligned_path": str(image_path), "bbox": None, "landmarks": None}
    mtcnn = MTCNN(keep_all=False, image_size=160)
    img_pil = Image.open(image_path).convert("RGB")
    face, prob = mtcnn(img_pil, return_prob=True)
    if face is None:
        return {"aligned_path": str(image_path), "bbox": None, "landmarks": None}
    # face is CHW float tensor in [0,1]; convert to HWC uint8
    aligned = face.permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
    aligned_img = Image.fromarray(aligned, mode="RGB")
    if output_path.exists() and output_path.is_dir():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned_img.save(output_path)
    return {"aligned_path": str(output_path), "bbox": None, "landmarks": None, "prob": prob}


def build_manifest_from_raw(raw_dir: Path, output_dir: Path, logger: logging.Logger | None = None) -> List[Dict[str, Any]]:
    """Process raw LFW images (nested person folders) into aligned outputs; returns manifest items with checksums."""
    output_dir.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, Any]] = []
    images = sorted(raw_dir.rglob("*.jpg"))
    total = len(images)
    for idx, img_path in enumerate(images, start=1):
        rel = img_path.relative_to(raw_dir)
        aligned_target = output_dir / rel
        result = detect_and_align(img_path, aligned_target)
        aligned_path = Path(result["aligned_path"])
        if aligned_path.exists() and aligned_path.is_dir():
            shutil.rmtree(aligned_path)
            aligned_path = aligned_target
        if not aligned_path.exists():
            aligned_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, aligned_target)
            aligned_path = aligned_target
        # If detection failed and pointed to raw path, mirror the raw file into processed to keep manifest relative.
        if not aligned_path.is_relative_to(output_dir):
            aligned_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(aligned_path, aligned_target)
            aligned_path = aligned_target
        checksum = compute_checksum(aligned_path) if aligned_path.exists() else ""
        item_id = rel.with_suffix("").as_posix().replace("/", "_")
        items.append({"id": item_id, "path": aligned_path.relative_to(output_dir).as_posix(), "checksum": checksum})
        if logger and idx % 200 == 0:
            logger.info("Processed %d/%d images (%.1f%%)", idx, total, idx * 100.0 / max(total, 1))
    if logger:
        logger.info("Finished alignment for %d images", total)
    return items


def write_manifest(
    items: List[Dict[str, Any]],
    manifest_path: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    version: str = "1.0.0",
    meta: Dict[str, Any] = None,
    pairs: List[Dict[str, str]] = None,
) -> Path:
    """Shuffle items, split into train/val/test, and write manifest."""
    meta = meta or {}
    pairs = pairs or []
    ids = [item["id"] for item in items]
    random.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    splits = {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }
    items_by_id = {item["id"]: item for item in items}
    checksums = {item["id"]: item.get("checksum", "") for item in items}
    manifest = {
        "version": version,
        "items": [items_by_id[i] for i in ids],
        "splits": splits,
        "checksums": checksums,
        "meta": {"source": "LFW", "aligned": True, **meta},
        "pairs": pairs,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True))
    return manifest_path
