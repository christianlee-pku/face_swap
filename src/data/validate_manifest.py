import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def compute_checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path) -> Dict:
    return json.loads(path.read_text())


def validate_items(manifest: Dict, root: Path) -> Tuple[List[str], List[str]]:
    missing = []
    bad_checksum = []
    checksums = manifest.get("checksums", {})
    items = manifest.get("items", [])
    for item in items:
        rel_path = item.get("path")
        item_id = item.get("id")
        if rel_path is None or item_id is None:
            continue
        full_path = root / rel_path
        if not full_path.exists():
            missing.append(rel_path)
            continue
        expected = checksums.get(item_id, "")
        if expected:
            actual = compute_checksum(full_path)
            if actual != expected:
                bad_checksum.append(rel_path)
    return missing, bad_checksum


def validate_splits(manifest: Dict) -> bool:
    splits = manifest.get("splits", {})
    items = manifest.get("items", [])
    ids = {item.get("id") for item in items if item.get("id") is not None}
    split_ids = set()
    for id_list in splits.values():
        split_ids.update(id_list)
    # ensure splits reference known ids and are non-empty
    return split_ids.issubset(ids) and all(len(v) >= 0 for v in splits.values())


def main():
    manifest_path = Path("data/lfw/manifest.json")
    root = Path("data/lfw/processed")
    if len(sys.argv) > 1:
        manifest_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        root = Path(sys.argv[2])

    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    manifest = load_manifest(manifest_path)
    missing, bad_checksum = validate_items(manifest, root)
    splits_ok = validate_splits(manifest)

    if missing:
        print("Missing files:", missing)
    if bad_checksum:
        print("Checksum mismatches:", bad_checksum)
    if not splits_ok:
        print("Split validation failed: splits reference unknown ids or are malformed.")

    if not missing and not bad_checksum and splits_ok:
        print("Manifest validation PASSED")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
