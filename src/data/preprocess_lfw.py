import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import yaml

from .retinaface_align import build_manifest_from_raw, write_manifest
from .download_lfw import main as download_lfw_main


def parse_pairs(pairs_root: str) -> List[Dict[str, str]]:
    pairs = []
    root = Path(pairs_root)
    for txt in root.glob("pairs*.txt"):
        for line in txt.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) == 3:
                # same person pair: name idx1 idx2
                name, idx1, idx2 = parts
                pairs.append({"type": "same", "person": name, "img1": idx1, "img2": idx2})
            elif len(parts) == 4:
                # different person: name1 idx1 name2 idx2
                n1, i1, n2, i2 = parts
                pairs.append({"type": "diff", "person1": n1, "img1": i1, "person2": n2, "img2": i2})
    return pairs


def run_preprocess(
    download: bool,
    dataset: str,
    raw_dir: str,
    proc_dir: str,
    manifest: str,
    train_ratio: float,
    val_ratio: float,
    version: str,
    pairs_dir: str = "data/lfw/raw",
) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("preprocess_lfw")
    raw_path = Path(raw_dir)
    proc_path = Path(proc_dir)
    manifest_path = Path(manifest)

    if download:
        log.info("Downloading LFW from Kaggle dataset %s ...", dataset)
        download_lfw_main(dataset)

    log.info("Aligning and building manifest from raw=%s to processed=%s ...", raw_path, proc_path)
    items = build_manifest_from_raw(raw_path, proc_path, logger=log)
    pairs = parse_pairs(pairs_dir)
    log.info("Loaded %d pairs files total entries=%d", len(list(Path(pairs_dir).glob('pairs*.txt'))), len(pairs))
    write_manifest(items, manifest_path, train_ratio=train_ratio, val_ratio=val_ratio, version=version, pairs=pairs)
    log.info("Wrote manifest to %s with %d items", manifest_path, len(items))
    return len(items)


def main(argv: Optional[list] = None):
    parser = argparse.ArgumentParser(description="Download (optional), align, and build LFW manifest (config-driven).")
    parser.add_argument("--config", required=True, help="YAML config for data preparation")
    args = parser.parse_args(argv)

    cfg: Dict[str, Any] = yaml.safe_load(Path(args.config).read_text()) or {}

    run_preprocess(
        download=cfg.get("download", False),
        dataset=cfg.get("dataset", "ashishpatel26/lfw-dataset"),
        raw_dir=cfg.get("raw_dir", "data/lfw/raw"),
        proc_dir=cfg.get("proc_dir", "data/lfw/processed"),
        manifest=cfg.get("manifest", "data/lfw/manifest.json"),
        train_ratio=cfg.get("train_ratio", 0.8),
        val_ratio=cfg.get("val_ratio", 0.1),
        version=cfg.get("version", "1.0.0"),
    )


if __name__ == "__main__":
    main()
