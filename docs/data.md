# Data & Manifests

## Sources

- LFW raw images in `data/lfw/raw/`

## Download, Alignment & Manifests

- Expected raw layout: `data/lfw/raw/<person>/<image>.jpg` plus `pairs.txt` and `pairs_01~pairs_10.txt` (pairs are ingested into the manifest).
- Kaggle download: use Kaggle API to fetch LFW into `data/lfw/raw/`; handle auth/token. Command (config-driven):
  ```bash
  PYTHONPATH=src python -m interfaces.cli prepare-data --config configs/face_swap/data_prepare.yaml
  ```
- Structure is preserved: aligned outputs go to `data/lfw/processed/<person>/<image>.jpg`.
- Detection/alignment: MTCNN (RetinaFace disabled); crops at MTCNN default size (160); if detection fails, the raw image is copied to processed; checksums captured in manifest.
- Progress logging: data prep logs INFO every ~200 images and reports total pairs/items when manifest is written.
- Manifest (`data/lfw/manifest.json`) includes version, items (id, path, checksum), splits (80/10/10), checksums, pairs metadata, and meta.
- Update utility: `src/data/update_dataset.py` to bump manifest version/changelog.

## Splits

- Recommended: train/val/test = 80/10/10; keep consistent across runs.

## Augmentations

- Light: color jitter, flip, mild blur; deterministic seeds.

## Validation

- Run validation:
  ```bash
  PYTHONPATH=src python -m interfaces.cli validate-manifest --manifest data/lfw/manifest.json --processed-dir data/lfw/processed
  ```
- Ensure aligned images exist at `data/lfw/processed/` paths referenced in manifest; fix or regenerate if missing.
