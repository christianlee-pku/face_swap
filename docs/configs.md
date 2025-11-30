# Configs & Conventions

## Layout

- `configs/face_swap/baseline.yaml` — train/eval baseline
- `configs/face_swap/eval.yaml` — eval-only
- `configs/face_swap/export_edge.yaml` — export/benchmark
- `configs/face_swap/ablation_*.yaml` — ablations

## Naming

- Experiments: `<task>-<model>-<data>-<id>`
- Work dirs: `work_dirs/<exp-name>-<timestamp>/`
- Registry keys: `pkg.component.name`

## Required Fields

- `dataset`: type, root, manifest, split, optional augmentations
- `model`: type, params (e.g., channels)
- `loss`: type, weights
- `runner`: type
- Optional: optimizer/scheduler/hooks

## Reproducibility

- `seed` set; `config.snapshot.json` and `env.hash` saved to work_dir.

## Adding New Configs

- Place under `configs/face_swap/`
- Reference registered components only (no absolute imports)
- Include minimal runnable defaults; avoid hidden defaults.
