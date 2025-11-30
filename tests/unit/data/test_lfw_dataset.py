import json
from pathlib import Path

from src.data.lfw_dataset import LFWDataset
from src.data.transforms import LightAugmentation
from src.data.manifest import DatasetManifest


def test_lfw_dataset_reads_manifest(tmp_path: Path):
    manifest = DatasetManifest(
        version="1.0",
        items=[{"id": "a", "path": "a.jpg"}, {"id": "b", "path": "b.jpg"}],
        splits={"train": ["a"], "val": ["b"]},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest.save(manifest_path)

    ds = LFWDataset(root=str(tmp_path), split="train", manifest=str(manifest_path))
    assert len(ds) == 1
    sample = ds[0]
    assert sample["id"] == "a"


def test_lfw_dataset_with_transform(tmp_path: Path):
    manifest = DatasetManifest(version="1.0", items=[{"id": "a", "path": "a.jpg"}], splits={"train": ["a"]})
    manifest_path = tmp_path / "manifest.json"
    manifest.save(manifest_path)
    aug = LightAugmentation(seed=123)
    ds = LFWDataset(root=str(tmp_path), split="train", manifest=str(manifest_path), transform=aug)
    sample = ds[0]
    assert sample["aug_seed"] == 123
