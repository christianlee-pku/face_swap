import subprocess
import sys
from pathlib import Path
from typing import Optional


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_kaggle_dataset(dataset: str, out_dir: Path) -> None:
    """Download LFW from Kaggle using the CLI; requires KAGGLE_USERNAME/KAGGLE_KEY."""
    ensure_dir(out_dir)
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(out_dir),
        "--unzip",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def normalize_structure(raw_dir: Path) -> None:
    """Normalize folder structure if archive unzips into nested directories."""
    for child in raw_dir.glob("*"):
        if child.is_dir() and child.name.lower() == "lfw":
            for img in child.rglob("*.jpg"):
                target = raw_dir / img.name
                if not target.exists():
                    target.write_bytes(img.read_bytes())


def main(dataset_ref: Optional[str] = None) -> None:
    dataset_ref = dataset_ref or "ashishpatel26/lfw-dataset"
    raw_dir = Path("data/lfw/raw")
    ensure_dir(raw_dir)
    try:
        download_kaggle_dataset(dataset_ref, raw_dir)
        normalize_structure(raw_dir)
        print(f"Download complete. Raw images under {raw_dir}")
    except FileNotFoundError:
        print("kaggle CLI not found. Install via `pip install kaggle` and ensure it is on PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Kaggle download failed: {e}. Check credentials and dataset reference.")
        sys.exit(e.returncode)


if __name__ == "__main__":
    dataset_ref = sys.argv[1] if len(sys.argv) > 1 else None
    main(dataset_ref)
