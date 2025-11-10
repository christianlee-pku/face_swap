import csv
import logging
from pathlib import Path

def setup_logger(log_path: Path, level: str = "INFO"):
    """
    Create a console+file logger with a clear format.
    """
    logger = logging.getLogger("train")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    # file
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    fh.setFormatter(fmt)
    # console
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger

def human_bytes(n: int) -> str:
    """
    Nicely format bytes.
    """
    units = ["B","KB","MB","GB","TB","PB"]
    i = 0; f = float(n)
    while f >= 1024.0 and i < len(units)-1:
        f /= 1024.0; i += 1
    return f"{f:.1f}{units[i]}"

def save_plot(history: dict, save_path: Path, title: str, logger):
    """
    Save loss curves. If matplotlib not available, warn and continue.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,4))
    for k in ["g","id","perc","l1f","l1bg"]:
        if k in history and len(history[k]) > 0:
            plt.plot(history[k], label=k)
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss"); plt.title(title)
    plt.tight_layout(); plt.savefig(save_path); plt.close()

class CSVLogger:
    """
    Simple CSV logger that writes a header once and appends rows.
    """
    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self._file = None
        self._writer = None
        self._ensure_file()

    def _ensure_file(self):
        newfile = not self.path.exists()
        self._file = open(self.path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        if newfile:
            self._writer.writeheader()

    def write(self, row: dict):
        self._writer.writerow(row)

    def close(self):
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        self.close()
