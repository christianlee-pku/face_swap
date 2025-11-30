import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _json_formatter(record: logging.LogRecord) -> str:
    payload: Dict[str, Any] = {
        "level": record.levelname,
        "name": record.name,
        "message": record.getMessage(),
    }
    if record.exc_info:
        payload["exc_info"] = logging.Formatter().formatException(record.exc_info)
    return json.dumps(payload, ensure_ascii=True)


class JsonFormatter(logging.Formatter):
    """Simple JSON formatter to keep logs structured."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        return _json_formatter(record)


def _has_file_handler(logger: logging.Logger, log_file: Path) -> bool:
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")) == log_file:
            return True
    return False


def setup_logger(
    name: str,
    level: int = logging.INFO,
    json_format: bool = True,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Create or retrieve a logger; default JSON output, optional plain text, optional file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        if json_format:
            handler.setFormatter(JsonFormatter())
        else:
            handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if not _has_file_handler(logger, log_file):
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            if json_format:
                fh.setFormatter(JsonFormatter())
            else:
                fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            logger.addHandler(fh)

    logger.propagate = False
    return logger


def log_dict(logger: logging.Logger, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """Helper to emit a structured log with extra fields."""
    extra = extra or {}
    payload = {"message": message, **extra}
    logger.info(json.dumps(payload, ensure_ascii=True))


def log_error(logger: logging.Logger, message: str, code: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """Emit an error with a code to aid observability."""
    extra = extra or {}
    payload = {"message": message, "code": code, **extra}
    logger.error(json.dumps(payload, ensure_ascii=True))
