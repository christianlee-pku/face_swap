from typing import Any, Callable, Dict, Optional


class Registry:
    """Minimal registry for registering callable components."""

    def __init__(self, name: str):
        self.name = name
        self._items: Dict[str, Any] = {}

    def register(self, name: Optional[str] = None) -> Callable[[Any], Any]:
        def decorator(obj: Any) -> Any:
            key = name or obj.__name__
            if key in self._items:
                raise KeyError(f"{self.name} registry already has key: {key}")
            self._items[key] = obj
            return obj

        return decorator

    def get(self, key: str) -> Any:
        if key not in self._items:
            raise KeyError(f"{self.name} registry missing key: {key}")
        return self._items[key]

    def build(self, cfg: Dict[str, Any]) -> Any:
        """Instantiate a registered callable/class using cfg['type'] and remaining kwargs."""
        if "type" not in cfg:
            raise KeyError("Config missing 'type' for registry build")
        cfg = cfg.copy()
        obj_type = cfg.pop("type")
        target = self.get(obj_type)
        if isinstance(target, type):
            return target(**cfg)
        if callable(target):
            return target(**cfg)
        raise TypeError(f"Registry entry for {obj_type} is not callable or class")


# Global registries
DATASETS = Registry("datasets")
MODELS = Registry("models")
LOSSES = Registry("losses")
AUGMENTATIONS = Registry("augmentations")
PIPELINES = Registry("pipelines")
RUNNERS = Registry("runners")
EXPORTERS = Registry("exporters")
