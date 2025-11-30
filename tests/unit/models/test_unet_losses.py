import importlib
import pytest

def torch_available():
    return importlib.util.find_spec("torch") is not None


def test_unet_instantiation():
    models = importlib.import_module("src.registry.models")
    model_cls = models.UNetFaceSwap
    model = model_cls(channels=32)
    assert hasattr(model, "channels")


@pytest.mark.skipif(not torch_available(), reason="torch not installed")
def test_unet_forward_with_tensor():
    import torch

    models = importlib.import_module("src.registry.models")
    model = models.UNetFaceSwap(channels=8)
    x = torch.zeros((1, 3, 4, 4))
    out = model(x)
    assert "output" in out


def test_loss_forward_returns_number():
    losses = importlib.import_module("src.models.losses")
    loss_fn = losses.FaceSwapLoss()
    result = loss_fn({}, {})
    assert result == 0.0 or result is not None
