import pytest

from src.registry import Registry


def test_register_and_get():
    reg = Registry("test")

    @reg.register()
    class Foo:
        pass

    assert reg.get("Foo") is Foo


def test_register_duplicate_raises():
    reg = Registry("test")

    @reg.register("dup")
    def foo():
        return 1

    with pytest.raises(KeyError):
        reg.register("dup")(lambda: 2)


def test_build_callable_and_class():
    reg = Registry("test")

    @reg.register()
    def make_item(x):
        return {"x": x}

    @reg.register()
    class Item:
        def __init__(self, y):
            self.y = y

    assert reg.build({"type": "make_item", "x": 5}) == {"x": 5}
    obj = reg.build({"type": "Item", "y": 7})
    assert isinstance(obj, Item)
    assert obj.y == 7


def test_build_missing_type():
    reg = Registry("test")
    with pytest.raises(KeyError):
        reg.build({})
