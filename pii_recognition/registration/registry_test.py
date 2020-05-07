from typing import Any

from .registry import Registry


def test_Registry_register():
    class ToyClass:
        pass

    # Any is equivalent to Type[Any]
    actual = Registry[Any]()

    actual.register(ToyClass)
    assert isinstance(actual, dict)
    assert actual["ToyClass"] == ToyClass

    actual.register(ToyClass, "TClass")
    assert isinstance(actual, dict)
    assert actual["TClass"] == ToyClass


def test_Registry_create_instance():
    class ToyClass:
        def __init__(self, a):
            self.a = a

    # Any is equivalent to Type[Any]
    registry = Registry[Any]()
    registry.register(ToyClass)

    actual = registry.create_instance(name="ToyClass", config={"a": "value_a"})
    assert isinstance(actual, ToyClass)
    assert actual.a == "value_a"
