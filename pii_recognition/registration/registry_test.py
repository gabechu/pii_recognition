from typing import Any

from .registry import Registry


def test_Registry_add_item():
    class ToyClass:
        pass

    # Any is equivalent to Type[Any]
    actual = Registry[Any]()

    actual.add_item(ToyClass)
    assert isinstance(actual, dict)
    assert actual["ToyClass"] == ToyClass

    actual.add_item(ToyClass, "TClass")
    assert isinstance(actual, dict)
    assert actual["TClass"] == ToyClass


def test_Registry_create_instance():
    class ToyClass:
        def __init__(self, a):
            self.a = a

    # Any is equivalent to Type[Any]
    registry = Registry[Any]()
    registry.add_item(ToyClass)

    actual = registry.create_instance(name="ToyClass", config={"a": "value_a"})
    assert isinstance(actual, ToyClass)
    assert actual.a == "value_a"
