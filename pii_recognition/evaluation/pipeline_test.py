from unittest.mock import patch

from .pipeline import get_recogniser
from pii_recognition.registration.registry import Registry
from typing import Any


class RegistryNoConfig:
    # class can be instantiated without passing any args
    pass


class RegistryWithConfig:
    # must pass args to instantiate the class
    def __init__(self, config_a):
        self.config_a = config_a


def mock_registry():
    # Any is equivalent to Type[Any]
    regsitry: Registry[Any] = Registry()
    regsitry.add_item(RegistryNoConfig)
    regsitry.add_item(RegistryWithConfig)
    return regsitry


@patch("pii_recognition.evaluation.pipeline.recogniser_registry", new=mock_registry())
def test_get_recogniser():
    actual = get_recogniser("RegistryNoConfig")["recogniser"]
    assert isinstance(actual, RegistryNoConfig)

    actual = get_recogniser("RegistryWithConfig", {"config_a": "attr_a"})["recogniser"]
    assert isinstance(actual, RegistryWithConfig)
    assert actual.config_a == "attr_a"
