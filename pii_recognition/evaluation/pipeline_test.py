from typing import Any
from unittest.mock import patch

from pii_recognition.registration.registry import Registry

from .pipeline import get_recogniser, get_tokeniser


class RegistryNoConfig:
    # class can be instantiated without passing any args
    pass


class RegistryWithConfig:
    # must pass args to instantiate the class
    def __init__(self, param_a):
        self.param_a = param_a


def mock_registry():
    # Any is equivalent to Type[Any]
    regsitry: Registry[Any] = Registry()
    regsitry.add_item(RegistryNoConfig)
    regsitry.add_item(RegistryWithConfig)
    return regsitry


@patch("pii_recognition.evaluation.pipeline.recogniser_registry", new=mock_registry())
def test_get_recogniser():
    setup_no_config = {"name": "RegistryNoConfig"}
    actual = get_recogniser(setup_no_config)["recogniser"]  # it's in meta
    assert isinstance(actual, RegistryNoConfig)

    setup_with_config = {"name": "RegistryWithConfig", "config": {"param_a": "value_a"}}
    actual = get_recogniser(setup_with_config)["recogniser"]  # it's in meta
    assert isinstance(actual, RegistryWithConfig)
    assert actual.param_a == "value_a"


@patch("pii_recognition.evaluation.pipeline.tokeniser_registry", new=mock_registry())
def test_get_tokeniser():
    setup_no_config = {"name": "RegistryNoConfig"}
    actual = get_tokeniser(setup_no_config)["tokeniser"]  # it's in meta
    assert isinstance(actual, RegistryNoConfig)

    setup_with_config = {"name": "RegistryWithConfig", "config": {"param_a": "value_a"}}
    actual = get_tokeniser(setup_with_config)["tokeniser"]  # it's in meta
    assert isinstance(actual, RegistryWithConfig)
    assert actual.param_a == "value_a"
