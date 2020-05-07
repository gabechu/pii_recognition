from typing import Any
from unittest.mock import patch

from pii_recognition.registration.registry import Registry

from .pakkr_pipeline import (
    get_recogniser,
    get_tokeniser,
    load_test_data,
    reader_registry,
)


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
    regsitry.register(RegistryNoConfig)
    regsitry.register(RegistryWithConfig)
    return regsitry


@patch(
    "pii_recognition.evaluation.pakkr_pipeline.recogniser_registry", new=mock_registry()
)
def test_get_recogniser():
    setup_no_config = {"name": "RegistryNoConfig"}
    actual = get_recogniser(setup_no_config)["recogniser"]  # it's in meta
    assert isinstance(actual, RegistryNoConfig)

    setup_with_config = {"name": "RegistryWithConfig", "config": {"param_a": "value_a"}}
    actual = get_recogniser(setup_with_config)["recogniser"]  # it's in meta
    assert isinstance(actual, RegistryWithConfig)
    assert actual.param_a == "value_a"


@patch(
    "pii_recognition.evaluation.pakkr_pipeline.tokeniser_registry", new=mock_registry()
)
def test_get_tokeniser():
    setup_no_config = {"name": "RegistryNoConfig"}
    actual = get_tokeniser(setup_no_config)["tokeniser"]  # it's in meta
    assert isinstance(actual, RegistryNoConfig)

    setup_with_config = {"name": "RegistryWithConfig", "config": {"param_a": "value_a"}}
    actual = get_tokeniser(setup_with_config)["tokeniser"]  # it's in meta
    assert isinstance(actual, RegistryWithConfig)
    assert actual.param_a == "value_a"


def test_load_test_data():
    data_path = "pii_recognition/datasets/conll2003/eng.testa"
    detokeniser_setup = {"name": "SimpleDetokeniser"}
    with patch.object(reader_registry, "create_instance") as mock_registry:
        load_test_data(data_path, detokeniser_setup)
        mock_registry.assert_called_with("ConllReader", detokeniser_setup)
