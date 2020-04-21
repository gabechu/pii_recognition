from unittest.mock import patch

from .pipeline import get_recogniser


class RegistryNoConfig:
    # class can be instantiated without passing any args
    pass


class RegistryWithConfig:
    # must pass args to instantiate the class
    def __init__(self, config_a):
        self.config_a = config_a


def mock_registry():
    return {
        "RegistryNoConfig": RegistryNoConfig,
        "RegistryWithConfig": RegistryWithConfig,
    }


@patch("pii_recognition.evaluation.pipeline.recogniser_registry", new=mock_registry())
def test_get_recogniser():
    actual = get_recogniser("RegistryNoConfig")["recogniser"]
    assert isinstance(actual, RegistryNoConfig)

    actual = get_recogniser("RegistryWithConfig", {"config_a": "attr_a"})["recogniser"]
    assert isinstance(actual, RegistryWithConfig)
    assert actual.config_a == "attr_a"
