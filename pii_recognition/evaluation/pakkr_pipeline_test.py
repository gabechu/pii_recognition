from typing import Any
from unittest.mock import Mock, call, patch

from pii_recognition.data_readers.reader import Data
from pii_recognition.registration.registry import Registry

from .pakkr_pipeline import (
    evaluate,
    get_detokeniser,
    get_recogniser,
    get_tokeniser,
    load_test_data,
    log_config_yaml_path,
    mlflow,
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


@patch.object(mlflow, "log_param")
def test_log_config_yaml_path(mock_log_param):
    log_config_yaml_path("fake_path")
    mock_log_param.assert_called_with("config_yaml_path", "fake_path")


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


@patch(
    "pii_recognition.evaluation.pakkr_pipeline.detokeniser_registry",
    new=mock_registry(),
)
def test_get_detokeniser():
    setup_no_config = {"name": "RegistryNoConfig"}
    actual = get_detokeniser(setup_no_config)["detokeniser"]  # it's in meta
    assert isinstance(actual, RegistryNoConfig)

    setup_with_config = {"name": "RegistryWithConfig", "config": {"param_a": "value_a"}}
    actual = get_detokeniser(setup_with_config)["detokeniser"]  # it's in meta
    assert isinstance(actual, RegistryWithConfig)
    assert actual.param_a == "value_a"


@patch.object(mlflow, "log_param", new=Mock())
def test_load_test_data():
    data_path = "pii_recognition/datasets/conll2003/eng.testa"
    detokeniser = Mock()
    with patch.object(reader_registry, "create_instance") as mock_registry:
        load_test_data(data_path, ["I-LOC"], True, detokeniser)
        mock_registry.assert_called_with("ConllReader", {"detokeniser": detokeniser})


@patch("pii_recognition.evaluation.pakkr_pipeline.log_entities_metric")
@patch.object(mlflow, "log_artifact", new=Mock())
def test_evaluate(mock_log):
    X_test = ["This is Bob from Melbourne ."]
    y_test = [["O", "O", "I-PER", "O", "O", "O"]]
    data = Data(X_test, y_test, ["I-PER"], True)

    evaluator = Mock()
    evaluator.evaluate_all.return_value = "fake_counter", "fake_mistakes"
    evaluator.calculate_score.return_value = (
        {"I-PER": 0.5},
        {"I-PER": 0.4},
        {"I-PER": 0.3},
    )

    evaluate(data, evaluator)
    mock_log.assert_has_calls(
        [
            call({"I-PER": 0.5}, "recall"),
            call({"I-PER": 0.4}, "precision"),
            call({"I-PER": 0.3}, "f1"),
        ]
    )
