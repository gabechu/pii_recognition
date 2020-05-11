import os
import tempfile
from typing import Dict, List, Optional

import mlflow
from pakkr import Pipeline, returns

from pii_recognition.data_readers import reader_registry
from pii_recognition.data_readers.reader import Data
from pii_recognition.evaluation.model_evaluator import ModelEvaluator
from pii_recognition.paths.data_path import DataPath
from pii_recognition.recognisers import registry as recogniser_registry
from pii_recognition.recognisers.entity_recogniser import EntityRecogniser
from pii_recognition.tokenisation import detokeniser_registry, tokeniser_registry
from pii_recognition.tokenisation.detokenisers import Detokeniser
from pii_recognition.tokenisation.tokenisers import Tokeniser
from pii_recognition.utils import load_yaml_file, write_iterable_to_file

from .tracking import end_tracker, log_entities_metric, start_tracker


@returns()
def enable_tracker(experiment_name: str, run_name: str):
    start_tracker(experiment_name, run_name)


@returns()
def log_config_yaml_path(config_yaml_path: str):
    mlflow.log_param("config_yaml_path", config_yaml_path)


# tokeniser has been injected to meta
@returns(tokeniser=Tokeniser)
def get_tokeniser(tokeniser_setup: Dict) -> Dict[str, Tokeniser]:
    return {
        "tokeniser": tokeniser_registry.create_instance(
            tokeniser_setup["name"], tokeniser_setup.get("config")
        )
    }


# detokeniser has been injected to meta
@returns(detokeniser=Detokeniser)
def get_detokeniser(detokeniser_setup: Dict) -> Dict[str, Detokeniser]:
    return {
        "detokeniser": detokeniser_registry.create_instance(
            detokeniser_setup["name"], detokeniser_setup.get("config")
        )
    }


# recogniser has been injected to meta
@returns(recogniser=EntityRecogniser)
def get_recogniser(recogniser_setup: Dict) -> Dict[str, EntityRecogniser]:
    recogniser_instance = recogniser_registry.create_instance(
        recogniser_setup["name"], recogniser_setup.get("config")
    )
    return {"recogniser": recogniser_instance}


# evaluator has been injected to meta
@returns(evaluator=ModelEvaluator)
def get_evaluator(
    recogniser: EntityRecogniser,
    tokeniser: Tokeniser,
    target_recogniser_entities: List[str],
    convert_labels: Optional[Dict[str, str]] = None,
) -> Dict[str, ModelEvaluator]:
    return {
        "evaluator": ModelEvaluator(
            recogniser, tokeniser, target_recogniser_entities, convert_labels
        )
    }


# Multiple outputs
@returns(Data)
def load_test_data(
    test_data_path: str,
    test_data_support_entities: List[str],
    test_is_io_schema: bool,
    detokeniser: Detokeniser,
) -> Data:
    data_path = DataPath(test_data_path)
    reader_config = {"detokeniser": detokeniser}
    reader = reader_registry.create_instance(data_path.reader_name, reader_config)
    return reader.get_test_data(
        data_path.path, test_data_support_entities, test_is_io_schema
    )


@returns()
def evaluate(
    data: Data, evaluator: ModelEvaluator,
):
    counters, mistakes = evaluator.evaluate_all(data.sentences, data.labels)
    recall, precision, f1 = evaluator.calculate_score(counters)

    log_entities_metric(recall, "recall")
    log_entities_metric(precision, "precision")
    log_entities_metric(f1, "f1")

    # save wrong predicitons to artifact
    with tempfile.TemporaryDirectory() as tempdir:
        error_file_path = os.path.join(tempdir, "prediction_mistakes.txt")
        write_iterable_to_file(mistakes, error_file_path)
        mlflow.log_artifact(error_file_path)


@returns()
def disable_tracker():
    end_tracker()


def execute_evaluation_pipeline(config_yaml: str):
    eval_pipeline = Pipeline(
        enable_tracker,
        log_config_yaml_path,
        get_tokeniser,
        get_detokeniser,
        get_recogniser,
        get_evaluator,
        load_test_data,
        evaluate,
        disable_tracker,
        name="pii_evaluation_pipeline",
        _suppress_timing_logs=False,
    )

    config = load_yaml_file(config_yaml)

    if config:
        config["config_yaml_path"] = config_yaml
        return eval_pipeline(**config)
    else:
        raise ValueError("Config YAML is empty.")
