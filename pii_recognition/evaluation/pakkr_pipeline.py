from typing import Dict, List, Tuple

from mlflow import ActiveRun
from pakkr import returns

from pii_recognition.data_readers import reader_registry
from pii_recognition.evaluation.model_evaluator import ModelEvaluator
from pii_recognition.paths.data_path import DataPath
from pii_recognition.recognisers import registry as recogniser_registry
from pii_recognition.recognisers.entity_recogniser import EntityRecogniser
from pii_recognition.tokenisation import tokeniser_registry
from pii_recognition.tokenisation.tokenisers import Tokeniser

from .tracking import end_tracker, log_entities_metric, start_tracker


@returns(ActiveRun)
def enable_tracker(experiment_name: str, run_name: str) -> ActiveRun:
    return start_tracker(experiment_name, run_name)


# recogniser has been injected to meta
@returns(recogniser=EntityRecogniser)
def get_recogniser(recogniser_setup: Dict) -> Dict[str, EntityRecogniser]:
    recogniser_instance = recogniser_registry.create_instance(
        recogniser_setup["name"], recogniser_setup.get("config")
    )
    return {"recogniser": recogniser_instance}


# tokeniser has been injected to meta
@returns(tokeniser=Tokeniser)
def get_tokeniser(tokeniser_setup: Dict) -> Dict[str, Tokeniser]:
    return {
        "tokeniser": tokeniser_registry.create_instance(
            tokeniser_setup["name"], tokeniser_setup.get("config")
        )
    }


# evaluator has been injected to meta
@returns(evaluator=ModelEvaluator)
def get_evaluator(
    recogniser: EntityRecogniser,
    tokeniser: Tokeniser,
    target_recogniser_entities: List[str],
    convert_labels: Dict[str, str],
) -> Dict[str, ModelEvaluator]:
    return {
        "evaluator": ModelEvaluator(
            recogniser, tokeniser, target_recogniser_entities, convert_labels
        )
    }


# Multiple outputs
@returns(List, List)
def load_test_data(
    test_data_path: str, detokeniser_setup: Dict
) -> Tuple[List[str], List[List[str]]]:
    data_path = DataPath(test_data_path)
    reader = reader_registry.create_instance(data_path.reader_name, detokeniser_setup)
    return reader.get_test_data(data_path.path)


@returns()
def evaluate(
    X_test: List[str], y_test: List[List[str]], evaluator: ModelEvaluator,
):
    counters, mistakes = evaluator.evaluate_all(X_test, y_test)
    recall, precision, f1 = evaluator.calculate_score(counters)

    log_entities_metric(recall)
    log_entities_metric(precision)
    log_entities_metric(f1)


@returns()
def disable_tracker():
    end_tracker()
