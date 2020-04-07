from typing import Dict, List, Tuple, Type

from pakkr import returns

from pii_recognition.evaluation.model_evaluator import ModelEvaluator
from pii_recognition.recognisers.entity_recogniser import EntityRecogniser


@returns(Type[EntityRecogniser])
def get_recogniser(recogniser_name: str) -> Type[EntityRecogniser]:
    ...


# recogniser has been injected to meta
@returns(recogniser=EntityRecogniser)
def initialise_recogniser(
    recogniser: EntityRecogniser, recogniser_config: Dict[str, str]
) -> Dict[str, EntityRecogniser]:
    ...


# evaluator has been injected to meta
@returns(evaluator=ModelEvaluator)
def initialise_evaluator(
    recogniser: EntityRecogniser,
    target_entities: List[str],
    tokeniser_name: str,
    convert_labels: Dict[str, str],
) -> Dict[str, ModelEvaluator]:
    ...


# Multiple outputs
@returns(List, List)
def prepare_evaluation_data(eval_data_path: str) -> Tuple[List[str], List[List[str]]]:
    ...


@returns
def evaluate_and_logging(
    X_test: List[str],
    y_test: List[List[str]],
    recogniser: EntityRecogniser,
    evaluator: ModelEvaluator,
    experiment_name: str,
    run_name: str,
):
    ...
