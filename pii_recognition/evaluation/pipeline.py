from typing import Dict, List, Tuple

from pakkr import returns

from pii_recognition import registry
from pii_recognition.evaluation.model_evaluator import ModelEvaluator
from pii_recognition.recognisers.entity_recogniser import EntityRecogniser


# recogniser has been injected to meta
@returns(recogniser=EntityRecogniser)
def get_recogniser(
    recogniser_name: str, recogniser_config: Dict = {}
) -> Dict[str, EntityRecogniser]:
    recogniser_class = registry.recogniser[recogniser_name]
    recogniser_instance = recogniser_class(**recogniser_config)
    return {"recogniser": recogniser_instance}


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
