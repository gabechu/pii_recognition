from dagster import execute_pipeline, pipeline, solid

from recognisers.crf_recogniser import CrfRecogniser
from recognisers.first_letter_uppercase_recogniser import FirstLetterUppercaseRecogniser
from recognisers.flair_recogniser import FlairRecogniser
from recognisers.recogniser_registry import RecogniserRegistry
from recognisers.spacy_recogniser import SpacyRecogniser
from recognisers.stanza_recogniser import StanzaRecogniser

from .types import RecogniserRegistryDT


@solid
def register_recognisers(context) -> RecogniserRegistryDT:
    registry = RecogniserRegistry()

    registry.add_recogniser(CrfRecogniser)
    registry.add_recogniser(FirstLetterUppercaseRecogniser)
    registry.add_recogniser(FlairRecogniser)
    registry.add_recogniser(SpacyRecogniser)
    registry.add_recogniser(StanzaRecogniser)

    return registry


@solid
def get_recogniser(
    context, recogniser_registry: RecogniserRegistryDT, recogniser_name: str
):
    return recogniser_registry.registry[recogniser_name]


@solid
def initialise_recogniser(context):
    ...


@solid
def get_evaluation_data(context):
    ...


@solid
def initialise_evaluator(context):
    ...


@solid
def evaluate_and_logging(context):
    ...


@pipeline
def evaluation_pipeline():
    return get_recogniser(register_recognisers())


if __name__ == "__main__":
    environment_dict = {
        "solids": {
            "get_recogniser": {
                "inputs": {"recogniser_name": {"value": "CrfRecogniser"}}
            }
        }
    }

    result = execute_pipeline(evaluation_pipeline, environment_dict=environment_dict)
    assert result.success
