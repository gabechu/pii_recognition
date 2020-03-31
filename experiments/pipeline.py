import inspect

from dagster import Field, List, String, execute_pipeline, pipeline, solid

from recognisers.recogniser_registry import RecogniserRegistry

from .types import RecogniserRegistryDT


@solid
def get_recogniser(context, recogniser_name: str):
    registry = RecogniserRegistry()
    return registry.get_recogniser(recogniser_name)


@solid(
    config={
        "supported_entities": Field(list, is_required=True),
        "supported_languages": Field(list, is_required=True),
        "model_path": Field(String, is_required=False),
        "tokeniser": Field(
            String, is_required=False, default_value="nltk_word_tokenizer"
        ),
        "model_name": Field(String, is_required=False),
    }
)
def initialise_recogniser(context, recogniser_cls):
    required_args = inspect.getfullargspec(recogniser_cls).args
    required_args.pop(0)  # remove self

    filtered_config = {
        key: value
        for key, value in context.solid_config.items()
        if key in required_args
    }
    return recogniser_cls(**filtered_config)


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
    return initialise_recogniser(get_recogniser())


if __name__ == "__main__":
    environment_dict = {
        "solids": {
            "get_recogniser": {
                "inputs": {"recogniser_name": {"value": "SpacyRecogniser"}}
            },
            "initialise_recogniser": {
                "config": {
                    "supported_entities": ["LOC", "MISC", "ORG", "PER"],
                    "supported_languages": ["en", "de", "es", "fr", "it", "pt", "ru"],
                    "model_name": "xx_ent_wiki_sm",
                }
            },
        }
    }

    result = execute_pipeline(evaluation_pipeline, environment_dict=environment_dict)
    assert result.success
