import inspect

from dagster import (
    Field,
    Output,
    OutputDefinition,
    String,
    execute_pipeline,
    pipeline,
    solid,
)

from data_reader.data_reader_registry import DataReaderRegistry
from evaluation.model_evaluator import ModelEvaluator
from paths import DataPath
from recognisers.recogniser_registry import RecogniserRegistry
from tokeniser.detokeniser import DetokeniserRegistry
from tokeniser.tokeniser import TokeniserRegistry

from experiments.mlflow_tracking import log_evaluation_to_mlflow


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


@solid(
    config={
        "detokeniser": Field(
            String, is_required=False, default_value="space_join_detokensier"
        )
    },
    output_defs=[OutputDefinition(list, "X_test"), OutputDefinition(list, "y_test")],
)
def get_evaluation_data(context, data_path: str):
    path = DataPath(data_path)
    reader_registry = DataReaderRegistry()
    detokeniser_registry = DetokeniserRegistry()

    reader = reader_registry.registry[path.data_name]  # type: ignore
    detokeniser = detokeniser_registry.registry[context.solid_config["detokeniser"]]
    X_test, y_test = reader(path.path, detokeniser)
    yield Output(X_test, "X_test")
    yield Output(y_test, "y_test")


@solid(
    config={
        "tokeniser": Field(
            String, is_required=False, default_value="nltk_word_tokenizer"
        )
    }
)
def initialise_evaluator(context, recogniser, target_entities, to_eval_labels=None):
    tokeniser_registry = TokeniserRegistry()
    return ModelEvaluator(
        recogniser,
        target_entities,
        tokeniser_registry.registry[context.solid_config["tokeniser"]],
        to_eval_labels,
    )


@solid
def evaluate_and_logging(
    context, recogniser, evaluator, X_test, y_test, experiment_name, run_name
):
    log_evaluation_to_mlflow(
        experiment_name, recogniser, evaluator, X_test, y_test, run_name
    )


@pipeline
def evaluation_pipeline():
    recogniser = initialise_recogniser(get_recogniser())
    X_test, y_test = get_evaluation_data()
    evaluator = initialise_evaluator(recogniser)
    evaluate_and_logging(recogniser, evaluator, X_test, y_test)


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
            "get_evaluation_data": {
                "inputs": {"data_path": {"value": "datasets/conll2003/eng.testb"}},
            },
            "initialise_evaluator": {
                "inputs": {
                    "target_entities": {"value": ["PER"]},
                    "to_eval_labels": {"value": {"PER": "I-person"}},
                }
            },
            "evaluate_and_logging": {
                "inputs": {
                    "experiment_name": {"value": "Spacy"},
                    "run_name": {"value": "pipeline"},
                }
            },
        }
    }

    execute_pipeline(evaluation_pipeline, environment_dict=environment_dict)
