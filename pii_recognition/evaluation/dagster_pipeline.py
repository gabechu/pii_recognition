import os
import tempfile

import mlflow
from dagster import Field, Shape, pipeline, solid

from pii_recognition.data_readers import reader_registry
from pii_recognition.evaluation.model_evaluator import ModelEvaluator
from pii_recognition.evaluation.tracking import (
    end_tracker,
    log_entities_metric,
    start_tracker,
)
from pii_recognition.paths.data_path import DataPath
from pii_recognition.recognisers import registry as recogniser_registry
from pii_recognition.tokenisation import detokeniser_registry, tokeniser_registry
from pii_recognition.utils import write_iterable_to_file


@solid
def enable_tracker(context, experiment_name, run_name, tracker_uri=None):
    start_tracker(experiment_name, run_name, tracker_uri)


@solid(
    config={
        "tokeniser_name": Field(
            str, is_required=False, default_value="TreebankWordTokeniser"
        ),
        "tokeniser_config": Field(Shape(fields={}), is_required=False),
    }
)
def get_tokeniser(context):
    return tokeniser_registry.create_instance(
        context.solid_config["tokeniser_name"],
        context.solid_config.get("tokeniser_config"),
    )


@solid(
    config={
        "detokeniser_name": Field(
            str, is_required=False, default_value="TreebankWordDetokeniser"
        ),
        "detokeniser_config": Field(Shape(fields={}), is_required=False),
    }
)
def get_detokeniser(context):
    return detokeniser_registry.create_instance(
        context.solid_config["detokeniser_name"],
        context.solid_config.get("detokeniser_config"),
    )


@solid
def get_recogniser(context, recogniser_name, recogniser_config):
    return recogniser_registry.create_instance(recogniser_name, recogniser_config)


@solid
def get_evaluator(
    context, recogniser, tokeniser, predict_on, switch_labels=None,
):
    return ModelEvaluator(recogniser, tokeniser, predict_on, switch_labels)


@solid
def load_test_data(
    context, detokeniser, test_data_path, test_data_support_entities, test_is_io_schema
):
    data_path = DataPath(test_data_path)
    if not data_path.valid:
        raise Exception(
            f"Got invalid data path, make sure it follow the "
            f"pattern {data_path.pattern_str}."
        )

    reader_config = {"detokeniser": detokeniser}
    reader = reader_registry.create_instance(data_path.reader_name, reader_config)
    return reader.get_test_data(
        data_path.path, test_data_support_entities, test_is_io_schema
    )


@solid
def evaluate(
    context, data, evaluator,
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


@solid
def disable_tracker(context):
    end_tracker()


@pipeline
def evaluation_pipeline():
    enable_tracker()

    evaluator = get_evaluator(get_recogniser(), get_tokeniser())
    data = load_test_data(get_detokeniser())
    evaluate(data, evaluator)

    disable_tracker()
