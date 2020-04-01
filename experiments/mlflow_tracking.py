import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.exceptions import MlflowException

from constants import BASE_DIR, LABEL_COMPLIANCE
from evaluation.model_evaluator import ModelEvaluator
from recognisers.entity_recogniser import Rec_co
from utils import write_iterable_to_text


def activate_experiment(exp_name: str, artifact_location: str):
    # TODO: add test!
    try:
        mlflow.create_experiment(
            name=exp_name, artifact_location=artifact_location,
        )
    except MlflowException:
        logging.info(f"Experiment {exp_name} already exists.")


def delete_experiment(exp_name: str):
    # TODO: add test!
    try:
        experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id
    except AttributeError:
        logging.info(f"No {exp_name} experiment found.")
        return

    try:
        mlflow.delete_experiment(experiment_id)
    except MlflowException:
        logging.info(f"Experiment has already been deleted.")


def log_evaluation_to_mlflow(
    experiment_name: str,
    recogniser: Rec_co,
    evaluator: ModelEvaluator,
    X_test: List[str],
    y_test: List[List[str]],
    run_name: str = "default",
):
    artifact_path = os.path.join(BASE_DIR, "artifacts", f"{experiment_name}")

    activate_experiment(experiment_name, artifact_path)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        counters, mistakes = evaluator.evaulate_all(X_test, y_test)
        # remove returns with no mistakes
        mistakes = list(filter(lambda x: x.token_errors, mistakes))

        recall, precision, f1 = evaluator.calculate_score(counters, f_beta=1.0)
        _, _, f2 = evaluator.calculate_score(counters, f_beta=2.0)

        with tempfile.TemporaryDirectory() as tempdir:
            error_file_path = os.path.join(tempdir, f"{run_name}.mis")
            write_iterable_to_text(mistakes, error_file_path)
            mlflow.log_artifact(error_file_path)

        # log_params(params)

        log_metrics(recall, suffix="recall")
        log_metrics(precision, suffix="precision")
        log_metrics(f1, suffix="f1")
        log_metrics(f2, suffix="f2")


def log_metrics(metrics: Dict, suffix: Optional[str] = None):
    # TODO: add test!
    for key, value in metrics.items():
        assert isinstance(key, str), f"Metric key must be string but got {type(key)}"

        if key in LABEL_COMPLIANCE:
            if suffix:
                key_name = LABEL_COMPLIANCE[key] + f"_{suffix}"
            else:
                key_name = LABEL_COMPLIANCE[key]
        else:
            if suffix:
                key_name = key + f"_{suffix}"
            else:
                key_name = key

        mlflow.log_metric(key_name, value)


def log_params(params: Dict[str, Any]):
    # TODO: add test!
    for key, value in params.items():
        if callable(value):
            mlflow.log_param(key, value.__name__)
        elif isinstance(value, list):
            mlflow.log_param(key, ", ".join(map(str, value)))
        else:
            mlflow.log_param(key, value)
