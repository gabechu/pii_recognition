import os
import tempfile
from typing import Dict, List

import mlflow

from evaluation.model_evaluator import ModelEvaluator
from recognisers.entity_recogniser import Rec_co
from utils import write_iterable_to_text

from .manage_experiments import activate_experiment
from config import BASE_DIR


def log_evaluation_to_mlflow(
    experiment_name: str,
    param: Dict,
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

        for key, value in param.items():
            if not isinstance(value, str):
                mlflow.log_param(key, value.__name__)
            else:
                mlflow.log_param(key, value)

        # TODO: for now only focusing on I-PER and label is form CONLL 2003
        mlflow.log_metric("PER_recall", recall["I-PER"])
        mlflow.log_metric("PER_precision", precision["I-PER"])
        mlflow.log_metric("PER_f1", f1["I-PER"])
        mlflow.log_metric("PER_f2", f2["I-PER"])
