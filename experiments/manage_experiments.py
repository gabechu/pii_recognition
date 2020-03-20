import logging
import mlflow
from mlflow.exceptions import MlflowException


CRF_EXP = "PythonCRF"
Spacy_EXP = "Spacy"
FIRST_LETTER_UPPERCASE = "FirstLetterUppercase"


def activate_experiment(exp_name: str, artifact_location: str):
    try:
        mlflow.create_experiment(name=exp_name, artifact_location=artifact_location)
    except MlflowException:
        logging.info(f"Experiment {exp_name} already exists.")


def delete_experiment(exp_name: str):
    try:
        experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id
    except AttributeError:
        logging.info(f"No {exp_name} experiment found.")
        return

    try:
        mlflow.delete_experiment(experiment_id)
    except MlflowException:
        logging.info(f"Experiment has already been deleted.")
