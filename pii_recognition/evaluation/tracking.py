"""
Tracker module implementing Mlflow API.
"""

import os
from typing import Dict

import mlflow

from pii_recognition.constants import BASE_DIR

DEFAULT_TRACKER_URI = os.path.join(BASE_DIR, "mlruns")


def _get_experiment_id(experiment_name: str):
    return mlflow.get_experiment_by_name(experiment_name).experiment_id


def start_tracker(
    experiment_name: str,
    run_name: str = "default",
    tracker_uri: str = DEFAULT_TRACKER_URI,
) -> mlflow.ActiveRun:
    """
    Start a new tracker. This tracker stays active under which metrics and parameters
    will be logged. To terminate the current tracker, call `end_tracker()` or use a
    context manager within a `with` block.
    """
    # Connect to a tracking URI.
    # URI can either be a HTTP/HTTPS URI for a remote server, a database connection
    # string, or a local path to log data to a directory.
    mlflow.set_tracking_uri(tracker_uri)

    # Set an experiment active, this experiment will be created under the tracking URI.
    mlflow.set_experiment(experiment_name)

    # Start a new run and set it active, where the new run will be launched under the
    # active experiment.
    return mlflow.start_run(run_name=run_name)


def end_tracker():
    """
    Terminate an active tracker.
    """
    mlflow.end_run()


def log_metric_per_entity(metric: Dict[str, float], metric_name: str = None):
    for entity_name, entity_score in metric.items():
        mlflow.log_metric(entity_name + f"_{metric_name}", entity_score)
