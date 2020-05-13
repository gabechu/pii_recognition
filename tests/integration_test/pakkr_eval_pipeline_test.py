from tempfile import TemporaryDirectory

import yaml
import os

from pii_recognition.evaluation.pakkr_pipeline import \
    execute_evaluation_pipeline
from pii_recognition.utils import dump_yaml_file, load_yaml_file


def test_execute_evaluation_pipeline():
    config_yaml = "tests/assets/eval_config.yaml"

    with TemporaryDirectory() as tempdir:
        temp_config_yaml = os.path.join(tempdir, "config.yaml")
        tracker_uri = os.path.join(tempdir, "mlruns")

        # add tracker_uri field to config_yaml
        data = load_yaml_file(config_yaml)
        data["tracker_uri"] = tracker_uri
        dump_yaml_file(temp_config_yaml, data)

        execute_evaluation_pipeline(temp_config_yaml)
        assert set(os.listdir(tempdir)) == {"config.yaml", "mlruns"}
        assert "meta.yaml" in os.listdir(os.path.join(tempdir, "mlruns", "1"))
