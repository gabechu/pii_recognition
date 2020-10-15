import os
from tempfile import TemporaryDirectory

from mock import patch
from pii_recognition.labels.schema import Entity
from pii_recognition.pipelines.pii_validation_pipeline import exec_pipeline
from pii_recognition.utils import dump_yaml_file, load_json_file, load_yaml_file


@patch("pii_recognition.pipelines.pii_validation_pipeline.recogniser_registry")
def test_execute_pii_validation_pipeline(mock_registry):
    mock_registry.create_instance.return_value.analyse.return_value = [
        Entity("ORGANIZATION", 5, 10),
        Entity("PERSON", 13, 25),
        Entity("CREDIT_CARD", 27, 46),
    ]
    config_yaml = "tests/assets/config/pii_validation.yaml"

    with TemporaryDirectory() as tempdir:
        temp_config_yaml_path = os.path.join(tempdir, "config.yaml")
        temp_json_dump_path = os.path.join(tempdir, "test_report.json")

        # reconfigurate to dump files to temp dir
        config = load_yaml_file(config_yaml)
        config["dump_file"] = temp_json_dump_path
        dump_yaml_file(temp_config_yaml_path, config)

        exec_pipeline(temp_config_yaml_path)
        report = load_json_file(temp_json_dump_path)
        assert set(os.listdir(tempdir)) == {"config.yaml", "test_report.json"}
        assert report == {
            "exact_match_f1": 2 / 9,
            "partial_match_f1_threshold_at_50%": 2 / 9,
        }
