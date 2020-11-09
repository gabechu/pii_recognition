import os
from tempfile import TemporaryDirectory

from mock import patch
from pii_recognition.labels.schema import Entity
from pii_recognition.pipelines.pii_validation_pipeline import exec_pipeline
from pii_recognition.utils import dump_yaml_file, load_json_file, load_yaml_file


def predictions():
    # this case tests on recall threshold
    entities_1st_text = [
        Entity("LOCATION", 36, 48),
        Entity("OTHER", 75, 91),
    ]

    # this case tests empty precisions
    entities_2nd_text = [
        Entity("TITLE", 84, 87),
        Entity("EVENT", 88, 105),
    ]

    entities_3rd_text = [
        Entity("PERSON", 13, 25),
        Entity("PERSON", 28, 35),
    ]

    # this case tests empty recalls
    entities_4th_text = [
        Entity("PERSON", 11, 17),
    ]

    # this case tests empty precisions and recalls
    entities_5th_text = [
        Entity("ORGANIZATION", 11, 21),
    ]

    return [
        entities_1st_text,
        entities_2nd_text,
        entities_3rd_text,
        entities_4th_text,
        entities_5th_text,
    ]


@patch("pii_recognition.pipelines.pii_validation_pipeline.recogniser_registry")
def test_execute_pii_validation_pipeline(mock_registry):
    mock_registry.create_instance.return_value.analyse.side_effect = predictions()
    config_yaml = "tests/assets/config/pii_validation.yaml"

    with TemporaryDirectory() as tempdir:
        # direct file writings to temp dir
        temp_config_yaml = os.path.join(tempdir, "config.yaml")
        temp_json_dump = os.path.join(tempdir, "test_report.json")

        config = load_yaml_file(config_yaml)
        config["dump_file"] = temp_json_dump
        dump_yaml_file(temp_config_yaml, config)

        exec_pipeline(temp_config_yaml)
        report = load_json_file(temp_json_dump)
        assert set(os.listdir(tempdir)) == {"config.yaml", "test_report.json"}
        assert len(report.keys()) == 5
        assert report["exact_match_f1"] == 0.5062
        assert report["partial_match_f1_threshold_at_50%"] == 0.5333
        assert report["frozenset({'PERSON'})"] == 0.4
        assert report["frozenset({'LOCATION'})"] == 0.6857
        assert (
            report.get("frozenset({'CREDIT_CARD', 'OTHER'})") == 1.0
            or report.get("frozenset({'OTHER', 'CREDIT_CARD'})") == 1.0
        )
