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

        config = load_yaml_file(config_yaml)
        config["predictions_dump_path"] = preds_dump_path = os.path.join(
            tempdir, "test_predictions.json"
        )
        config["scores_dump_path"] = scores_dump_path = os.path.join(
            tempdir, "test_scores.json"
        )
        dump_yaml_file(temp_config_yaml, config)

        exec_pipeline(temp_config_yaml)
        scores = load_json_file(scores_dump_path)
        preds = load_json_file(preds_dump_path)

        assert set(os.listdir(tempdir)) == {
            "config.yaml",
            "test_predictions.json",
            "test_scores.json",
        }
        assert len(scores.keys()) == 5
        assert scores["exact_match_f1"] == 0.5062
        assert scores["partial_match_f1_threshold_at_50%"] == 0.5333
        assert scores["frozenset({'PERSON'})"] == 0.4
        assert scores["frozenset({'LOCATION'})"] == 0.6857
        assert (
            scores.get("frozenset({'CREDIT_CARD', 'OTHER'})") == 1.0
            or scores.get("frozenset({'OTHER', 'CREDIT_CARD'})") == 1.0
        )

        assert len(preds) == 5
        item_one = preds[
            "Please update billing addrress with Markt 84, "
            "MÜLLNERN 9123 for this card: 5550253262199449"
        ]
        item_two = preds[
            "My name appears incorrectly on credit card statement could you "
            "please correct it to Ms. Aybika Rushisvili?"
        ]
        item_three = preds["A tribute to Joshua Lewis – sadly, she wasn't impressed."]
        item_four = preds["I work for Flightview"]
        item_five = preds["I work for Flight"]

        assert item_one["predicted"] == {
            "Markt 84, MÜ": {"type": "LOCATION", "score": 1.0, "start": 36},
            "5550253262199449": {"type": "OTHER", "score": 1.0, "start": 75},
        }
        assert item_one["ground_truth"] == {
            "Markt 84, MÜLLNERN 9123": {
                "type": "LOCATION",
                "score": 0.52,
                "start": 36,
            },
            "5550253262199449": {"type": "CREDIT_CARD", "score": 1.0, "start": 75},
        }

        assert item_two["predicted"] == {}
        assert item_two["ground_truth"] == {
            "Aybika Rushisvili": {"type": "PERSON", "score": 0.0, "start": 88}
        }

        assert item_three["predicted"] == {
            "Joshua Lewis": {"type": "PERSON", "score": 1.0, "start": 13},
            "sadly, ": {"type": "PERSON", "score": 0.0, "start": 28},
        }
        assert item_three["ground_truth"] == {
            "Joshua Lewis": {"type": "PERSON", "score": 1.0, "start": 13}
        }

        assert item_four["predicted"] == {
            "Flight": {"type": "PERSON", "score": 0.0, "start": 11}
        }
        assert item_four["ground_truth"] == {}

        assert item_five["predicted"] == {}
        assert item_five["ground_truth"] == {}
