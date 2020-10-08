import json

from mock import mock_open, patch
from pii_recognition.data_readers.data import Entity

from .presidio_fake_pii_reader import PresidioFakePiiReader


def presidio_fake_pii_json():
    return json.dumps(
        [
            {
                "full_text": "It's like that since 12/17/1967",
                "spans": [
                    {
                        "entity_type": "BIRTHDAY",
                        "entity_value": "12/17/1967",
                        "start_position": 21,
                        "end_position": 31,
                    }
                ],
            },
            {
                "full_text": (
                    "The address of Balefire Global is Valadouro 3, Ubide 48145"
                ),
                "spans": [
                    {
                        "entity_type": "ORGANIZATION",
                        "entity_value": "Balefire Global",
                        "start_position": 15,
                        "end_position": 30,
                    },
                    {
                        "entity_type": "LOCATION",
                        "entity_value": "Valadouro 3, Ubide 48145",
                        "start_position": 34,
                        "end_position": 58,
                    },
                ],
            },
        ]
    )


@patch("builtins.open", new_callable=mock_open, read_data=presidio_fake_pii_json())
def test_build_data_for_presidio_fake_pii_reader(mock_file):
    reader = PresidioFakePiiReader()
    data = reader.build_data("fake_path/file.json")
    assert [item.text for item in data.items] == [
        "It's like that since 12/17/1967",
        "The address of Balefire Global is Valadouro 3, Ubide 48145",
    ]
    assert [item.true_labels for item in data.items] == [
        [Entity("BIRTHDAY", 21, 31)],
        [Entity("ORGANIZATION", 15, 30), Entity("LOCATION", 34, 58)],
    ]
    assert [item.pred_labels for item in data.items] == [None, None]
    assert data.supported_entities == {"BIRTHDAY", "ORGANIZATION", "LOCATION"}
