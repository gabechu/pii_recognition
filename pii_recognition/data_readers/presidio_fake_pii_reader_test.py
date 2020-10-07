import json

from mock import mock_open, patch

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
                "full_text": "A tribute to Joshua Lewis – sadly, she wasn't impressed.",
                "spans": [
                    {
                        "entity_type": "PERSON",
                        "entity_value": "Joshua Lewis",
                        "start_position": 13,
                        "end_position": 25,
                    }
                ],
            },
        ]
    )


@patch("builtins.open", new_callable=mock_open, read_data=presidio_fake_pii_json())
def test_PresidioFakePiiReader(mock_file):
    reader = PresidioFakePiiReader()
    data = reader.build_data("fake_path/file.json")
    assert [x.text for x in data.items] == [
        "It's like that since 12/17/1967",
        "A tribute to Joshua Lewis – sadly, she wasn't impressed.",
    ]
    assert [x.true_label for x in data.items] == [
        [
            {
                "entity_type": "BIRTHDAY",
                "entity_value": "12/17/1967",
                "start_position": 21,
                "end_position": 31,
            }
        ],
        [
            {
                "entity_type": "PERSON",
                "entity_value": "Joshua Lewis",
                "start_position": 13,
                "end_position": 25,
            }
        ],
    ]
    assert [x.pred_label for x in data.items] == [None, None]
    assert data.supported_entities == {"BIRTHDAY", "PERSON"}
