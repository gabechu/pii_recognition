from typing import List
from unittest.mock import Mock, mock_open, patch

from pytest import fixture, raises

from .wnut_reader import WnutReader


@fixture
def mock_detokeniser():
    def simple_detokeniser(tokens: List[str]) -> str:
        return " ".join(tokens)

    mock = Mock()
    mock.detokenise = simple_detokeniser

    return mock


def test_get_wnut_eval_data(mock_detokeniser):
    patch_target = "pii_recognition.data_readers.wnut_reader.open"
    reader = WnutReader(detokeniser=mock_detokeniser)

    # test 1: empty file
    text = ""
    with patch(patch_target, new=mock_open(read_data=text)):
        data = reader.get_test_data("fake_data", supported_entities=[])
    assert data.sentences == []
    assert data.labels == []
    assert data.supported_entities == []
    assert data.is_io_schema is True

    # test 2: one sentence and end without new line
    text = "This\tO\nis\tO\nBob\tI-person\nfrom\tO\nMelbourne\tI-location\n.\tO\n"
    with patch(patch_target, new=mock_open(read_data=text)):
        data = reader.get_test_data(
            "fake_data", supported_entities=["I-person", "I-location"]
        )
    assert data.sentences == ["This is Bob from Melbourne ."]
    assert data.labels == [["O", "O", "I-person", "O", "I-location", "O"]]
    assert data.supported_entities == ["I-person", "I-location"]
    assert data.is_io_schema is True

    # test 3: one sentence and end with new line
    text = "This\tO\nis\tO\nBob\tI-person\nfrom\tO\nMelbourne\tI-location\n.\tO\n\n"
    with patch(patch_target, new=mock_open(read_data=text)):
        data = reader.get_test_data(
            "fake_data", supported_entities=["I-person", "I-location"]
        )
    assert data.sentences == ["This is Bob from Melbourne ."]
    assert data.labels == [["O", "O", "I-person", "O", "I-location", "O"]]
    assert data.supported_entities == ["I-person", "I-location"]
    assert data.is_io_schema is True

    # test 4: two sentences
    text = (
        "This\tO\nis\tO\nBob\tI-person\nfrom\tO\nMelbourne\tI-location\n.\tO\n\n"
        "This\tO\nis\tO\nBob\tI-person\nfrom\tO\nMelbourne\tI-location\n.\tO\n\n"
    )
    with patch(patch_target, new=mock_open(read_data=text)):
        data = reader.get_test_data(
            "fake_data", supported_entities=["I-person", "I-location"]
        )
    assert data.sentences == [
        "This is Bob from Melbourne .",
        "This is Bob from Melbourne .",
    ]
    assert data.labels == [
        ["O", "O", "I-person", "O", "I-location", "O"],
        ["O", "O", "I-person", "O", "I-location", "O"],
    ]
    assert data.supported_entities == ["I-person", "I-location"]
    assert data.is_io_schema is True

    # test 4: contains unsupported entity
    with patch(patch_target, new=mock_open(read_data=text)):
        with raises(ValueError) as err:
            data = reader.get_test_data("fake_data", supported_entities=["I-person"])
        assert str(err.value) == (
            "Found unsupported entity {'I-location'} in data. "
            "You may need to update your supported entity list."
        )
