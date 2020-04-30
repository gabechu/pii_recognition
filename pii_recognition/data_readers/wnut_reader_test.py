from typing import List
from unittest.mock import Mock, mock_open, patch

from .wnut_reader import WnutReader, detokeniser_registry


def get_mock_detokeniser():
    def simple_detokeniser(tokens: List[str]) -> str:
        return " ".join(tokens)

    mock = Mock()
    mock.return_value.detokenise = simple_detokeniser

    return mock


@patch.object(
    target=detokeniser_registry,
    attribute="create_instance",
    new_callable=get_mock_detokeniser,
)
def test_get_wnut_eval_data(mock_detokeniser):
    patch_target = "pii_recognition.data_readers.wnut_reader.open"
    reader = WnutReader(detokeniser_setup={"name": "fake_reader"})
    mock_detokeniser.assert_called_once_with(config=None, name="fake_reader")

    # test 1: empty file
    text = ""
    with patch(patch_target, new=mock_open(read_data=text)):
        sents, labels = reader.get_test_data("fake_data")
    assert sents == []
    assert labels == []

    # test 2: one sentence and end without new line
    text = "This\tO\nis\tO\nBob\tI-person\nfrom\tO\nMelbourne\tI-location\n.\tO\n"
    with patch(patch_target, new=mock_open(read_data=text)):
        sents, labels = reader.get_test_data("fake_data")
    assert sents == ["This is Bob from Melbourne ."]
    assert labels == [["O", "O", "I-person", "O", "I-location", "O"]]

    # test 3: one sentence and end with new line
    text = "This\tO\nis\tO\nBob\tI-person\nfrom\tO\nMelbourne\tI-location\n.\tO\n\n"
    with patch(patch_target, new=mock_open(read_data=text)):
        sents, labels = reader.get_test_data("fake_data")
    assert sents == ["This is Bob from Melbourne ."]
    assert labels == [["O", "O", "I-person", "O", "I-location", "O"]]

    # test 4: two sentences
    text = (
        "This\tO\nis\tO\nBob\tI-person\nfrom\tO\nMelbourne\tI-location\n.\tO\n\n"
        "This\tO\nis\tO\nBob\tI-person\nfrom\tO\nMelbourne\tI-location\n.\tO\n\n"
    )
    with patch(patch_target, new=mock_open(read_data=text)):
        sents, labels = reader.get_test_data("fake_data")
    assert sents == ["This is Bob from Melbourne .", "This is Bob from Melbourne ."]
    assert labels == [
        ["O", "O", "I-person", "O", "I-location", "O"],
        ["O", "O", "I-person", "O", "I-location", "O"],
    ]
