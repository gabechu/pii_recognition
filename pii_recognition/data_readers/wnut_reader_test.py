from unittest.mock import mock_open, patch

from pii_recognition.tokenisation.detokenisers import space_join_detokensier

from .wnut_reader import get_wnut_eval_data


def test_get_wnut_eval_data():
    patch_target = "pii_recognition.data_readers.wnut_reader.open"
    # test 1: empty file
    text = ""
    with patch(patch_target, new=mock_open(read_data=text)):
        sents, labels = get_wnut_eval_data(
            "fake_data", detokenizer=space_join_detokensier
        )
    assert sents == []
    assert labels == []

    # test 2: one sentence and end without new line
    text = "This\tO\nis\tO\nBob\tI-person\nfrom\tO\nMelbourne\tI-location\n.\tO\n"
    with patch(patch_target, new=mock_open(read_data=text)):
        sents, labels = get_wnut_eval_data(
            "fake_data", detokenizer=space_join_detokensier
        )
    assert sents == ["This is Bob from Melbourne ."]
    assert labels == [["O", "O", "I-person", "O", "I-location", "O"]]

    # test 3: one sentence and end with new line
    text = "This\tO\nis\tO\nBob\tI-person\nfrom\tO\nMelbourne\tI-location\n.\tO\n\n"
    with patch(patch_target, new=mock_open(read_data=text)):
        sents, labels = get_wnut_eval_data(
            "fake_data", detokenizer=space_join_detokensier
        )
    assert sents == ["This is Bob from Melbourne ."]
    assert labels == [["O", "O", "I-person", "O", "I-location", "O"]]

    # test 4: two sentences
    text = (
        "This\tO\nis\tO\nBob\tI-person\nfrom\tO\nMelbourne\tI-location\n.\tO\n\n"
        "This\tO\nis\tO\nBob\tI-person\nfrom\tO\nMelbourne\tI-location\n.\tO\n\n"
    )
    with patch(patch_target, new=mock_open(read_data=text)):
        sents, labels = get_wnut_eval_data(
            "fake_data", detokenizer=space_join_detokensier
        )
    assert sents == ["This is Bob from Melbourne .", "This is Bob from Melbourne ."]
    assert labels == [
        ["O", "O", "I-person", "O", "I-location", "O"],
        ["O", "O", "I-person", "O", "I-location", "O"],
    ]
