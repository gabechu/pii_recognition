from unittest.mock import Mock, patch

import pytest

from pii_recognition.labels.schema import SpanLabel
from pii_recognition.tokenisation.token_schema import Token

from .crf_recogniser import CrfRecogniser, tokeniser_registry


def get_mock_model():
    model = Mock()
    model.tag.return_value = ["O", "O", "PER", "O", "LOC", "O"]

    return model


def get_mock_tokeniser():
    # reference: text = "This is Bob from Melbourne."
    tokeniser = Mock()
    tokeniser.return_value.tokenise.return_value = [
        Token("This", 0, 4),
        Token("is", 5, 7),
        Token("Bob", 8, 11),
        Token("from", 12, 16),
        Token("Melbourne", 17, 26),
        Token(".", 26, 27),
    ]
    return tokeniser


@patch.object(target=CrfRecogniser, attribute="model", new=get_mock_model())
@patch.object(
    target=tokeniser_registry,
    attribute="create_instance",
    new_callable=get_mock_tokeniser,
)
def test_crf_recogniser_analyse(mock_tokeniser):
    recogniser = CrfRecogniser(
        ["PER", "LOC"],
        ["en"],
        "fake_path",
        tokeniser_setup={
            "name": "fake_tokeniser",
            "config": {"fake_param": "fake_value"},
        },
    )

    actual = recogniser.analyse("fake_text", entities=["PER"])
    mock_tokeniser.assert_called_with("fake_tokeniser", {"fake_param": "fake_value"})
    assert actual == [SpanLabel("PER", 8, 11)]

    actual = recogniser.analyse("fake_text", entities=["PER", "LOC"])
    assert actual == [SpanLabel("PER", 8, 11), SpanLabel("LOC", 17, 26)]

    with pytest.raises(AssertionError) as err:
        recogniser.analyse("fake_text", entities=["PER", "LOC", "TIME"])
    assert (
        str(err.value) == "Only support ['PER', 'LOC'], but got ['PER', 'LOC', 'TIME']"
    )
