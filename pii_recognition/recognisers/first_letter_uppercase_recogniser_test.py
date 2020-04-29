from unittest.mock import Mock, patch

from pii_recognition.labels.schema import SpanLabel
from pii_recognition.tokenisation.token_schema import Token

from .first_letter_uppercase_recogniser import (
    FirstLetterUppercaseRecogniser,
    tokeniser_registry,
)


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


@patch.object(
    target=tokeniser_registry,
    attribute="create_instance",
    new_callable=get_mock_tokeniser,
)
def test_first_letter_uppercase_analyse(mock_tokeniser):
    recogniser = FirstLetterUppercaseRecogniser(
        ["en"],
        tokeniser={"name": "fake_tokeniser", "config": {"fake_param": "fake_value"}},
    )
    mock_tokeniser.assert_called_with("fake_tokeniser", {"fake_param": "fake_value"})
    actual = recogniser.analyse("fake_text", entities=["PER"])
    assert actual == [
        SpanLabel("PER", 0, 4),
        SpanLabel("PER", 8, 11),
        SpanLabel("PER", 17, 26),
    ]
