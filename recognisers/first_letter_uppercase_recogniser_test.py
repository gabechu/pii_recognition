from mock import Mock
from pytest import fixture

from label.label_schema import SpanLabel
from tokeniser.token import Token

from .first_letter_uppercase_recogniser import FirstLetterUppercaseRecogniser


@fixture
def mock_tokeniser():
    tokeniser = Mock()
    tokeniser.return_value = [
        Token("This", 0, 4),
        Token("is", 5, 7),
        Token("Bob", 8, 11),
        Token("from", 12, 16),
        Token("Melbourne", 17, 26),
        Token(".", 26, 27),
    ]
    return tokeniser


@fixture
def text():
    return "This is Bob from Melbourne."


def test_first_letter_uppercase_analyse(text, mock_tokeniser):
    recogniser = FirstLetterUppercaseRecogniser(
        supported_languages=["en"], tokeniser=mock_tokeniser
    )
    actual = recogniser.analyse(text, entities=["PER"])
    assert actual == [
        SpanLabel("PER", 0, 4),
        SpanLabel("PER", 8, 11),
        SpanLabel("PER", 17, 26),
    ]
