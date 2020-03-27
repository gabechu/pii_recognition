from mock import Mock, patch
from pytest import fixture

from tokeniser.token import Token
import pytest

from .crf_recogniser import CrfRecogniser
from label.label_schema import SpanLabel


def get_mock_model():
    model = Mock()
    model.tag.return_value = ["O", "O", "PER", "O", "LOC", "O"]

    load_model = Mock()
    load_model.return_value = model
    return load_model


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


@patch.object(target=CrfRecogniser, attribute="model", new_callable=get_mock_model())
def test_crf_recogniser_analyse(text, mock_tokeniser):
    recogniser = CrfRecogniser(["PER", "LOC"], ["en"], "fake_path", mock_tokeniser)

    actual = recogniser.analyse(text, entities=["PER"])
    assert actual == [SpanLabel("PER", 8, 11)]

    actual = recogniser.analyse(text, entities=["PER", "LOC"])
    assert actual == [SpanLabel("PER", 8, 11), SpanLabel("LOC", 17, 26)]

    with pytest.raises(AssertionError) as err:
        recogniser.analyse(text, entities=["PER", "LOC", "TIME"])
    assert (
        str(err.value) == "Only support ['PER', 'LOC'], but got ['PER', 'LOC', 'TIME']"
    )
