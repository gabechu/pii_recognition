from mock import Mock, patch
from pytest import fixture

from tokeniser.token import Token
import pytest

from .crf_recogniser import CrfRecogniser
from .recogniser_result import RecogniserResult


def get_mock_load_model():
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
        Token(".", 27, 28),
    ]
    return tokeniser


@fixture
def text():
    return "This is Bob from Melbourne."


@patch.object(target=CrfRecogniser, attribute="load_model", new=get_mock_load_model())
def test_crf_recogniser_analyze(text, mock_tokeniser):
    recogniser = CrfRecogniser(["PER", "LOC"], ["en"], "fake_path", mock_tokeniser)

    actual = recogniser.analyze(text, entities=["PER"])
    assert actual == [RecogniserResult("PER", 8, 11)]

    actual = recogniser.analyze(text, entities=["PER", "LOC"])
    assert actual == [RecogniserResult("PER", 8, 11), RecogniserResult("LOC", 17, 26)]

    with pytest.raises(AssertionError) as err:
        recogniser.analyze(text, entities=["PER", "LOC", "TIME"])
    assert str(err.value) == "Only support ['PER', 'LOC'], but got ['PER', 'LOC', 'TIME']"
