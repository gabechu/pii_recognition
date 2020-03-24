from typing import NamedTuple

import pytest
from mock import Mock, patch
from pytest import fixture

from label.label_schema import SpanLabel

from .flair import Flair


class MockFlairSpan(NamedTuple):
    tag: str
    start_pos: int
    end_pos: int


def mock_sentence():
    sentence = Mock()
    sentence.return_value.get_spans.return_value = [
        MockFlairSpan("PER", 8, 11),
        MockFlairSpan("LOC", 17, 26),
    ]
    return sentence


@fixture
def text():
    return "This is Bob from Melbourne."


@patch("recognisers.flair.Sentence", new=mock_sentence())
@patch("recognisers.flair.SequenceTagger", new=Mock())
def test_flair_analyse():
    recogniser = Flair(
        supported_entities=["PER", "LOC", "ORG", "MISC"],
        supported_languages=["en"],
        model_name="fake_model",
    )

    actual = recogniser.analyse(text, entities=["PER"])
    assert actual == [SpanLabel("PER", 8, 11)]

    actual = recogniser.analyse(text, entities=["PER", "LOC"])
    assert actual == [SpanLabel("PER", 8, 11), SpanLabel("LOC", 17, 26)]

    with pytest.raises(AssertionError) as err:
        recogniser.analyse(text, entities=["PER", "LOC", "TIME"])
    assert (
        str(err.value)
        == "Only support ['PER', 'LOC', 'ORG', 'MISC'], but got ['PER', 'LOC', 'TIME']"
    )
