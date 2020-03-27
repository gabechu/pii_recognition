from typing import NamedTuple

import pytest
from mock import Mock, patch
from pytest import fixture

from label.label_schema import SpanLabel

from .stanza_recogniser import StanzaRecogniser


class MockStanzaSpan(NamedTuple):
    type: str
    start_char: int
    end_char: int


def mock_pipeline():
    pipeline = Mock()
    pipeline.return_value.return_value.entities = [
        MockStanzaSpan("PERSON", 8, 11),
        MockStanzaSpan("LOC", 17, 26),
    ]
    return pipeline


@fixture
def text():
    return "This is Bob from Melbourne."


@patch("recognisers.stanza_recogniser.Pipeline", new=mock_pipeline())
def test_stanza_analyse(text):
    recogniser = StanzaRecogniser(
        supported_entities=["PERSON", "LOC", "ORG"],
        supported_languages=["en"],
        model_name="en",
    )

    actual = recogniser.analyse(text, entities=["PERSON"])
    assert actual == [SpanLabel("PERSON", 8, 11)]

    actual = recogniser.analyse(text, entities=["PERSON", "LOC"])
    assert actual == [SpanLabel("PERSON", 8, 11), SpanLabel("LOC", 17, 26)]

    with pytest.raises(AssertionError) as err:
        recogniser.analyse(text, entities=["PERSON", "LOC", "TIME"])
    assert (
        str(err.value)
        == "Only support ['PERSON', 'LOC', 'ORG'], but got ['PERSON', 'LOC', 'TIME']"
    )
