from typing import NamedTuple
from unittest.mock import Mock, patch

import pytest
from pytest import fixture

from pii_recognition.labels.schema import Entity

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


@patch("pii_recognition.recognisers.stanza_recogniser.Pipeline", new=mock_pipeline())
def test_stanza_analyse(text):
    recogniser = StanzaRecogniser(
        supported_entities=["PERSON", "LOC", "ORG"],
        supported_languages=["en"],
        model_name="en",
    )

    actual = recogniser.analyse(text, entities=["PERSON"])
    assert actual == [Entity("PERSON", 8, 11)]

    actual = recogniser.analyse(text, entities=["PERSON", "LOC"])
    assert actual == [Entity("PERSON", 8, 11), Entity("LOC", 17, 26)]

    with pytest.raises(AssertionError) as err:
        recogniser.analyse(text, entities=["PERSON", "LOC", "TIME"])
    assert (
        str(err.value)
        == "Only support ['PERSON', 'LOC', 'ORG'], but got ['PERSON', 'LOC', 'TIME']"
    )
