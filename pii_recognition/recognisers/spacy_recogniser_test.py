from typing import NamedTuple
from unittest.mock import Mock, patch

import pytest
from pytest import fixture

from pii_recognition.labels.schema import Entity

from .spacy_recogniser import SpacyRecogniser


class MockEntity(NamedTuple):
    label_: str
    start_char: int
    end_char: int


def get_mock_model():
    model = Mock()
    model.return_value.ents = [MockEntity("PER", 8, 11), MockEntity("LOC", 17, 26)]

    load_model = Mock()
    load_model.return_value = model
    return load_model


@fixture
def text():
    return "This is Bob from Melbourne."


@patch.object(target=SpacyRecogniser, attribute="model", new_callable=get_mock_model())
def test_spacy_recogniser(text):
    recogniser = SpacyRecogniser(["PER", "LOC"], ["en"], model_name="fake_model")

    actual = recogniser.analyse(text, entities=["PER"])
    assert actual == [Entity("PER", 8, 11)]

    actual = recogniser.analyse(text, entities=["PER", "LOC"])
    assert actual == [Entity("PER", 8, 11), Entity("LOC", 17, 26)]

    with pytest.raises(AssertionError) as err:
        recogniser.analyse(text, entities=["PER", "LOC", "TIME"])
    assert (
        str(err.value) == "Only support ['PER', 'LOC'], but got ['PER', 'LOC', 'TIME']"
    )
