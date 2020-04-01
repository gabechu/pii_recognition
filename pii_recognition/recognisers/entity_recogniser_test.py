from unittest.mock import patch

import pytest

from .entity_recogniser import EntityRecogniser


@patch.object(target=EntityRecogniser, attribute="__abstractmethods__", new=set())
def test_entity_recogniser_attributes():
    actual = EntityRecogniser(
        supported_entities=["PER"],
        supported_languages=["en"],
        name="TEST",
        version="0.0.0",
    )
    assert actual.supported_entities == ["PER"]
    assert actual.supported_languages == ["en"]
    assert actual.name == "TEST"
    assert actual.version == "0.0.0"

    actual = EntityRecogniser(supported_entities=["PER"], supported_languages=["en"])
    assert actual.name == "EntityRecogniser"
    assert actual.version == "0.0.1"


@patch.object(target=EntityRecogniser, attribute="__abstractmethods__", new=set())
def test_entity_recogniser_validation():
    actual = EntityRecogniser(supported_entities=["PER"], supported_languages=["en"])
    actual.validate_entities(["PER"])
    actual.validate_languages(["en"])

    with pytest.raises(AssertionError) as err:
        actual.validate_entities(["PER", "LOC"])
    assert str(err.value) == "Only support ['PER'], but got ['PER', 'LOC']"

    with pytest.raises(AssertionError) as err:
        actual.validate_languages(["en", "pt"])
    assert str(err.value) == "Only support ['en'], but got ['en', 'pt']"
