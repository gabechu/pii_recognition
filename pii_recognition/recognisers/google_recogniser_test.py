from typing import Dict

from google.cloud.language_v1 import AnalyzeEntitiesResponse
from mock import patch
from pii_recognition.labels.schema import Entity
from pytest import fixture, raises

from .google_recogniser import GoogleRecogniser


@fixture
def text() -> str:
    return (
        "Please update billing addrress with Markt 84, "
        "MÜLLNERN 9123 for this card: 5550253262199449"
    )


@fixture
def response() -> AnalyzeEntitiesResponse:
    response: Dict = {
        "language": "en",
        "entities": [
            {
                "name": "billing",
                "type_": "OTHER",
                "salience": 0.4248213469982147,
                "mentions": [{"text": {"content": "billing", "begin_offset": 14}}],
            },
            {
                "name": "84",
                "type_": "NUMBER",
                "metadata": {"key": "value", "value": "84"},
                "mentions": [{"text": {"content": "84", "begin_offset": 42}}],
            },
        ],
    }
    return AnalyzeEntitiesResponse(response)


def test_google_recogniser_for_initialisation():
    actual = GoogleRecogniser(
        supported_entities=["test_PERSON", "test_LOCATION"],
        supported_languages=["test_en", "test_fr"],
    )

    actual.supported_entities == ["test_PERSON", "test_LOCATION"]
    actual.supported_languages == ["test_en", "test_fr"]


def test_google_recogniser_for_analyse_unsupported_entity_type(text):
    recogniser = GoogleRecogniser(
        supported_entities=["test_PERSON", "test_LOCATION"], supported_languages=["en"]
    )

    with raises(AssertionError) as err:
        recogniser.analyse(text, entities=["test_UNSUPPORTED"])
    assert str(err.value) == (
        "Only support ['test_PERSON', 'test_LOCATION'], but got ['test_UNSUPPORTED']"
    )


@patch("pii_recognition.recognisers.google_recogniser.GoogleRecogniser.client")
def test_google_recogniser_for_analyse(mock_client, text, response):
    mock_client.analyze_entities.return_value = response

    recogniser = GoogleRecogniser(
        supported_entities=[
            "UNKNOWN",
            "PERSON",
            "LOCATION",
            "ORGANIZATION",
            "EVENT",
            "WORK_OF_ART",
            "CONSUMER_GOOD",
            "OTHER",
            "PHONE_NUMBER",
            "ADDRESS",
            "DATE",
            "NUMBER",
            "PRICE",
        ],
        supported_languages=["en"],
    )
    actual = recogniser.analyse(text, recogniser.supported_entities)
    assert actual == [Entity("OTHER", 14, 21), Entity("NUMBER", 42, 44)]
