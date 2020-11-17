from mock import patch
from pii_recognition.labels.schema import Entity
from pii_recognition.recognisers import registry as recogniser_registry


@patch("pii_recognition.recognisers.comprehend_recogniser.config_cognito_session")
@patch("pii_recognition.recognisers.comprehend_recogniser.ComprehendRecogniser.analyse")
def test_registry_for_comprehend(mock_analyse, mock_session):
    mock_analyse.return_value = [Entity("test", 0, 4)]

    recogniser_name = "ComprehendRecogniser"
    recogniser_params = {
        "supported_entities": [
            "COMMERCIAL_ITEM",
            "DATE",
            "EVENT",
            "LOCATION",
            "ORGANIZATION",
            "OTHER",
            "PERSON",
            "QUANTITY",
            "TITLE",
        ],
        "supported_languages": ["en"],
        "model_name": "pii",
    }

    recogniser = recogniser_registry.create_instance(recogniser_name, recogniser_params)
    actual = recogniser.analyse("test text", recogniser.supported_entities)

    assert actual == [Entity("test", 0, 4)]


@patch("pii_recognition.recognisers.google_recogniser.GoogleRecogniser.client")
@patch("pii_recognition.recognisers.google_recogniser.GoogleRecogniser.analyse")
def test_registry_for_google_recogniser(mock_analyse, mock_client):
    mock_analyse.return_value = [Entity("test", 0, 4)]

    recogniser_name = "GoogleRecogniser"
    recogniser_params = {
        "supported_entities": [
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
        "supported_languages": ["en"],
    }

    recogniser = recogniser_registry.create_instance(recogniser_name, recogniser_params)
    actual = recogniser.analyse("test text", recogniser.supported_entities)

    assert actual == [Entity("test", 0, 4)]
