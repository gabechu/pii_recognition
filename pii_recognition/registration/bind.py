from pii_recognition import recognisers
from pii_recognition.registration import registry


def _get_default_recognisers():
    return [
        recognisers.crf_recogniser.CrfRecogniser,
        recognisers.first_letter_uppercase_recogniser.FirstLetterUppercaseRecogniser,
        recognisers.flair_recogniser.FlairRecogniser,
        recognisers.spacy_recogniser.SpacyRecogniser,
        recognisers.stanza_recogniser.StanzaRecogniser,
    ]


def _register_recognisers():
    for item in _get_default_recognisers():
        registry.recogniser.add_item(item)
