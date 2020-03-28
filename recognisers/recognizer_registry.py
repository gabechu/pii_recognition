from typing import Dict, Generic, Type

from .crf_recogniser import CrfRecogniser
from .entity_recogniser import EntityRecogniser
from .first_letter_uppercase_recogniser import FirstLetterUppercaseRecogniser
from .flair_recogniser import FlairRecogniser
from .spacy_recogniser import SpacyRecogniser
from .stanza_recogniser import StanzaRecogniser


def get_recogniser(name: str, params: Dict) -> EntityRecogniser:
    # recogniser lazy initialised
    available_recognisers = {
        "Crf": CrfRecogniser(**params),
        "FirstLetterUppercase": FirstLetterUppercaseRecogniser(**params),
        "Flair": FlairRecogniser(**params),
        "Spacy": SpacyRecogniser(**params),
        "Stanza": StanzaRecogniser(**params),
    }

    assert (
        name in available_recognisers
    ), f"Recogniser not found, available recognisers are {available_recognisers.keys()}"

    # TODO: fix mypy error
    return available_recognisers[name]  # type: ignore
