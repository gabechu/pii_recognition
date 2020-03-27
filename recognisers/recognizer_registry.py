from typing import Dict, Type, Generic

from .crf_recogniser import CrfRecogniser
from .entity_recogniser import Rec_co, EntityRecogniser
from .first_letter_uppercase_recogniser import FirstLetterUppercaseRecogniser
from .flair_recogniser import FlairRecogniser
from .spacy_recogniser import SpacyRecogniser
from .stanza_recogniser import StanzaRecogniser


def get_recogniser(name: str, params: Dict) -> Rec_co:
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
