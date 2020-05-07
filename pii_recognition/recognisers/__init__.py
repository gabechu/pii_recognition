from .entity_recogniser import EntityRecogniser
from pii_recognition.registration.registry import Registry


def init() -> Registry:
    from .crf_recogniser import CrfRecogniser
    from .first_letter_uppercase_recogniser import FirstLetterUppercaseRecogniser
    from .flair_recogniser import FlairRecogniser
    from .spacy_recogniser import SpacyRecogniser
    from .stanza_recogniser import StanzaRecogniser

    registry = Registry[EntityRecogniser]()
    registry.register(CrfRecogniser)
    registry.register(FirstLetterUppercaseRecogniser)
    registry.register(FlairRecogniser)
    registry.register(SpacyRecogniser)
    registry.register(StanzaRecogniser)

    return registry


registry: Registry[EntityRecogniser] = init()
