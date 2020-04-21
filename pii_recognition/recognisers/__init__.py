from .entity_recogniser import EntityRecogniser
from pii_recognition.registration.registry import Registry


def init():
    from .crf_recogniser import CrfRecogniser
    from .first_letter_uppercase_recogniser import FirstLetterUppercaseRecogniser
    from .flair_recogniser import FlairRecogniser
    from .spacy_recogniser import SpacyRecogniser
    from .stanza_recogniser import StanzaRecogniser

    registry = Registry[EntityRecogniser]()
    registry.add_item(CrfRecogniser)
    registry.add_item(FirstLetterUppercaseRecogniser)
    registry.add_item(FlairRecogniser)
    registry.add_item(SpacyRecogniser)
    registry.add_item(StanzaRecogniser)

    return registry


registry: Registry[EntityRecogniser] = init()
