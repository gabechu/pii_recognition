from .crf_recogniser import CrfRecogniser
from .entity_recogniser import EntityRecogniser
from .first_letter_uppercase_recogniser import FirstLetterUppercaseRecogniser
from .flair_recogniser import FlairRecogniser
from .spacy_recogniser import SpacyRecogniser
from .stanza_recogniser import StanzaRecogniser


class RecogniserRegistry:
    def __init__(self):
        self.registry = {}

    def add_recogniser(self, recogniser: EntityRecogniser):
        # TODO: myerror
        # https://github.com/python/mypy/issues/3728
        self.registry[recogniser.__name__] = recogniser  # type: ignore

    def get_recogniser(self, name: str) -> EntityRecogniser:
        if name not in self.registry:
            raise ValueError(
                f"Recogniser not found, available recognisers are"
                f"{self.registry.keys()}"
            )

        return self.registry[name]
