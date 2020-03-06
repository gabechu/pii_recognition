from typing import List

import spacy
from spacy.lang.xx import MultiLanguage

from .entity_recogniser import EntityRecogniser
from .recogniser_result import RecogniserResult


class SpacyRecogniser(EntityRecogniser):
    def __init__(
        self, supported_entities: List, supported_languages: List, model_name: str
    ):
        self._model_name = model_name
        super().__init__(
            supported_entities=supported_entities,
            supported_languages=supported_languages,
        )

    def load_model(self) -> MultiLanguage:
        return spacy.load(self._model_name, disable=["parser", "tagger"])

    def analyze(self, text: str, entities: List) -> List:
        self.validate_entities(entities)

        # TODO: validate languages
        # TODO: add support for batch

        doc = self._model(text)
        spacy_entities = [entity for entity in doc.ents]

        filtered_entities = list(filter(lambda x: x.label_ in entities, spacy_entities))

        return [
            RecogniserResult(
                entity_type=entity.label_, start=entity.start_char, end=entity.end_char
            )
            for entity in filtered_entities
        ]
