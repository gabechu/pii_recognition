from typing import List

import spacy
from spacy.lang.xx import MultiLanguage

from pii_recognition.labels.schema import SpanLabel
from pii_recognition.utils import cached_property

from .entity_recogniser import EntityRecogniser


class SpacyRecogniser(EntityRecogniser):
    """
    Spacy named entity recogniser.

    Attributes:
        supported_entities: the entities supported by this recogniser.
        supported_languages: the languages supported by this recogniser.
        model_name: pretrained NER models, more available model at
            https://spacy.io/models
    """

    def __init__(
        self,
        supported_entities: List[str],
        supported_languages: List[str],
        model_name: str,
    ):
        self._model_name = model_name
        super().__init__(
            supported_entities=supported_entities,
            supported_languages=supported_languages,
        )

    @cached_property
    def model(self) -> MultiLanguage:
        return spacy.load(self._model_name, disable=["parser", "tagger"])

    def analyse(self, text: str, entities: List[str]) -> List[SpanLabel]:
        self.validate_entities(entities)

        doc = self.model(text)
        spacy_entities = [entity for entity in doc.ents]

        filtered_entities = list(filter(lambda x: x.label_ in entities, spacy_entities))

        return [
            SpanLabel(
                entity_type=entity.label_, start=entity.start_char, end=entity.end_char
            )
            for entity in filtered_entities
        ]
