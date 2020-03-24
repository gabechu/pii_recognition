from typing import List

import stanza

from label.label_schema import SpanLabel
from stanza import Pipeline

from .entity_recogniser import EntityRecogniser


class StanzaEn(EntityRecogniser):
    def __init__(self, supported_entities: List[str]):
        super().__init__(supported_entities=supported_entities, supported_languages=['en'])

    def load_model(self) -> Pipeline:
        return Pipeline('en')

    def analyse(self, text: str, entities: List[str]) -> List[SpanLabel]:
        results = self._model(text)

        span_labels = []
        for entity in results.entities:
            if entity.type in entities:
                span_labels.append(
                    SpanLabel(entity.type, entity.start_char, entity.end_char))
        return span_labels
