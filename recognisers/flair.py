from typing import List

from flair.data import Sentence
from flair.models import SequenceTagger

from label.label_schema import SpanLabel

from .entity_recogniser import EntityRecogniser


class Flair(EntityRecogniser):
    def __init__(
        self, supported_entities: List[str], supported_languages: List[str],
    ):
        super().__init__(
            supported_entities=supported_entities,
            supported_languages=supported_languages,
        )

    def load_model(self):
        return SequenceTagger.load("ner")

    def analyse(self, text: str, entities: List[str]) -> List[SpanLabel]:
        sentence = Sentence(text)
        self._model.predict(sentence)

        span_labels = []
        for entity in sentence.get_spans("ner"):
            if entity.tag in entities:
                span_labels.append(
                    SpanLabel(entity.tag, entity.start_pos, entity.end_pos)
                )

        return span_labels
