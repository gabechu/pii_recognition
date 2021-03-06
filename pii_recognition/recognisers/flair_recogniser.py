from typing import List

from flair.data import Sentence
from flair.models import SequenceTagger

from pii_recognition.labels.schema import Entity
from pii_recognition.utils import cached_property

from .entity_recogniser import EntityRecogniser


class FlairRecogniser(EntityRecogniser):
    """
    Flair named entity recogniser.

    Attributes:
        supported_entities: the entities supported by this recogniser.
        supported_languages: the languages supported by this recogniser.
        model_name: pretrained NER models, more available model at
            https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_2_TAGGING.md#list-of-pre-trained-sequence-tagger-models
    """

    def __init__(
        self,
        supported_entities: List[str],
        supported_languages: List[str],
        model_name: str,
    ):
        self.model_name = model_name
        super().__init__(
            supported_entities=supported_entities,
            supported_languages=supported_languages,
        )

    @cached_property
    def model(self):
        return SequenceTagger.load(self.model_name)

    def analyse(self, text: str, entities: List[str]) -> List[Entity]:
        self.validate_entities(entities)

        sentence = Sentence(text)
        self.model.predict(sentence)

        span_labels = []
        for entity in sentence.get_spans("ner"):
            if entity.tag in entities:
                span_labels.append(Entity(entity.tag, entity.start_pos, entity.end_pos))

        return span_labels
