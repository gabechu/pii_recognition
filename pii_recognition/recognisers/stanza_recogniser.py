from typing import List

from stanza import Pipeline

from pii_recognition.labels.schema import Entity
from pii_recognition.utils import cached_property

from .entity_recogniser import EntityRecogniser


class StanzaRecogniser(EntityRecogniser):
    """
    Stanza named entity recogniser.

    Attributes:
        supported_entities: the entities supported by this recogniser.
        supported_languages: the languages supported by this recogniser.
        model_name: pretrained NER models, more available model at
            https://stanfordnlp.github.io/stanza/models.html#available-ner-models
    """

    def __init__(
        self,
        supported_entities: List[str],
        supported_languages: List[str],
        model_name: str,
    ):
        self.model_name = model_name  # model name is defined by language code
        super().__init__(
            supported_entities=supported_entities,
            supported_languages=supported_languages,
        )

    @cached_property
    def model(self) -> Pipeline:
        # TODO: override the default location ~/stanza_resources by setting an
        # environmental variable called STANZA_RESOURCES_DIR
        return Pipeline(self.model_name)

    def analyse(self, text: str, entities: List[str]) -> List[Entity]:
        self.validate_entities(entities)

        results = self.model(text)

        span_labels = []
        for entity in results.entities:
            if entity.type in entities:
                span_labels.append(
                    Entity(entity.type, entity.start_char, entity.end_char)
                )
        return span_labels
