from typing import Dict, List

from pii_recognition.labels.schema import Entity, TokenLabel
from pii_recognition.labels.span import token_labels_to_span_labels
from pii_recognition.tokenisation import tokeniser_registry

from .entity_recogniser import EntityRecogniser


class FirstLetterUppercaseRecogniser(EntityRecogniser):
    """
    A heuristic model that check the first letter of every word/token and determine if
    that token is PERSON. Only PERSON entity is supported.
    """

    PER = "PER"

    def __init__(
        self,
        supported_entities: List[str],
        supported_languages: List[str],
        tokeniser_setup: Dict,
    ):
        self._tokeniser = tokeniser_registry.create_instance(
            tokeniser_setup["name"], tokeniser_setup.get("config")
        )

        # supported_entities argument has been kept for consistency with other
        # recognisers
        if not supported_entities == [self.PER]:
            raise ValueError(f"{type(self).__name__} only supports {self.PER} entity.")

        super().__init__(
            supported_entities=[self.PER], supported_languages=supported_languages
        )

    def analyse(self, text: str, entities: List[str]) -> List[Entity]:
        tokens = self._tokeniser.tokenise(text)
        entity_tags = [self.PER if token.text.istitle() else "O" for token in tokens]
        assert len(tokens) == len(entity_tags)

        token_labels = [
            TokenLabel(
                entity_type=entity_tags[i], start=tokens[i].start, end=tokens[i].end
            )
            for i in range(len(tokens))
        ]

        spans = token_labels_to_span_labels(token_labels)
        return [span for span in spans if span.entity_type in entities]
