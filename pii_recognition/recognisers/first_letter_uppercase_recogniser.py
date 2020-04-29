from typing import Dict, List

from pii_recognition.labels.schema import SpanLabel
from pii_recognition.labels.span import token_labels_to_span_labels
from pii_recognition.tokenisation import tokeniser_registry

from .entity_recogniser import EntityRecogniser


class FirstLetterUppercaseRecogniser(EntityRecogniser):
    """
    A heuristic model that check the first letter of every word/token and determine if
    that token is PERSON. Only PERSON entity is supported.
    """

    PER = "PER"

    def __init__(self, supported_languages: List[str], tokeniser: Dict):
        self._tokeniser = tokeniser_registry.create_instance(
            tokeniser["name"], tokeniser["config"]
        )
        super().__init__(
            supported_entities=[self.PER], supported_languages=supported_languages
        )

    def analyse(self, text: str, entities: List[str]) -> List[SpanLabel]:
        tokens = self._tokeniser.tokenise(text)
        entity_tags = [self.PER if token.text.istitle() else "O" for token in tokens]

        spans = token_labels_to_span_labels(tokens, entity_tags)
        return [span for span in spans if span.entity_type in entities]
