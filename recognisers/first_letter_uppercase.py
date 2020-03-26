from typing import Callable, List

from label.label_schema import SpanLabel
from label.span import token_labels_to_span_labels
from tokeniser.tokeniser import Token

from .entity_recogniser import EntityRecogniser

# TODO: add suffix recogniser
class FirstLetterUppercase(EntityRecogniser):
    """
    A heuristic model that check the first letter of every word/token and determine if
    that token is PERSON. Only PERSON entity is supported.
    """

    PER = "PER"
    default_tag = "O"  # a token belongs to no entity

    def __init__(
        self, supported_languages: List[str], tokeniser: Callable[[str], List[Token]]
    ):
        self.tokeniser = tokeniser
        super().__init__(
            supported_entities=[self.PER], supported_languages=supported_languages
        )

    def load_model(self):
        pass

    def analyse(self, text: str, entities: List[str]) -> List[SpanLabel]:
        tokens = self.tokeniser(text)
        entity_tags = [
            self.PER if token.text.istitle() else self.default_tag for token in tokens
        ]

        spans = token_labels_to_span_labels(tokens, entity_tags)
        return [span for span in spans if span.entity_type in entities]
