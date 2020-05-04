from typing import Dict, List

from pycrfsuite import Tagger

from pii_recognition.features.word_to_features import word2features
from pii_recognition.labels.schema import SpanLabel, TokenLabel
from pii_recognition.labels.span import token_labels_to_span_labels
from pii_recognition.tokenisation import tokeniser_registry
from pii_recognition.tokenisation.token_schema import Token
from pii_recognition.utils import cached_property

from .entity_recogniser import EntityRecogniser


class CrfRecogniser(EntityRecogniser):
    def __init__(
        self,
        supported_entities: List[str],
        supported_languages: List[str],
        model_path: str,
        tokeniser_setup: Dict,
    ):
        self._model_path = model_path
        self._tokeniser = tokeniser_registry.create_instance(
            tokeniser_setup["name"], tokeniser_setup.get("config")
        )
        super().__init__(
            supported_entities=supported_entities,
            supported_languages=supported_languages,
        )

    @cached_property
    def model(self) -> Tagger:
        tagger = Tagger()
        tagger.open(self._model_path)
        return tagger

    def preprocess_text(self, text: str) -> List[Token]:
        return self._tokeniser.tokenise(text)

    def build_features(self, tokenised_sentence: List[str]) -> List[List[str]]:
        return [
            word2features(tokenised_sentence, i) for i in range(len(tokenised_sentence))
        ]

    def analyse(self, text: str, entities: List[str]) -> List[SpanLabel]:
        self.validate_entities(entities)
        # TODO: validate languages

        preprocessed_text = self.preprocess_text(text)
        tokens = [token.text for token in preprocessed_text]
        features = self.build_features(tokens)
        entity_tags = self.model.tag(features)

        assert len(entity_tags) == len(tokens) == len(preprocessed_text)
        token_labels = [
            TokenLabel(
                entity_type=entity_tags[i],
                start=preprocessed_text[i].start,
                end=preprocessed_text[i].end,
            )
            for i in range(len(entity_tags))
        ]

        spans = token_labels_to_span_labels(token_labels)
        return [span for span in spans if span.entity_type in entities]
